# genie_agent.py
import json
import logging
from typing import Any, Callable, Dict, List, Optional, Union

import mlflow
from databricks.sdk import WorkspaceClient
from databricks_ai_bridge.genie import Genie, GenieResponse
from openai import OpenAI
from pydantic import BaseModel, Field

_logger = logging.getLogger(__name__)

class GenieAgentInput(BaseModel):
    """Input schema for GenieAgent tool."""
    
    query: str = Field(
        description="The natural language query to ask the Genie agent about the data"
    )
    conversation_id: Optional[str] = Field(
        default=None,
        description="Optional conversation ID to continue an existing conversation"
    )

class GenieAgentOutput(BaseModel):
    """Output schema for GenieAgent tool."""
    
    result: str = Field(description="The response from the Genie agent")
    query_reasoning: Optional[str] = Field(
        default=None, 
        description="The reasoning behind the query (if include_context=True)"
    )
    sql_query: Optional[str] = Field(
        default=None,
        description="The SQL query generated (if include_context=True)"
    )
    conversation_id: Optional[str] = Field(
        default=None,
        description="The conversation ID for continuing the conversation"
    )

@mlflow.trace()
def _concat_messages_array(messages: List[Dict[str, Any]]) -> str:
    """Concatenate OpenAI messages into a single query string."""
    concatenated_message = "\n".join([
        f"{message.get('role', 'unknown')}: {message.get('content', '')}"
        for message in messages
    ])
    return concatenated_message

@mlflow.trace()
def _query_genie_as_agent(
    query: str,
    genie: Genie,
    genie_agent_name: str,
    include_context: bool = False,
    conversation_id: Optional[str] = None,
    message_processor: Optional[Callable] = None,
) -> GenieAgentOutput:
    """Query Genie as an agent and return structured output."""
    
    # Apply message processor if provided
    if message_processor:
        processed_query = message_processor(query)
    else:
        processed_query = f"I will provide you a query as {genie_agent_name}. Please help with the described information.\nQuery: {query}"

    # Send the message and wait for a response
    genie_response = genie.ask_question(processed_query, conversation_id=conversation_id)

    query_reasoning = genie_response.description or ""
    query_sql = genie_response.query or ""
    query_result = genie_response.result or ""
    returned_conversation_id = genie_response.conversation_id

    return GenieAgentOutput(
        result=query_result,
        query_reasoning=query_reasoning if include_context else None,
        sql_query=query_sql if include_context else None,
        conversation_id=returned_conversation_id
    )

class GenieAgent:
    """
    A Genie agent that can be used as an OpenAI tool for querying Databricks data.
    """
    
    def __init__(
        self,
        genie_space_id: str,
        genie_agent_name: str = "Genie",
        description: str = "",
        include_context: bool = False,
        message_processor: Optional[Callable] = None,
        client: Optional[WorkspaceClient] = None,
    ):
        """
        Initialize the GenieAgent.
        
        Args:
            genie_space_id: The ID of the genie space to use
            genie_agent_name: Name for the agent (default: "Genie")
            description: Custom description for the agent
            include_context: Whether to include query reasoning and SQL in the response
            message_processor: Optional function to process queries before sending to Genie.
                              Should accept a string query and return a processed string.
            client: Optional WorkspaceClient instance
        """
        if not genie_space_id:
            raise ValueError("genie_space_id is required to create a GenieAgent")
        
        self.genie_space_id = genie_space_id
        self.genie_agent_name = genie_agent_name
        self.include_context = include_context
        self.message_processor = message_processor
        
        # Initialize Genie
        self.genie = Genie(genie_space_id, client=client)
        
        # Set description
        self.description = description or self.genie.description
        
        # Create the tool definition
        self.tool = self._create_tool()
    
    def _create_tool(self) -> Dict[str, Any]:
        """Create the OpenAI tool definition."""
        return {
            "type": "function",
            "function": {
                "name": self.genie_agent_name.lower().replace(" ", "_"),
                "description": self.description,
                "parameters": GenieAgentInput.model_json_schema()
            }
        }
    
    @mlflow.trace(span_type="AGENT")
    def execute(
        self,
        query: str,
        conversation_id: Optional[str] = None,
    ) -> str:
        """
        Execute a query using the Genie agent.
        
        Args:
            query: The natural language query to ask
            conversation_id: Optional conversation ID to continue an existing conversation
            
        Returns:
            JSON string containing the agent's response
        """
        try:
            result = _query_genie_as_agent(
                query=query,
                genie=self.genie,
                genie_agent_name=self.genie_agent_name,
                include_context=self.include_context,
                conversation_id=conversation_id,
                message_processor=self.message_processor,
            )
            
            # Convert to dict and then to JSON string
            result_dict = result.model_dump(exclude_none=True)
            return json.dumps(result_dict, indent=2)
            
        except Exception as e:
            _logger.error(f"Error executing Genie agent query: {e}")
            error_result = GenieAgentOutput(
                result=f"Error executing query: {str(e)}",
                conversation_id=conversation_id
            )
            return json.dumps(error_result.model_dump(exclude_none=True), indent=2)
    
    def __repr__(self) -> str:
        return f"GenieAgent(space_id='{self.genie_space_id}', name='{self.genie_agent_name}')"
