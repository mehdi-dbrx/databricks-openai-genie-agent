# Databricks GenieAgent for OpenAI Integration

## Overview

This project adds GenieAgent support to the OpenAI integration package.  
The original Databricks AI Bridge only provided GenieAgent for LangChain users.

The original AI Bridge had:
```python
# Only available in databricks-langchain
from databricks_langchain import GenieAgent
```

But no equivalent for OpenAI SDK users. This fills that gap.

### Core File :

**`genie_agent.py`** - Core GenieAgent class with OpenAI tool integration

## Usage


Add the **`genie_agent.py`** to your project 

import it in your agent.py

```python
from genie_agent import GenieAgent
```

Instanciate youre Genie Agent

```python
from genie_agent import GenieAgent

GENIE_SPACE_ID = "your-genie-space-id"
genie_agent_description = "Teva Genie"

genie_agent = GenieAgent(
    genie_space_id=GENIE_SPACE_ID,
    genie_agent_name="Genie",
    description=genie_agent_description,
    client=WorkspaceClient(
        host=os.getenv("DB_MODEL_SERVING_HOST_URL"),
        token=os.getenv("DATABRICKS_GENIE_PAT"),
    ),
)
```

Add your Genie Agent to your toolset

```python
# Add GenieAgent to tools
genie_tool_info = ToolInfo(
    name=genie_agent.tool["function"]["name"],
    spec=genie_agent.tool,
    exec_fn=genie_agent.execute
)
TOOL_INFOS.append(genie_tool_info)
```

Test your agent :  
Send a request that will trigger the Genie Agent

```python
from agent import AGENT

AGENT.predict({"input": [{"role": "user", "content": "what is the NAMCS dataset about"}]})
```

### Prerequisites

```python
%pip install openai pydantic databricks-ai-bridge databricks-sdk mlflow
```

### Configuration

Modify these in `agent.py`:
```python
LLM_ENDPOINT_NAME = "databricks-claude-sonnet-4"
GENIE_SPACE_ID = "your-space-id-here"
genie_agent_description = "Your Genie Description"
```


## GenieAgent Implementation

### Core Components

#### GenieAgent Class
- **Input/Output Schemas**: Uses Pydantic models (`GenieAgentInput`, `GenieAgentOutput`) for type safety
- **Tool Definition**: Creates OpenAI function calling spec with `_create_tool()`
- **Execution**: `execute()` method calls Genie API and returns JSON string
- **Conversation Management**: Supports conversation IDs for context continuity

#### Key Methods

```python
def _create_tool(self) -> Dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": self.genie_agent_name.lower().replace(" ", "_"),
            "description": self.description,
            "parameters": GenieAgentInput.model_json_schema()
        }
    }

def execute(self, query: str, conversation_id: Optional[str] = None) -> str:
    # Calls Genie API and returns structured JSON response
```

### Key Features

- **OpenAI Compatible**: Uses OpenAI function calling specification
- **Structured Output**: JSON responses with result, SQL, reasoning
- **Error Handling**: Catches exceptions and returns error messages
- **MLflow Tracing**: Integrated with MLflow for observability

