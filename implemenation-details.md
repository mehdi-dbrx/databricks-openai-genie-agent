# GenieAgent: OpenAI vs LangChain Implementation Differences

## Key Differences

### **Return Format**
**LangChain:**
```python
# Returns LangChain messages
return {"messages": [AIMessage(content=query_result, name="query_result")]}
```

**OpenAI:**
```python
# Returns JSON string
return json.dumps(result_dict, indent=2)
```

### **Tool Integration**
**LangChain:**
```python
# Returns LangChain Runnable
runnable = RunnableLambda(partial_genie_agent)
return runnable
```

**OpenAI:**
```python
# Returns OpenAI tool specification
return {
    "type": "function",
    "function": {
        "name": self.genie_agent_name.lower().replace(" ", "_"),
        "description": self.description,
        "parameters": GenieAgentInput.model_json_schema()
    }
}
```

### **Input Handling**
**LangChain:**
```python
# Processes message arrays
def _query_genie_as_agent(input, genie, genie_agent_name, ...):
    messages = input.get("messages", [])
    query = _concat_messages_array(messages)
```

**OpenAI:**
```python
# Processes single query string
def _query_genie_as_agent(query: str, genie, genie_agent_name, ...):
    # Direct query processing
```

### **Schema Definition**
**LangChain:**
- No explicit input/output schemas
- Uses LangChain message objects

**OpenAI:**
```python
class GenieAgentInput(BaseModel):
    query: str
    conversation_id: Optional[str] = None

class GenieAgentOutput(BaseModel):
    result: str
    query_reasoning: Optional[str] = None
    sql_query: Optional[str] = None
    conversation_id: Optional[str] = None
```

### **Usage Pattern**
**LangChain:**
```python
# Used as LangChain Runnable
agent = GenieAgent("space-123")
result = agent.invoke({"messages": [HumanMessage("query")]})
```

**OpenAI:**
```python
# Used as OpenAI tool
genie_agent = GenieAgent("space-123")
result = genie_agent.execute(query="query")
```

