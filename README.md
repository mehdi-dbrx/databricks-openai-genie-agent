# Databricks GenieAgent for OpenAI Integration

## Overview

This project adds GenieAgent support to the OpenAI integration package.  
The original Databricks AI Bridge only provided GenieAgent for LangChain users.

## Why This Was Needed

The original AI Bridge had:
```python
# Only available in databricks-langchain
from databricks_langchain import GenieAgent
```

But no equivalent for OpenAI SDK users. This fills that gap.

## Implementation

### Files Created

1. **`genie_agent.py`** - Core GenieAgent class with OpenAI tool integration
2. **`agent.py`** - Modified agent code that includes GenieAgent as a tool

### Key Changes to Original Agent Code



## Usage

Add these lines after GenieAgent creation:

```python
# Add GenieAgent to tools
genie_tool_info = ToolInfo(
    name=genie_agent.tool["function"]["name"],
    spec=genie_agent.tool,
    exec_fn=genie_agent.execute
)
TOOL_INFOS.append(genie_tool_info)
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

