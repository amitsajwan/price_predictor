# --- Imports ---
from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional, Dict, Any
from your_tool_setup import run_tool  # Custom tool wrapper function
from your_llm_setup import azure_model  # LLM with tool use enabled

# --- Define State ---
class EDAState(TypedDict):
    eda_prompt: Optional[str]
    eda_result: Optional[Dict[str, Any]]

# --- Wrap run_tool as LangChain tool ---
from langchain.tools import Tool

python_tool = Tool(
    name="python_tool",
    func=lambda code: run_tool(instructions=code, context={"df_name": "df"}),
    description="Executes Python code for data exploration, using `df` as the input DataFrame."
)

# --- Define ReAct Prompt Template ---
EDA_REACT_PROMPT = '''
You are an expert data scientist conducting EDA (Exploratory Data Analysis) for a stock price prediction problem.

You are given a pandas DataFrame named `df` with columns like:
- 'Date', 'Open', 'High', 'Low', 'Close', 'Volume'

The 'Date' column is already a datetime index, sorted chronologically.

Your task is to:
1. Analyze columns (type, nulls, stats)
2. Find patterns (correlation, seasonality, autocorr)
3. Suggest improvements (feature engineering ideas)

You can call Python code using the tool like:
```python
df.describe()
df.isnull().sum()
df.corr()
df['Close'].autocorr(lag=1)
df.index.to_series().diff().mode()
```

Think and act step-by-step.
When finished, return a JSON object like:
```json
{
  "eda_summary": {
    "columns": {
      "Close": {
        "autocorr_lag_1": 0.82,
        "notes": "Strong autocorrelation"
      },
      "Volume": {
        "corr_with_Close": -0.68,
        "notes": "Negatively correlated"
      }
    },
    "time": {
      "indexed": true,
      "sorted": true,
      "start": "2019-01-01",
      "end": "2024-06-01",
      "frequency": "business_day"
    }
  },
  "recommendations": [
    "Use lag features",
    "Add rolling statistics",
    "Include day-of-week and month",
    "Explore RSI and EMA"
  ]
}
```
Return only the final JSON.
'''

# --- Define Node Function ---
def run_eda_llm_step(state: EDAState) -> EDAState:
    response = azure_model.invoke(EDA_REACT_PROMPT, tools=[python_tool])
    return {"eda_prompt": EDA_REACT_PROMPT, "eda_result": response}

# --- Build LangGraph ---
graph = StateGraph(EDAState)
graph.add_node("run_eda", run_eda_llm_step)
graph.set_entry_point("run_eda")
graph.set_finish_point("run_eda")
EDA_GRAPH = graph.compile()

# --- Run It ---
result = EDA_GRAPH.invoke({})
print(result["eda_result"])
