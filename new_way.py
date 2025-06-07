# langgraph_eda_pipeline.py (extended with FE and Modeling)

import os
import pandas as pd
import io
import traceback
import contextlib

from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda
from langchain_openai import AzureChatOpenAI
from langchain.agents import Tool, create_tool_calling_agent

# --- File Config ---
input_files = {
    "clean": "clean.csv",
    "val": "val.csv",
    "test": "test.csv",
}
output_files = {
    k: f"eda_{k}.csv" for k in input_files
}

# --- LangGraph State Definition ---
class EDAState(TypedDict):
    status: Literal["start", "eda_complete", "fe_complete", "model_complete"]
    summary: str

# --- Tool to run Python code across all DataFrames ---
def python_tool_func(code: str) -> str:
    try:
        dataframes = {}
        for name, path in input_files.items():
            if not os.path.exists(path):
                return f"❌ Missing file: {path}"
            df = pd.read_csv(path)
            dataframes[name] = df

        globals_ = {f"df_{name}": df for name, df in dataframes.items()}
        globals_["pd"] = pd

        with io.StringIO() as buf, contextlib.redirect_stdout(buf):
            exec(code, globals_)
            for name in dataframes:
                if f"df_{name}" in globals_:
                    globals_[f"df_{name}"].to_csv(output_files[name], index=True)
            return buf.getvalue().strip() or "✅ Code ran successfully"

    except Exception as e:
        return f"❌ Error: {e}\n{traceback.format_exc()}"

# --- Tool as LangChain Tool ---
python_tool = Tool(
    name="python_tool",
    func=python_tool_func,
    description="Run pandas code on df_clean, df_val, df_test. Clean date, describe, nulls, correlation."
)

# --- Azure Chat LLM Agent ---
llm = AzureChatOpenAI(
    openai_api_key=os.environ["AZURE_OPENAI_API_KEY"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    deployment_name=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
    openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    temperature=0,
)

agent = create_tool_calling_agent(llm, [python_tool])

# --- LangGraph Nodes ---
def run_eda(state: EDAState) -> EDAState:
    print("\n🔍 Running EDA Agent on all 3 files...")
    input_text = (
        "Perform full EDA on df_clean, df_val, df_test:\n"
        "- Convert 'Date' to datetime and set as index\n"
        "- Show shape, nulls, df.describe(), correlation\n"
        "- Save cleaned files automatically\n"
        "- Summarize all findings clearly."
    )
    result = agent.invoke({"input": input_text})
    return {"status": "eda_complete", "summary": result["output"]}

def run_fe(state: EDAState) -> EDAState:
    print("\n🛠️ Running Feature Engineering...")
    input_text = (
        "Create useful features from df_clean, df_val, df_test:\n"
        "- Add lag features, rolling means, pct changes\n"
        "- Avoid leakage, keep output same size\n"
        "- Save new versions to same eda_* files\n"
        "- Summarize features added"
    )
    result = agent.invoke({"input": input_text})
    return {"status": "fe_complete", "summary": state["summary"] + "\n\n" + result["output"]}

def run_modeling(state: EDAState) -> EDAState:
    print("\n🤖 Running Modeling...")
    input_text = (
        "Train regression model on df_clean to predict target (e.g., 'close')\n"
        "- Use features generated\n"
        "- Evaluate on df_val and df_test\n"
        "- Show RMSE, R2, summary\n"
        "- Don't do plots"
    )
    result = agent.invoke({"input": input_text})
    return {"status": "model_complete", "summary": state["summary"] + "\n\n" + result["output"]}

# --- Graph Construction ---
workflow = StateGraph(EDAState)
workflow.add_node("EDA", RunnableLambda(run_eda))
workflow.add_node("FE", RunnableLambda(run_fe))
workflow.add_node("Model", RunnableLambda(run_modeling))

workflow.set_entry_point("EDA")
workflow.add_edge("EDA", "FE")
workflow.add_edge("FE", "Model")
workflow.set_finish_point("Model")

eda_app = workflow.compile()

# --- Run pipeline ---
if __name__ == "__main__":
    for file in input_files.values():
        if not os.path.exists(file):
            print(f"❌ Required input missing: {file}")
            exit(1)

    print("\n🚀 Starting LangGraph EDA → FE → Modeling pipeline...")
    final_state = eda_app.invoke({"status": "start", "summary": ""})

    print("\n✅ Final Summary:\n")
    print(final_state["summary"])

    with open("eda_summary.txt", "w") as f:
        f.write(final_state["summary"])
        print("\n📄 Summary saved to eda_summary.txt")
