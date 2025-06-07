import os
import io
import pandas as pd
import traceback
import contextlib

from langchain.agents import Tool, initialize_agent
from langchain.agents.agent_types import AgentType
from langchain_openai import AzureChatOpenAI

# --- Python Tool to execute EDA instructions ---
def python_tool_func(code: str) -> str:
    """Executes Python code against df loaded from stock_data.csv"""
    try:
        if not os.path.exists("stock_data.csv"):
            return "❌ File 'stock_data.csv' not found."

        df = pd.read_csv("stock_data.csv")
        global_vars = {"df": df, "pd": pd}
        with io.StringIO() as buf, contextlib.redirect_stdout(buf):
            exec(code, global_vars)
            return buf.getvalue().strip() or "✅ Code ran but returned no output."
    except Exception as e:
        return f"❌ Error: {e}\n{traceback.format_exc()}"

# --- Tool declaration for the agent ---
python_tool = Tool(
    name="python_tool",
    func=python_tool_func,
    description=(
        "Executes Python code on a time-series stock dataframe called `df`. "
        "Useful for df.describe(), df.info(), missing value checks, correlations, etc. No plots. Text output only."
    )
)

# --- Azure LLM configuration ---
llm = AzureChatOpenAI(
    openai_api_key=os.environ["AZURE_OPENAI_API_KEY"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    deployment_name=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
    openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    temperature=0,
)

# --- Agent initialization ---
agent = initialize_agent(
    tools=[python_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)

# --- Run the EDA ---
if __name__ == "__main__":
    if not os.path.exists("stock_data.csv"):
        print("❗ Please provide 'stock_data.csv' in current directory.")
        exit()

    print("\n🚀 Starting ReAct-style EDA...\n")
    result = agent.invoke({"input": "Do full exploratory data analysis (EDA) on the stock data. Summarize insights clearly, don't show raw numbers unless needed."})

    print("\n✅ Final EDA Summary:\n")
    print(result["output"])
