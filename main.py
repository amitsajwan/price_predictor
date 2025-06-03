import os
import asyncio
import nest_asyncio
import pandas as pd
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from openai import AsyncAzureOpenAI
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib

# Configure Azure OpenAI client (ensure these env vars are set)
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
azure_key = os.getenv("AZURE_OPENAI_API_KEY")
deployment_name = os.getenv("AZURE_OPENAI_CHATGPT_DEPLOYMENT")
client = AsyncAzureOpenAI(
    api_key=azure_key,
    azure_endpoint=azure_endpoint,
    api_version="2024-02-01"
)

# Define the workflow state schema
class WorkflowState(TypedDict):
    raw_data_path: str
    preliminary_insights: str
    eda_insights: str
    features_data_path: str
    features_insights: str
    model_path: str
    rmse: float
    final_summary: str

# Stage 1: Preliminary Analysis Node
async def preliminary_analysis(state: WorkflowState) -> dict:
    df = pd.read_csv(state["raw_data_path"])
    n_rows, n_cols = df.shape
    missing_vals = {col: int(df[col].isna().sum()) for col in df.columns}
    prompt = (
        f"Dataset has {n_rows} rows and {n_cols} columns. "
        f"Missing values per column: {missing_vals}. "
        "Provide a brief preliminary data analysis summary."
    )
    response = await client.chat.completions.create(
        model=deployment_name,
        messages=[
            {"role": "system", "content": "You are an AI data scientist assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    prelim_text = response.choices[0].message.content
    # Write analysis report to disk
    with open("preliminary_analysis.txt", "w") as f:
        f.write(prelim_text)
    return {"preliminary_insights": prelim_text}

# Stage 2: Exploratory Data Analysis (EDA) Node
async def exploratory_data_analysis(state: WorkflowState) -> dict:
    df = pd.read_csv(state["raw_data_path"])
    desc = df.describe().to_dict()
    top_corr = []
    if "target" in df.columns:
        corr = df.corr()["target"].drop("target", errors='ignore').to_dict()
        # Select top 3 correlated features
        top_corr = sorted(corr.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
    prompt = (
        f"Basic statistics: {desc}. "
        f"Top feature correlations with target: {top_corr}. "
        "Provide key insights from the EDA."
    )
    response = await client.chat.completions.create(
        model=deployment_name,
        messages=[
            {"role": "system", "content": "You are an AI data analysis assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    eda_text = response.choices[0].message.content
    with open("eda_report.txt", "w") as f:
        f.write(eda_text)
    return {"eda_insights": eda_text}

# Stage 3: Feature Engineering Node
async def feature_engineering(state: WorkflowState) -> dict:
    df = pd.read_csv(state["raw_data_path"])
    # Example: create square of the first feature
    feature_cols = [col for col in df.columns if col != "target"]
    new_cols = []
    if feature_cols:
        col = feature_cols[0]
        new_col = f"{col}_squared"
        df[new_col] = df[col] ** 2
        new_cols.append(new_col)
    features_path = "features_data.csv"
    df.to_csv(features_path, index=False)
    prompt = (
        f"Created new feature columns: {new_cols} from the original features. "
        "Explain the purpose of these engineered features."
    )
    response = await client.chat.completions.create(
        model=deployment_name,
        messages=[
            {"role": "system", "content": "You are an AI feature engineering assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    feat_text = response.choices[0].message.content
    with open("feature_engineering_report.txt", "w") as f:
        f.write(feat_text)
    return {"features_data_path": features_path, "features_insights": feat_text}

# Stage 4: Modeling Node
async def modeling(state: WorkflowState) -> dict:
    df = pd.read_csv(state["features_data_path"])
    X = df.drop(columns=["target"])
    y = df["target"]
    prompt = (
        f"We have {X.shape[1]} input features to predict the target. "
        "Which regression model should we try first?"
    )
    response = await client.chat.completions.create(
        model=deployment_name,
        messages=[
            {"role": "system", "content": "You are an AI modeling assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    suggestion = response.choices[0].message.content.lower()
    # Choose a model based on suggestion
    if "forest" in suggestion or "trees" in suggestion:
        model = RandomForestRegressor(random_state=42)
    else:
        model = LinearRegression()
    # Train the model
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model.fit(X_train, y_train)
    model_path = "model.pkl"
    joblib.dump(model, model_path)
    return {"model_path": model_path}

# Stage 5: Evaluation Node (compute RMSE and generate final summary)
async def evaluation(state: WorkflowState) -> dict:
    df = pd.read_csv(state["features_data_path"])
    X = df.drop(columns=["target"])
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = joblib.load(state["model_path"])
    preds = model.predict(X_test)
    rmse = mean_squared_error(y_test, preds, squared=False)
    threshold = 0.3
    if rmse > threshold:
        # Ask LLM for suggestions when performance is poor
        prompt = (
            f"The model achieved RMSE = {rmse:.3f}, above the threshold. "
            "What could we try next?"
        )
        response = await client.chat.completions.create(
            model=deployment_name,
            messages=[
                {"role": "system", "content": "You are an AI modeling advisor."},
                {"role": "user", "content": prompt}
            ]
        )
        with open("evaluation_interim.txt", "w") as f:
            f.write(response.choices[0].message.content)
        return {"rmse": rmse}
    else:
        # Generate final summary
        prompt = (
            f"Preliminary insights: {state['preliminary_insights']} "
            f"EDA insights: {state['eda_insights']} "
            f"Feature insights: {state['features_insights']} "
            f"Final model RMSE: {rmse:.3f}. "
            "Provide a concise summary of the findings."
        )
        response = await client.chat.completions.create(
            model=deployment_name,
            messages=[
                {"role": "system", "content": "You are an AI summarization assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        final_text = response.choices[0].message.content
        with open("final_summary.txt", "w") as f:
            f.write(final_text)
        return {"rmse": rmse, "final_summary": final_text}

# Build the LangGraph state machine
builder = StateGraph(WorkflowState)
builder.add_node("preliminary_analysis", preliminary_analysis)
builder.add_node("eda", exploratory_data_analysis)
builder.add_node("feature_engineering", feature_engineering)
builder.add_node("modeling", modeling)
builder.add_node("evaluation", evaluation)

# Define edges between stages
builder.add_edge(START, "preliminary_analysis")
builder.add_edge("preliminary_analysis", "eda")
builder.add_edge("eda", "feature_engineering")
builder.add_edge("feature_engineering", "modeling")
builder.add_edge("modeling", "evaluation")

# Conditional loop: if RMSE > 0.3, go back to modeling; else end.
def route_fn(state: WorkflowState):
    rmse_val = state.get("rmse", float("inf"))
    return "modeling" if rmse_val > 0.3 else END

builder.add_conditional_edges("evaluation", route_fn)
graph = builder.compile()

# Execution entry point with dummy data
if __name__ == "__main__":
    nest_asyncio.apply()
    # Create synthetic regression data for demonstration
    from sklearn.datasets import make_regression
    X, y = make_regression(n_samples=100, n_features=5, noise=0.2, random_state=1)
    df_raw = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    df_raw["target"] = y
    df_raw.to_csv("raw_data.csv", index=False)
    # Initialize state with dummy values
    initial_state = WorkflowState(
        raw_data_path="raw_data.csv",
        preliminary_insights="",
        eda_insights="",
        features_data_path="",
        features_insights="",
        model_path="",
        rmse=0.0,
        final_summary=""
    )
    # Run the workflow
    result = asyncio.run(graph.ainvoke(initial_state))
    print("Pipeline completed. Final summary:\n", result.get("final_summary", ""))
