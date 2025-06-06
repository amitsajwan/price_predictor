import os
import re
import json
import random
from typing import List, Dict, Optional, Literal, Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field # Use pydantic_v1 for compatibility
from langchain.agents.format_scratchpad import format_to_openai_tool_messages
from langchain.agents.output_parsers.tools import OpenAIToolsAgentOutputParser
from langchain.agents import AgentExecutor
from langchain_openai import AzureChatOpenAI
from langchain_core.tools import tool

from langgraph.graph import StateGraph, END

# --- 1. Define the State with Pydantic for Cleaner Access & Grouping ---
# NOTE: Using pydantic_v1's BaseModel from langchain_core is a good practice
# for ensuring compatibility within the LangChain ecosystem.

class EDAState(BaseModel):
    report: Optional[Dict[str, Any]] = None
    model_suggestions: List[str] = Field(default_factory=list)
    fe_suggestions: List[str] = Field(default_factory=list)
    final_cleaned_train_ref: Optional[str] = None
    final_cleaned_val_ref: Optional[str] = None
    final_cleaned_test_ref: Optional[str] = None

class FEState(BaseModel):
    applied_steps_summary: Optional[str] = None
    final_feature_list: List[str] = Field(default_factory=list)
    numerical_features: List[str] = Field(default_factory=list)
    categorical_features: List[str] = Field(default_factory=list)
    X_train_ref: Optional[str] = None
    y_train_ref: Optional[str] = None
    X_val_ref: Optional[str] = None
    y_val_ref: Optional[str] = None
    X_test_ref: Optional[str] = None

class ModelingState(BaseModel):
    top_model_configurations: List[Dict[str, Any]] = Field(default_factory=list)
    decision_rationale: Optional[str] = None
    training_summary: Optional[str] = None
    best_model_ref_so_far: Optional[str] = None
    best_rmse_so_far: float = float('inf')
    best_model_config_so_far: Optional[Dict[str, Any]] = None
    modeling_config_index: int = 0
    max_modeling_configs_to_try: int = 2
    strategy_log: List[str] = Field(default_factory=list)

class EvaluationState(BaseModel):
    summary: Optional[str] = None
    metrics: Dict[str, float] = Field(default_factory=dict)
    test_set_prediction_status: Optional[str] = None

# Main State Class
class MultiAgentPipelineState(BaseModel):
    data_paths: Dict[str, str]
    target_column_name: str
    problem_type: Literal["classification", "regression"]
    target_rmse: float
    max_react_iterations: int = 8
    eda: EDAState = Field(default_factory=EDAState)
    feature_engineering: FEState = Field(default_factory=FEState)
    modeling: ModelingState = Field(default_factory=ModelingState)
    evaluation: EvaluationState = Field(default_factory=EvaluationState)
    current_stage_completed: Optional[str] = None

# --- 2. Define the LangChain Tool (Implementation Abstracted) ---

@tool
def python_data_science_tool(instruction: str) -> str:
    """
    Executes a Python script for data science tasks like EDA, feature engineering, and modeling.
    Takes a natural language instruction for a data-related task.
    Handles data loading, transformation, analysis, model training (Scikit-learn pipelines),
    and evaluation. Returns a string observation, which MUST include references (filenames)
    to artifacts produced, like datasets (.pkl), models (.joblib), or plots (.png).
    """
    print(f"\n   [TOOL EXECUTED] Instruction: '{instruction}'")
    # In a real-world scenario, this function would contain the complex code to
    # interpret the instruction and execute it.
    return "Tool execution acknowledged. The agent will now process the imagined result."

# --- 3. Define the LLM and Agent Executor (Latest Pattern) ---

# Set up the Azure Chat Model
# IMPORTANT: Ensure your AZURE_OPENAI_API_KEY environment variable is set.
llm = AzureChatOpenAI(
    temperature=0,
    openai_api_version="2024-05-01-preview", # Use a recent, valid API version
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "your-deployment-name"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", "https://your-endpoint.openai.azure.com"),
    max_retries=3,
)

tools = [python_data_science_tool]

# This is the modern way to make the LLM aware of tools.
# It creates a new model object that knows how to call the provided tools.
llm_with_tools = llm.bind_tools(tools)

# The prompt now uses a structured list of messages and a placeholder for intermediate steps.
# This is a more robust pattern than manually formatting a text string.
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a powerful and methodical data science assistant. You must use the provided tools to answer the user's question. Think step-by-step and produce a final JSON answer when you are done."),
        ("user", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

# This is the modern way to construct the agent runnable using LCEL.
agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_to_openai_tool_messages(
            x["intermediate_steps"]
        ),
    }
    | prompt
    | llm_with_tools
    | OpenAIToolsAgentOutputParser()
)

# The AgentExecutor remains the runtime for our agent.
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True # Crucial for robustness
)


def parse_llm_json_final_answer(final_answer_json_string: str, default_error_message: str = "Report generation failed.") -> Dict:
    try:
        match = re.search(r"```json\s*(.*?)\s*```", final_answer_json_string, re.DOTALL)
        json_string_cleaned = match.group(1).strip() if match else final_answer_json_string.strip()
        if not json_string_cleaned:
            return {"error": "LLM returned an empty JSON string.", "summary": default_error_message}
        return json.loads(json_string_cleaned)
    except Exception as e:
        print(f"   ERROR parsing LLM JSON: {e}, String: '{final_answer_json_string}'")
        return {"error": str(e), "summary": default_error_message}

# --- 4. Define Agent Nodes ---

def eda_agent_node(state: MultiAgentPipelineState) -> Dict:
    print("\n--- EDA Agent Node ---")
    input_prompt = f"""
    Perform a full EDA on the data specified in these paths: {state.data_paths}.
    The target column is '{state.target_column_name}'. The problem type is '{state.problem_type}'.
    Your tasks:
    1. Load the datasets.
    2. Perform analysis: check missing values, data types, etc.
    3. Generate suggestions for feature engineering and model types.
    4. Clean the data (handle dates, simple imputation) and report the new file references.

    Your Final Answer must be a single JSON object with the keys: "eda_summary", "fe_suggestions", "model_suggestions", "artifact_references": {{"final_cleaned_train_data": "<ref.pkl>", "final_cleaned_val_data": "<ref.pkl>", "final_cleaned_test_data": "<ref.pkl>"}}.
    """
    response = agent_executor.invoke({"input": input_prompt})
    # In the latest versions, the agent's final answer is in the 'output' key.
    parsed_output = parse_llm_json_final_answer(response.get("output", "{}"))

    state.eda.report = parsed_output
    state.eda.fe_suggestions = parsed_output.get("fe_suggestions", [])
    state.eda.model_suggestions = parsed_output.get("model_suggestions", [])
    artifacts = parsed_output.get("artifact_references", {})
    state.eda.final_cleaned_train_ref = artifacts.get("final_cleaned_train_data")
    state.eda.final_cleaned_val_ref = artifacts.get("final_cleaned_val_data")
    state.eda.final_cleaned_test_ref = artifacts.get("final_cleaned_test_data")
    state.current_stage_completed = "EDA"
    return state.model_dump()


def feature_engineering_agent_node(state: MultiAgentPipelineState) -> Dict:
    print("\n--- Feature Engineering Agent Node ---")
    input_prompt = f"""
    Prepare the final features for modeling based on the EDA.
    - Cleaned training data: '{state.eda.final_cleaned_train_ref}'
    - Target column: '{state.target_column_name}'
    - EDA suggestions: {state.eda.fe_suggestions}
    Tasks:
    1. Load cleaned data.
    2. Apply feature engineering suggestions.
    3. Separate features (X) from target (y).
    4. Save final datasets (X_train, y_train, etc.) and report their references.
    5. List all numerical and categorical feature names.

    Your Final Answer must be a JSON object with keys: "fe_summary", "final_feature_list", "numerical_features", "categorical_features", and "data_references": {{"X_train": "<ref.pkl>", ...}}.
    """
    response = agent_executor.invoke({"input": input_prompt})
    parsed_output = parse_llm_json_final_answer(response.get("output", "{}"))

    fe = state.feature_engineering
    fe.applied_steps_summary = parsed_output.get("fe_summary")
    fe.final_feature_list = parsed_output.get("final_feature_list", [])
    fe.numerical_features = parsed_output.get("numerical_features", [])
    fe.categorical_features = parsed_output.get("categorical_features", [])
    data_refs = parsed_output.get("data_references", {})
    fe.X_train_ref = data_refs.get("X_train")
    fe.y_train_ref = data_refs.get("y_train")
    fe.X_val_ref = data_refs.get("X_val")
    fe.y_val_ref = data_refs.get("y_val")
    fe.X_test_ref = data_refs.get("X_test")
    state.current_stage_completed = "FeatureEngineering"
    return state.model_dump()


def model_selection_decision_agent_node(state: MultiAgentPipelineState) -> Dict:
    print("\n--- Model Selection Decision Agent Node ---")
    fe = state.feature_engineering
    input_prompt = f"""
    Decide on 2-3 promising Scikit-learn model configurations for this regression task based on the context:
    - EDA Model Suggestions: {state.eda.model_suggestions}
    - Numerical features: {len(fe.numerical_features)}, Categorical features: {len(fe.categorical_features)}

    Your Final Answer must be a JSON object with keys: "decision_rationale" and "top_model_configurations": [{{ "model_type": "e.g., RandomForestRegressor", ... }}].
    """
    response = agent_executor.invoke({"input": input_prompt})
    parsed_output = parse_llm_json_final_answer(response.get("output", "{}"))

    state.modeling.decision_rationale = parsed_output.get("decision_rationale")
    state.modeling.top_model_configurations = parsed_output.get("top_model_configurations", [])
    state.current_stage_completed = "ModelSelectionDecision"
    return state.model_dump()


def modeling_agent_node(state: MultiAgentPipelineState) -> Dict:
    print(f"\n--- Modeling Agent Node (Trying Config #{state.modeling.modeling_config_index}) ---")
    fe = state.feature_engineering
    m_state = state.modeling
    config_idx = m_state.modeling_config_index
    current_config = m_state.top_model_configurations[config_idx]
    model_type = current_config.get("model_type")
    hyperparams = current_config.get("initial_hyperparameters", {})

    input_prompt = f"""
    Execute one modeling trial.
    - Configuration: Train a '{model_type}' with hyperparameters: {json.dumps(hyperparams)}.
    - Features: Use these numerical features for StandardScaler ({fe.numerical_features}) and these for OneHotEncoder ({fe.categorical_features}).
    - Data: Use X_train '{fe.X_train_ref}', y_train '{fe.y_train_ref}', X_val '{fe.X_val_ref}', and y_val '{fe.y_val_ref}'.
    Tasks:
    1. Create an untrained Scikit-learn pipeline (ColumnTransformer + estimator).
    2. Train the entire pipeline.
    3. Save the TRAINED pipeline and report its new reference.
    4. Evaluate on validation data and report the RMSE.

    Your Final Answer must be a JSON object with keys: "config_trial_summary", "config_trained_pipeline_ref", and "config_rmse".
    """
    response = agent_executor.invoke({"input": input_prompt})
    parsed_output = parse_llm_json_final_answer(response.get("output", "{}"))

    current_rmse = parsed_output.get("config_rmse")
    trained_ref = parsed_output.get("config_trained_pipeline_ref")
    m_state.strategy_log.append(f"Trial {config_idx} ({model_type}): RMSE={current_rmse}, Ref={trained_ref}")

    if current_rmse and isinstance(current_rmse, (int, float)) and current_rmse < m_state.best_rmse_so_far:
        print(f"   New best RMSE found: {current_rmse} (previous: {m_state.best_rmse_so_far})")
        m_state.best_rmse_so_far = current_rmse
        m_state.best_model_ref_so_far = trained_ref
        m_state.best_model_config_so_far = current_config

    m_state.modeling_config_index += 1
    state.current_stage_completed = "Modeling_Config_Trial"
    return state.model_dump()


def evaluation_node(state: MultiAgentPipelineState) -> Dict:
    print("\n--- Evaluation Agent Node ---")
    input_prompt = f"""
    Perform a final evaluation of the best model.
    - Best Trained Pipeline Reference: '{state.modeling.best_model_ref_so_far}'
    - Test Data Reference: '{state.feature_engineering.X_test_ref}'
    Task: Load the pipeline, predict on the test set, and calculate final performance metrics (RMSE, R-squared).

    Your Final Answer must be a JSON with keys: "evaluation_summary" and "final_metrics": {{"rmse": 0.123, ...}}.
    """
    response = agent_executor.invoke({"input": input_prompt})
    parsed_output = parse_llm_json_final_answer(response.get("output", "{}"))

    state.evaluation.summary = parsed_output.get("evaluation_summary")
    state.evaluation.metrics = parsed_output.get("final_metrics", {})
    state.current_stage_completed = "Evaluation"
    return state.model_dump()


# --- 5. Graph Definition and Conditional Logic ---

def modeling_iteration_decision(state_dict: dict) -> str:
    # LangGraph passes state as a dict, so we convert it back to our Pydantic model
    state = MultiAgentPipelineState.model_validate(state_dict)
    print("\n--- Checking Modeling Iteration Decision ---")
    m_state = state.modeling

    if not m_state.top_model_configurations:
        print("   No model configurations to try. Skipping to evaluation.")
        return "evaluation_agent"
    if m_state.best_rmse_so_far <= state.target_rmse:
        print(f"   Target RMSE ({state.target_rmse}) achieved. Skipping to evaluation.")
        return "evaluation_agent"
    if m_state.modeling_config_index >= len(m_state.top_model_configurations) or m_state.modeling_config_index >= m_state.max_modeling_configs_to_try:
        print("   All model configurations tried or limit reached. Proceeding to evaluation.")
        return "evaluation_agent"

    print(f"   Proceeding to try model configuration at index {m_state.modeling_config_index}.")
    return "modeling_agent"


workflow = StateGraph(MultiAgentPipelineState)

workflow.add_node("eda_agent", eda_agent_node)
workflow.add_node("feature_engineering_agent", feature_engineering_agent_node)
workflow.add_node("model_selection_decision_agent", model_selection_decision_agent_node)
workflow.add_node("modeling_agent", modeling_agent_node)
workflow.add_node("evaluation_agent", evaluation_node)

workflow.set_entry_point("eda_agent")
workflow.add_edge("eda_agent", "feature_engineering_agent")
workflow.add_edge("feature_engineering_agent", "model_selection_decision_agent")
workflow.add_edge("model_selection_decision_agent", "modeling_agent")
workflow.add_conditional_edges(
    "modeling_agent",
    modeling_iteration_decision,
    {
        "modeling_agent": "modeling_agent",
        "evaluation_agent": "evaluation_agent"
    }
)
workflow.add_edge("evaluation_agent", END)

pipeline_app = workflow.compile()

# --- 6. Example Invocation ---
if __name__ == "__main__":
    print("Starting ML Pipeline with latest LangChain and Azure OpenAI...")

    initial_data_paths = {
        "train": "train_clean.csv",
        "val": "val_clean.csv",
        "test": "test_clean.csv"
    }

    print("Creating dummy input files...")
    for path in initial_data_paths.values():
        with open(path, "w") as f:
            f.write("Date,Feature1,Feature2,Target\n")
            f.write("2025-06-07,10,20,15\n")
            f.write("2025-06-08,12,22,17\n")

    initial_pipeline_state = {
        "data_paths": initial_data_paths,
        "target_column_name": "Target",
        "problem_type": "regression",
        "target_rmse": 0.85,
        "modeling": {"max_modeling_configs_to_try": 2}
    }

    config = {"configurable": {"thread_id": f"ml_pipeline_run_{random.randint(1000, 9999)}"}}

    # In latest langgraph, the stream provides chunks with the node name as the key
    for chunk in pipeline_app.stream(initial_pipeline_state, config=config):
        for node_name, node_output in chunk.items():
            print(f"\n<<< Update from Node: {node_name} >>>")
            # The node_output is the full state dictionary after the node has run
            print(json.dumps(node_output, indent=2, default=str))

    # To get the final state, you can invoke the graph
    print("\n\n--- Final Pipeline State ---")
    final_state = pipeline_app.invoke(initial_pipeline_state, config=config)
    print(json.dumps(final_state, indent=2, default=str))


    print("\nCleaning up dummy files...")
    for path in initial_data_paths.values():
        if os.path.exists(path):
            os.remove(path)

    print("\nMulti-Agent Pipeline Finished.")
