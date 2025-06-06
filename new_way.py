import os
import re
import json
import random
from typing import List, Dict, Optional, Literal, Any

from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import tool
# Correctly import AzureChatOpenAI
from langchain_openai import AzureChatOpenAI
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END

# --- 1. Define the State with Pydantic for Cleaner Access & Grouping ---

class EDAState(BaseModel):
    # Using Dict[str, Any] is often necessary for the initial, unstructured report from the LLM.
    # The robustness of the parse_llm_json_final_answer function is key to handling this.
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

# Main State Class using dot notation (e.g., state.eda.report)
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
    Key capabilities:
    - Loading data: "load the dataset from 'path/to/train.csv' and report its reference as 'train_df_ref.pkl'"
    - Data manipulation: "extract year, month, day from the 'date' column in 'df_ref.pkl' and report new reference as 'df_ref_fe.pkl'"
    - Model Pipeline: "create an untrained scikit-learn pipeline with a StandardScaler for numerical_features and OneHotEncoder for categorical_features, followed by a RandomForestRegressor. Save it and report its reference."
    - Training & Evaluation: "train the pipeline 'untrained_pipe.joblib' using X_train 'X_train.pkl' and y_train 'y_train.pkl'. Report the trained pipeline reference and the validation RMSE using X_val and y_val."
    """
    print(f"\n   [TOOL EXECUTED] Instruction: '{instruction}'")
    # In a real-world scenario, this function would contain the complex code to
    # interpret the instruction and execute it. For this example, we assume
    # the LLM will hallucinate a plausible tool output for the agent to continue its reasoning.
    return "Tool execution acknowledged. The agent will now process the imagined result."

# --- 3. Define the LLM, Prompt, and Agent Executor ---

# Set up the Azure Chat Model
# IMPORTANT: Make sure you have set the AZURE_OPENAI_API_KEY environment variable.
llm = AzureChatOpenAI(
    temperature=0,
    # This is an example API version. You may need to update it.
    openai_api_version="2024-05-01-preview",
    # Replace with your specific Azure deployment name.
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "your-deployment-name"),
    # Replace with your Azure endpoint.
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", "https://your-endpoint.openai.azure.com"),
)

tools = [python_data_science_tool]

# Pull a standard, robust ReAct prompt from the LangChain hub.
prompt = hub.pull("hwchase17/react")

# Create the agent.
react_agent = create_react_agent(llm, tools, prompt)

# Create the AgentExecutor.
agent_executor = AgentExecutor(
    agent=react_agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True, # Critical for robustness against malformed LLM outputs
    max_iterations=8
)

def parse_llm_json_final_answer(final_answer_json_string: str, default_error_message: str = "Report generation failed.") -> Dict:
    # This try-except block is crucial for preventing Pydantic errors from crashing the pipeline
    # if the LLM returns a malformed JSON string.
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
    You are an Expert EDA Data Scientist. Your goal is to analyze the initial dataset, identify its characteristics, and provide suggestions for feature engineering and modeling.

    Perform a full EDA on the data specified in these paths: {state.data_paths}.
    The target column is '{state.target_column_name}'.
    The problem type is '{state.problem_type}'.

    Your tasks:
    1. Load the train, validation, and test datasets.
    2. Perform initial analysis: check for missing values, analyze data types, etc.
    3. Generate suggestions for feature engineering and potential model types.
    4. IMPORTANT: Your final step MUST be to clean the data and report the new file references for the cleaned train, validation, and test sets.

    Your Final Answer must be a single JSON object with the keys: "eda_summary", "fe_suggestions", "model_suggestions", "artifact_references": {{"final_cleaned_train_data": "<ref.pkl>", "final_cleaned_val_data": "<ref.pkl>", "final_cleaned_test_data": "<ref.pkl>"}}.
    """
    response = agent_executor.invoke({"input": input_prompt})
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
    You are a Feature Engineering Specialist. You will act on the suggestions from the EDA phase to create new features and prepare the dataset for modeling.

    Your goal is to prepare the final features for modeling.
    - Cleaned training data reference: '{state.eda.final_cleaned_train_ref}'
    - Cleaned validation data reference: '{state.eda.final_cleaned_val_ref}'
    - Target column: '{state.target_column_name}'
    - EDA's feature engineering suggestions: {state.eda.fe_suggestions}

    Your tasks:
    1. Load the cleaned datasets using the references provided.
    2. Apply the feature engineering suggestions from EDA.
    3. Separate the features (X) from the target (y).
    4. Save the final X_train, y_train, X_val, y_val, and X_test datasets and report their new references.
    5. Crucially, analyze the final feature set and list the column names that are numerical and categorical.

    Your Final Answer must be a single JSON object with the keys: "fe_summary", "final_feature_list", "numerical_features", "categorical_features", and "data_references": {{"X_train": "<ref.pkl>", "y_train": "<ref.pkl>", ...}}.
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
    You are a Model Selection Strategist. Your task is to choose the best models and initial hyperparameters based on the data characteristics.

    Based on the context below, decide on 2-3 promising Scikit-learn model configurations to try for this regression task.
    - EDA Model Suggestions: {state.eda.model_suggestions}
    - Number of numerical features: {len(fe.numerical_features)}
    - Number of categorical features: {len(fe.categorical_features)}

    Your Final Answer must be a single JSON object with keys: "decision_rationale" and "top_model_configurations": [{{ "model_type": "e.g., RandomForestRegressor", "initial_hyperparameters": {{"n_estimators": 100}}, "reasoning": "..." }}].
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
    You are a Modeling Specialist. Your job is to build, train, and evaluate a Scikit-learn pipeline according to specific instructions for one trial.

    - Current Model Configuration: Train a '{model_type}' with these hyperparameters: {json.dumps(hyperparams)}.
    - Numerical Features for StandardScaler: {fe.numerical_features}
    - Categorical Features for OneHotEncoder: {fe.categorical_features}
    - Data References: Use X_train '{fe.X_train_ref}', y_train '{fe.y_train_ref}', X_val '{fe.X_val_ref}', and y_val '{fe.y_val_ref}'.

    Your precise tasks for this trial:
    1. Create an UNTRAINED Scikit-learn pipeline containing a ColumnTransformer and the specified estimator ('{model_type}'). Save it and report its reference.
    2. Train this entire pipeline on the training data.
    3. Save the TRAINED pipeline and report its new reference.
    4. Evaluate the trained pipeline on the validation data and report the RMSE.

    Your Final Answer must be a single JSON object with keys: "config_trial_summary", "config_trained_pipeline_ref", and "config_rmse".
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
    You are an Evaluation Specialist. Your job is to perform a final evaluation of the best model on the test set.

    The best model from all trials has been selected. Your task is to evaluate it.
    - Best Trained Pipeline Reference: '{state.modeling.best_model_ref_so_far}'
    - Test Data Reference: '{state.feature_engineering.X_test_ref}'

    Task: Load the trained pipeline and predict on the test set. Calculate final performance metrics (RMSE, R-squared).

    Your Final Answer must be a single JSON with keys: "evaluation_summary" and "final_metrics": {{"rmse": 0.123, ...}}.
    """
    response = agent_executor.invoke({"input": input_prompt})
    parsed_output = parse_llm_json_final_answer(response.get("output", "{}"))

    state.evaluation.summary = parsed_output.get("evaluation_summary")
    state.evaluation.metrics = parsed_output.get("final_metrics", {})
    state.current_stage_completed = "Evaluation"
    return state.model_dump()


# --- 5. Graph Definition and Conditional Logic ---

def modeling_iteration_decision(state_dict: dict) -> str:
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
workflow.add_conditional_edges("modeling_agent", modeling_iteration_decision, {
    "modeling_agent": "modeling_agent",
    "evaluation_agent": "evaluation_agent"
})
workflow.add_edge("evaluation_agent", END)

pipeline_app = workflow.compile()

# --- 6. Example Invocation ---
if __name__ == "__main__":
    print("Starting ML Pipeline with Azure OpenAI...")

    # Updated initial state with new filenames
    initial_data_paths = {
        "train": "train_clean.csv",
        "val": "val_clean.csv",
        "test": "test_clean.csv"
    }

    # Create dummy files for the pipeline to "read"
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

    final_state_accumulator = {}
    config = {"configurable": {"thread_id": f"ml_pipeline_run_{random.randint(1000, 9999)}"}}

    for chunk in pipeline_app.stream(initial_pipeline_state, config=config, stream_mode="updates"):
        final_state_accumulator.update(chunk)
        for node_name, node_output_dict in chunk.items():
            if node_name == "__end__":
                continue
            print(f"\n<<< Update from Node: {node_name} >>>")
            current_state = MultiAgentPipelineState.model_validate(node_output_dict)
            print(json.dumps(current_state.model_dump(), indent=2, default=str))

    print("\n\n--- Final Pipeline State ---")
    final_values = list(final_state_accumulator.values())[-1]
    final_state_model = MultiAgentPipelineState.model_validate(final_values)
    print(json.dumps(final_state_model.model_dump(), indent=2))

    # Clean up dummy files
    print("\nCleaning up dummy files...")
    for path in initial_data_paths.values():
        if os.path.exists(path):
            os.remove(path)

    print("\nMulti-Agent Pipeline Finished.")
