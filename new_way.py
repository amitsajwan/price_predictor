import os
import re
import json
import random
from typing import List, Dict, Optional, Sequence, Literal

from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI # Using ChatOpenAI as a stand-in for AzureChatOpenAI
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END

# --- 1. Define the State with Pydantic for Cleaner Access & Grouping ---

# Pydantic models for grouping related state properties
class EDAState(BaseModel):
    report: Optional[Dict[str, any]] = None
    model_suggestions: List[str] = Field(default_factory=list)
    fe_suggestions: List[str] = Field(default_factory=list)
    # Reference to the final cleaned data that FE should use
    final_cleaned_train_ref: Optional[str] = None
    final_cleaned_val_ref: Optional[str] = None
    final_cleaned_test_ref: Optional[str] = None

class FEState(BaseModel):
    applied_steps_summary: Optional[str] = None
    final_feature_list: List[str] = Field(default_factory=list)
    numerical_features: List[str] = Field(default_factory=list)
    categorical_features: List[str] = Field(default_factory=list)
    # Data references ready for the modeling pipeline
    X_train_ref: Optional[str] = None
    y_train_ref: Optional[str] = None
    X_val_ref: Optional[str] = None
    y_val_ref: Optional[str] = None
    X_test_ref: Optional[str] = None

class ModelingState(BaseModel):
    top_model_configurations: List[Dict[str, any]] = Field(default_factory=list)
    decision_rationale: Optional[str] = None
    training_summary: Optional[str] = None
    # Reference to the single best-trained pipeline
    best_model_ref_so_far: Optional[str] = None
    best_rmse_so_far: float = float('inf')
    best_model_config_so_far: Optional[Dict[str, any]] = None
    # Iteration tracking
    modeling_config_index: int = 0
    max_modeling_configs_to_try: int = 2
    strategy_log: List[str] = Field(default_factory=list)

class EvaluationState(BaseModel):
    summary: Optional[str] = None
    metrics: Dict[str, float] = Field(default_factory=dict)
    test_set_prediction_status: Optional[str] = None

# Main State Class using dot notation (e.g., state.eda.report)
class MultiAgentPipelineState(BaseModel):
    # Overall pipeline inputs
    data_paths: Dict[str, str]
    target_column_name: str
    problem_type: Literal["classification", "regression"]
    target_rmse: float
    max_react_iterations: int = 8

    # Grouped states for each major stage
    eda: EDAState = Field(default_factory=EDAState)
    feature_engineering: FEState = Field(default_factory=FEState)
    modeling: ModelingState = Field(default_factory=ModelingState)
    evaluation: EvaluationState = Field(default_factory=EvaluationState)

    # Tracks the current stage completion
    current_stage_completed: Optional[str] = None

# --- 2. Define a LangChain Tool ---
# The logic from your agno_python_tool_interface is moved into a function
# decorated with @tool. The docstring is crucial for the agent to understand its purpose.

SIMULATED_MODEL_PERFORMANCE_REGISTRY = {}

@tool
def python_data_science_tool(instruction: str) -> str:
    """
    Executes a Python script for data science tasks like EDA, feature engineering, and modeling.
    Takes a natural language instruction for a data-related task.
    Handles data loading, transformation, analysis, model training (Scikit-learn pipelines),
    and evaluation. Returns a string observation, which often includes references (filenames)
    to artifacts produced, like datasets (.pkl), models (.joblib), or plots (.png).
    Key capabilities:
    - Loading data: "load the dataset from 'path/to/train.csv' and report its reference as 'train_df_ref.pkl'"
    - Data manipulation: "extract year, month, day from the 'date' column in 'df_ref.pkl' and report new reference as 'df_ref_fe.pkl'"
    - Model Pipeline: "create an untrained scikit-learn pipeline with a StandardScaler for numerical_features and OneHotEncoder for categorical_features, followed by a RandomForestRegressor. Save it and report its reference."
    - Training & Evaluation: "train the pipeline 'untrained_pipe.joblib' using X_train 'X_train.pkl' and y_train 'y_train.pkl'. Report the trained pipeline reference and the validation RMSE using X_val and y_val."
    """
    global SIMULATED_MODEL_PERFORMANCE_REGISTRY
    print(f"\n   [TOOL EXECUTED] Instruction: '{instruction}'")
    sim_observation = "" # Start with an empty observation
    instruction_lower = instruction.lower()

    # Simulate the behavior of your original interface
    if "load the dataset from" in instruction_lower:
        ref = re.search(r"reference as '([^']+)'", instruction_lower).group(1)
        sim_observation += f"Dataset loaded. Tool reports reference as '{ref}'."
    elif "extract year, month, day, dayofweek from the 'date' column" in instruction_lower:
        sim_observation += ("Date features extracted, original 'Date' column dropped. "
                            "New data refs: train='train_date_fe.pkl', val='val_date_fe.pkl', test='test_date_fe.pkl'.")
    elif "separate target" in instruction_lower:
        sim_observation += ("Target separated. Tool reports X/y refs: X_train='X_train_final.pkl', "
                            "y_train='y_train_final.pkl', X_val='X_val_final.pkl', y_val='y_val_final.pkl', "
                            "X_test='X_test_final.pkl'. Final feature list: ['Price_log', 'Volume', "
                            "'Year', 'Month', 'DayOfWeek', 'Category_A', 'Category_B']. "
                            "Numerical features: ['Price_log', 'Volume', 'Year', 'Month', 'DayOfWeek']. "
                            "Categorical features: ['Category_A', 'Category_B'].")
    elif "create an untrained scikit-learn pipeline" in instruction_lower:
        ref_match = re.search(r"report its reference as '([^']+)'", instruction_lower)
        ref_name = ref_match.group(1) if ref_match else "untrained_full_pipeline.joblib"
        sim_observation += f"Untrained Scikit-learn pipeline created. Reference is '{ref_name}'."
    elif "train this entire pipeline" in instruction_lower:
        untrained_pipe_ref = re.search(r"pipeline '([^']+)'", instruction_lower).group(1)
        trained_model_ref_match = re.search(r"report its reference as '([^']+)'", instruction_lower)
        trained_model_ref = trained_model_ref_match.group(1) if trained_model_ref_match else f"trained_{untrained_pipe_ref}"
        params_key = "default"; model_type_sim = "RandomForest"
        full_model_key = f"{model_type_sim}_{params_key}"
        if full_model_key not in SIMULATED_MODEL_PERFORMANCE_REGISTRY:
            SIMULATED_MODEL_PERFORMANCE_REGISTRY[full_model_key] = random.uniform(0.6, 2.5)
        else:
            SIMULATED_MODEL_PERFORMANCE_REGISTRY[full_model_key] *= random.uniform(0.85, 0.99)
        current_sim_rmse = SIMULATED_MODEL_PERFORMANCE_REGISTRY[full_model_key]
        sim_observation += (f"Pipeline '{untrained_pipe_ref}' trained. Trained pipeline reference is '{trained_model_ref}'. "
                            f"Validation RMSE from this pipeline: {current_sim_rmse:.4f}.")
    elif "calculate" in instruction_lower and "metrics" in instruction_lower:
         sim_observation += f"Final evaluation complete. Metrics: {{'rmse': 0.8123, 'r_squared': 0.78}}."
    else:
        sim_observation += "Task completed successfully. References to any created artifacts are included in this message."

    print(f"   [TOOL RESULT] Observation: '{sim_observation}'")
    return sim_observation

# --- 3. Define the LLM, Prompt, and Agent Executor ---

# Use your actual AzureChatOpenAI model here. We'll use ChatOpenAI as a placeholder.
# Ensure your environment variables (AZURE_OPENAI_API_KEY, etc.) are set.
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# The tools our agent has access to
tools = [python_data_science_tool]

# A generic ReAct-style prompt template that will be used by all agent nodes
# The agent will fill the `agent_scratchpad` with its thoughts and tool outputs
REACT_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", "{system_prompt}"),
    ("user", "{input_prompt}"),
    ("placeholder", "{agent_scratchpad}"),
])

# Create the agent. This binds the LLM, the prompt, and the tools together.
react_agent = create_react_agent(llm, tools, REACT_PROMPT_TEMPLATE)

# The AgentExecutor is the runtime for the agent. It's what invokes the agent,
# executes the chosen tools, and runs the loop until completion.
agent_executor = AgentExecutor(
    agent=react_agent,
    tools=tools,
    verbose=True, # Set to True to see the agent's thoughts
    handle_parsing_errors=True, # Gracefully handle any LLM output parsing errors
    max_iterations=8
)

def parse_llm_json_final_answer(final_answer_json_string: str, default_error_message: str = "Report generation failed.") -> Dict:
    # A robust parser for the JSON output from the LLM
    try:
        # Remove markdown code fences
        match = re.search(r"```json\s*(.*?)\s*```", final_answer_json_string, re.DOTALL)
        if match:
            json_string_cleaned = match.group(1).strip()
        else:
            json_string_cleaned = final_answer_json_string.strip()
        return json.loads(json_string_cleaned)
    except Exception as e:
        print(f"   ERROR parsing LLM JSON: {e}, String: '{final_answer_json_string}'")
        return {"error": str(e), "summary": default_error_message}


# --- 4. Define Agent Nodes (Now much simpler) ---

def eda_agent_node(state: MultiAgentPipelineState) -> Dict:
    print("\n--- EDA Agent Node ---")
    system_prompt = "You are an Expert EDA Data Scientist. Your goal is to analyze the initial dataset, identify its characteristics, and provide suggestions for feature engineering and modeling. You must use the provided tool to perform all data operations."
    input_prompt = f"""
    Perform a full EDA on the data specified in these paths: {state.data_paths}.
    The target column is '{state.target_column_name}'.
    The problem type is '{state.problem_type}'.
    Your tasks:
    1. Load the train, validation, and test datasets.
    2. Perform initial analysis: check for missing values, analyze data types, and get basic statistics.
    3. Generate suggestions for feature engineering (e.g., 'extract year from date', 'log transform skewed features').
    4. Generate suggestions for potential model types (e.g., 'RandomForestRegressor for tabular data').
    5. IMPORTANT: Your final step MUST be to clean the data (e.g., handle dates, simple imputation) and report the new file references for the cleaned train, validation, and test sets.

    Your Final Answer must be a single JSON object with the keys: "eda_summary", "fe_suggestions", "model_suggestions", "artifact_references": {{"final_cleaned_train_data": "<ref.pkl>", "final_cleaned_val_data": "<ref.pkl>", "final_cleaned_test_data": "<ref.pkl>"}}.
    """
    response = agent_executor.invoke({
        "system_prompt": system_prompt,
        "input_prompt": input_prompt,
    })
    parsed_output = parse_llm_json_final_answer(response.get("output", "{}"))

    # Update the Pydantic state model
    state.eda.report = parsed_output
    state.eda.fe_suggestions = parsed_output.get("fe_suggestions", [])
    state.eda.model_suggestions = parsed_output.get("model_suggestions", [])
    artifacts = parsed_output.get("artifact_references", {})
    state.eda.final_cleaned_train_ref = artifacts.get("final_cleaned_train_data")
    state.eda.final_cleaned_val_ref = artifacts.get("final_cleaned_val_data")
    state.eda.final_cleaned_test_ref = artifacts.get("final_cleaned_test_data")
    state.current_stage_completed = "EDA"
    return state.model_dump() # Return the updated state as a dictionary


def feature_engineering_agent_node(state: MultiAgentPipelineState) -> Dict:
    print("\n--- Feature Engineering Agent Node ---")
    system_prompt = "You are a Feature Engineering Specialist. You will act on the suggestions from the EDA phase to create new features and prepare the dataset for modeling."
    input_prompt = f"""
    Your goal is to prepare the final features for modeling.
    - Cleaned training data reference: '{state.eda.final_cleaned_train_ref}'
    - Cleaned validation data reference: '{state.eda.final_cleaned_val_ref}'
    - Target column: '{state.target_column_name}'
    - EDA's feature engineering suggestions: {state.eda.fe_suggestions}

    Your tasks:
    1. Load the cleaned datasets using the references provided.
    2. Apply the feature engineering suggestions from EDA. For date columns, extract components like year, month, and day, then drop the original.
    3. After all transformations, separate the features (X) from the target (y) for the train and validation sets.
    4. Save the final X_train, y_train, X_val, y_val, and X_test datasets and report their new references.
    5. Crucially, analyze the final feature set (X_train) and list the column names that are numerical and categorical.

    Your Final Answer must be a single JSON object with the keys: "fe_summary", "final_feature_list", "numerical_features", "categorical_features", and "data_references": {{"X_train": "<ref.pkl>", "y_train": "<ref.pkl>", ...}}.
    """
    response = agent_executor.invoke({
        "system_prompt": system_prompt,
        "input_prompt": input_prompt,
    })
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
    system_prompt = "You are a Model Selection Strategist. Your task is to choose the best models and initial hyperparameters based on the data characteristics."
    input_prompt = f"""
    Based on the context below, decide on 2-3 promising Scikit-learn model configurations to try for this regression task.
    - EDA Model Suggestions: {state.eda.model_suggestions}
    - Number of numerical features: {len(fe.numerical_features)}
    - Number of categorical features: {len(fe.categorical_features)}
    - Total features: {len(fe.final_feature_list)}

    Your Final Answer must be a single JSON object with keys: "decision_rationale" and "top_model_configurations": [{{ "model_type": "e.g., RandomForestRegressor", "initial_hyperparameters": {{"n_estimators": 100}}, "reasoning": "..." }}].
    """
    response = agent_executor.invoke({
        "system_prompt": system_prompt,
        "input_prompt": input_prompt,
    })
    parsed_output = parse_llm_json_final_answer(response.get("output", "{}"))

    state.modeling.decision_rationale = parsed_output.get("decision_rationale")
    state.modeling.top_model_configurations = parsed_output.get("top_model_configurations", [])
    state.current_stage_completed = "ModelSelectionDecision"
    return state.model_dump()


def modeling_agent_node(state: MultiAgentPipelineState) -> Dict:
    print(f"\n--- Modeling Agent Node (Trying Config #{state.modeling.modeling_config_index}) ---")
    fe = state.feature_engineering
    m_state = state.modeling

    # Get the configuration for the current trial
    config_idx = m_state.modeling_config_index
    current_config = m_state.top_model_configurations[config_idx]
    model_type = current_config.get("model_type")
    hyperparams = current_config.get("initial_hyperparameters", {})

    system_prompt = "You are a Modeling Specialist. Your job is to build, train, and evaluate a Scikit-learn pipeline according to specific instructions."
    input_prompt = f"""
    You are executing one trial in a series of model experiments.
    - Current Model Configuration: Train a '{model_type}' with these hyperparameters: {json.dumps(hyperparams)}.
    - Numerical Features for StandardScaler: {fe.numerical_features}
    - Categorical Features for OneHotEncoder: {fe.categorical_features}
    - Data References: Use X_train '{fe.X_train_ref}', y_train '{fe.y_train_ref}', X_val '{fe.X_val_ref}', and y_val '{fe.y_val_ref}'.

    Your precise tasks for this trial:
    1. Create an UNTRAINED Scikit-learn pipeline containing a ColumnTransformer (for scaling numerical and encoding categorical features) and the specified estimator ('{model_type}'). Save it and report its reference.
    2. Load that untrained pipeline, train it on the training data.
    3. Save the TRAINED pipeline and report its new reference.
    4. Evaluate the trained pipeline on the validation data and report the RMSE.

    Your Final Answer must be a single JSON object with keys: "config_trial_summary", "config_trained_pipeline_ref", and "config_rmse".
    """
    response = agent_executor.invoke({
        "system_prompt": system_prompt,
        "input_prompt": input_prompt,
    })
    parsed_output = parse_llm_json_final_answer(response.get("output", "{}"))

    # Update state based on this trial's results
    current_rmse = parsed_output.get("config_rmse")
    trained_ref = parsed_output.get("config_trained_pipeline_ref")
    m_state.strategy_log.append(f"Trial {config_idx} ({model_type}): RMSE={current_rmse}, Ref={trained_ref}")

    if current_rmse and current_rmse < m_state.best_rmse_so_far:
        print(f"   New best RMSE found: {current_rmse} (previous: {m_state.best_rmse_so_far})")
        m_state.best_rmse_so_far = current_rmse
        m_state.best_model_ref_so_far = trained_ref
        m_state.best_model_config_so_far = current_config

    m_state.modeling_config_index += 1 # Increment for the next loop
    state.current_stage_completed = "Modeling_Config_Trial"
    return state.model_dump()


def evaluation_node(state: MultiAgentPipelineState) -> Dict:
    print("\n--- Evaluation Agent Node ---")
    system_prompt = "You are an Evaluation Specialist. Your job is to perform a final evaluation of the best model on the test set."
    input_prompt = f"""
    The best model from all trials has been selected. Your task is to evaluate it.
    - Best Trained Pipeline Reference: '{state.modeling.best_model_ref_so_far}'
    - Test Data Reference: '{state.feature_engineering.X_test_ref}'

    Task: Load the trained pipeline and predict on the test set. Calculate final performance metrics (RMSE, R-squared).

    Your Final Answer must be a single JSON with keys: "evaluation_summary" and "final_metrics": {{"rmse": 0.123, ...}}.
    """
    response = agent_executor.invoke({
        "system_prompt": system_prompt,
        "input_prompt": input_prompt,
    })
    parsed_output = parse_llm_json_final_answer(response.get("output", "{}"))

    state.evaluation.summary = parsed_output.get("evaluation_summary")
    state.evaluation.metrics = parsed_output.get("final_metrics", {})
    state.current_stage_completed = "Evaluation"
    return state.model_dump()

# --- 5. Graph Definition and Conditional Logic ---

def modeling_iteration_decision(state_dict: dict) -> str:
    # LangGraph passes state as a dict, so we convert it back to our Pydantic model for easy access
    state = MultiAgentPipelineState.model_validate(state_dict)
    print("\n--- Checking Modeling Iteration Decision ---")
    m_state = state.modeling

    # Stop conditions
    if not m_state.top_model_configurations:
        print("   No model configurations to try. Skipping to evaluation.")
        return "evaluation_agent"
    if m_state.best_rmse_so_far <= state.target_rmse:
        print(f"   Target RMSE ({state.target_rmse}) achieved. Skipping to evaluation.")
        return "evaluation_agent"
    if m_state.modeling_config_index >= len(m_state.top_model_configurations) or m_state.modeling_config_index >= m_state.max_modeling_configs_to_try:
        print("   All model configurations tried or limit reached. Proceeding to evaluation.")
        return "evaluation_agent"

    # Continue condition
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
    "modeling_agent": "modeling_agent", # Loop back to modeling for the next config
    "evaluation_agent": "evaluation_agent" # Go to evaluation if loop is done
})
workflow.add_edge("evaluation_agent", END)

pipeline_app = workflow.compile()

# --- 6. Example Invocation ---
if __name__ == "__main__":
    print("Starting ML Pipeline with LangChain AgentExecutor...")
    initial_pipeline_state = {
        "data_paths": {"train": "path/train.csv", "val": "path/val.csv", "test": "path/test.csv"},
        "target_column_name": "Target",
        "problem_type": "regression",
        "target_rmse": 0.85,
        "modeling": {"max_modeling_configs_to_try": 2}
    }

    # Stream the execution and print updates from each node
    final_state_accumulator = {}
    for chunk in pipeline_app.stream(initial_pipeline_state, stream_mode="updates"):
        for node_name, node_output_dict in chunk.items():
            print(f"\n<<< Update from Node: {node_name} >>>")
            final_state_accumulator.update(node_output_dict)
            for k, v in node_output_dict.items():
                 # Pretty print nested dictionaries
                if isinstance(v, dict):
                    print(f"  {k}:")
                    for sub_k, sub_v in v.items():
                        print(f"    {sub_k}: {str(sub_v)[:300]}...")
                else:
                    print(f"  {k}: {str(v)[:350]}...")

    print("\n\n--- Final Pipeline State ---")
    # Validate the final state with our Pydantic model and print nicely
    final_state_model = MultiAgentPipelineState.model_validate(final_state_accumulator)
    print(json.dumps(final_state_model.model_dump(), indent=2))
    print("\nMulti-Agent Pipeline Finished.")
