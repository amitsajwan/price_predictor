import os
import re
import json 
import random 
from typing import TypedDict, Annotated, List, Dict, Optional, Sequence, Literal
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI 
from langgraph.graph import StateGraph, END

# --- 1. Define the State for the Pipeline ---
class MultiAgentPipelineState(TypedDict):
    # Input
    data_paths: Dict[str, str] 
    target_column_name: Optional[str] 
    problem_type: Optional[Literal["classification", "regression"]] 
    target_rmse: Optional[float] 

    # Output from EdaAgentNode
    eda_report: Optional[Dict[str, any]] 

    # Output from FeatureEngineeringAgentNode
    fe_applied_steps_summary: Optional[str]
    fe_final_feature_list: Optional[List[str]] # All features in X before preproc pipeline
    fe_numerical_features: Optional[List[str]] # Identified numerical features for scaling
    fe_categorical_features: Optional[List[str]] # Identified categorical for encoding
    fe_X_train_ref: Optional[str] # Data ready for sklearn Pipeline (dates handled, etc.)
    fe_y_train_ref: Optional[str]
    fe_X_val_ref: Optional[str]   
    fe_y_val_ref: Optional[str]
    fe_X_test_ref: Optional[str]  
    fe_custom_transformer_module: Optional[str] 
    
    # Output from ModelSelectionDecisionNode
    top_model_configurations: Optional[List[Dict[str, any]]] 

    # Modeling Node State & Output
    model_training_summary: Optional[str] 
    # This will now be the reference to the FULL pipeline (preprocessors + model), trained
    model_trained_pipeline_ref: Optional[str] 
    current_rmse: Optional[float]            
    best_rmse_so_far: Optional[float]        
    best_model_config_so_far: Optional[Dict[str,any]] 
    best_model_ref_so_far: Optional[str]      
    modeling_config_index: Optional[int]      
    max_modeling_configs_to_try: Optional[int]
    modeling_strategy_log: Optional[List[str]]

    # Output from Evaluation Node
    evaluation_summary: Optional[str] 
    evaluation_metrics: Optional[Dict[str, float]]
    test_set_prediction_status: Optional[str]

    current_stage_completed: Optional[str]
    max_react_iterations: Optional[int]


# --- 2. Interface for your Agnostic PythonTool ---
# REPLACE THIS FUNCTION WITH THE ACTUAL CALL TO YOUR AGNO_PYTHON_TOOL
SIMULATED_MODEL_PERFORMANCE_REGISTRY = {} 
def agno_python_tool_interface(instruction: str, agent_context_hint: Optional[str] = None) -> str:
    global SIMULATED_MODEL_PERFORMANCE_REGISTRY
    print(f"    [AGNO_PYTHON_TOOL INTERFACE] Instruction: '{instruction}'")
    if agent_context_hint: print(f"    Agent Context Hint: {agent_context_hint}")
    
    sim_observation = run_python(instruction,agent_context_hint )
    instruction_lower = instruction.lower()

    if "load the dataset from" in instruction_lower and "report its reference as" in instruction_lower:
        ref = re.search(r"reference as '([^']+)'", instruction_lower).group(1)
        sim_observation += f"Dataset loaded. Tool reports reference as '{ref}'."
    elif "identify the 'date' column in" in instruction_lower and "parse it as datetime" in instruction_lower and "report new reference as" in instruction_lower:
        ref_out = re.search(r"new reference as '([^']+)'", instruction_lower).group(1)
        sim_observation += f"'Date' column parsed. New data reference is '{ref_out}'."
    elif "create a temporary numeric-only version of" in instruction_lower and "report the new reference as" in instruction_lower:
        ref_out = re.search(r"new reference as '([^']+)'", instruction_lower).group(1)
        sim_observation += f"Numeric-only version created. New reference: '{ref_out}'."
    elif "extract year, month, day, dayofweek from the 'date' column" in instruction_lower and "drop the original 'date' column" in instruction_lower and "report new data references for train as" in instruction_lower:
        sim_observation += ("Date features extracted, original 'Date' column dropped. "
                           "New data refs: train='train_date_fe.pkl', val='val_date_fe.pkl', test='test_date_fe.pkl'.")
    elif "perform final cleaning on" in instruction_lower and "save the resulting dataset and report the new reference as" in instruction_lower:
        ref_out = re.search(r"new reference as '([^']+)'", instruction_lower).group(1)
        sim_observation += f"Data cleaned. New reference: '{ref_out}'."
    elif "generate a histogram for" in instruction_lower and "save it as" in instruction_lower and "report the filename, and provide a textual description" in instruction_lower:
        fname = re.search(r"save it as '([^']+)'", instruction_lower).group(1)
        sim_observation += f"Histogram saved as '{fname}'. Description: [Simulated description]."
    elif "separate target" in instruction_lower and "save them as .pkl files and report new references for x_train as" in instruction_lower:
        sim_observation += "Target separated. Tool reports X/y refs: X_train='X_train_final.pkl', y_train='y_train_final.pkl', X_val='X_val_final.pkl', y_val='y_val_final.pkl', X_test='X_test_final.pkl'. Final feature list: ['Price_log', 'Volume', 'Year', 'Month', 'DayOfWeek', 'Category_A', 'Category_B']." # Example feature list
    
    # Modeling specific - constructing and fitting a FULL pipeline
    elif "create an untrained scikit-learn pipeline with a columntransformer" in instruction_lower and "standardscaler for numerical features" in instruction_lower and "onehotencoder for categorical features" in instruction_lower and "estimator type" in instruction_lower and "save it as a .joblib file and report its reference as" in instruction_lower:
        ref_match = re.search(r"report its reference as '([^']+)'", instruction_lower)
        ref_name = ref_match.group(1) if ref_match else "untrained_full_pipeline.joblib"
        sim_observation += f"Untrained Scikit-learn pipeline with ColumnTransformer (Scaler+OHE) and specified estimator created. Saved. Reference is '{ref_name}'."
    
    elif "load the untrained pipeline" in instruction_lower and "train this entire pipeline using x_train" in instruction_lower and "save the trained pipeline as a .joblib file and report its reference as" in instruction_lower:
        untrained_pipe_ref = re.search(r"untrained pipeline '([^']+)'", instruction_lower).group(1)
        trained_model_ref_match = re.search(r"report its reference as '([^']+)'", instruction_lower)
        trained_model_ref = trained_model_ref_match.group(1) if trained_model_ref_match else f"trained_{untrained_pipe_ref}"
        
        params_key = "default"; model_type_sim = "DefaultModel" # Simplified simulation
        if "randomforest" in untrained_pipe_ref: model_type_sim="RandomForest" 
        if "n_estimators=100" in instruction_lower : params_key="n100_d10" # Example param key
            
        full_model_key = f"{model_type_sim}_{params_key}"
        if full_model_key not in SIMULATED_MODEL_PERFORMANCE_REGISTRY: SIMULATED_MODEL_PERFORMANCE_REGISTRY[full_model_key] = random.uniform(0.6, 2.5) 
        else: SIMULATED_MODEL_PERFORMANCE_REGISTRY[full_model_key] *= random.uniform(0.85, 0.99) 
        current_sim_rmse = SIMULATED_MODEL_PERFORMANCE_REGISTRY[full_model_key]

        sim_observation += (f"Untrained pipeline '{untrained_pipe_ref}' loaded and ENTIRE pipeline trained. Trained pipeline saved. Reference is '{trained_model_ref}'.")
        if "report rmse" in instruction_lower: sim_observation += f" Validation RMSE from this pipeline: {current_sim_rmse:.4f}."
    
    elif "load trained pipeline" in instruction_lower and "calculate" in instruction_lower and "metrics" in instruction_lower: 
        final_pipe_ref = re.search(r"pipeline '([^']+)'", instruction_lower).group(1)
        final_sim_rmse = 0.45 
        for key, val_rmse in SIMULATED_MODEL_PERFORMANCE_REGISTRY.items():
            if key in final_pipe_ref: final_sim_rmse = val_rmse; break 
        sim_observation += f"Final evaluation using '{final_pipe_ref}'. Metrics: {{'rmse': {final_sim_rmse:.4f}, 'r_squared': {max(0, 1 - final_sim_rmse / 2.0):.2f}}}."
    else:
        sim_observation += "Task completed. If specific artifacts were requested, their references are included."
            
    print(f"    [AGNO_PYTHON_TOOL INTERFACE] Returning Observation:\n    '{sim_observation}'")
    return sim_observation

# --- 3. Generic ReAct Loop Engine & JSON Parser (Assumed defined as before) ---

def run_generic_react_loop(initial_prompt_content: str, max_iterations: int, agent_context_hint_for_tool: Optional[str] = None) -> str: 
    # ... (Implementation from ml_pipeline_agent_managed_refs_json_v9_robust_json_and_refs) ...
    react_messages: List[BaseMessage] = [SystemMessage(content=initial_prompt_content)]
    final_answer_text = json.dumps({"error": "Agent did not produce a Final Answer within iteration limit."}) 
    for i in range(max_iterations):
        print(f"  [GenericReActLoop] Iteration {i+1}/{max_iterations}")
        ai_response = llm.invoke(react_messages); ai_content = ai_response.content.strip()
        react_messages.append(ai_response) ; print(f"    LLM: {ai_content[:450]}...")
        final_answer_match = re.search(r"Final Answer:\s*(```json\s*(.*?)\s*```|{\s*.*})", ai_content, re.DOTALL | re.IGNORECASE)
        action_match = re.search(r"Action:\s*Python\s*Action Input:\s*(.+)", ai_content, re.DOTALL | re.IGNORECASE) 
        if final_answer_match:
            json_block_content = final_answer_match.group(2) if final_answer_match.group(2) else final_answer_match.group(1)
            final_answer_text = json_block_content.strip(); print(f"    Loop Concluded. Final Answer (JSON string) obtained:\n{final_answer_text}"); break 
        elif action_match:
            nl_instruction_for_tool = action_match.group(1).strip()
            tool_observation = agno_python_tool_interface(nl_instruction_for_tool, agent_context_hint_for_tool)
            react_messages.append(HumanMessage(content=f"Observation: {tool_observation}"))
        else:
            react_messages.append(HumanMessage(content="System hint: Ensure valid format ('Action: Python...' or 'Final Answer: ```json...```'). No extra text outside JSON block."))
            if i > 1: final_answer_text = json.dumps({"error": "Agent format error."}); print(f"    Agent format error."); break 
        if i == max_iterations - 1: print(f"    Max ReAct iterations reached."); final_answer_text = json.dumps({"error": f"Max iterations. Last thought: {ai_content}"})
    return final_answer_text

def parse_llm_json_final_answer(final_answer_json_string: str, default_error_message: str = "Report generation failed.") -> Dict:
    # ... (Implementation from ml_pipeline_agent_managed_refs_json_v9_robust_json_and_refs) ...
    try:
        match = re.search(r"```json\s*(.*?)\s*```", final_answer_json_string, re.DOTALL);
        if match: json_string_cleaned = match.group(1).strip()
        else: json_string_cleaned = final_answer_json_string.strip();
        if json_string_cleaned.startswith("```") and json_string_cleaned.endswith("```"): json_string_cleaned = json_string_cleaned[3:-3].strip()
        json_string_cleaned = re.sub(r",\s*([}\]])", r"\1", json_string_cleaned); json_string_cleaned = re.sub(r"//.*?\n", "\n", json_string_cleaned); json_string_cleaned = re.sub(r"^\s*#.*?\n", "\n", json_string_cleaned, flags=re.MULTILINE)
        if not json_string_cleaned: return {"error": "JSON empty after cleaning", "summary": default_error_message, "original_string": final_answer_json_string}
        return json.loads(json_string_cleaned)
    except Exception as e: print(f"    ERROR parsing LLM JSON: {e}, String: '{final_answer_json_string}'"); return {"error": str(e), "summary": default_error_message, "original_string": final_answer_json_string}

# --- 5. Define Agent Nodes ---

def eda_agent_node(state: MultiAgentPipelineState) -> Dict[str, any]:
    # ... (Prompt mostly same as v9, but ensure it requests FINAL data refs for FE)
    print("\n--- EDA Agent Node Running (Standard Pipeline Focus) ---")
    data_paths = state["data_paths"]; target_col = state.get("target_column_name", "Target"); problem_type = state.get("problem_type", "regression") 
    eda_tool_context_hint = f"Initial data paths: {json.dumps(data_paths)}. Target: '{target_col}'. Problem: {problem_type}. Task: EDA for stock price prediction."
    prompt_content = f"""You are an Expert EDA Data Scientist... (Prompt as in v9, ensuring it asks for FINAL cleaned data references for train, val, test to pass to FE, and that date columns are parsed early and numeric-only versions are used for correlation).
    "Final Answer:" JSON object...
    "artifact_references": {{ 
        "final_cleaned_train_data": "<tool_reported_ref.pkl>", // Key for FE
        "final_cleaned_val_data": "<tool_reported_ref.pkl>",   // Key for FE
        "final_cleaned_test_data": "<tool_reported_ref.pkl>",  // Key for FE
        "numeric_train_for_correlation": "<tool_reported_ref.pkl>", // Temp for EDA itself
        "plots": {{ ... }} }}
    Begin.
    """ # Ensure EDA prompt emphasizes FINAL cleaned data refs for FE under these keys.
    final_answer_json_string = run_generic_react_loop(prompt_content, state.get("max_react_iterations", 20), eda_tool_context_hint)
    eda_report_output = parse_llm_json_final_answer(final_answer_json_string, "EDA report failed.")
    output_to_state = {"current_stage_completed": "EDA", "eda_report": eda_report_output}
    if "error" not in eda_report_output:
        artifact_refs = eda_report_output.get("artifact_references", {})
        output_to_state.update({
            "eda_model_suggestions": eda_report_output.get("model_suggestions", []), 
            "eda_fe_suggestions": eda_report_output.get("fe_suggestions", []),
            "eda_processed_train_ref": artifact_refs.get("final_cleaned_train_data"), # Use this key
            "eda_processed_val_ref": artifact_refs.get("final_cleaned_val_data"),
            "eda_processed_test_ref": artifact_refs.get("final_cleaned_test_data"),
            "eda_plot_references": artifact_refs.get("plots", {})
        })
    else: # Defaults on error
        output_to_state.update({k: [] for k in ["eda_model_suggestions", "eda_fe_suggestions"]})
        output_to_state.update({k: f"error_default_eda_{k.split('_')[-2]}.pkl" for k in ["eda_processed_train_ref", "eda_processed_val_ref", "eda_processed_test_ref"]})
        output_to_state["eda_plot_references"] = {}
    return output_to_state


def feature_engineering_agent_node(state: MultiAgentPipelineState) -> Dict[str, any]:
    print("\n--- Feature Engineering Agent Node (Focus on Feature Lists for Pipeline) ---")
    eda_report = state.get("eda_report", {}) 
    if eda_report.get("error"): return {"current_stage_completed": "FeatureEngineering", "fe_applied_steps_summary": "FE skipped due to EDA errors."}

    # Use FINAL processed data references from EDA
    train_ref_from_eda = eda_report.get("artifact_references", {}).get("final_cleaned_train_data", "default_train_eda.pkl") 
    val_ref_from_eda = eda_report.get("artifact_references", {}).get("final_cleaned_val_data", "default_val_eda.pkl")
    test_ref_from_eda = eda_report.get("artifact_references", {}).get("final_processed_test_data", "default_test_eda.pkl")
    
    suggestions_from_eda = eda_report.get("fe_suggestions", [])
    target_col = state.get("target_column_name", "Target")

    date_fe_suggestion = "PRIORITY: Implement EDA's suggestion for 'Date' column: extract numerical date features (Year, Month, Day, DayOfWeek) from parsed 'Date' column AND THEN DROP ORIGINAL 'Date' (datetime object) COLUMN. Tool must report NEW data references (as .pkl) after this."
    for suggestion in suggestions_from_eda: # Check for specific EDA suggestion
        if "date column" in suggestion.lower() and "extract" in suggestion.lower() and "drop" in suggestion.lower():
            date_fe_suggestion = suggestion; break

    fe_tool_context_hint = (f"Input data from EDA (expect .pkl): train='{train_ref_from_eda}', val='{val_ref_from_eda}', test='{test_ref_from_eda}'. "
                            f"Target: '{target_col}'. EDA FE Suggestions: {json.dumps(suggestions_from_eda)}.")

    prompt_content = f"""You are a Feature Engineering Specialist for stock price prediction. PythonTool takes NL.
    Context from EDA:
    - Input Train Data Ref (cleaned & date-parsed by EDA): {train_ref_from_eda}
    - Input Val Data Ref: {val_ref_from_eda}
    - Input Test Data Ref: {test_ref_from_eda}
    - EDA FE Suggestions: {json.dumps(suggestions_from_eda)}
    - Specific EDA instruction for 'Date' column (YOUR FIRST PRIORITY): "{date_fe_suggestion}"
    - Target Column: '{target_col}'

    Your tasks:
    1. Instruct tool to use/load datasets using FINAL PROCESSED .pkl references from EDA.
    2. **CRITICAL Date Handling:** Execute EDA's suggestion for 'Date' column: "{date_fe_suggestion}". Tool MUST report NEW .pkl data references for train, val, test after this. Use these NEW references for subsequent FE.
    3. Implement other FE steps from EDA suggestions (e.g., custom interactions, log transforms if not part of a standard scaler later) on these NEW date-handled datasets. Apply consistently. Tool must report NEW data refs if data changes.
    4. **Identify Feature Types:** After all feature creation/transformation, instruct PythonTool to analyze the LATEST training data reference (e.g., from step 2 or 3) and report two lists of column names: one for 'numerical_features' (that would need scaling) and one for 'categorical_features' (that would need encoding). EXCLUDE the target column ('{target_col}') from these lists.
    5. Separate features (X) and target ('{target_col}') from the LATEST version of train, val, test data. Ask tool to save X_train, y_train, etc., as .pkl files and report their full .pkl filename references. Also ask for the final feature list (column names in X datasets).

    ReAct Format: Action: Python, Action Input: <NL instruction>.
    "Final Answer:" MUST be a single well-formed JSON object string, enclosed ONLY in ```json ... ```. NO OTHER TEXT.
    **All string values within JSON MUST be properly quoted and escaped.**
    JSON keys: "fe_summary", 
               "numerical_features": [ (list of strings) ],
               "categorical_features": [ (list of strings) ],
               "final_feature_list": [ (list of strings for X) ], // This is X.columns
               "data_references": {{ "X_train": "<X_train_ref.pkl>", "y_train": "<y_train_ref.pkl>", ... }}
               // NO "transformer_references" or "untrained_full_pipeline_ref" here. Modeling will build it.
    Begin. Address date feature engineering first.
    """
    final_answer_json_string = run_generic_react_loop(prompt_content, state.get("max_react_iterations", 15), fe_tool_context_hint) # FE can be iterative
    parsed_data = parse_llm_json_final_answer(final_answer_json_string, "FE report generation failed.")
    
    output_to_state = {"current_stage_completed": "FeatureEngineering"}
    if "error" not in parsed_data:
        data_refs = parsed_data.get("data_references", {})
        output_to_state.update({
            "fe_applied_steps_summary": parsed_data.get("fe_summary"),
            "fe_final_feature_list": parsed_data.get("final_feature_list", []),
            "fe_numerical_features": parsed_data.get("numerical_features", []), # New
            "fe_categorical_features": parsed_data.get("categorical_features", []), # New
            "fe_X_train_ref": data_refs.get("X_train"), 
            "fe_y_train_ref": data_refs.get("y_train"),
            "fe_X_val_ref": data_refs.get("X_val"),
            "fe_y_val_ref": data_refs.get("y_val"),
            "fe_X_test_ref": data_refs.get("X_test"),
            # fe_y_test_ref is often not applicable or not created at this stage
        })
    else: 
        output_to_state.update({k: f"error_in_fe_for_{k.split('_',1)[1] if '_' in k else k}" for k in ["fe_applied_steps_summary", "fe_final_feature_list", "fe_numerical_features", "fe_categorical_features", "fe_X_train_ref", "fe_y_train_ref", "fe_X_val_ref", "fe_y_val_ref", "fe_X_test_ref"]})
    return output_to_state


def model_selection_decision_agent_node(state: MultiAgentPipelineState) -> Dict[str, any]:
    print("\n--- Model Selection Decision Agent Node Running ---")
    eda_report = state.get("eda_report", {}); 
    if eda_report.get("error"): return {"current_stage_completed": "ModelSelectionDecision", "top_model_configurations": []}
    
    eda_model_suggestions = eda_report.get("model_suggestions", [])
    # Use feature lists from FE state now
    fe_numerical_features = state.get("fe_numerical_features", []) 
    fe_categorical_features = state.get("fe_categorical_features", [])
    fe_final_feature_list_count = len(state.get("fe_final_feature_list", []))
    problem_type = state.get("problem_type", "regression") 
    
    decision_tool_context_hint = (f"EDA Model Suggestions: {json.dumps(eda_model_suggestions)}. "
                                  f"From FE: {len(fe_numerical_features)} numerical features (e.g., {json.dumps(fe_numerical_features[:3])}), "
                                  f"{len(fe_categorical_features)} categorical features (e.g., {json.dumps(fe_categorical_features[:3])}). "
                                  f"Total features: {fe_final_feature_list_count}. Problem type: {problem_type}.")

    prompt_content = f"""You are a Model Selection Strategist for stock price prediction (a regression task).
    Context: {decision_tool_context_hint}
    Your Task: Based on EDA suggestions, feature types (numerical/categorical counts), and total number of features, select up to 2-3 promising Scikit-learn REGRESSION model types.
    For each, suggest initial hyperparameters or state to use defaults.
    "Final Answer:" MUST be a single well-formed JSON object string, enclosed ONLY in ```json ... ```. NO OTHER TEXT.
    **All string values within JSON MUST be properly quoted and escaped.**
    JSON keys: "decision_rationale", "top_model_configurations": [ {{ "model_type": (string) "e.g.RandomForestRegressor", "initial_hyperparameters": {{ (object) "n_estimators":100 }}, "reasoning": "..." }} ]. Begin."""
    final_answer_json_string = run_generic_react_loop(prompt_content, 3, decision_tool_context_hint) 
    parsed_data = parse_llm_json_final_answer(final_answer_json_string, "Model selection decision failed.")
    return {"current_stage_completed": "ModelSelectionDecision", "top_model_configurations": parsed_data.get("top_model_configurations", [])}


def modeling_agent_node(state: MultiAgentPipelineState) -> Dict[str, any]:
    print("\n--- Modeling Agent Node Running (Standard Sklearn Pipeline Workflow) ---")
    top_model_configurations = state.get("top_model_configurations", [])
    # Get feature lists from FE stage for ColumnTransformer
    numerical_features = state.get("fe_numerical_features", [])
    categorical_features = state.get("fe_categorical_features", [])
    
    x_train_ref = state.get("fe_X_train_ref"); y_train_ref = state.get("fe_y_train_ref")
    x_val_ref = state.get("fe_X_val_ref"); y_val_ref = state.get("fe_y_val_ref") 
    target_rmse = state.get("target_rmse", 0.002) 
    config_idx = state.get("modeling_config_index", 0) 
    overall_best_rmse = state.get("best_rmse_so_far", float('inf'))
    overall_best_model_ref = state.get("best_model_ref_so_far"); overall_best_model_config = state.get("best_model_config_so_far")
    strategy_log_for_all_configs = state.get("modeling_strategy_log", [])
    max_configs_to_try = state.get("max_modeling_configs_to_try", len(top_model_configurations))

    if not top_model_configurations or not x_train_ref or \
       config_idx >= len(top_model_configurations) or config_idx >= max_configs_to_try:
        summary_msg = f"Completed trying {config_idx} model configurations or critical inputs missing. Best RMSE: {overall_best_rmse:.4f}."
        return {"current_stage_completed": "Modeling", "model_training_summary": summary_msg, "model_trained_pipeline_ref": overall_best_model_ref, "best_rmse_so_far": overall_best_rmse, "best_model_ref_so_far": overall_best_model_ref, "best_model_config_so_far": overall_best_model_config, "modeling_strategy_log": strategy_log_for_all_configs, "modeling_config_index": config_idx }

    current_config = top_model_configurations[config_idx]; chosen_model_type = current_config.get("model_type", "RandomForestRegressor"); initial_hyperparams = current_config.get("initial_hyperparameters", {})
    
    model_tool_context_hint = (f"Config(Idx{config_idx}): Type='{chosen_model_type}', Params='{json.dumps(initial_hyperparams)}'. "
                               f"Numerical features for StandardScaler: {json.dumps(numerical_features)}. "
                               f"Categorical features for OneHotEncoder: {json.dumps(categorical_features)}. "
                               f"Data(.pkl): X_train='{x_train_ref}', etc. Target RMSE: {target_rmse}.")
    
    prompt_content = f"""You are a Modeling Specialist for stock price prediction. PythonTool takes NL instructions. Context: {model_tool_context_hint}
    Task for THIS Configuration (Type: '{chosen_model_type}', Params: {json.dumps(initial_hyperparams)}):
    1. Instruct PythonTool to:
        a. Create an UNTRAINED Scikit-learn `Pipeline`. This pipeline should first use a `ColumnTransformer`.
           The `ColumnTransformer` should apply:
           - `StandardScaler()` to the following numerical features: {json.dumps(numerical_features)}.
           - `OneHotEncoder(handle_unknown='ignore')` to the following categorical features: {json.dumps(categorical_features)}.
           The second step in the main `Pipeline` should be the estimator: '{chosen_model_type}' with initial hyperparameters: {json.dumps(initial_hyperparams)}.
        b. Save this UNTRAINED full pipeline (ColumnTransformer + Estimator) as a .joblib file (e.g., 'untrained_pipeline_config{config_idx}.joblib') and report its .joblib reference.
        c. Load X_train ('{x_train_ref}') and y_train ('{y_train_ref}') (expected as .pkl). Train this entire pipeline.
        d. Save the TRAINED pipeline as a .joblib file (e.g., 'trained_pipeline_config{config_idx}.joblib') and report its .joblib reference.
        e. Load X_val ('{x_val_ref}') and y_val ('{y_val_ref}') (.pkl). Predict using the trained pipeline and calculate RMSE. Report the RMSE.
    "Final Answer:" MUST be a single well-formed JSON object string for THIS trial, enclosed ONLY in ```json ... ```. NO OTHER TEXT.
    **All string values within JSON MUST be properly quoted and escaped.**
    JSON keys: "config_trial_summary", "config_trained_pipeline_ref" (MUST be .joblib), "config_rmse", "model_type_tried", "hyperparameters_tried". Begin."""
    config_trial_final_answer_json = run_generic_react_loop(prompt_content, state.get("max_react_iterations", 7), model_tool_context_hint)
    parsed_config_trial_data = parse_llm_json_final_answer(config_trial_final_answer_json, f"Modeling trial {config_idx} failed.")
    output_to_state = {"modeling_config_index": config_idx + 1, "current_stage_completed": "Modeling_Config_Trial" }
    if "error" in parsed_config_trial_data: output_to_state.update({"current_rmse": None, "model_training_summary": parsed_config_trial_data.get("summary",f"Trial {config_idx} failed."), "modeling_strategy_log": strategy_log_for_all_configs + [f"CfgIdx{config_idx}({chosen_model_type}): FAILED"]})
    else:
        config_rmse = parsed_config_trial_data.get("config_rmse"); config_model_ref = parsed_config_trial_data.get("config_trained_pipeline_ref")
        new_best_rmse, new_best_model_ref, new_best_model_config = overall_best_rmse, overall_best_model_ref, overall_best_model_config
        if config_rmse is not None and isinstance(config_rmse, (int, float)) and config_rmse < overall_best_rmse: new_best_rmse, new_best_model_ref, new_best_model_config = config_rmse, config_model_ref, current_config
        output_to_state.update({"current_rmse": config_rmse, "best_rmse_so_far": new_best_rmse, "best_model_ref_so_far": new_best_model_ref, "best_model_config_so_far": new_best_model_config, "model_training_summary": parsed_config_trial_data.get("config_trial_summary"), "model_trained_pipeline_ref": config_model_ref, "modeling_strategy_log": strategy_log_for_all_configs + [f"CfgIdx{config_idx}({chosen_model_type}): RMSE={config_rmse}, Ref={config_model_ref}"]})
    return output_to_state


def evaluation_node(state: MultiAgentPipelineState) -> Dict[str, any]:
    print("\n--- Evaluation Agent Node Running ---")
    trained_pipeline_ref = state.get("best_model_ref_so_far", state.get("model_trained_pipeline_ref")) 
    if not trained_pipeline_ref or "error" in str(trained_pipeline_ref).lower() or ("default" in str(trained_pipeline_ref).lower()): # Basic check
         return {"current_stage_completed": "Evaluation", "evaluation_summary": "Skipped: No valid trained model reference.", "evaluation_metrics": {}}
    x_val_ref = state.get("fe_X_val_ref"); y_val_ref = state.get("fe_y_val_ref"); x_test_ref = state.get("fe_X_test_ref") 
    problem_type = state.get("problem_type"); custom_transformer_module = state.get("fe_custom_transformer_module") 
    best_model_config_info = state.get("best_model_config_so_far", {})
    eval_tool_context_hint = (f"Trained pipeline ref (best): '{trained_pipeline_ref}' (.joblib). Config: {json.dumps(best_model_config_info)}. Val X (.pkl): '{x_val_ref}', Val y (.pkl): '{y_val_ref}'. Test X (.pkl): '{x_test_ref if x_test_ref else 'N/A'}'. Problem: {problem_type}.")
    if custom_transformer_module: eval_tool_context_hint += f" Custom transformer module: '{custom_transformer_module}'."
    metrics_to_request = "MSE, RMSE, MAE, R-squared" 
    prompt_content = f"""You are an Evaluation Specialist... Context: {eval_tool_context_hint}
    Tasks: Load trained pipeline '{trained_pipeline_ref}'. Load val data. Predict. Calculate metrics: {metrics_to_request}. Report as dict string. (Opt) Predict on X_test.
    "Final Answer:" JSON keys: "evaluation_summary", "validation_metrics": {{metric:value}}, "test_set_prediction_status". Begin."""
    final_answer_json_string = run_generic_react_loop(prompt_content, state.get("max_react_iterations", 5), eval_tool_context_hint)
    parsed_data = parse_llm_json_final_answer(final_answer_json_string, "Evaluation report failed."); metrics_val = parsed_data.get("validation_metrics", {}); metrics = {}
    if isinstance(metrics_val, str): 
        try: metrics_str_cleaned = re.sub(r"^[^\(\{]*", "", metrics_val); metrics = json.loads(metrics_str_cleaned.replace("'", "\"")) if metrics_str_cleaned else {"error":"empty"}
        except: metrics = {"error": f"failed to parse metrics string: {metrics_val}"}
    else: metrics = metrics_val if isinstance(metrics_val, dict) else {}
    return {"current_stage_completed": "Evaluation", "evaluation_summary": parsed_data.get("evaluation_summary"), "evaluation_metrics": metrics, "test_set_prediction_status": parsed_data.get("test_set_prediction_status"), "model_trained_pipeline_ref": trained_pipeline_ref }


# --- 6. Graph Definition with Modeling Loop based on Configurations ---
def modeling_iteration_decision(state: MultiAgentPipelineState):
    print("\n--- Checking Modeling Iteration Decision ---")
    config_idx = state.get("modeling_config_index", 0); top_configurations = state.get("top_model_configurations", [])
    max_configs_to_try = state.get("max_modeling_configs_to_try", len(top_configurations) if top_configurations else 0) 
    current_best_rmse = state.get("best_rmse_so_far", float('inf')); target_rmse = state.get("target_rmse", 0.002)
    if not state.get("fe_X_train_ref") or "error" in str(state.get("fe_X_train_ref","")) or (config_idx == 0 and not top_configurations):
        print("  Critical error in FE/Decision or no model configs. Bypassing modeling iterations."); return "evaluation_agent"
    print(f"  Config Idx to try: {config_idx} / Total Configs: {len(top_configurations)} (Max to try: {max_configs_to_try}). Best RMSE: {current_best_rmse}, Target: {target_rmse}")
    if current_best_rmse <= target_rmse: print(f"  Target RMSE achieved. Evaluating."); return "evaluation_agent"
    if config_idx < len(top_configurations) and config_idx < max_configs_to_try: print(f"  Proceeding to try model configuration at index {config_idx}."); return "modeling_agent"
    else: print(f"  All model configs tried or limit reached. Evaluating."); return "evaluation_agent"

workflow = StateGraph(MultiAgentPipelineState)
# ... (Graph definition remains the same as previous version) ...
workflow.add_node("eda_agent", eda_agent_node) 
workflow.add_node("feature_engineering_agent", feature_engineering_agent_node)
workflow.add_node("model_selection_decision_agent", model_selection_decision_agent_node) 
workflow.add_node("modeling_agent", modeling_agent_node) 
workflow.add_node("evaluation_agent", evaluation_node) 
workflow.set_entry_point("eda_agent")
workflow.add_edge("eda_agent", "feature_engineering_agent")
workflow.add_edge("feature_engineering_agent", "model_selection_decision_agent") 
workflow.add_edge("model_selection_decision_agent", "modeling_agent") 
workflow.add_conditional_edges("modeling_agent", modeling_iteration_decision, { "modeling_agent": "modeling_agent", "evaluation_agent": "evaluation_agent" })
workflow.add_edge("evaluation_agent", END)
pipeline_app = workflow.compile()


# --- 7. Example Invocation ---
if __name__ == "__main__":
    # ... (Main block same as previous version) ...
    print("Starting ML Pipeline with Enhanced JSON/Ref Robustness & Standard Sklearn Workflow...")
    os.makedirs("dummy_pipeline_data", exist_ok=True)
    initial_data_paths = { "train": "dummy_pipeline_data/train_data.csv", "val": "dummy_pipeline_data/val_data.csv", "test": "dummy_pipeline_data/test_data.csv"}
    dummy_header = "Date,Price,Volume,FeatureA,FeatureB,Category,CustomText,Target\n"; dummy_row_template = "{date_val},{price},{volume},{fA},{fB},{cat},{text},{target}\n"
    for k, v_path in initial_data_paths.items():
        with open(v_path, "w") as f: f.write(dummy_header)
        for i in range(10): 
            year_str, month_str, day_str = "2023", f"{((i%12)+1):02d}", f"{((i%28)+1):02d}" 
            date_val = f"{year_str}-{month_str}-{day_str}" 
            f.write(dummy_row_template.format(date_val=date_val, price=100+i*random.uniform(-2,2) + (i*0.5), volume=10000+i*100 + random.randint(-500,500), fA=0.5+i*0.01, fB=1.2-i*0.01, cat='TypeA' if i%3==0 else ('TypeB' if i%3==1 else 'TypeC'), text=f"Txt{i}", target= (101+i*0.25 + random.uniform(-1,1)))) 
    initial_pipeline_state = {
        "data_paths": initial_data_paths, "target_column_name": "Target", "problem_type": "regression",   
        "max_react_iterations": 6, "target_rmse": 0.75, "max_modeling_configs_to_try": 2, 
        "modeling_config_index": 0, "best_rmse_so_far": float('inf'), 
    }
    config = {"configurable": {"thread_id": f"ml_pipeline_robust_json_refs_{random.randint(1000,9999)}"}}
    print("\nInvoking pipeline stream:"); final_state_accumulator = {} 
    for chunk in pipeline_app.stream(initial_pipeline_state, config=config, stream_mode="updates"):
        for node_name_in_chunk, node_output_dict in chunk.items(): 
            print(f"\n<<< Update from Node: {node_name_in_chunk} >>>")
            if isinstance(node_output_dict, dict): final_state_accumulator.update(node_output_dict) 
            for k_item, v_item in node_output_dict.items(): print(f"  {k_item}: {str(v_item)[:350]}...")
            else: print(f"  Unexpected output format from node {node_name_in_chunk}: {str(node_output_dict)[:350]}...")
    print("\n\n--- Final Pipeline State (from accumulated stream) ---")
    if final_state_accumulator: print(json.dumps(final_state_accumulator, indent=2, default=str))
    for v_path in initial_data_paths.values():
        if os.path.exists(v_path): os.remove(v_path)
    if os.path.exists("dummy_pipeline_data"): os.rmdir("dummy_pipeline_data")
    print("\nMulti-Agent Pipeline Finished.")
