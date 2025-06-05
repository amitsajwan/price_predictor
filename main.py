import os
import re
import json 
import random # For simulating RMSE changes
from typing import TypedDict, Annotated, List, Dict, Optional, Sequence, Literal
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI 
from langgraph.graph import StateGraph, END
# from langgraph.checkpoint.sqlite import SqliteSaver # Example for persistence

# --- 1. Define the State for the Pipeline ---
class MultiAgentPipelineState(TypedDict):
    # Input
    data_paths: Dict[str, str] 
    target_column_name: Optional[str] 
    problem_type: Optional[Literal["classification", "regression"]] 
    target_rmse: Optional[float] 

    # Output from EdaAgentNode: A structured report
    eda_report: Optional[Dict[str, any]] 

    # Output from FeatureEngineeringAgentNode
    fe_applied_steps_summary: Optional[str]
    fe_final_feature_list: Optional[List[str]] 
    fe_X_train_ref: Optional[str] # Expect .pkl if DataFrame/Series
    fe_y_train_ref: Optional[str] # Expect .pkl
    fe_X_val_ref: Optional[str]   # Expect .pkl
    fe_y_val_ref: Optional[str]   # Expect .pkl
    fe_X_test_ref: Optional[str]  # Expect .pkl
    fe_transformer_references: Optional[Dict[str, str]] # Expect .joblib
    fe_custom_transformer_module: Optional[str] 
    
    # Output from ModelSelectionDecisionNode
    top_model_configurations: Optional[List[Dict[str, any]]] 

    # Modeling Node State & Output
    model_training_summary: Optional[str] 
    model_trained_pipeline_ref: Optional[str] # Expect .joblib
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

    # Control and tracking
    current_stage_completed: Optional[str]
    max_react_iterations: Optional[int]


# --- 2. Interface for your Agnostic PythonTool ---
# REPLACE THIS FUNCTION WITH THE ACTUAL CALL TO YOUR AGNO_PYTHON_TOOL
SIMULATED_MODEL_PERFORMANCE_REGISTRY = {} 

def agno_python_tool_interface(instruction: str, agent_context_hint: Optional[str] = None) -> str:
    global SIMULATED_MODEL_PERFORMANCE_REGISTRY
    print(f"    [AGNO_PYTHON_TOOL INTERFACE] Sending Instruction to your tool:\n    '{instruction}'")
    if agent_context_hint:
        print(f"    Agent Context Hint (passed to your tool): {agent_context_hint}")
    
    sim_observation = f"Observation: PythonTool processed instruction: '{instruction}'. "
    instruction_lower = instruction.lower()

    if "load the dataset from" in instruction_lower and "report its reference" in instruction_lower:
        ref = "unknown_loaded_ref.pkl"
        if "train_data.csv" in instruction: ref = "train_df_initial_ref.pkl"
        elif "val_data.csv" in instruction: ref = "val_df_initial_ref.pkl"
        elif "test_data.csv" in instruction: ref = "test_df_initial_ref.pkl"
        sim_observation += f"Dataset loaded. Tool reports its reference as '{ref}'."
    elif "identify the 'date' column in" in instruction_lower and "parse it as datetime using format 'yyyy-mm-dd'" in instruction_lower:
        ref_match = re.search(r"in '([^']+)'", instruction_lower)
        data_ref = ref_match.group(1) if ref_match else "unknown_ref"
        sim_observation += f"'Date' column in '{data_ref}' identified and parsed to datetime64[ns] using YYYY-MM-DD format. Confirmed."
    elif "create a temporary numeric-only version of" in instruction_lower and "excluding the original 'date' column" in instruction_lower and "report the new reference" in instruction_lower:
        ref_match = re.search(r"version of '([^']+)'", instruction_lower)
        data_ref = ref_match.group(1) if ref_match else "unknown_ref"
        new_numeric_ref = f"numeric_version_of_{data_ref.replace('.pkl','')}_for_corr.pkl"
        sim_observation += f"Numeric-only version of '{data_ref}' created. New reference reported by tool: '{new_numeric_ref}'."
    elif "extract year, month, day, dayofweek from the 'date' column" in instruction_lower and "drop the original 'date' column" in instruction_lower and "report new data references as .pkl files" in instruction_lower: # Explicit about .pkl
        sim_observation += ("Date features extracted. Original 'Date' column dropped. "
                           "New data references reported by tool: train='train_with_date_features.pkl', val='val_with_date_features.pkl', test='test_with_date_features.pkl'.")
    elif "clean data referenced by" in instruction_lower and "report the new reference as a .pkl file" in instruction_lower: 
        ref_match = re.search(r"referenced by '([^']+)'", instruction_lower)
        data_ref = ref_match.group(1) if ref_match else "unknown_ref"
        new_cleaned_ref = f"cleaned_{data_ref.replace('.pkl','_eda_final.pkl')}"
        sim_observation += f"Data '{data_ref}' cleaned. New reference reported by tool: '{new_cleaned_ref}'."
    elif "generate a histogram for" in instruction_lower and "save it, report the filename, and provide a textual description" in instruction_lower:
        sim_observation += "Histogram generated. Plot saved by tool as 'sim_histogram.png'. Description: [Simulated detailed description]."
    elif "fit a standardscaler" in instruction_lower and "save it as a .joblib file and report its reference" in instruction_lower:
        sim_observation += "StandardScaler fitted. Saved as 'fitted_scaler.joblib'. This is its reference."
    elif "create an untrained scikit-learn pipeline" in instruction_lower and "save it as a .joblib file and report its reference" in instruction_lower:
        model_type_match = re.search(r"estimator type '([^']+)'", instruction_lower)
        model_type = model_type_match.group(1) if model_type_match else "UnknownModel"
        ref = f"untrained_pipeline_{model_type}.joblib"
        sim_observation += f"Untrained Scikit-learn pipeline with {model_type} created. Saved. Reference is '{ref}'."
    elif "load the untrained pipeline" in instruction_lower and "train it using x_train" in instruction_lower:
        untrained_pipe_ref = re.search(r"untrained pipeline '([^']+)'", instruction_lower).group(1) if re.search(r"untrained pipeline '([^']+)'", instruction_lower) else "unknown_pipe.joblib"
        params_key = "default"; model_type_sim = "DefaultModel"
        if "randomforest" in untrained_pipe_ref: model_type_sim="RandomForest" 
        if "n_estimators=200" in instruction_lower: params_key = "n200"
        full_model_key = f"{model_type_sim}_{params_key}"
        if full_model_key not in SIMULATED_MODEL_PERFORMANCE_REGISTRY: SIMULATED_MODEL_PERFORMANCE_REGISTRY[full_model_key] = random.uniform(0.6, 2.5) 
        else: SIMULATED_MODEL_PERFORMANCE_REGISTRY[full_model_key] *= random.uniform(0.85, 0.99) 
        current_sim_rmse = SIMULATED_MODEL_PERFORMANCE_REGISTRY[full_model_key]
        trained_model_ref = f"trained_{untrained_pipe_ref.replace('.pkl','.joblib').replace('untrained_','')}" 
        sim_observation += (f"Pipeline '{untrained_pipe_ref}' trained. Saved. Reference is '{trained_model_ref}'.")
        if "report rmse" in instruction_lower: sim_observation += f" Validation RMSE: {current_sim_rmse:.4f}."
    elif "separate target" in instruction_lower and "save them as .pkl files and report new references" in instruction_lower: # Explicit about .pkl
        sim_observation += "Target separated. Tool reports new references: X_train='X_train_final.pkl', y_train='y_train_final.pkl', X_val='X_val_final.pkl', y_val='y_val_final.pkl', X_test='X_test_final.pkl'."
    elif "calculate regression metrics" in instruction_lower: 
        sim_observation += "Metrics reported by tool: {{'rmse': {random.uniform(0.1,0.8):.4f}, 'r_squared': {random.uniform(0.6,0.9):.2f}}}."
    else:
        sim_observation += "Task completed. If specific artifacts were requested to be saved (e.g. as .pkl or .joblib) and their references reported, those details are included above."
            
    print(f"    [AGNO_PYTHON_TOOL INTERFACE] Returning Observation:\n    '{sim_observation}'")
    return sim_observation

# --- 3. Generic ReAct Loop Engine ---
llm = ChatOpenAI(model="gpt-4o", temperature=0.1) 
def run_generic_react_loop(initial_prompt_content: str, max_iterations: int, agent_context_hint_for_tool: Optional[str] = None) -> str: 
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
            react_messages.append(HumanMessage(content="System hint: Ensure valid format ('Action: Python...' or 'Final Answer: ```json...```'). No extra text outside the JSON block for Final Answer."))
            if i > 1: final_answer_text = json.dumps({"error": "Agent failed to follow output format consistently."}); print(f"    Agent failed to follow output format."); break 
        if i == max_iterations - 1: print(f"    Max ReAct iterations reached."); final_answer_text = json.dumps({"error": f"Max iterations. Last thought: {ai_content}"})
    return final_answer_text

# --- 4. Helper to Parse LLM's JSON Final Answer ---
def parse_llm_json_final_answer(final_answer_json_string: str, default_error_message: str = "Report generation failed.") -> Dict:
    try:
        match = re.search(r"```json\s*(.*?)\s*```", final_answer_json_string, re.DOTALL)
        if match: json_string_cleaned = match.group(1).strip()
        else: 
            json_string_cleaned = final_answer_json_string.strip()
            if json_string_cleaned.startswith("```") and json_string_cleaned.endswith("```"): json_string_cleaned = json_string_cleaned[3:-3].strip()
        json_string_cleaned = re.sub(r",\s*([}\]])", r"\1", json_string_cleaned); json_string_cleaned = re.sub(r"//.*?\n", "\n", json_string_cleaned) 
        json_string_cleaned = re.sub(r"^\s*#.*?\n", "\n", json_string_cleaned, flags=re.MULTILINE)
        if not json_string_cleaned: return {"error": "JSON string empty after cleaning", "summary": default_error_message, "original_string": final_answer_json_string}
        return json.loads(json_string_cleaned)
    except Exception as e: print(f"    ERROR parsing LLM JSON: {e}, String: {final_answer_json_string}"); return {"error": str(e), "summary": default_error_message, "original_string": final_answer_json_string}


# --- 5. Define Agent Nodes ---

def eda_agent_node(state: MultiAgentPipelineState) -> Dict[str, any]:
    print("\n--- EDA Agent Node Running (Reinforced Date & JSON Handling) ---")
    data_paths = state["data_paths"]
    target_col = state.get("target_column_name", "Target")
    problem_type = state.get("problem_type", "regression") 
    eda_tool_context_hint = f"Initial data paths: {json.dumps(data_paths)}. Target column: '{target_col}'. Problem type: {problem_type}. Task: EDA for stock price prediction."

    prompt_content = f"""You are an Expert EDA Data Scientist creating an 'EDA Manual'.
    The PythonTool you use accepts natural language instructions. It will report back references (FULL FILENAMES including extensions like .pkl for data, .png for plots, .joblib for sklearn objects).
    When you instruct the PythonTool to generate a plot, you MUST ALSO instruct it to provide a textual description of the plot's key features.
    You MUST instruct the tool to report references for ALL created/modified data and plots.

    Initial context for PythonTool: {eda_tool_context_hint}

    Your EDA Process (CRITICAL: Address 'Date' column named 'Date' with format 'YYYY-MM-DD' explicitly and EARLY):
    1.  **Load Datasets:** Instruct tool to load train, val, test datasets from paths in context. Ask it to report the references it assigns (e.g., 'initial_train_ref.pkl').
    2.  **Parse 'Date' Column:** Instruct tool: "For dataset 'initial_train_ref.pkl', identify 'Date' column. Parse it as datetime objects using format 'YYYY-MM-DD'. Confirm parsing and report the new dtype." Repeat for val/test.
    3.  **Initial Structure & Quality:** Using date-parsed references, check shapes, dtypes (confirm 'Date' is datetime), head/tail, missing values, outliers (request plot refs & descriptions for '{target_col}', 'Price', 'Volume').
    4.  **Prepare for Numeric EDA:** Instruct tool: "Using 'initial_train_ref.pkl' (with parsed 'Date'), create a TEMPORARY numeric-only version FOR CORRELATION by EXCLUDING 'Date' (datetime object) and any other non-numeric columns (except target '{target_col}'). Report NEW reference (e.g., 'train_numeric_corr.pkl')."
    5.  **Correlations:** Using 'train_numeric_corr.pkl', compute correlations. Ask for heatmap plot ref & description.
    6.  **Distribution & Time Series Analysis:** Using main data refs (with parsed dates), analyze distributions of '{target_col}', 'Price'. Plot '{target_col}' over parsed 'Date'. Plot refs & descriptions.
    7.  **Final Cleaning & References:** If general cleaning is done on main data refs, instruct tool: "Perform final cleaning on 'initial_train_ref.pkl'. Save resulting dataset as a .pkl file and report its reference as 'cleaned_train_final_eda.pkl'." Similarly for val/test.
    8.  **Model & FE Suggestions.** For dates, state: "FE Suggestion: From parsed 'Date' column in 'cleaned_train_final_eda.pkl', extract numerical features (Year, Month, Day, DayOfWeek). Then, the original 'Date' (datetime object) column MUST BE DROPPED before modeling."
    9.  Conclude.

    ReAct Format: Action: Python, Action Input: <NL instruction>.
    "Final Answer:" MUST be a single well-formed JSON object string, enclosed ONLY in ```json ... ```. NO OTHER TEXT OR COMMENTS OUTSIDE THIS BLOCK.
    **All string values within the JSON MUST be properly quoted and escaped (e.g., newlines as \\n, double quotes as \\\", backslashes as \\\\).**
    (JSON structure as defined previously - "eda_summary", "data_profile", "data_quality_report", "key_insights", "model_suggestions", "fe_suggestions", "artifact_references" with "processed_train_data":"<ref.pkl>", "plots":{{...}})
    The "artifact_references" MUST contain tool-reported references for all final processed data (as .pkl) and key plots (as .png).
    Begin.
    """
    final_answer_json_string = run_generic_react_loop(prompt_content, state.get("max_react_iterations", 20), eda_tool_context_hint)
    eda_report_output = parse_llm_json_final_answer(final_answer_json_string, "EDA report generation failed.")
    output_to_state = {"current_stage_completed": "EDA", "eda_report": eda_report_output}
    if "error" not in eda_report_output:
        artifact_refs = eda_report_output.get("artifact_references", {})
        output_to_state.update({
            "eda_model_suggestions": eda_report_output.get("model_suggestions", []), 
            "eda_fe_suggestions": eda_report_output.get("fe_suggestions", []),
            "eda_processed_train_ref": artifact_refs.get("processed_train_data"),
            "eda_processed_val_ref": artifact_refs.get("processed_val_data"),
            "eda_processed_test_ref": artifact_refs.get("processed_test_data"),
            "eda_numeric_train_ref_for_correlation": artifact_refs.get("numeric_train_for_correlation"), # If EDA creates this
            "eda_plot_references": artifact_refs.get("plots", {})
        })
    else: 
        output_to_state.update({k: [] for k in ["eda_model_suggestions", "eda_fe_suggestions"]})
        output_to_state.update({k: f"error_in_eda_{k.replace('eda_','').replace('_ref','')}.pkl" for k in ["eda_processed_train_ref", "eda_processed_val_ref", "eda_processed_test_ref", "eda_numeric_train_ref_for_correlation"]})
        output_to_state["eda_plot_references"] = {}
    return output_to_state

def feature_engineering_agent_node(state: MultiAgentPipelineState) -> Dict[str, any]:
    print("\n--- Feature Engineering Agent Node Running (Strict Date & PKL/Joblib Handling) ---")
    eda_report = state.get("eda_report", {}) 
    if eda_report.get("error"): return {"current_stage_completed": "FeatureEngineering", "fe_applied_steps_summary": "FE skipped due to EDA errors."}

    train_ref_from_eda = eda_report.get("artifact_references", {}).get("processed_train_data", "default_train_eda.pkl") 
    val_ref_from_eda = eda_report.get("artifact_references", {}).get("processed_val_data", "default_val_eda.pkl")
    test_ref_from_eda = eda_report.get("artifact_references", {}).get("processed_test_data", "default_test_eda.pkl")
    suggestions_from_eda = eda_report.get("fe_suggestions", [])
    target_col = state.get("target_column_name", "Target")

    date_fe_suggestion_from_eda = "PRIORITY: If EDA suggested extracting date features and DROPPING original 'Date' column, implement this first. Tool must report NEW data references (as .pkl)."
    for suggestion in suggestions_from_eda:
        if "date column" in suggestion.lower() and "extract" in suggestion.lower() and "drop" in suggestion.lower():
            date_fe_suggestion_from_eda = suggestion; break

    fe_tool_context_hint = (f"Input data refs from EDA (expected as .pkl): train='{train_ref_from_eda}', val='{val_ref_from_eda}', test='{test_ref_from_eda}'. Target: '{target_col}'. EDA FE Suggestions: {json.dumps(suggestions_from_eda)}.")

    prompt_content = f"""You are a Feature Engineering Specialist for stock price prediction. PythonTool takes NL instructions.
    Context from EDA:
    - Input Train Data Ref (cleaned by EDA, .pkl): {train_ref_from_eda}
    - Input Val Data Ref (.pkl): {val_ref_from_eda}
    - Input Test Data Ref (.pkl): {test_ref_from_eda}
    - EDA FE Suggestions: {json.dumps(suggestions_from_eda)}
    - Specific EDA instruction for 'Date' column (PRIORITIZE THIS): "{date_fe_suggestion_from_eda}"
    - Target Column: '{target_col}'

    Your tasks:
    1. Instruct tool to use/load datasets using FINAL PROCESSED .pkl references from EDA.
    2. **CRITICAL FIRST FE STEP (Date Handling):** Execute EDA's suggestion for 'Date' column: "{date_fe_suggestion_from_eda}". Instruct tool to extract numerical date features AND THEN DROP ORIGINAL 'Date' (datetime object) COLUMN. Tool must report new data references for these (as .pkl files). Use these NEW references for subsequent FE steps.
    3. Implement other FE steps. Apply consistently.
    4. Create and SAVE individual FITTED transformers (scalers, encoders) as .joblib files. Ask tool to report full .joblib filename references.
    5. After ALL transformations, separate features (X) and target ('{target_col}'). Ask tool to save X_train, y_train, etc., as .pkl files and report their full .pkl filename references. Also ask for the final feature list (NO original 'Date' column).

    ReAct Format: Action: Python, Action Input: <NL instruction>.
    "Final Answer:" MUST be a single well-formed JSON object string, enclosed ONLY in ```json ... ```. NO OTHER TEXT.
    **All string values within JSON MUST be properly quoted and escaped.**
    JSON keys: "fe_summary", "final_feature_list", 
               "transformer_references": {{ "scaler_price": "<ref.joblib>", ... }}, 
               "custom_transformer_module": (string or null),
               "data_references": {{ "X_train": "<X_train_ref.pkl>", "y_train": "<y_train_ref.pkl>", ... }}
    Begin. Address date feature engineering first.
    """
    final_answer_json_string = run_generic_react_loop(prompt_content, state.get("max_react_iterations", 12), fe_tool_context_hint)
    parsed_data = parse_llm_json_final_answer(final_answer_json_string, "FE report generation failed.")
    
    output_to_state = {"current_stage_completed": "FeatureEngineering"}
    if "error" not in parsed_data:
        data_refs = parsed_data.get("data_references", {})
        output_to_state.update({
            "fe_applied_steps_summary": parsed_data.get("fe_summary"),
            "fe_final_feature_list": parsed_data.get("final_feature_list", []),
            "fe_transformer_references": parsed_data.get("transformer_references", {}),
            "fe_custom_transformer_module": parsed_data.get("custom_transformer_module"), 
            "fe_X_train_ref": data_refs.get("X_train", "default_X_train_fe.pkl"), 
            "fe_y_train_ref": data_refs.get("y_train", "default_y_train_fe.pkl"),
            "fe_X_val_ref": data_refs.get("X_val", "default_X_val_fe.pkl"),
            "fe_y_val_ref": data_refs.get("y_val", "default_y_val_fe.pkl"),
            "fe_X_test_ref": data_refs.get("X_test", "default_X_test_fe.pkl"),
            "fe_y_test_ref": data_refs.get("y_test") 
        })
    else: 
        output_to_state.update({k: f"error_in_fe_{k.split('_',1)[1] if '_' in k else k}" for k in ["fe_applied_steps_summary", "fe_final_feature_list", "fe_transformer_references", "fe_X_train_ref", "fe_y_train_ref", "fe_X_val_ref", "fe_y_val_ref", "fe_X_test_ref"]})
    return output_to_state


def model_selection_decision_agent_node(state: MultiAgentPipelineState) -> Dict[str, any]:
    print("\n--- Model Selection Decision Agent Node Running ---")
    eda_report = state.get("eda_report", {}); 
    if eda_report.get("error"): return {"current_stage_completed": "ModelSelectionDecision", "top_model_configurations": []} 
    eda_model_suggestions = eda_report.get("model_suggestions", [])
    fe_final_feature_list = state.get("fe_final_feature_list", [])
    problem_type = state.get("problem_type", "regression") 
    decision_tool_context_hint = (f"EDA Model Suggestions: {json.dumps(eda_model_suggestions)}. Final Features from FE ({len(fe_final_feature_list)} features, expect .pkl): {json.dumps(fe_final_feature_list[:5])}. Problem: {problem_type}.")
    prompt_content = f"""You are a Model Selection Strategist for predicting stock prices (regression). Context: {decision_tool_context_hint}
    Task: Based on EDA suggestions and final features, select up to 2-3 promising Scikit-learn REGRESSION model types. For each, suggest initial hyperparameters or defaults.
    "Final Answer:" MUST be a single well-formed JSON object string, enclosed ONLY in ```json ... ```. NO OTHER TEXT.
    **All string values within JSON MUST be properly quoted and escaped.**
    JSON keys: "decision_rationale", "top_model_configurations": [ {{ "model_type": "e.g.RandomForestRegressor", "initial_hyperparameters": {{}}, "reasoning": "..." }} ]. Begin."""
    final_answer_json_string = run_generic_react_loop(prompt_content, 3, decision_tool_context_hint) 
    parsed_data = parse_llm_json_final_answer(final_answer_json_string, "Model selection decision failed.")
    return {"current_stage_completed": "ModelSelectionDecision", "top_model_configurations": parsed_data.get("top_model_configurations", [])}


def modeling_agent_node(state: MultiAgentPipelineState) -> Dict[str, any]:
    print("\n--- Modeling Agent Node Running (Iterates Top Configurations for RMSE) ---")
    top_model_configurations = state.get("top_model_configurations", [])
    transformer_refs_from_fe = state.get("fe_transformer_references", {})
    x_train_ref = state.get("fe_X_train_ref", "X_train_fe.pkl"); y_train_ref = state.get("fe_y_train_ref", "y_train_fe.pkl") # Expect .pkl
    x_val_ref = state.get("fe_X_val_ref", "X_val_fe.pkl"); y_val_ref = state.get("fe_y_val_ref", "y_val_fe.pkl") # Expect .pkl
    custom_module = state.get("fe_custom_transformer_module")
    target_rmse = state.get("target_rmse", 0.002) 
    config_idx = state.get("modeling_config_index", 0) 
    overall_best_rmse = state.get("best_rmse_so_far", float('inf'))
    overall_best_model_ref = state.get("best_model_ref_so_far"); overall_best_model_config = state.get("best_model_config_so_far")
    strategy_log_for_all_configs = state.get("modeling_strategy_log", [])
    max_configs_to_try = state.get("max_modeling_configs_to_try", len(top_model_configurations))

    if not top_model_configurations or config_idx >= len(top_model_configurations) or config_idx >= max_configs_to_try:
        summary_msg = f"Completed trying {config_idx} model configurations. Best RMSE: {overall_best_rmse:.4f}."
        # Ensure the overall best model reference is correctly propagated
        final_output = {"current_stage_completed": "Modeling", "model_training_summary": summary_msg, 
                        "model_trained_pipeline_ref": overall_best_model_ref, # This should be the best one
                        "best_rmse_so_far": overall_best_rmse, 
                        "best_model_ref_so_far": overall_best_model_ref, 
                        "best_model_config_so_far": overall_best_model_config, 
                        "modeling_strategy_log": strategy_log_for_all_configs, 
                        "modeling_config_index": config_idx }
        if not overall_best_model_ref and top_model_configurations: # If all attempts failed to report a ref
             final_output["model_trained_pipeline_ref"] = "error_no_valid_model_trained.joblib"
        return final_output


    current_config = top_model_configurations[config_idx]; chosen_model_type = current_config.get("model_type", "RandomForestRegressor"); initial_hyperparams = current_config.get("initial_hyperparameters", {})
    model_tool_context_hint = (f"Config (Idx {config_idx}): Type='{chosen_model_type}', Params='{json.dumps(initial_hyperparams)}'. Transformers (expect .joblib): {json.dumps(transformer_refs_from_fe)}. Custom module: '{custom_module}'. Data (expect .pkl): X_train='{x_train_ref}', etc. Target RMSE: {target_rmse}.")
    prompt_content = f"""You are a Modeling Specialist for stock price prediction. PythonTool takes NL. Context: {model_tool_context_hint}
    Task for THIS Config ('{chosen_model_type}', Params: {json.dumps(initial_hyperparams)}):
    1. Instruct PythonTool to:
        a. Create UNTRAINED Scikit-learn pipeline: combine preprocessors loaded using .joblib references from '{json.dumps(transformer_refs_from_fe)}' with estimator '{chosen_model_type}' using params '{json.dumps(initial_hyperparams)}'. (Custom module '{custom_module}' if specified).
        b. Save this UNTRAINED pipeline (e.g., 'untrained_cfg{config_idx}.joblib') & report .joblib ref.
        c. Load X_train ('{x_train_ref}') and y_train ('{y_train_ref}') (expected as .pkl). Train pipeline.
        d. Save TRAINED pipeline (e.g., 'trained_cfg{config_idx}.joblib') & report .joblib ref.
        e. Load X_val ('{x_val_ref}') and y_val ('{y_val_ref}') (.pkl). Predict & calculate RMSE. Report RMSE.
    "Final Answer:" JSON for THIS trial: "config_trial_summary", "config_trained_pipeline_ref" (MUST be .joblib), "config_rmse", "model_type_tried", "hyperparameters_tried". Begin."""
    config_trial_final_answer_json = run_generic_react_loop(prompt_content, state.get("max_react_iterations", 7), model_tool_context_hint)
    parsed_config_trial_data = parse_llm_json_final_answer(config_trial_final_answer_json, f"Modeling config trial {config_idx} failed.")
    
    output_to_state = {"modeling_config_index": config_idx + 1, "current_stage_completed": "Modeling_Config_Trial" }
    if "error" in parsed_config_trial_data:
        output_to_state.update({"current_rmse": None, "model_training_summary": parsed_config_trial_data.get("summary", f"Trial {config_idx} failed."), "modeling_strategy_log": strategy_log_for_all_configs + [f"ConfigIdx{config_idx}({chosen_model_type}): FAILED - {parsed_config_trial_data.get('error')}" ]})
    else:
        config_rmse = parsed_config_trial_data.get("config_rmse"); config_model_ref = parsed_config_trial_data.get("config_trained_pipeline_ref")
        new_best_rmse, new_best_model_ref, new_best_model_config = overall_best_rmse, overall_best_model_ref, overall_best_model_config
        if config_rmse is not None and isinstance(config_rmse, (int, float)) and config_rmse < overall_best_rmse: 
            new_best_rmse, new_best_model_ref, new_best_model_config = config_rmse, config_model_ref, current_config
        output_to_state.update({
            "current_rmse": config_rmse, "best_rmse_so_far": new_best_rmse, "best_model_ref_so_far": new_best_model_ref, 
            "best_model_config_so_far": new_best_model_config, "model_training_summary": parsed_config_trial_data.get("config_trial_summary"), 
            "model_trained_pipeline_ref": config_model_ref, # This attempt's model
            "modeling_strategy_log": strategy_log_for_all_configs + [f"ConfigIdx{config_idx}({chosen_model_type}): RMSE={config_rmse}, Ref={config_model_ref}"]
        })
    return output_to_state


def evaluation_node(state: MultiAgentPipelineState) -> Dict[str, any]:
    print("\n--- Evaluation Agent Node Running ---")
    trained_pipeline_ref = state.get("best_model_ref_so_far", state.get("model_trained_pipeline_ref")) 
    if not trained_pipeline_ref or "error" in str(trained_pipeline_ref).lower() or "default" in str(trained_pipeline_ref).lower() and not os.path.exists(str(trained_pipeline_ref)): # More robust check
        return {"current_stage_completed": "Evaluation", "evaluation_summary": "Skipped: No valid trained model reference from modeling.", "evaluation_metrics": {}}

    x_val_ref = state.get("fe_X_val_ref"); y_val_ref = state.get("fe_y_val_ref"); x_test_ref = state.get("fe_X_test_ref") 
    problem_type = state.get("problem_type"); custom_transformer_module = state.get("fe_custom_transformer_module") 
    best_model_config_info = state.get("best_model_config_so_far", {})
    eval_tool_context_hint = (f"Trained pipeline ref (best from tuning): '{trained_pipeline_ref}' (expect .joblib). Config: {json.dumps(best_model_config_info)}. Val X (.pkl): '{x_val_ref}', Val y (.pkl): '{y_val_ref}'. Test X (.pkl): '{x_test_ref if x_test_ref else 'N/A'}'. Problem: {problem_type}.")
    if custom_transformer_module: eval_tool_context_hint += f" Custom transformer module: '{custom_transformer_module}'."
    metrics_to_request = "MSE, RMSE, MAE, R-squared" 
    prompt_content = f"""You are an Evaluation Specialist for a stock price prediction model. PythonTool takes NL instructions. Context: {eval_tool_context_hint}
    Tasks: Load trained pipeline '{trained_pipeline_ref}' (a .joblib file). Load validation data X_val from '{x_val_ref}' and y_val from '{y_val_ref}' (as .pkl). Make predictions on X_val. Calculate metrics: {metrics_to_request}. Report as dict string. (Optional) Predict on X_test.
    "Final Answer:" JSON keys: "evaluation_summary", "validation_metrics": {{metric:value}}, "test_set_prediction_status". Begin."""
    final_answer_json_string = run_generic_react_loop(prompt_content, state.get("max_react_iterations", 5), eval_tool_context_hint)
    parsed_data = parse_llm_json_final_answer(final_answer_json_string, "Evaluation report failed.")
    metrics_val = parsed_data.get("validation_metrics", {}); metrics = {}
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
    
    # Check for critical errors from previous stages
    if not state.get("fe_X_train_ref") or not top_configurations:
        print("  Critical error in FE or Model Selection, cannot proceed with modeling iterations. Going to evaluation with current best (if any).")
        return "evaluation_agent"

    print(f"  Config Idx to try: {config_idx} / Total Configs Provided: {len(top_configurations)} (Max to try: {max_configs_to_try}). Best RMSE: {current_best_rmse}, Target: {target_rmse}")
    if current_best_rmse <= target_rmse: print(f"  Target RMSE achieved. Evaluating."); return "evaluation_agent"
    if config_idx < len(top_configurations) and config_idx < max_configs_to_try: print(f"  Proceeding to try model configuration at index {config_idx}."); return "modeling_agent"
    else: print(f"  All model configs tried or limit reached. Evaluating."); return "evaluation_agent"

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
workflow.add_conditional_edges("modeling_agent", modeling_iteration_decision, { "modeling_agent": "modeling_agent", "evaluation_agent": "evaluation_agent" })
workflow.add_edge("evaluation_agent", END)
pipeline_app = workflow.compile()


# --- 7. Example Invocation ---
if __name__ == "__main__":
    print("Starting ML Pipeline with Robust Error Handling for JSON and File Refs...")

    os.makedirs("dummy_pipeline_data", exist_ok=True)
    initial_data_paths = { "train": "dummy_pipeline_data/train_data.csv", "val": "dummy_pipeline_data/val_data.csv", "test": "dummy_pipeline_data/test_data.csv"}
    dummy_header = "Date,Price,Volume,FeatureA,FeatureB,Category,CustomText,Target\n" 
    dummy_row_template = "{date_val},{price},{volume},{fA},{fB},{cat},{text},{target}\n"
    for k, v_path in initial_data_paths.items():
        with open(v_path, "w") as f: f.write(dummy_header)
        for i in range(10): 
            year_str, month_str, day_str = "2023", f"{((i%12)+1):02d}", f"{((i%28)+1):02d}" 
            date_val = f"{year_str}-{month_str}-{day_str}" 
            f.write(dummy_row_template.format(date_val=date_val, price=100+i*random.uniform(-2,2) + (i*0.5), volume=10000+i*100 + random.randint(-500,500), 
                                             fA=0.5+i*0.01, fB=1.2-i*0.01, cat='TypeA' if i%3==0 else ('TypeB' if i%3==1 else 'TypeC'), 
                                             text=f"Txt{i}", target= (101+i*0.25 + random.uniform(-1,1)))) 

    initial_pipeline_state = {
        "data_paths": initial_data_paths, "target_column_name": "Target", "problem_type": "regression",   
        "max_react_iterations": 6, "target_rmse": 0.75, "max_modeling_configs_to_try": 2, 
        "modeling_config_index": 0, "best_rmse_so_far": float('inf'), 
    }
    
    config = {"configurable": {"thread_id": f"ml_pipeline_robust_run_{random.randint(1000,9999)}"}}

    print("\nInvoking pipeline stream:")
    final_state_accumulator = {} 

    for chunk in pipeline_app.stream(initial_pipeline_state, config=config, stream_mode="updates"):
        for node_name_in_chunk, node_output_dict in chunk.items(): 
            print(f"\n<<< Update from Node: {node_name_in_chunk} >>>")
            if isinstance(node_output_dict, dict):
                final_state_accumulator.update(node_output_dict) 
                for k_item, v_item in node_output_dict.items(): 
                    print(f"  {k_item}: {str(v_item)[:350]}...")
            else:
                print(f"  Unexpected output format from node {node_name_in_chunk}: {str(node_output_dict)[:350]}...")

    print("\n\n--- Final Pipeline State (from accumulated stream) ---")
    if final_state_accumulator:
        print(json.dumps(final_state_accumulator, indent=2, default=str))
    
    for v_path in initial_data_paths.values():
        if os.path.exists(v_path): os.remove(v_path)
    if os.path.exists("dummy_pipeline_data"): os.rmdir("dummy_pipeline_data")

    print("\nMulti-Agent Pipeline Finished.")
