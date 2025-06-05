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
    fe_X_train_ref: Optional[str] 
    fe_y_train_ref: Optional[str]
    fe_X_val_ref: Optional[str]   
    fe_y_val_ref: Optional[str]
    fe_X_test_ref: Optional[str]  
    fe_transformer_references: Optional[Dict[str, str]] 
    fe_custom_transformer_module: Optional[str] 
    
    # Output from ModelSelectionDecisionNode
    top_model_configurations: Optional[List[Dict[str, any]]] 

    # Modeling Node State & Output
    model_training_summary: Optional[str] 
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
    
    sim_observation = run_python(instruction, agent_context_hint)
    instruction_lower = instruction.lower()

    # LLM must explicitly ask the tool to "report the reference" or "report the filename"
    # and "provide a description" for plots. Tool confirms and provides these.
    if "load the dataset from" in instruction_lower and "report its reference as" in instruction_lower:
        ref_as_match = re.search(r"report its reference as '([^']+)'", instruction_lower)
        ref_name = ref_as_match.group(1) if ref_as_match else "unknown_initial_load_ref.pkl"
        if "train_data.csv" in instruction:
            sim_observation += f"Dataset 'dummy_pipeline_data/train_data.csv' loaded. Tool reports its reference as '{ref_name}'."
        elif "val_data.csv" in instruction:
            sim_observation += f"Dataset 'dummy_pipeline_data/val_data.csv' loaded. Tool reports its reference as '{ref_name}'."
        elif "test_data.csv" in instruction:
            sim_observation += f"Dataset 'dummy_pipeline_data/test_data.csv' loaded. Tool reports its reference as '{ref_name}'."
        else:
            sim_observation += f"Dataset loaded. Tool reports its reference as '{ref_name}'."
    
    elif "identify the 'date' column in" in instruction_lower and "parse it as datetime using format 'yyyy-mm-dd'" in instruction_lower and "report the new reference" in instruction_lower:
        ref_in_match = re.search(r"in '([^']+)'", instruction_lower)
        ref_out_match = re.search(r"new reference as '([^']+)'", instruction_lower)
        data_ref_in = ref_in_match.group(1) if ref_in_match else "unknown_ref.pkl"
        data_ref_out = ref_out_match.group(1) if ref_out_match else f"parsed_date_{data_ref_in}"
        sim_observation += f"'Date' column in '{data_ref_in}' identified and parsed to datetime64[ns]. New data reference with parsed date is '{data_ref_out}'. Confirmed."
    
    elif "create a temporary numeric-only version of" in instruction_lower and "excluding the original 'date' column" in instruction_lower and "report the new reference as" in instruction_lower:
        ref_in_match = re.search(r"version of '([^']+)'", instruction_lower)
        ref_out_match = re.search(r"new reference as '([^']+)'", instruction_lower)
        data_ref_in = ref_in_match.group(1) if ref_in_match else "unknown_ref.pkl"
        data_ref_out = ref_out_match.group(1) if ref_out_match else f"numeric_version_of_{data_ref_in}"
        sim_observation += f"Numeric-only version of '{data_ref_in}' created (original 'Date' column excluded). New reference reported by tool: '{data_ref_out}'."
    
    elif "extract year, month, day, dayofweek from the 'date' column in" in instruction_lower and "drop the original 'date' column" in instruction_lower and "report new data references as" in instruction_lower:
        # LLM needs to specify refs for train, val, test if it wants them separately
        ref_out_train_match = re.search(r"new data references as train='([^']+)', val='([^']+)', test='([^']+)'", instruction_lower)
        if ref_out_train_match:
            train_ref, val_ref, test_ref = ref_out_train_match.groups()
            sim_observation += ("Date features (Year, Month, Day, DayOfWeek) extracted. Original 'Date' column dropped. "
                               f"New data references reported: train='{train_ref}', val='{val_ref}', test='{test_ref}'.")
        else:
            sim_observation += "Date features extracted and original 'Date' column dropped. Tool reports new general reference: 'data_with_date_features.pkl'."

    elif "clean data referenced by" in instruction_lower and "save the cleaned data and report the new reference as" in instruction_lower: 
        ref_in_match = re.search(r"referenced by '([^']+)'", instruction_lower)
        ref_out_match = re.search(r"new reference as '([^']+)'", instruction_lower)
        data_ref_in = ref_in_match.group(1) if ref_in_match else "unknown_ref.pkl"
        data_ref_out = ref_out_match.group(1) if ref_out_match else f"cleaned_{data_ref_in}"
        sim_observation += f"Data '{data_ref_in}' cleaned. New reference reported by tool: '{data_ref_out}'."

    elif "generate a histogram for" in instruction_lower and "save it as" in instruction_lower and "report the filename, and provide a textual description" in instruction_lower:
        filename_match = re.search(r"save it as '([^']+)'", instruction_lower)
        filename = filename_match.group(1) if filename_match else "sim_histogram.png"
        sim_observation += f"Histogram generated. Plot saved by tool as '{filename}'. Description: [Simulated detailed description]."
    
    elif "fit a standardscaler" in instruction_lower and "save it as a .joblib file and report its reference as" in instruction_lower:
        ref_match = re.search(r"report its reference as '([^']+)'", instruction_lower)
        ref_name = ref_match.group(1) if ref_match else "fitted_scaler.joblib"
        sim_observation += f"StandardScaler fitted. Saved as '{ref_name}'. This is its reference."
    
    elif "create an untrained scikit-learn pipeline" in instruction_lower and "save it as a .joblib file and report its reference as" in instruction_lower:
        ref_match = re.search(r"report its reference as '([^']+)'", instruction_lower)
        ref_name = ref_match.group(1) if ref_match else "untrained_pipeline.joblib"
        sim_observation += f"Untrained Scikit-learn pipeline created. Saved. Reference is '{ref_name}'."
    
    elif "load the untrained pipeline" in instruction_lower and "train it using x_train" in instruction_lower and "save the trained pipeline as a .joblib file and report its reference as" in instruction_lower:
        ref_match = re.search(r"report its reference as '([^']+)'", instruction_lower)
        trained_model_ref = ref_match.group(1) if ref_match else "trained_model_pipeline.joblib"
        current_sim_rmse = random.uniform(0.1, 1.0) # Simplified
        sim_observation += (f"Pipeline trained. Trained pipeline saved. Reference is '{trained_model_ref}'.")
        if "report rmse" in instruction_lower: sim_observation += f" Validation RMSE: {current_sim_rmse:.4f}."

    elif "separate target" in instruction_lower and "save them as .pkl files and report new references like x_train_ref=" in instruction_lower:
        sim_observation += "Target separated. Tool reports new references: X_train='X_train_final.pkl', y_train='y_train_final.pkl', X_val='X_val_final.pkl', y_val='y_val_final.pkl', X_test='X_test_final.pkl'."
    
    elif "calculate regression metrics" in instruction_lower: 
        sim_observation += "Metrics reported by tool: {{'rmse': {random.uniform(0.1,0.8):.4f}, 'r_squared': {random.uniform(0.6,0.9):.2f}}}."
    else:
        sim_observation += "Task completed. If specific artifacts were requested to be saved and their references reported, those details are included above."
            
    print(f"    [AGNO_PYTHON_TOOL INTERFACE] Returning Observation:\n    '{sim_observation}'")
    return sim_observation

# --- 3. Generic ReAct Loop Engine ---
 
# ... (run_generic_react_loop definition remains the same as in ml_pipeline_agent_managed_refs_json_v8_error_fixes)
def run_generic_react_loop(
    initial_prompt_content: str,
    max_iterations: int,
    agent_context_hint_for_tool: Optional[str] = None 
) -> str: 
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
            if i > 1: final_answer_text = json.dumps({"error": "Agent failed to follow output format consistently."}); print(f"    Agent failed to follow output format."); break 
        if i == max_iterations - 1: print(f"    Max ReAct iterations reached."); final_answer_text = json.dumps({"error": f"Max iterations. Last thought: {ai_content}"})
    return final_answer_text

# --- 4. Helper to Parse LLM's JSON Final Answer ---
def parse_llm_json_final_answer(final_answer_json_string: str, default_error_message: str = "Report generation failed.") -> Dict:
    # ... (Same robust implementation as in ml_pipeline_agent_managed_refs_json_v8_error_fixes)
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
    except Exception as e: print(f"    ERROR parsing LLM JSON: {e}, String: '{final_answer_json_string}'"); return {"error": str(e), "summary": default_error_message, "original_string": final_answer_json_string}


# --- 5. Define Agent Nodes ---

def eda_agent_node(state: MultiAgentPipelineState) -> Dict[str, any]:
    print("\n--- EDA Agent Node Running (Strict Reference & Date Handling Focus) ---")
    data_paths = state["data_paths"]
    target_col = state.get("target_column_name", "Target")
    problem_type = state.get("problem_type", "regression") 
    eda_tool_context_hint = f"Initial data paths: {json.dumps(data_paths)}. Target column: '{target_col}'. Problem type: {problem_type}. Task: EDA for stock price prediction."

    prompt_content = f"""You are an Expert EDA Data Scientist creating an 'EDA Manual'.
    The PythonTool you use accepts natural language instructions.
    **CRITICAL**: When you instruct PythonTool to load, create, or modify data/plots/models, you MUST explicitly ask it to "report the new reference as 'reference_name.ext'" or "report the filename as 'filename.ext'". Use these EXACT tool-reported references in your thoughts and Final Answer. Assume `.pkl` for dataframes, `.joblib` for sklearn objects, and `.png` for plots unless tool specifies otherwise.

    Initial context for PythonTool: {eda_tool_context_hint}

    Your EDA Process (STRICTLY FOLLOW THIS ORDER FOR DATE HANDLING):
    1.  **Load Datasets:** Instruct tool: "Load train dataset from '{data_paths.get('train')}' and report its reference as 'initial_train_ref.pkl'." Repeat for val ('initial_val_ref.pkl') and test ('initial_test_ref.pkl').
    2.  **Parse 'Date' Column:** Instruct tool: "For data referenced by 'initial_train_ref.pkl', identify the 'Date' column (format 'YYYY-MM-DD') and parse it to datetime. Save this modified dataset and report its new reference as 'parsed_date_train_ref.pkl'." Repeat for 'initial_val_ref.pkl' (output 'parsed_date_val_ref.pkl') and 'initial_test_ref.pkl' (output 'parsed_date_test_ref.pkl').
    3.  **Initial Structure & Quality:** Using 'parsed_date_train_ref.pkl' (and val/test refs), check shapes, dtypes (confirm 'Date' is datetime), head/tail, missing values, outliers. For plots, instruct tool: "For 'parsed_date_train_ref.pkl', generate a boxplot for '{target_col}', save it, report its filename as 'target_boxplot.png', AND provide a textual description."
    4.  **Prepare for Numeric EDA:** Instruct tool: "Using 'parsed_date_train_ref.pkl', create a temporary numeric-only version FOR CORRELATION by EXCLUDING the 'Date' column (datetime object) and other non-numerics (except target '{target_col}'). Report NEW reference as 'train_numeric_for_corr.pkl'."
    5.  **Correlations:** Using 'train_numeric_for_corr.pkl', instruct tool: "Compute correlations, generate a heatmap, save it, report filename as 'correlation_heatmap.png', AND describe strong correlations."
    6.  **Distribution & Time Series Analysis:** Using 'parsed_date_train_ref.pkl', analyze distributions ('{target_col}'). Plot '{target_col}' over parsed 'Date'. Ask for plot refs, filenames (e.g., 'target_vs_time.png'), AND descriptions.
    7.  **Final Cleaning & References:** If further general cleaning is performed on 'parsed_date_train_ref.pkl', instruct tool: "Perform final cleaning on 'parsed_date_train_ref.pkl'. Save and report its reference as 'cleaned_train_final_eda.pkl'." Similarly for val ('cleaned_val_final_eda.pkl') and test ('cleaned_test_final_eda.pkl'). These are the primary outputs for FE.
    8.  **Model & FE Suggestions.** For dates, ensure suggestion is: "FE Suggestion: From the 'Date' column in 'cleaned_train_final_eda.pkl' (which should be datetime objects), extract numerical features like Year, Month, Day, DayOfWeek. After extraction, the original 'Date' column MUST BE DROPPED before modeling."
    9.  Conclude.

    ReAct Format: Action: Python, Action Input: <NL instruction>.
    "Final Answer:" MUST be a single well-formed JSON object string, enclosed ONLY in ```json ... ```. NO OTHER TEXT.
    **All string values within JSON MUST be properly quoted and escaped (newlines as \\n, double quotes as \\\", backslashes as \\\\).**
    (JSON structure: "eda_summary", "data_profile" (with initial_refs), "data_quality_report", "key_insights", "model_suggestions", "fe_suggestions", 
     "artifact_references": {{ "processed_train_data": "<TOOL_REPORTED_cleaned_train_final_eda.pkl>", ... , "numeric_train_for_correlation": "<TOOL_REPORTED_train_numeric_for_corr.pkl>", "plots": {{...}} }})
    Begin.
    """
    final_answer_json_string = run_generic_react_loop(prompt_content, state.get("max_react_iterations", 25), eda_tool_context_hint) # EDA is complex
    
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
            "eda_numeric_train_ref_for_correlation": artifact_refs.get("numeric_train_for_correlation"),
            "eda_plot_references": artifact_refs.get("plots", {})
        })
    else: 
        output_to_state.update({k: [] for k in ["eda_model_suggestions", "eda_fe_suggestions"]})
        output_to_state.update({k: f"error_in_eda_{k.replace('eda_','').replace('_ref','')}.pkl" for k in ["eda_processed_train_ref", "eda_processed_val_ref", "eda_processed_test_ref", "eda_numeric_train_ref_for_correlation"]})
        output_to_state["eda_plot_references"] = {}
    return output_to_state


def feature_engineering_agent_node(state: MultiAgentPipelineState) -> Dict[str, any]:
    print("\n--- Feature Engineering Agent Node Running (Strict Reference Usage) ---")
    eda_report = state.get("eda_report", {}) 
    if eda_report.get("error"): return {"current_stage_completed": "FeatureEngineering", "fe_applied_steps_summary": "FE skipped due to EDA errors."}

    # CRITICAL: Use the EXACT references reported by EDA for its FINAL processed data
    train_ref_from_eda = eda_report.get("artifact_references", {}).get("processed_train_data", "default_train_eda.pkl") 
    val_ref_from_eda = eda_report.get("artifact_references", {}).get("processed_val_data", "default_val_eda.pkl")
    test_ref_from_eda = eda_report.get("artifact_references", {}).get("processed_test_data", "default_test_eda.pkl")
    
    suggestions_from_eda = eda_report.get("fe_suggestions", [])
    target_col = state.get("target_column_name", "Target")

    date_fe_suggestion_from_eda = "PRIORITY: If EDA suggested: \"From the parsed 'Date' column in '<some_eda_train_ref.pkl>', extract numerical features (Year, Month, Day, DayOfWeek). After extraction, the original 'Date' (datetime object) column MUST BE DROPPED before modeling.\", then implement this first. Tool must report NEW data references for train, val, and test after these date operations (e.g., 'train_with_date_features_dropped.pkl')."
    # Find the exact suggestion if present
    for suggestion in suggestions_from_eda:
        if "date column" in suggestion.lower() and "extract" in suggestion.lower() and "drop" in suggestion.lower():
            date_fe_suggestion_from_eda = suggestion; break 

    fe_tool_context_hint = (f"Input data refs from EDA (these are .pkl files containing DataFrames with parsed 'Date' columns): train='{train_ref_from_eda}', val='{val_ref_from_eda}', test='{test_ref_from_eda}'. "
                            f"Target: '{target_col}'. EDA FE Suggestions: {json.dumps(suggestions_from_eda)}.")

    prompt_content = f"""You are a Feature Engineering Specialist for stock price prediction. PythonTool takes NL instructions.
    Context from EDA:
    - Input Train Data Ref (cleaned & date-parsed by EDA): {train_ref_from_eda}
    - Input Val Data Ref (cleaned & date-parsed by EDA): {val_ref_from_eda}
    - Input Test Data Ref (cleaned & date-parsed by EDA): {test_ref_from_eda}
    - EDA FE Suggestions: {json.dumps(suggestions_from_eda)}
    - Specific EDA instruction for 'Date' column (YOUR ABSOLUTE FIRST PRIORITY): "{date_fe_suggestion_from_eda}"
    - Target Column: '{target_col}'

    Your tasks:
    1. Instruct PythonTool to use/load datasets using the EXACT FINAL PROCESSED references from EDA (e.g., "{train_ref_from_eda}").
    2. **CRITICAL FIRST FE STEP (Date Handling):** Carefully execute the EDA's suggestion for the 'Date' column: "{date_fe_suggestion_from_eda}". This means instructing the tool to load the data using the EDA-provided references (e.g., '{train_ref_from_eda}'), then extract numerical date features (Year, Month, Day, DayOfWeek, etc.) from its 'Date' column, AND THEN INSTRUCT THE TOOL TO DROP THE ORIGINAL 'Date' (datetime object) COLUMN from train, val, and test datasets. The tool MUST report NEW data references after these operations (e.g., 'train_date_features_final.pkl'). Use these NEW references for all subsequent FE steps.
    3. Implement other FE steps from EDA suggestions on these NEW date-handled datasets. Apply consistently.
    4. Create and SAVE individual FITTED transformers (scalers, encoders) as .joblib files using training data. Ask tool to report full .joblib filename references.
    5. After ALL transformations, separate features (X) and target ('{target_col}'). Ask tool to save X_train, y_train, etc., as .pkl files and report their full .pkl filename references. Also ask for the final feature list (this list MUST NOT contain the original 'Date' column).

    ReAct Format: Action: Python, Action Input: <NL instruction>.
    "Final Answer:" MUST be a single well-formed JSON object string, enclosed ONLY in ```json ... ```. NO OTHER TEXT.
    **All string values within JSON MUST be properly quoted and escaped.**
    (JSON structure: "fe_summary", "final_feature_list", "transformer_references": {{ "scaler_price": "<ref.joblib>" }}, "custom_transformer_module", "data_references": {{ "X_train": "<X_train_ref.pkl>" }})
    Begin. Address date feature engineering as the absolute first priority using the latest references from EDA.
    """
    final_answer_json_string = run_generic_react_loop(prompt_content, state.get("max_react_iterations", 15), fe_tool_context_hint) # FE can also be iterative
    parsed_data = parse_llm_json_final_answer(final_answer_json_string, "FE report generation failed.")
    
    output_to_state = {"current_stage_completed": "FeatureEngineering"}
    if "error" not in parsed_data:
        data_refs = parsed_data.get("data_references", {})
        output_to_state.update({
            "fe_applied_steps_summary": parsed_data.get("fe_summary"),
            "fe_final_feature_list": parsed_data.get("final_feature_list", []),
            "fe_transformer_references": parsed_data.get("transformer_references", {}),
            "fe_custom_transformer_module": parsed_data.get("custom_transformer_module"), 
            "fe_X_train_ref": data_refs.get("X_train"), 
            "fe_y_train_ref": data_refs.get("y_train"),
            "fe_X_val_ref": data_refs.get("X_val"),
            "fe_y_val_ref": data_refs.get("y_val"),
            "fe_X_test_ref": data_refs.get("X_test"),
            "fe_y_test_ref": data_refs.get("y_test") 
        })
    else: 
        output_to_state.update({k: f"error_in_fe_parsing_for_{k.split('_',1)[1] if '_' in k else k}" for k in ["fe_applied_steps_summary", "fe_final_feature_list", "fe_transformer_references", "fe_X_train_ref", "fe_y_train_ref", "fe_X_val_ref", "fe_y_val_ref", "fe_X_test_ref"]})
        output_to_state["fe_untrained_full_pipeline_ref"] = "error_fe_pipeline.joblib" # Ensure this is also set for modeling to know FE failed
    return output_to_state


def model_selection_decision_agent_node(state: MultiAgentPipelineState) -> Dict[str, any]:
    # ... (Prompt includes strict JSON instruction) ...
    # ... (Error check for fe_final_feature_list being empty/error)
    print("\n--- Model Selection Decision Agent Node Running ---")
    eda_report = state.get("eda_report", {}); 
    if eda_report.get("error"): return {"current_stage_completed": "ModelSelectionDecision", "top_model_configurations": []}
    fe_final_feature_list = state.get("fe_final_feature_list", [])
    if not fe_final_feature_list or "error" in str(state.get("fe_X_train_ref","")) : # Check if FE was successful
        print("    WARN: FE stage might have failed or produced no features. Model selection will be based on limited info.")
        # Allow to proceed but it might suggest very generic models
    
    eda_model_suggestions = eda_report.get("model_suggestions", [])
    problem_type = state.get("problem_type", "regression") 
    decision_tool_context_hint = (f"EDA Model Suggestions: {json.dumps(eda_model_suggestions)}. Final Features from FE ({len(fe_final_feature_list)} features): {json.dumps(fe_final_feature_list[:5])}. Problem: {problem_type}.")
    prompt_content = f"""You are a Model Selection Strategist for stock price prediction (regression). Context: {decision_tool_context_hint}
    Task: Based on EDA suggestions and final features, select up to 2-3 promising Scikit-learn REGRESSION model types. For each, suggest initial hyperparameters or defaults.
    "Final Answer:" MUST be a single well-formed JSON object string, enclosed ONLY in ```json ... ```. NO OTHER TEXT.
    **All string values within JSON MUST be properly quoted and escaped.**
    JSON keys: "decision_rationale", "top_model_configurations": [ {{ "model_type": "e.g.RandomForestRegressor", "initial_hyperparameters": {{}}, "reasoning": "..." }} ]. Begin."""
    final_answer_json_string = run_generic_react_loop(prompt_content, 3, decision_tool_context_hint) 
    parsed_data = parse_llm_json_final_answer(final_answer_json_string, "Model selection decision failed.")
    return {"current_stage_completed": "ModelSelectionDecision", "top_model_configurations": parsed_data.get("top_model_configurations", [])}


def modeling_agent_node(state: MultiAgentPipelineState) -> Dict[str, any]:
    # ... (Prompt includes strict JSON instruction) ...
    # ... (Robust error check for missing top_model_configurations or essential X/y refs from FE) ...
    print("\n--- Modeling Agent Node Running (Iterates Top Configurations for RMSE) ---")
    top_model_configurations = state.get("top_model_configurations", [])
    if not top_model_configurations or state.get("fe_X_train_ref") is None or "error" in str(state.get("fe_X_train_ref","")):
        error_msg = "Critical inputs missing for modeling (no model configurations from decision node or FE failed). Cannot proceed."
        print(f"    ERROR: {error_msg}")
        # Crucially, ensure this node "completes" by setting its outputs to indicate failure or skip
        return {"current_stage_completed": "Modeling", 
                "model_training_summary": error_msg, 
                "modeling_config_index": state.get("modeling_config_index",0)+1, # Increment to avoid infinite loop if condition is based on this
                "best_rmse_so_far": state.get("best_rmse_so_far", float('inf')), # Keep current best
                "model_trained_pipeline_ref": state.get("best_model_ref_so_far") 
                }

    # ... (Rest of modeling_agent_node logic as in ml_pipeline_agent_managed_refs_json_v7 - it already expects .joblib/.pkl based on context)
    # The key is that fe_X_train_ref etc. are now correctly populated with .pkl (or .npy if your tool saves that way and reports it)
    # and transformer_references are .joblib
    transformer_refs_from_fe = state.get("fe_transformer_references", {}); x_train_ref = state.get("fe_X_train_ref"); y_train_ref = state.get("fe_y_train_ref"); x_val_ref = state.get("fe_X_val_ref"); y_val_ref = state.get("fe_y_val_ref") 
    custom_module = state.get("fe_custom_transformer_module"); target_rmse = state.get("target_rmse", 0.002) 
    config_idx = state.get("modeling_config_index", 0); overall_best_rmse = state.get("best_rmse_so_far", float('inf'))
    overall_best_model_ref = state.get("best_model_ref_so_far"); overall_best_model_config = state.get("best_model_config_so_far")
    strategy_log_for_all_configs = state.get("modeling_strategy_log", []); max_configs_to_try = state.get("max_modeling_configs_to_try", len(top_model_configurations))

    if config_idx >= len(top_model_configurations) or config_idx >= max_configs_to_try: # Check if all configs tried
        summary_msg = f"Completed trying {config_idx} model configurations. Best RMSE: {overall_best_rmse:.4f}."; 
        return {"current_stage_completed": "Modeling", "model_training_summary": summary_msg, "model_trained_pipeline_ref": overall_best_model_ref, "best_rmse_so_far": overall_best_rmse, "best_model_ref_so_far": overall_best_model_ref, "best_model_config_so_far": overall_best_model_config, "modeling_strategy_log": strategy_log_for_all_configs, "modeling_config_index": config_idx }

    current_config = top_model_configurations[config_idx]; chosen_model_type = current_config.get("model_type", "RandomForestRegressor"); initial_hyperparams = current_config.get("initial_hyperparameters", {})
    model_tool_context_hint = (f"Config(Idx{config_idx}): Type='{chosen_model_type}', Params='{json.dumps(initial_hyperparams)}'. Transformers(.joblib): {json.dumps(transformer_refs_from_fe)}. CustomMod: '{custom_module}'. Data(.pkl): X_train='{x_train_ref}'. TargetRMSE: {target_rmse}.")
    prompt_content = f"""You are a Modeling Specialist... Context: {model_tool_context_hint}
    Task for THIS Config ('{chosen_model_type}', Params: {json.dumps(initial_hyperparams)}):
    1. Instruct PythonTool to:
        a. Create UNTRAINED sklearn pipeline: combine preprocessors (from '{json.dumps(transformer_refs_from_fe)}') with estimator '{chosen_model_type}' (params '{json.dumps(initial_hyperparams)}'). (Custom module '{custom_module}' if specified).
        b. Save UNTRAINED pipeline (e.g., 'untrained_cfg{config_idx}.joblib') & report .joblib ref.
        c. Load X_train ('{x_train_ref}') & y_train ('{y_train_ref}') (.pkl). Train pipeline.
        d. Save TRAINED pipeline (e.g., 'trained_cfg{config_idx}.joblib') & report .joblib ref.
        e. Load X_val ('{x_val_ref}') & y_val ('{y_val_ref}') (.pkl). Predict & calculate RMSE. Report RMSE.
    "Final Answer:" JSON for THIS trial: "config_trial_summary", "config_trained_pipeline_ref" (MUST be .joblib), "config_rmse", "model_type_tried", "hyperparameters_tried". Begin."""
    config_trial_final_answer_json = run_generic_react_loop(prompt_content, state.get("max_react_iterations", 7), model_tool_context_hint)
    parsed_config_trial_data = parse_llm_json_final_answer(config_trial_final_answer_json, f"Modeling trial {config_idx} failed.")
    # ... (rest of logic for updating best RMSE and returning state from v7)
    output_to_state = {"modeling_config_index": config_idx + 1, "current_stage_completed": "Modeling_Config_Trial" }
    if "error" in parsed_config_trial_data: output_to_state.update({"current_rmse": None, "model_training_summary": parsed_config_trial_data.get("summary",f"Trial {config_idx} failed."), "modeling_strategy_log": strategy_log_for_all_configs + [f"CfgIdx{config_idx}({chosen_model_type}): FAILED"]})
    else:
        config_rmse = parsed_config_trial_data.get("config_rmse"); config_model_ref = parsed_config_trial_data.get("config_trained_pipeline_ref")
        new_best_rmse, new_best_model_ref, new_best_model_config = overall_best_rmse, overall_best_model_ref, overall_best_model_config
        if config_rmse is not None and isinstance(config_rmse, (int, float)) and config_rmse < overall_best_rmse: new_best_rmse, new_best_model_ref, new_best_model_config = config_rmse, config_model_ref, current_config
        output_to_state.update({"current_rmse": config_rmse, "best_rmse_so_far": new_best_rmse, "best_model_ref_so_far": new_best_model_ref, "best_model_config_so_far": new_best_model_config, "model_training_summary": parsed_config_trial_data.get("config_trial_summary"), "model_trained_pipeline_ref": config_model_ref, "modeling_strategy_log": strategy_log_for_all_configs + [f"CfgIdx{config_idx}({chosen_model_type}): RMSE={config_rmse}, Ref={config_model_ref}"]})
    return output_to_state

def evaluation_node(state: MultiAgentPipelineState) -> Dict[str, any]:
    # ... (Same as in ml_pipeline_agent_managed_refs_json_v7, ensures it uses best_model_ref_so_far) ...
    # ... (Prompt includes strict JSON instruction)
    print("\n--- Evaluation Agent Node Running ---")
    trained_pipeline_ref = state.get("best_model_ref_so_far", state.get("model_trained_pipeline_ref", "default_best_trained.joblib"))
    if not trained_pipeline_ref or "error" in str(trained_pipeline_ref).lower() or ("default" in str(trained_pipeline_ref).lower() and not os.path.exists(str(trained_pipeline_ref))):
         return {"current_stage_completed": "Evaluation", "evaluation_summary": "Skipped: No valid trained model ref.", "evaluation_metrics": {}}
    # ... (rest of eval node from v7)
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
    # ... (Same as in ml_pipeline_agent_managed_refs_json_v7_explicit_date_handling) ...
    print("\n--- Checking Modeling Iteration Decision ---")
    config_idx = state.get("modeling_config_index", 0); top_configurations = state.get("top_model_configurations", [])
    max_configs_to_try = state.get("max_modeling_configs_to_try", len(top_configurations) if top_configurations else 0) 
    current_best_rmse = state.get("best_rmse_so_far", float('inf')); target_rmse = state.get("target_rmse", 0.002)
    # Critical check: if FE failed or no model configs, go to eval (which will likely report skip)
    if not state.get("fe_X_train_ref") or "error" in str(state.get("fe_X_train_ref","")) or (config_idx == 0 and not top_configurations):
        print("  Critical error in prior stage (FE) or no model configurations. Bypassing further modeling iterations.")
        return "evaluation_agent"
    print(f"  Config Idx to try: {config_idx} / Total Configs: {len(top_configurations)} (Max to try: {max_configs_to_try}). Best RMSE: {current_best_rmse}, Target: {target_rmse}")
    if current_best_rmse <= target_rmse: print(f"  Target RMSE achieved. Evaluating."); return "evaluation_agent"
    if config_idx < len(top_configurations) and config_idx < max_configs_to_try: print(f"  Proceeding to try model configuration at index {config_idx}."); return "modeling_agent"
    else: print(f"  All model configs tried or limit reached. Evaluating."); return "evaluation_agent"

workflow = StateGraph(MultiAgentPipelineState)
# ... (Graph definition remains the same as ml_pipeline_agent_managed_refs_json_v7_explicit_date_handling) ...
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
    # ... (Same main block as in ml_pipeline_agent_managed_refs_json_v7_explicit_date_handling) ...
    print("Starting ML Pipeline with Stricter JSON & Date Handling (Final Version)...")
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
    config = {"configurable": {"thread_id": f"ml_pipeline_strict_json_v2_{random.randint(1000,9999)}"}} # Unique thread ID
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
