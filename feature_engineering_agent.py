import json
import logging
from typing import Dict, List, Optional
from common_utils import MultiAgentPipelineState, run_generic_react_loop, parse_llm_json_final_answer

logger = logging.getLogger(__name__)

def feature_engineering_agent_node(state: MultiAgentPipelineState) -> Dict[str, any]:
    node_name = "FeatureEngineering_Agent"
    logger.info(f"--- {node_name} Node Running ---")
    eda_report = state.get("eda_report", {}) 
    
    if not eda_report or eda_report.get("error") or not state.get("eda_processed_train_ref") or "ERROR" in str(state.get("eda_processed_train_ref","")).upper():
        error_msg = f"{node_name}_SKIPPED: Preceding EDA stage failed or critical data references missing. EDA report: {json.dumps(eda_report)}"
        logger.error(error_msg)
        return {"current_stage_completed": f"{node_name}_SKIPPED_PREV_ERROR", "pipeline_status_message": error_msg,
                "fe_X_train_ref": "ERROR_FE_skipped.pkl"} # Ensure this key exists for graph condition

    # Use the FINAL processed data references from the EDA's artifact_references
    train_ref_from_eda = eda_report.get("artifact_references", {}).get("processed_train_data", "default_train_eda.pkl") 
    val_ref_from_eda = eda_report.get("artifact_references", {}).get("processed_val_data", "default_val_eda.pkl")
    test_ref_from_eda = eda_report.get("artifact_references", {}).get("processed_test_data", "default_test_eda.pkl")
    
    suggestions_from_eda = eda_report.get("fe_suggestions", [])
    target_col = state.get("target_column_name", "Target")

    date_fe_suggestion_from_eda = "PRIORITY: If EDA suggested extracting date features (Year, Month, Day, etc.) from a parsed 'Date' column and then DROPPING the original 'Date' column, implement this first on all datasets (train, val, test) using their references from EDA. Ensure tool reports NEW data references (as .pkl) after this step."
    for suggestion in suggestions_from_eda:
        if "date column" in suggestion.lower() and "extract" in suggestion.lower() and "drop" in suggestion.lower():
            date_fe_suggestion_from_eda = suggestion; break

    fe_tool_context_hint = (f"Input data refs from EDA (these are .pkl files containing DataFrames with parsed 'Date' columns): train='{train_ref_from_eda}', val='{val_ref_from_eda}', test='{test_ref_from_eda}'. "
                            f"Target: '{target_col}'. EDA FE Suggestions: {json.dumps(suggestions_from_eda)}.")

    # Prompt as defined in ml_pipeline_agent_managed_refs_json_v9_robust_json_and_refs
    # with STRICT JSON output instructions
    prompt_content = f"""You are a Feature Engineering Specialist for stock price prediction. PythonTool takes NL instructions.
    Context from EDA:
    - Input Train Data Ref (cleaned & date-parsed by EDA): {train_ref_from_eda}
    - Input Val Data Ref: {val_ref_from_eda}
    - Input Test Data Ref: {test_ref_from_eda}
    - EDA FE Suggestions: {json.dumps(suggestions_from_eda)}
    - Specific EDA instruction for 'Date' column (YOUR FIRST PRIORITY): "{date_fe_suggestion_from_eda}"
    - Target Column: '{target_col}'

    Your tasks:
    1. Instruct tool to use/load datasets using FINAL PROCESSED .pkl references from EDA (e.g., '{train_ref_from_eda}').
    2. **CRITICAL FIRST FE STEP (Date Handling):** Execute EDA's suggestion for 'Date' column: "{date_fe_suggestion_from_eda}". Instruct tool to extract numerical date features AND THEN DROP ORIGINAL 'Date' (datetime object) COLUMN from train, val, and test. Tool MUST report new .pkl data references for train, val, and test after these operations. Use these NEW references for subsequent FE.
    3. Implement other FE steps from EDA suggestions (transformations, imputation) on these NEW date-handled datasets. Apply consistently.
    4. Create and SAVE individual FITTED transformers (scalers, encoders for categoricals like 'Category', imputers) as .joblib files using training data. Ask tool to report full .joblib filename references.
    5. After ALL transformations, separate features (X) and target ('{target_col}'). Ask tool to save X_train, y_train, etc., as .pkl files and report their full .pkl filename references. Also ask for the final feature list (this list MUST NOT contain the original 'Date' column).

    ReAct Format: Action: Python, Action Input: <NL instruction>.
    "Final Answer:" MUST be a single well-formed JSON object string, enclosed ONLY in ```json ... ```. NO OTHER TEXT.
    **All string values within JSON MUST be properly quoted and escaped (e.g., newlines as \\n, double quotes as \\\", backslashes as \\\\).**
    JSON keys: "fe_summary", "numerical_features", "categorical_features", "final_feature_list", 
               "transformer_references": {{ "scaler_price": "<ref.joblib>", ... }}, 
               "custom_transformer_module": (string or null),
               "data_references": {{ "X_train": "<X_train_ref.pkl>", ... }}
    Begin. Address date feature engineering first.
    """
    final_answer_json_string = run_generic_react_loop(prompt_content, state.get("max_react_iterations", 15), fe_tool_context_hint)
    parsed_data = parse_llm_json_final_answer(final_answer_json_string, f"{node_name} report generation failed.")
    
    output_to_state = {"current_stage_completed": f"{node_name}_SUCCESS"}
    if "error" in parsed_data:
        logger.error(f"{node_name} failed to produce valid JSON report. Error: {parsed_data.get('error')}")
        output_to_state["current_stage_completed"] = f"{node_name}_FAILED_JSON_OUTPUT"
        output_to_state["pipeline_status_message"] = f"Critical error in {node_name}: {parsed_data.get('error')}"
        output_to_state.update({k: f"ERROR_FE_JSON_failed" for k in ["fe_applied_steps_summary", "fe_final_feature_list", "fe_numerical_features", "fe_categorical_features","fe_transformer_references", "fe_X_train_ref", "fe_y_train_ref", "fe_X_val_ref", "fe_y_val_ref", "fe_X_test_ref"]})
    else:
        logger.info(f"{node_name} produced FE report successfully.")
        data_refs = parsed_data.get("data_references", {})
        # Validate presence of critical X_train_ref
        x_train_parsed_ref = data_refs.get("X_train")
        if not x_train_parsed_ref:
            err_msg = f"{node_name}_FAILED_MISSING_XT몇RAIN_REF: FE report missing X_train reference. Original JSON: {json.dumps(parsed_data)}"
            logger.error(err_msg)
            output_to_state["current_stage_completed"] = f"{node_name}_FAILED_MISSING_XT몇RAIN_REF"
            output_to_state["pipeline_status_message"] = err_msg
            # Ensure defaults are set
            output_to_state.update({k: f"ERROR_FE_missing_{k}" for k in ["fe_X_train_ref", "fe_y_train_ref", "fe_X_val_ref", "fe_y_val_ref", "fe_X_test_ref"]})
            output_to_state["fe_final_feature_list"] = []
            output_to_state["fe_transformer_references"] = {}
            output_to_state["fe_applied_steps_summary"] = parsed_data.get("fe_summary", "Summary available but X_train ref missing.")
            return output_to_state

        output_to_state.update({
            "fe_applied_steps_summary": parsed_data.get("fe_summary"),
            "fe_final_feature_list": parsed_data.get("final_feature_list", []),
            "fe_numerical_features": parsed_data.get("numerical_features", []),
            "fe_categorical_features": parsed_data.get("categorical_features", []),
            "fe_transformer_references": parsed_data.get("transformer_references", {}),
            "fe_custom_transformer_module": parsed_data.get("custom_transformer_module"), 
            "fe_X_train_ref": x_train_parsed_ref, 
            "fe_y_train_ref": data_refs.get("y_train"),
            "fe_X_val_ref": data_refs.get("X_val"),
            "fe_y_val_ref": data_refs.get("y_val"),
            "fe_X_test_ref": data_refs.get("X_test"),
            "fe_y_test_ref": data_refs.get("y_test") 
        })
    return output_to_state
