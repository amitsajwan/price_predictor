import json
import logging
from typing import Dict, List, Optional
from common_utils import MultiAgentPipelineState, run_generic_react_loop, parse_llm_json_final_answer # Assuming common_utils.py is in PYTHONPATH

logger = logging.getLogger(__name__)

def eda_agent_node(state: MultiAgentPipelineState) -> Dict[str, any]:
    node_name = "EDA_Agent"
    logger.info(f"--- {node_name} Node Running (Strict Reference & Date Handling Focus) ---")
    data_paths = state["data_paths"]
    target_col = state.get("target_column_name", "Target")
    problem_type = state.get("problem_type", "regression") 
    eda_tool_context_hint = f"Initial data paths: {json.dumps(data_paths)}. Target column: '{target_col}'. Problem type: {problem_type}. Task: EDA for stock price prediction."

    # Prompt as defined in ml_pipeline_agent_managed_refs_json_v9_robust_json_and_refs
    # with STRICT JSON output instructions
    prompt_content = f"""You are an Expert EDA Data Scientist creating an 'EDA Manual'.
    The PythonTool you use accepts natural language instructions.
    **CRITICAL**: When you instruct PythonTool to load, create, or modify data/plots/models, you MUST explicitly ask it to "report the new reference as 'reference_name.ext'" or "report the filename as 'filename.ext'". Use these EXACT tool-reported references in your thoughts and Final Answer. Assume `.pkl` for dataframes, `.joblib` for sklearn objects, and `.png` for plots unless tool specifies otherwise. When requesting plots, ALSO ask for textual descriptions of their key features.

    Initial context for PythonTool: {eda_tool_context_hint}

    Your EDA Process (CRITICAL: Address 'Date' column named 'Date' with format 'YYYY-MM-DD' explicitly and EARLY):
    1.  **Load Datasets:** Instruct tool: "Load train dataset from '{data_paths.get('train')}' and report its reference as 'initial_train_ref.pkl'." Repeat for val ('initial_val_ref.pkl') and test ('initial_test_ref.pkl').
    2.  **Parse 'Date' Column:** Instruct tool: "For data referenced by 'initial_train_ref.pkl', identify the 'Date' column (format 'YYYY-MM-DD') and parse it to datetime. Save this modified dataset and report its new reference as 'parsed_date_train_ref.pkl'." Repeat for 'initial_val_ref.pkl' (output 'parsed_date_val_ref.pkl') and 'initial_test_ref.pkl' (output 'parsed_date_test_ref.pkl').
    3.  **Initial Structure & Quality:** Using 'parsed_date_train_ref.pkl' (and val/test refs), check shapes, dtypes (confirm 'Date' is datetime), missing values, outliers. For plots (e.g., for '{target_col}'), instruct tool: "...save plot, report filename as 'target_boxplot.png', AND provide textual description."
    4.  **Prepare for Numeric EDA:** Instruct tool: "Using 'parsed_date_train_ref.pkl', create a temporary numeric-only version FOR CORRELATION by EXCLUDING the 'Date' column (datetime object) and other non-numerics (except target '{target_col}'). Report NEW reference as 'train_numeric_for_corr.pkl'."
    5.  **Correlations:** Using 'train_numeric_for_corr.pkl', compute correlations. Ask for heatmap plot ref (e.g., 'correlation_heatmap.png') & description.
    6.  **Distribution & Time Series Analysis:** Using 'parsed_date_train_ref.pkl', analyze distributions ('{target_col}'). Plot '{target_col}' over parsed 'Date'. Plot refs (e.g., 'target_vs_time.png') & descriptions.
    7.  **Final Cleaning & References:** If general cleaning on 'parsed_date_train_ref.pkl', instruct tool: "Perform final cleaning on 'parsed_date_train_ref.pkl'. Save and report reference as 'cleaned_train_final_eda.pkl'." Similarly for val/test. These are PRIMARY outputs for FE.
    8.  **Model & FE Suggestions.** For dates, state: "FE Suggestion: From the 'Date' column in 'cleaned_train_final_eda.pkl', extract numerical features (Year, Month, Day, DayOfWeek). Then, the original 'Date' (datetime object) column MUST BE DROPPED before modeling."
    9.  Conclude.

    ReAct Format: Action: Python, Action Input: <NL instruction>.
    "Final Answer:" MUST be a single well-formed JSON object string, enclosed ONLY in ```json ... ```. NO OTHER TEXT OR COMMENTS OUTSIDE THIS JSON BLOCK.
    **All string values within the JSON MUST be properly quoted and escaped (e.g., newlines as \\n, double quotes as \\\", backslashes as \\\\).**
    The JSON object MUST have keys: "eda_summary", "data_profile", "data_quality_report", "key_insights", "model_suggestions", "fe_suggestions", "artifact_references".
    Example for "artifact_references": {{ "processed_train_data": "tool_reported_cleaned_train_final_eda.pkl", "plots": {{ "target_distribution": "tool_reported_plot.png" }} }}
    Begin.
    """
    final_answer_json_string = run_generic_react_loop(prompt_content, state.get("max_react_iterations", 20), eda_tool_context_hint)
    
    eda_report_output = parse_llm_json_final_answer(final_answer_json_string, node_name) # Pass node_name for logging
    output_to_state = {"current_stage_completed": f"{node_name}_SUCCESS", "eda_report": eda_report_output}

    if "error" in eda_report_output:
        logger.error(f"{node_name} failed. Error in JSON report: {eda_report_output.get('error')}")
        output_to_state["current_stage_completed"] = f"{node_name}_FAILED_JSON_OUTPUT"
        output_to_state["pipeline_status_message"] = f"Critical error in {node_name}: Could not generate valid EDA report. Error: {eda_report_output.get('error')}"
        # Populate with defaults to prevent NoneType errors downstream, clearly indicating failure
        output_to_state.update({ "eda_model_suggestions": [], "eda_fe_suggestions": [],
            "eda_processed_train_ref": "ERROR_EDA_failed_train.pkl", 
            "eda_processed_val_ref": "ERROR_EDA_failed_val.pkl",
            "eda_processed_test_ref": "ERROR_EDA_failed_test.pkl", 
            "eda_plot_references": {} })
    else:
        logger.info(f"{node_name} produced EDA report successfully.")
        artifact_refs = eda_report_output.get("artifact_references", {})
        # Validate presence of critical references
        required_eda_refs = {
            "processed_train_data": "default_eda_train_missing_ref.pkl",
            "processed_val_data": "default_eda_val_missing_ref.pkl",
            "processed_test_data": "default_eda_test_missing_ref.pkl"
        }
        missing_critical_refs = []
        for key, default_val in required_eda_refs.items():
            if not artifact_refs.get(key):
                missing_critical_refs.append(key)
                artifact_refs[key] = default_val # Use default if missing
        
        if missing_critical_refs:
            err_msg = f"{node_name}_FAILED_MISSING_CRITICAL_REFS: EDA report missing references for: {missing_critical_refs}."
            logger.error(err_msg)
            output_to_state["current_stage_completed"] = f"{node_name}_FAILED_MISSING_CRITICAL_REFS"
            output_to_state["pipeline_status_message"] = err_msg

        output_to_state.update({
            "eda_model_suggestions": eda_report_output.get("model_suggestions", []), 
            "eda_fe_suggestions": eda_report_output.get("fe_suggestions", []),
            "eda_processed_train_ref": artifact_refs.get("processed_train_data"),
            "eda_processed_val_ref": artifact_refs.get("processed_val_data"),
            "eda_processed_test_ref": artifact_refs.get("processed_test_data"),
            "eda_plot_references": artifact_refs.get("plots", {})
        })
    return output_to_state
