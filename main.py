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
    # Key EDA outputs for direct access by subsequent nodes if needed (derived from eda_report)
    eda_model_suggestions: Optional[List[str]] 
    eda_fe_suggestions: Optional[List[str]]
    eda_processed_train_ref: Optional[str] 
    eda_processed_val_ref: Optional[str]   
    eda_processed_test_ref: Optional[str]  
    eda_plot_references: Optional[Dict[str, str]]

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
    
    sim_observation = f"Observation: PythonTool processed instruction: '{instruction}'. "
    instruction_lower = instruction.lower()

    if "load the dataset from" in instruction_lower and "report its reference" in instruction_lower:
        if "train_data.csv" in instruction:
            sim_observation += "Dataset 'dummy_pipeline_data/train_data.csv' loaded. Tool reports its reference as 'train_df_loaded_ref.pkl'."
        elif "val_data.csv" in instruction:
            sim_observation += "Dataset 'dummy_pipeline_data/val_data.csv' loaded. Tool reports its reference as 'val_df_loaded_ref.pkl'."
        elif "test_data.csv" in instruction:
            sim_observation += "Dataset 'dummy_pipeline_data/test_data.csv' loaded. Tool reports its reference as 'test_df_loaded_ref.pkl'."
        else:
            sim_observation += "Dataset loaded. Tool assigned generic reference '<generic_loaded_data_ref.pkl>'."
    elif "identify the 'date' column" in instruction_lower and "parse it as datetime" in instruction_lower:
        sim_observation += "Identified 'Date' column in 'train_df_loaded_ref.pkl' and parsed to datetime64[ns] using format YYYY-MM-DD. Confirmed."
    elif "create a numeric-only version" in instruction_lower and "excluding the 'date' column" in instruction_lower and "report the new reference" in instruction_lower:
        sim_observation += "Numeric-only version created (e.g., 'train_df_numeric_for_corr.pkl'). Original 'Date' column (datetime object) excluded."
    elif "extract year, month, day, dayofweek from the 'date' column" in instruction_lower and "drop the original 'date' column" in instruction_lower and "report new data references" in instruction_lower:
        sim_observation += ("Date features (Year, Month, Day, DayOfWeek) extracted from 'Date' column. "
                           "Original 'Date' column (datetime object) has been dropped from train, val, and test data. "
                           "New data references reported by tool: 'train_with_date_features.pkl', 'val_with_date_features.pkl', 'test_with_date_features.pkl'.")
    elif "clean data referenced by" in instruction_lower and "report the new reference" in instruction_lower: 
        if "'train_df_loaded_ref.pkl'" in instruction_lower or "'train_df_numeric_for_corr.pkl'" in instruction_lower or "'train_with_date_features.pkl'" in instruction_lower :
             sim_observation += "Training data cleaned. New reference reported by tool: 'cleaned_train_final_eda.pkl'."
        elif "'val_df_loaded_ref.pkl'" in instruction_lower or "'val_with_date_features.pkl'" in instruction_lower :
             sim_observation += "Validation data cleaned. New reference reported by tool: 'cleaned_val_final_eda.pkl'."
        elif "'test_df_loaded_ref.pkl'" in instruction_lower or "'test_with_date_features.pkl'" in instruction_lower :
             sim_observation += "Test data cleaned. New reference reported by tool: 'cleaned_test_final_eda.pkl'."
        else:
            sim_observation += "Data cleaned. New reference reported by tool: 'cleaned_generic_data_ref.pkl'."

    elif "generate a histogram for" in instruction_lower and "save it, report the filename, and provide a textual description" in instruction_lower:
        sim_observation += "Histogram generated. Plot saved as 'sim_histogram.png'. Description: [Simulated detailed description, e.g., 'The distribution is moderately right-skewed with a mean of X and median of Y.']."
    elif "fit a standardscaler" in instruction_lower and "save it as a .joblib file and report its reference" in instruction_lower:
        sim_observation += "StandardScaler fitted. Saved by tool. Reference is 'fitted_scaler.joblib'."
    elif "create an untrained scikit-learn pipeline using preprocessors" in instruction_lower and "estimator type" in instruction_lower and "save it as a .joblib file and report its reference" in instruction_lower:
        model_type_match = re.search(r"estimator type '([^']+)'", instruction_lower)
        model_type = model_type_match.group(1) if model_type_match else "UnknownModel"
        config_idx_match = re.search(r"config_idx_(\d+)", instruction_lower) 
        config_idx_str = config_idx_match.group(1) if config_idx_match else "0"
        ref = f"untrained_pipeline_{model_type}_config{config_idx_str}.joblib"
        sim_observation += f"Untrained Scikit-learn pipeline with preprocessors and {model_type} estimator created. Saved by tool. Reference is '{ref}'."
    elif "load the untrained pipeline" in instruction_lower and "train it using x_train" in instruction_lower:
        untrained_pipe_ref = re.search(r"untrained pipeline '([^']+)'", instruction_lower).group(1) if re.search(r"untrained pipeline '([^']+)'", instruction_lower) else "unknown_pipe.joblib"
        params_key = "default" 
        if "n_estimators=200" in instruction_lower: params_key = "n200"
        elif "n_estimators=50" in instruction_lower: params_key = "n50"
        
        model_type_sim = "DefaultModel"
        if "randomforest" in untrained_pipe_ref: model_type_sim="RandomForest" 
        elif "linearregression" in untrained_pipe_ref: model_type_sim="LinearRegression"

        full_model_key = f"{model_type_sim}_{params_key}"
        if full_model_key not in SIMULATED_MODEL_PERFORMANCE_REGISTRY: SIMULATED_MODEL_PERFORMANCE_REGISTRY[full_model_key] = random.uniform(0.6, 2.5) 
        else: SIMULATED_MODEL_PERFORMANCE_REGISTRY[full_model_key] *= random.uniform(0.85, 0.99) 
        current_sim_rmse = SIMULATED_MODEL_PERFORMANCE_REGISTRY[full_model_key]
        
        trained_model_ref = f"trained_{untrained_pipe_ref.replace('untrained_', '')}"
        sim_observation += (f"Pipeline '{untrained_pipe_ref}' trained. Trained pipeline saved. Reference is '{trained_model_ref}'.")
        if "report rmse" in instruction_lower: sim_observation += f" Evaluation on validation set complete. Reported RMSE: {current_sim_rmse:.4f}."
    elif "separate target" in instruction_lower and "report new references" in instruction_lower:
        sim_observation += "Target separated. Tool reports new references: X_train='X_train_final.pkl', y_train='y_train_final.pkl', X_val='X_val_final.pkl', y_val='y_val_final.pkl', X_test='X_test_final.pkl'."
    elif "calculate regression metrics" in instruction_lower: 
        final_pipe_ref = re.search(r"pipeline '([^']+)'", instruction_lower).group(1) if re.search(r"pipeline '([^']+)'", instruction_lower) else "unknown_pipe.joblib"
        final_sim_rmse = 0.45 
        for key, val_rmse in SIMULATED_MODEL_PERFORMANCE_REGISTRY.items():
            if key in final_pipe_ref: final_sim_rmse = val_rmse; break 
        sim_observation += f"Final evaluation using '{final_pipe_ref}'. Metrics reported by tool: {{'rmse': {final_sim_rmse:.4f}, 'mae': {final_sim_rmse*0.8:.4f}, 'r_squared': {max(0, 1 - final_sim_rmse / 2.0):.2f}}}."
    else:
        sim_observation += "Task completed. If specific artifacts were requested to be saved and their references reported, those details are included above."
            
    print(f"    [AGNO_PYTHON_TOOL INTERFACE] Returning Observation:\n    '{sim_observation}'")
    return sim_observation

# --- 3. Generic ReAct Loop Engine ---
llm = ChatOpenAI(model="gpt-4o", temperature=0.1) 

def run_generic_react_loop(
    initial_prompt_content: str,
    max_iterations: int,
    agent_context_hint_for_tool: Optional[str] = None 
) -> str: 
    react_messages: List[BaseMessage] = [SystemMessage(content=initial_prompt_content)]
    final_answer_text = json.dumps({"error": "Agent did not produce a Final Answer within iteration limit."}) 
    
    for i in range(max_iterations):
        print(f"  [GenericReActLoop] Iteration {i+1}/{max_iterations}")
        ai_response = llm.invoke(react_messages)
        ai_content = ai_response.content.strip()
        react_messages.append(ai_response) 
        print(f"    LLM: {ai_content[:450]}...")

        final_answer_match = re.search(r"Final Answer:\s*(```json\s*(.*?)\s*```|{\s*.*})", ai_content, re.DOTALL | re.IGNORECASE)
        action_match = re.search(r"Action:\s*Python\s*Action Input:\s*(.+)", ai_content, re.DOTALL | re.IGNORECASE) 

        if final_answer_match:
            json_block_content = final_answer_match.group(2) if final_answer_match.group(2) else final_answer_match.group(1)
            final_answer_text = json_block_content.strip()
            print(f"    Loop Concluded. Final Answer (JSON string) obtained:\n{final_answer_text}")
            break 
        elif action_match:
            nl_instruction_for_tool = action_match.group(1).strip()
            tool_observation = agno_python_tool_interface(nl_instruction_for_tool, agent_context_hint_for_tool)
            react_messages.append(HumanMessage(content=f"Observation: {tool_observation}"))
        else:
            react_messages.append(HumanMessage(content="System hint: Your response was not in the expected format. Please use 'Action: Python\\nAction Input: <NL_instruction>' or 'Final Answer: ```json\\n{...}\\n```'."))
            if i > 1: 
                final_answer_text = json.dumps({"error": "Agent failed to follow output format consistently."})
                print(f"    Agent failed to follow output format.")
                break 
        if i == max_iterations - 1: 
            print(f"    Max ReAct iterations reached.")
            if ai_content and not final_answer_match: 
                final_answer_text = json.dumps({"error": f"Max iterations reached. Last AI thought: {ai_content}"})
    return final_answer_text

# --- 4. Helper to Parse LLM's JSON Final Answer ---
def parse_llm_json_final_answer(final_answer_json_string: str, default_error_message: str = "Report generation failed.") -> Dict:
    try:
        if final_answer_json_string.startswith("```json"):
            json_string_cleaned = final_answer_json_string[7:-3].strip()
        elif final_answer_json_string.startswith("```"):
             json_string_cleaned = final_answer_json_string[3:-3].strip()
        else:
            json_string_cleaned = final_answer_json_string
        
        json_string_cleaned = re.sub(r",\s*([}\]])", r"\1", json_string_cleaned) 
        json_string_cleaned = re.sub(r"//.*?\n", "\n", json_string_cleaned) 
        parsed_data = json.loads(json_string_cleaned)
        return parsed_data
    except json.JSONDecodeError as e:
        print(f"    ERROR: Failed to decode JSON from LLM's Final Answer: {e}")
        print(f"    Problematic JSON string was: {final_answer_json_string}")
        return {"error": f"JSON Decode Error: {e}", "summary": default_error_message} 
    except Exception as e:
        print(f"    ERROR: Unexpected error parsing LLM's Final Answer: {e}")
        return {"error": f"Unexpected Parsing Error: {e}", "summary": default_error_message}

# --- 5. Define Agent Nodes ---

def eda_agent_node(state: MultiAgentPipelineState) -> Dict[str, any]:
    print("\n--- EDA Agent Node Running (Explicit Date Handling & Comprehensive Report) ---")
    data_paths = state["data_paths"]
    target_col = state.get("target_column_name", "Target")
    problem_type = state.get("problem_type", "regression") 
    eda_tool_context_hint = f"Initial data paths: {json.dumps(data_paths)}. Target column: '{target_col}'. Problem type: {problem_type}. Task: EDA for stock price prediction."

    prompt_content = f"""You are an Expert EDA Data Scientist performing iterative research to create a comprehensive 'EDA Manual'.
    The PythonTool you use accepts natural language instructions. It will report back references (FULL FILENAMES including extensions like .pkl for data, .png for plots) to any data it loads/creates and plots it saves.
    When you instruct the PythonTool to generate a plot, you MUST ALSO instruct it to provide a textual description of the plot's key features (shape, trend, skewness, outliers, correlation strength) in its observation, along with the plot's reference/filename. Use these textual descriptions in your reasoning and summary.
    You MUST instruct the tool to report references for ALL created/modified data and plots.

    Initial context for PythonTool: {eda_tool_context_hint}

    Your EDA Process (CRITICAL: Address 'Date' column with format 'YYYY-MM-DD' explicitly and EARLY):
    1.  **Load Datasets:** Instruct tool to load train, val, test datasets from paths in context. Ask it to report the references it assigns to these loaded dataframes (e.g., 'initial_train_df.pkl').
    2.  **Identify and Parse 'Date' Column:** Instruct tool: "For each loaded dataset reference (e.g., 'initial_train_df.pkl'), identify the 'Date' column. Parse it as datetime objects using format 'YYYY-MM-DD'. Confirm parsing and report the new dtype for 'Date' column in each dataset."
    3.  **Initial Structure & Quality:** Using these date-parsed references, check shapes, dtypes (confirm 'Date' is datetime), head/tail, missing values, outliers (request plot refs & descriptions for '{target_col}', 'Price', 'Volume').
    4.  **Prepare for Numeric Analysis (IMPORTANT):** Before numeric operations like correlation:
        Instruct PythonTool: "For data referenced by 'initial_train_df.pkl' (with parsed 'Date'), create a TEMPORARY numeric-only version FOR CORRELATION ANALYSIS. This version MUST EXCLUDE the original 'Date' column (datetime object) and any other identified non-numeric columns (except the target '{target_col}' if it's numeric). Report the NEW reference for this temporary numeric-ready dataset (e.g., 'train_df_numeric_for_corr.pkl')."
    5.  **Numerical EDA (Correlations):** Using the NEW numeric-ready references (e.g., 'train_df_numeric_for_corr.pkl'), compute correlations. Ask for plot ref & description for heatmap.
    6.  **Distribution & Time Series Analysis:** Using main data references (e.g., 'initial_train_df.pkl' with parsed dates), analyze distributions of '{target_col}', 'Price', 'Volume'. Plot '{target_col}' over the parsed 'Date' column to observe trends/seasonality. Ask for plot refs & descriptions.
    7.  **Data Cleaning (if needed):** If cleaning (imputation, etc.) is performed on main data references (e.g., 'initial_train_df.pkl'), instruct PythonTool to save these cleaned datasets and report their FINAL references (e.g., 'cleaned_train_final_eda.pkl'). These are the refs FE will use.
    8.  **Model & FE Suggestions:** Based on all findings (including date patterns, and the problem type: {problem_type}), provide regression model suggestions and specific FE suggestions. For dates, explicitly suggest: "FE Suggestion: From the parsed 'Date' column in 'cleaned_train_final_eda.pkl' (and val/test), extract numerical features like Year, Month, Day, DayOfWeek. After extraction, the original 'Date' (datetime object) column MUST BE DROPPED from the feature set before modeling."
    9.  Conclude with your comprehensive report.

    ReAct Format: Action: Python, Action Input: <NL instruction>.
    "Final Answer:" MUST be a single well-formed JSON object string, enclosed in ```json ... ```.
    The JSON object (your EDA Manual) MUST have these top-level keys:
    "eda_summary": (string) Comprehensive narrative summary of all findings, integrating insights from plot descriptions.
    "data_profile": {{
        "train_data_initial_ref": "<tool_reported_initial_train_ref.pkl>",
        "val_data_initial_ref": "<tool_reported_initial_val_ref.pkl>",
        "test_data_initial_ref": "<tool_reported_initial_test_ref.pkl>",
        "initial_shapes": {{ "train": [rows, cols], ...}},
        "column_dtypes_after_parsing": {{ "column_name": "dtype_after_parsing", ... }}
    }},
    "data_quality_report": {{ 
        "missing_values": [ {{ "column": "col_name", "dataset_ref": "ref", "details": "description from tool" }} ], 
        "outliers": [ {{ "column": "col_name", "dataset_ref": "ref", "details": "description from tool", "plot_ref": "plot.png" }} ],
        "date_columns_analysis": [ (list of strings, e.g., "'Date' column in 'initial_train_df.pkl' parsed to datetime64[ns] using YYYY-MM-DD format.") ]
    }},
    "key_insights": [ (list of strings, e.g., "Insight: 'Price' distribution (see 'price_dist.png') is right-skewed, suggesting a log transform for modeling.") ],
    "model_suggestions": [ (list of strings, e.g., "RandomForestRegressor for non-linearities") ],
    "fe_suggestions": [ (list of strings, e.g., "FE Suggestion: For parsed date column 'Date' in 'cleaned_train_final_eda.pkl', extract Year, Month, DayOfWeek as features, then DROP the original 'Date' column.") ], 
    "artifact_references": {{ 
        "processed_train_data": "<tool_reported_FINAL_train_ref.pkl_after_all_eda_cleaning>",
        "processed_val_data": "<tool_reported_FINAL_val_ref.pkl_after_all_eda_cleaning>",
        "processed_test_data": "<tool_reported_FINAL_test_ref.pkl_after_all_eda_cleaning>",
        "plots": {{ "target_distribution": "<plot_ref.png>", "correlation_matrix": "<plot_ref.png>", ... }}
    }}
    Begin your EDA research.
    """
    final_answer_json_string = run_generic_react_loop(prompt_content, state.get("max_react_iterations", 20), eda_tool_context_hint)
    
    eda_report_output = parse_llm_json_final_answer(final_answer_json_string, "EDA report generation failed.")
    
    artifact_refs = eda_report_output.get("artifact_references", {}) # Corrected key from "artifact_catalog" to "artifact_references"
    
    return {
        "current_stage_completed": "EDA", 
        "eda_report": eda_report_output,
        "eda_model_suggestions": eda_report_output.get("model_suggestions", []), 
        "eda_fe_suggestions": eda_report_output.get("fe_suggestions", []),
        "eda_processed_train_ref": artifact_refs.get("processed_train_data", "default_eda_train.pkl"), # Corrected key
        "eda_processed_val_ref": artifact_refs.get("processed_val_data", "default_eda_val.pkl"),     # Corrected key
        "eda_processed_test_ref": artifact_refs.get("processed_test_data", "default_eda_test.pkl"),    # Corrected key
        "eda_plot_references": artifact_refs.get("plots", {})
    }

def feature_engineering_agent_node(state: MultiAgentPipelineState) -> Dict[str, any]:
    print("\n--- Feature Engineering Agent Node Running (Strict Date Handling) ---")
    eda_report = state.get("eda_report", {}) 
    # Use the FINAL processed data references from the EDA's artifact_references
    train_ref_from_eda = eda_report.get("artifact_references", {}).get("processed_train_data", "train_eda.pkl") 
    val_ref_from_eda = eda_report.get("artifact_references", {}).get("processed_val_data", "val_eda.pkl")
    test_ref_from_eda = eda_report.get("artifact_references", {}).get("processed_test_data", "test_eda.pkl")
    
    suggestions_from_eda = eda_report.get("fe_suggestions", [])
    target_col = state.get("target_column_name", "Target")

    date_fe_suggestion_from_eda = "Ensure date features are extracted and original 'Date' column is dropped if EDA suggested it or if a 'Date' (datetime) column exists in input."
    for suggestion in suggestions_from_eda: 
        if "date column" in suggestion.lower() and "extract" in suggestion.lower() and "drop" in suggestion.lower():
            date_fe_suggestion_from_eda = suggestion; break

    fe_tool_context_hint = (f"Input data refs from EDA: train='{train_ref_from_eda}', val='{val_ref_from_eda}', test='{test_ref_from_eda}'. "
                            f"Target: '{target_col}'. EDA FE Suggestions: {json.dumps(suggestions_from_eda)}.")

    prompt_content = f"""You are a Feature Engineering Specialist for stock price prediction. PythonTool takes NL instructions.
    Context from EDA:
    - Input Train Data Ref (already cleaned by EDA): {train_ref_from_eda}
    - Input Val Data Ref (already cleaned by EDA): {val_ref_from_eda}
    - Input Test Data Ref (already cleaned by EDA): {test_ref_from_eda}
    - EDA FE Suggestions: {json.dumps(suggestions_from_eda)}
    - Specific EDA suggestion for 'Date' column (PRIORITIZE THIS): "{date_fe_suggestion_from_eda}"
    - Target Column: '{target_col}'

    Your tasks:
    1. Instruct PythonTool to use/load datasets using FINAL PROCESSED references from EDA (e.g., '{train_ref_from_eda}').
    2. **CRITICAL FIRST STEP for Date Handling:** Implement the EDA suggestion for the 'Date' column: "{date_fe_suggestion_from_eda}". This means instructing the tool to extract numerical date features (Year, Month, Day, DayOfWeek, etc.) from the (already parsed by EDA) 'Date' column AND THEN INSTRUCT THE TOOL TO DROP THE ORIGINAL 'Date' (datetime object) COLUMN from train, val, and test datasets. The tool must report new data references after these operations.
    3. Implement other FE steps from EDA suggestions (transformations, imputation) on these date-handled datasets. Apply consistently.
    4. Create and SAVE individual FITTED transformers (scalers, encoders for categoricals like 'Category', imputers) as .joblib files using training data. Ask tool to report full filename references.
    5. After ALL transformations (including date handling), separate features (X) and target ('{target_col}'). Ask tool to report references for X_train, y_train, X_val, y_val, X_test (e.g., 'X_train_fe_final.pkl') and the final feature list (this list MUST NOT contain the original 'Date' column).

    ReAct Format: Action: Python, Action Input: <NL instruction>.
    "Final Answer:" MUST be a single well-formed JSON object string, enclosed in ```json ... ```.
    JSON keys: "fe_summary", "final_feature_list", 
               "transformer_references": {{ "scaler_price": "<ref.joblib>", "encoder_category": "<ref.joblib>" }}, 
               "custom_transformer_module": (string or null) "<tool_reported_module_for_custom_transformers.py>",
               "data_references": {{ "X_train": "<X_train_ref.pkl>", ... }}
    Begin. Address date feature engineering as the first priority.
    """
    final_answer_json_string = run_generic_react_loop(prompt_content, state.get("max_react_iterations", 12), fe_tool_context_hint)
    parsed_data = parse_llm_json_final_answer(final_answer_json_string, "FE report generation failed.")
    data_refs = parsed_data.get("data_references", {})

    return {
        "current_stage_completed": "FeatureEngineering",
        "fe_applied_steps_summary": parsed_data.get("fe_summary", "FE Summary not parsed."),
        "fe_final_feature_list": parsed_data.get("final_feature_list", []),
        "fe_transformer_references": parsed_data.get("transformer_references", {}),
        "fe_custom_transformer_module": parsed_data.get("custom_transformer_module"), 
        "fe_X_train_ref": data_refs.get("X_train", "default_X_train_fe.pkl"),
        "fe_y_train_ref": data_refs.get("y_train", "default_y_train_fe.pkl"),
        "fe_X_val_ref": data_refs.get("X_val", "default_X_val_fe.pkl"),
        "fe_y_val_ref": data_refs.get("y_val", "default_y_val_fe.pkl"),
        "fe_X_test_ref": data_refs.get("X_test", "default_X_test_fe.pkl"),
        "fe_y_test_ref": data_refs.get("y_test") 
    }

def model_selection_decision_agent_node(state: MultiAgentPipelineState) -> Dict[str, any]:
    print("\n--- Model Selection Decision Agent Node Running ---")
    eda_report = state.get("eda_report", {}); 
    eda_model_suggestions = eda_report.get("model_suggestions", [])
    fe_final_feature_list = state.get("fe_final_feature_list", [])
    problem_type = state.get("problem_type", "regression") 
    
    decision_tool_context_hint = (f"EDA Model Suggestions: {json.dumps(eda_model_suggestions)}. "
                                  f"Final Features from FE ({len(fe_final_feature_list)} features): {json.dumps(fe_final_feature_list[:5])}... " 
                                  f"Problem type: {problem_type}.")

    prompt_content = f"""You are a Model Selection Strategist for predicting stock prices (a regression task).
    Context: {decision_tool_context_hint}
    Your Task: Based on EDA model suggestions and final features, select up to 2-3 promising Scikit-learn REGRESSION model types. For each, suggest initial hyperparameters or state to use defaults.
    "Final Answer:" JSON keys: "decision_rationale", "top_model_configurations": [ {{ "model_type": "e.g.RandomForestRegressor", "initial_hyperparameters": {{}}, "reasoning": "..." }} ]. Begin."""
    final_answer_json_string = run_generic_react_loop(prompt_content, 3, decision_tool_context_hint) 
    parsed_data = parse_llm_json_final_answer(final_answer_json_string, "Model selection decision failed.")
    top_configs = parsed_data.get("top_model_configurations", [])
    return {"current_stage_completed": "ModelSelectionDecision", "top_model_configurations": top_configs}


def modeling_agent_node(state: MultiAgentPipelineState) -> Dict[str, any]:
    print("\n--- Modeling Agent Node Running (Iterates Top Configurations for RMSE) ---")
    top_model_configurations = state.get("top_model_configurations", [])
    transformer_refs_from_fe = state.get("fe_transformer_references", {})
    x_train_ref = state.get("fe_X_train_ref"); y_train_ref = state.get("fe_y_train_ref")
    x_val_ref = state.get("fe_X_val_ref"); y_val_ref = state.get("fe_y_val_ref") 
    custom_module = state.get("fe_custom_transformer_module")
    target_rmse = state.get("target_rmse", 0.002) 
    config_idx = state.get("modeling_config_index", 0) 
    overall_best_rmse = state.get("best_rmse_so_far", float('inf'))
    overall_best_model_ref = state.get("best_model_ref_so_far"); overall_best_model_config = state.get("best_model_config_so_far")
    strategy_log_for_all_configs = state.get("modeling_strategy_log", [])
    max_configs_to_try = state.get("max_modeling_configs_to_try", len(top_model_configurations))

    if not top_model_configurations or config_idx >= len(top_model_configurations) or config_idx >= max_configs_to_try:
        summary_msg = f"Completed trying {config_idx} model configurations. Best RMSE: {overall_best_rmse:.4f}."
        return {"current_stage_completed": "Modeling", "model_training_summary": summary_msg, "model_trained_pipeline_ref": overall_best_model_ref, "best_rmse_so_far": overall_best_rmse, "best_model_ref_so_far": overall_best_model_ref, "best_model_config_so_far": overall_best_model_config, "modeling_strategy_log": strategy_log_for_all_configs, "modeling_config_index": config_idx }

    current_config = top_model_configurations[config_idx]; chosen_model_type = current_config.get("model_type", "RandomForestRegressor"); initial_hyperparams = current_config.get("initial_hyperparameters", {})
    model_tool_context_hint = (f"Config (Idx {config_idx}): Type='{chosen_model_type}', Params='{json.dumps(initial_hyperparams)}'. Transformers: {json.dumps(transformer_refs_from_fe)}. Custom module: '{custom_module}'. Data: X_train='{x_train_ref}', etc. Target RMSE: {target_rmse}.")
    prompt_content = f"""You are a Modeling Specialist, trying one model configuration for stock price prediction. PythonTool takes NL instructions. Context: {model_tool_context_hint}
    Task for THIS Configuration (Type: '{chosen_model_type}', Params: {json.dumps(initial_hyperparams)}):
    1. Instruct PythonTool to:
        a. Create UNTRAINED Scikit-learn pipeline: combine preprocessors loaded using references from '{json.dumps(transformer_refs_from_fe)}' (e.g. 'scaler_ref.joblib', 'encoder_ref.joblib') with estimator '{chosen_model_type}' using params '{json.dumps(initial_hyperparams)}'. (Ensure custom_module '{custom_module}' is usable if specified).
        b. Save this UNTRAINED pipeline for this config (e.g., 'untrained_config{config_idx}.joblib') & report ref.
        c. Load X_train ('{x_train_ref}') and y_train ('{y_train_ref}'). Train pipeline.
        d. Save TRAINED pipeline (e.g., 'trained_config{config_idx}.joblib') & report ref.
        e. Load X_val ('{x_val_ref}') and y_val ('{y_val_ref}'). Predict & calculate RMSE. Report RMSE.
    "Final Answer:" JSON for THIS trial: "config_trial_summary", "config_trained_pipeline_ref", "config_rmse", "model_type_tried", "hyperparameters_tried". Begin."""
    config_trial_final_answer_json = run_generic_react_loop(prompt_content, state.get("max_react_iterations", 7), model_tool_context_hint)
    parsed_config_trial_data = parse_llm_json_final_answer(config_trial_final_answer_json, f"Modeling config trial {config_idx} failed.")
    config_rmse = parsed_config_trial_data.get("config_rmse"); config_model_ref = parsed_config_trial_data.get("config_trained_pipeline_ref")
    new_best_rmse, new_best_model_ref, new_best_model_config = overall_best_rmse, overall_best_model_ref, overall_best_model_config
    if config_rmse is not None and isinstance(config_rmse, (int, float)) and config_rmse < overall_best_rmse: 
        new_best_rmse, new_best_model_ref, new_best_model_config = config_rmse, config_model_ref, current_config
    return {"modeling_config_index": config_idx + 1, "current_rmse": config_rmse, "best_rmse_so_far": new_best_rmse, "best_model_ref_so_far": new_best_model_ref, "best_model_config_so_far": new_best_model_config, "model_training_summary": parsed_config_trial_data.get("config_trial_summary"), "model_trained_pipeline_ref": config_model_ref, "modeling_strategy_log": strategy_log_for_all_configs + [f"ConfigIdx{config_idx}({chosen_model_type}): RMSE={config_rmse}"], "current_stage_completed": "Modeling_Config_Trial" }


def evaluation_node(state: MultiAgentPipelineState) -> Dict[str, any]:
    print("\n--- Evaluation Agent Node Running ---")
    trained_pipeline_ref = state.get("best_model_ref_so_far", state.get("model_trained_pipeline_ref", "default_best_trained.joblib"))
    x_val_ref = state.get("fe_X_val_ref"); y_val_ref = state.get("fe_y_val_ref"); x_test_ref = state.get("fe_X_test_ref") 
    problem_type = state.get("problem_type"); custom_transformer_module = state.get("fe_custom_transformer_module") 
    best_model_config_info = state.get("best_model_config_so_far", {})
    eval_tool_context_hint = (f"Trained pipeline ref (best from tuning): '{trained_pipeline_ref}'. Config: {json.dumps(best_model_config_info)}. Val X: '{x_val_ref}', Val y: '{y_val_ref}'. Test X: '{x_test_ref if x_test_ref else 'N/A'}'. Problem: {problem_type}.")
    if custom_transformer_module: eval_tool_context_hint += f" Custom transformer module: '{custom_transformer_module}'."
    metrics_to_request = "MSE, RMSE, MAE, R-squared" 
    prompt_content = f"""You are an Evaluation Specialist for a stock price prediction model. PythonTool takes NL instructions. Context: {eval_tool_context_hint}
    Tasks: Load trained pipeline '{trained_pipeline_ref}'. Load val data. Predict on X_val. Calculate metrics: {metrics_to_request}. Report as dict string. (Optional) Predict on X_test.
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
    config_idx = state.get("modeling_config_index", 0)
    top_configurations = state.get("top_model_configurations", [])
    max_configs_to_try = state.get("max_modeling_configs_to_try", len(top_configurations) if top_configurations else 0) 
    current_best_rmse = state.get("best_rmse_so_far", float('inf'))
    target_rmse = state.get("target_rmse", 0.002)
    print(f"  Config Idx to try: {config_idx} / Total Configs Provided: {len(top_configurations)} (Max to try: {max_configs_to_try}). Best RMSE: {current_best_rmse}, Target: {target_rmse}")
    if current_best_rmse <= target_rmse: print(f"  Target RMSE achieved. Evaluating."); return "evaluation_agent"
    if not top_configurations: print("  No model configs. Evaluating."); return "evaluation_agent"
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

# To enable persistence (optional):
# memory = SqliteSaver.from_conn_string("your_langgraph_pipeline_vF.sqlite") # Use a unique db name
# pipeline_app = workflow.compile(checkpointer=memory)
pipeline_app = workflow.compile()


# --- 7. Example Invocation ---
if __name__ == "__main__":
    print("Starting ML Pipeline with Explicit Date Handling, Decision Node & Iterative Modeling (Final Version)...")

    os.makedirs("dummy_pipeline_data", exist_ok=True)
    initial_data_paths = { "train": "dummy_pipeline_data/train_data.csv", "val": "dummy_pipeline_data/val_data.csv", "test": "dummy_pipeline_data/test_data.csv"}
    dummy_header = "Date,Price,Volume,FeatureA,FeatureB,Category,CustomText,Target\n" 
    dummy_row_template = "{date_val},{price},{volume},{fA},{fB},{cat},{text},{target}\n"
    for k, v_path in initial_data_paths.items():
        with open(v_path, "w") as f: f.write(dummy_header)
        for i in range(10): 
            year_str, month_str, day_str = "2023", f"{((i//30)%12)+1:02d}", f"{(i%28)+1:02d}" 
            date_val = f"{year_str}-{month_str}-{day_str}" 
            f.write(dummy_row_template.format(date_val=date_val, price=100+i*random.uniform(-2,2) + (i*0.5), volume=10000+i*100 + random.randint(-500,500), 
                                             fA=0.5+i*0.01, fB=1.2-i*0.01, cat='TypeA' if i%3==0 else ('TypeB' if i%3==1 else 'TypeC'), 
                                             text=f"Txt{i}", target= (101+i*0.25 + random.uniform(-1,1)))) 

    initial_pipeline_state = {
        "data_paths": initial_data_paths,
        "target_column_name": "Target", 
        "problem_type": "regression",   
        "max_react_iterations": 6,      # Max ReAct steps within each agent's single decision/action turn
        "target_rmse": 0.75,            # Target RMSE for tuning loop 
        "max_modeling_configs_to_try": 2, # How many of the Decision Node's suggestions to try
        "modeling_config_index": 0,     # Initial value for modeling loop over configurations
        "best_rmse_so_far": float('inf'), # Initial value
    }
    
    config = {"configurable": {"thread_id": f"ml_pipeline_final_run_{random.randint(1000,9999)}"}} # Unique thread ID per run for checkpointer

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
