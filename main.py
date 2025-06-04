import os
import re
import json 
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

    # Output from EdaAgentNode: A structured report
    eda_report: Optional[Dict[str, any]] 

    # Output from FeatureEngineeringAgentNode
    fe_applied_steps_summary: Optional[str]
    fe_final_feature_list: Optional[List[str]] 
    fe_X_train_ref: Optional[str] # Typically .pkl for DataFrames
    fe_y_train_ref: Optional[str] # Typically .pkl
    fe_X_val_ref: Optional[str]   
    fe_y_val_ref: Optional[str]
    fe_X_test_ref: Optional[str]  
    fe_transformer_references: Optional[Dict[str, str]] # e.g., {"scaler": "scaler.joblib"}
    fe_untrained_full_pipeline_ref: Optional[str] # e.g., "untrained_pipeline.joblib"

    # Output from ModelingNode
    model_trained_pipeline_ref: Optional[str] # e.g., "trained_pipeline.joblib"
    model_training_summary: Optional[str]

    # Output from Evaluation Node
    evaluation_summary: Optional[str] 
    evaluation_metrics: Optional[Dict[str, float]]
    test_set_prediction_status: Optional[str]

    # Control and tracking
    current_stage_completed: Optional[str]
    max_react_iterations: Optional[int]


# --- 2. Interface for your Agnostic PythonTool ---
# REPLACE THIS FUNCTION WITH THE ACTUAL CALL TO YOUR AGNO_PYTHON_TOOL
def agno_python_tool_interface(instruction: str, agent_context_hint: Optional[str] = None) -> str:
    """
    This function is the integration point for your actual agno_python_tool.
    It takes a natural language instruction from the LLM.
    Your tool internally generates and executes Python code based on this instruction.
    It MUST return a string observation that includes clear references (FULL FILENAMES 
    including extensions like .pkl for data, .joblib for sklearn objects, .png for plots) 
    for ANY artifact that was created or saved as a result of the instruction, 
    IF THE LLM'S INSTRUCTION ASKED FOR THE ARTIFACT TO BE SAVED AND ITS REFERENCE REPORTED.
    """
    print(f"    [AGNO_PYTHON_TOOL INTERFACE] Sending Instruction to your tool:\n    '{instruction}'")
    if agent_context_hint:
        print(f"    Agent Context Hint (passed to your tool): {agent_context_hint}")
    
    sim_observation = f"Observation: PythonTool processed instruction: '{instruction}'. "
    instruction_lower = instruction.lower()

    if "load the dataset from" in instruction_lower and "report its reference" in instruction_lower:
        if "train_data.csv" in instruction:
            sim_observation += "Dataset 'dummy_pipeline_data/train_data.csv' loaded. Tool reports its reference as 'train_df_loaded_ref.pkl'."
        # ... other load simulations for val/test ...
    elif "clean data referenced by" in instruction_lower and "report the new reference" in instruction_lower:
        match = re.search(r"clean data referenced by '([^']+)'", instruction_lower)
        old_ref = match.group(1) if match else "unknown_ref"
        sim_observation += f"Data '{old_ref}' cleaned. New reference reported by tool: 'cleaned_{old_ref.replace('.pkl','_v2.pkl')}'."
    elif "generate a histogram for" in instruction_lower and "save it, report the filename, and provide a textual description" in instruction_lower:
        match = re.search(r"histogram for the '([^']+)' column", instruction_lower)
        col_name = match.group(1) if match else "unknown_col"
        sim_observation += f"Histogram for '{col_name}' generated. Plot saved by tool as '{col_name}_histogram.png'. Description: The '{col_name}' distribution is [simulated_description]."
    
    # FE specific: Saving sklearn objects as .joblib
    elif "fit a standardscaler" in instruction_lower and "save it as a .joblib file and report its reference" in instruction_lower:
        sim_observation += "StandardScaler fitted. Saved by tool. Reference is 'fitted_price_scaler.joblib'."
    elif "fit a onehotencoder" in instruction_lower and "save it as a .joblib file and report its reference" in instruction_lower:
        sim_observation += "OneHotEncoder fitted. Saved by tool. Reference is 'fitted_category_encoder.joblib'."
    elif "create a scikit-learn pipeline" in instruction_lower and "save this untrained pipeline as a .joblib file and report its reference" in instruction_lower:
        sim_observation += "Untrained Scikit-learn pipeline created. Saved by tool. Reference is 'untrained_full_ml_pipeline.joblib'."
    
    # Modeling specific: Saving trained pipeline as .joblib
    elif "train the pipeline" in instruction_lower and "save the trained pipeline as a .joblib file and report its reference" in instruction_lower:
        sim_observation += "Pipeline trained. Trained pipeline saved by tool. Reference is 'trained_model_pipeline.joblib'."
    
    elif "separate target" in instruction_lower and "report new references for x_train, y_train" in instruction_lower:
        sim_observation += "Target separated. New references reported by tool: X_train='X_train_final_ref.pkl', y_train='y_train_final_ref.pkl', X_val='X_val_final_ref.pkl', y_val='y_val_final_ref.pkl', X_test='X_test_final_ref.pkl'."
    elif "calculate classification metrics" in instruction_lower:
        sim_observation += "Metrics calculated by tool and reported as: {{'accuracy': 0.90, 'f1_score': 0.88, 'roc_auc': 0.94}}."
    elif "calculate regression metrics" in instruction_lower:
        sim_observation += "Metrics calculated by tool and reported as: {{'mse': 12.3, 'r_squared': 0.81}}."
    else:
        sim_observation += "Task completed. If specific artifacts were requested to be saved and their references reported (including .joblib for sklearn objects or .pkl for data), those details are included above."
            
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
    print("\n--- EDA Agent Node Running (Date Handling Emphasis) ---")
    data_paths = state["data_paths"]
    target_col = state.get("target_column_name", "Target")
    eda_tool_context_hint = f"Initial data paths: {json.dumps(data_paths)}. Target column: '{target_col}'. Task: EDA."

    prompt_content = f"""You are an Expert EDA Data Scientist performing iterative research.
    The PythonTool you use accepts natural language instructions. It will report back references (FULL FILENAMES including extensions like .pkl for data, .png for plots) to any data it loads/creates and plots it saves.
    When you instruct the PythonTool to generate a plot, ALSO instruct it to provide a textual description of the plot's key features.
    You MUST instruct the tool to report references for all created/modified data and plots.

    Initial context for PythonTool: {eda_tool_context_hint}

    Your EDA Process (Strict Order for Date Handling):
    1.  Load Datasets: Instruct PythonTool to load all datasets (train, val, test) using paths from context. Ask it to report the references it assigns (e.g., 'initial_train_df_ref.pkl').
    2.  Identify and Parse Date Columns: Using reported references, instruct PythonTool to identify date/timestamp columns and parse them to datetime objects. Ask for confirmation.
    3.  Basic Structure & Quality (Initial): Check structure, quality.
    4.  Prepare Data for Numeric Analysis: Instruct PythonTool: "For data referenced by 'initial_train_df_ref.pkl', create a new version suitable for numeric analysis (e.g. correlation heatmaps). Ensure only numeric columns (and target, if numeric) are present. If date columns were parsed, EXCLUDE the original datetime object column. Report the NEW reference for this numeric-ready dataset (e.g., 'train_df_numeric_ref.pkl')."
    5.  Numerical EDA: Using NEW numeric-ready references, compute correlations (ask for plot ref & description).
    6.  Distribution Analysis: Using initial or cleaned data references, analyze distributions (especially '{target_col}'). Ask for plot refs & descriptions.
    7.  Further Data Quality & Cleaning: If cleaning is done, instruct PythonTool to save cleaned datasets and report their FINAL references (e.g., 'cleaned_train_final_eda.pkl').
    8.  Iterative Research & Conclude.

    ReAct Format: Action: Python, Action Input: <NL instruction>.
    "Final Answer:" MUST be a single well-formed JSON object string, enclosed in ```json ... ```.
    JSON keys: "eda_summary", "data_quality_report": {{ "missing_values": [], "outliers": [], "date_columns_analysis": [] }}, "key_insights": [], "fe_suggestions": [], 
    "artifact_references": {{ 
        "processed_train_data": "<tool_reported_FINAL_train_ref.pkl>",
        "processed_val_data": "<tool_reported_FINAL_val_ref.pkl>",
        "processed_test_data": "<tool_reported_FINAL_test_ref.pkl>",
        "plots": {{ "target_distribution": "<plot_ref.png>", "correlation_matrix": "<plot_ref.png>" }}
    }}
    Begin.
    """
    final_answer_json_string = run_generic_react_loop(prompt_content, state.get("max_react_iterations", 15), eda_tool_context_hint)
    parsed_data = parse_llm_json_final_answer(final_answer_json_string, "EDA report generation failed.")
    
    eda_report_dict = {
        "comprehensive_summary": parsed_data.get("eda_summary", "Summary not parsed."),
        "data_quality": parsed_data.get("data_quality_report", {}),
        "key_insights": parsed_data.get("key_insights", []),
        "fe_suggestions": parsed_data.get("fe_suggestions", []),
        "artifact_references": parsed_data.get("artifact_references", {"plots": {}})
    }
    artifact_refs = eda_report_dict.get("artifact_references", {})
    plot_refs = artifact_refs.get("plots", {})
    
    return {
        "current_stage_completed": "EDA", "eda_report": eda_report_dict, 
        "eda_comprehensive_summary": eda_report_dict["comprehensive_summary"],
        "eda_identified_issues": eda_report_dict.get("data_quality",{}).get("missing_values", []) + eda_report_dict.get("data_quality",{}).get("outliers", []),
        "eda_fe_suggestions": eda_report_dict["fe_suggestions"],
        "eda_processed_train_ref": artifact_refs.get("processed_train_data", "default_eda_train.pkl"),
        "eda_processed_val_ref": artifact_refs.get("processed_val_data", "default_eda_val.pkl"),
        "eda_processed_test_ref": artifact_refs.get("processed_test_data", "default_eda_test.pkl"),
        "eda_plot_references": plot_refs
    }

def feature_engineering_agent_node(state: MultiAgentPipelineState) -> Dict[str, any]:
    print("\n--- Feature Engineering Agent Node Running ---")
    eda_report = state.get("eda_report", {}) 
    train_ref_from_eda = eda_report.get("artifact_references", {}).get("processed_train_data", "train_eda.pkl")
    val_ref_from_eda = eda_report.get("artifact_references", {}).get("processed_val_data", "val_eda.pkl")
    test_ref_from_eda = eda_report.get("artifact_references", {}).get("processed_test_data", "test_eda.pkl")
    suggestions_from_eda = eda_report.get("fe_suggestions", [])
    issues_from_eda = eda_report.get("data_quality", {}) 
    target_col = state.get("target_column_name", "Target")

    fe_tool_context_hint = (f"Input data refs from EDA: train='{train_ref_from_eda}', val='{val_ref_from_eda}', test='{test_ref_from_eda}'. "
                            f"Target: '{target_col}'. EDA FE Suggestions: {json.dumps(suggestions_from_eda)}. EDA Issues: {json.dumps(issues_from_eda)}")

    prompt_content = f"""You are a Feature Engineering Specialist. PythonTool takes NL instructions.
    Context from EDA:
    - Input Train Data Ref for PythonTool: {train_ref_from_eda} (tool should use this as 'current_train_df')
    - Input Val Data Ref for PythonTool: {val_ref_from_eda} (as 'current_val_df')
    - Input Test Data Ref for PythonTool: {test_ref_from_eda} (as 'current_test_df')
    - EDA FE Suggestions: {json.dumps(suggestions_from_eda) if suggestions_from_eda else 'Perform standard best-practice FE.'}
    - Target Column: '{target_col}'

    Your tasks:
    1. Instruct PythonTool to use/load datasets using EDA references.
    2. Based on EDA suggestions, instruct tool to:
        a. Fit transformers (scalers, encoders, imputers) on training data. Ask tool to SAVE each as a .joblib file and report its full filename reference (e.g., 'fitted_scaler.joblib').
        b. Create a Scikit-learn `Pipeline` object including these transformers AND an UNTRAINED model estimator. Ask tool to SAVE this untrained pipeline as a .joblib file and report its full filename reference.
    3. Separate features (X) and target ('{target_col}') from the (EDA processed or further transformed) train/val data. Create X_test. Ask tool to report references for X_train, y_train, etc. (e.g., 'X_train_fe.pkl') and the final feature list.

    ReAct Format: Action: Python, Action Input: <NL instruction>.
    "Final Answer:" MUST be a single well-formed JSON object string, enclosed in ```json ... ```.
    JSON should have keys:
    "fe_summary": (string) Summary of steps.
    "final_feature_list": [ (list of strings) ]
    "transformer_references": {{ "scaler": "<tool_reported_scaler_ref.joblib>", ... }}
    "untrained_full_pipeline_ref": (string) "<tool_reported_untrained_pipeline_ref.joblib>"
    "data_references": {{ "X_train": "<X_train_ref.pkl>", "y_train": "<y_train_ref.pkl>", ... }}
    Begin.
    """
    final_answer_json_string = run_generic_react_loop(prompt_content, state.get("max_react_iterations", 12), fe_tool_context_hint)
    parsed_data = parse_llm_json_final_answer(final_answer_json_string, "FE report generation failed.")
    data_refs = parsed_data.get("data_references", {})

    return {
        "current_stage_completed": "FeatureEngineering",
        "fe_applied_steps_summary": parsed_data.get("fe_summary", "FE Summary not parsed."),
        "fe_final_feature_list": parsed_data.get("final_feature_list", []),
        "fe_transformer_references": parsed_data.get("transformer_references", {}),
        "fe_untrained_full_pipeline_ref": parsed_data.get("untrained_full_pipeline_ref", "default_untrained_pipe.joblib"), # Default to .joblib
        "fe_X_train_ref": data_refs.get("X_train", "default_X_train_fe.pkl"),
        "fe_y_train_ref": data_refs.get("y_train", "default_y_train_fe.pkl"),
        "fe_X_val_ref": data_refs.get("X_val", "default_X_val_fe.pkl"),
        "fe_y_val_ref": data_refs.get("y_val", "default_y_val_fe.pkl"),
        "fe_X_test_ref": data_refs.get("X_test", "default_X_test_fe.pkl"),
        "fe_y_test_ref": data_refs.get("y_test") 
    }

def modeling_agent_node(state: MultiAgentPipelineState) -> Dict[str, any]:
    print("\n--- Modeling Agent Node Running ---")
    untrained_pipeline_ref = state.get("fe_untrained_full_pipeline_ref", "untrained_pipeline.joblib") # Expect .joblib
    x_train_ref = state.get("fe_X_train_ref", "X_train_fe.pkl") 
    y_train_ref = state.get("fe_y_train_ref", "y_train_fe.pkl")
    
    model_tool_context_hint = (f"Untrained pipeline ref: '{untrained_pipeline_ref}'. "
                               f"Train with X_train_ref: '{x_train_ref}', y_train_ref: '{y_train_ref}'.")

    prompt_content = f"""You are a Modeling Specialist. PythonTool takes NL instructions.
    Context from Feature Engineering:
    - Untrained Full Scikit-learn Pipeline Reference: {untrained_pipeline_ref} (expected to be a .joblib file)
    - X_train Reference (for training the pipeline): {x_train_ref}
    - y_train Reference (for training the pipeline): {y_train_ref}

    Your task:
    1. Instruct PythonTool to load the untrained pipeline from '{untrained_pipeline_ref}'.
    2. Instruct PythonTool to load X_train from '{x_train_ref}' and y_train from '{y_train_ref}'.
    3. Instruct PythonTool to train (fit) the loaded pipeline using this X_train and y_train.
    4. Instruct PythonTool to save the ENTIRE TRAINED PIPELINE as a .joblib file and report its exact reference including extension.

    ReAct Format: Action: Python, Action Input: <NL instruction>.
    "Final Answer:" MUST be a single well-formed JSON object string, enclosed in ```json ... ```.
    JSON should have keys:
    "model_training_summary": (string) Summary of training.
    "trained_pipeline_ref": (string) "<tool_reported_TRAINED_pipeline_ref.joblib>"
    Begin.
    """
    final_answer_json_string = run_generic_react_loop(prompt_content, state.get("max_react_iterations", 5), model_tool_context_hint)
    parsed_data = parse_llm_json_final_answer(final_answer_json_string, "Modeling report generation failed.")

    return {
        "current_stage_completed": "Modeling",
        "model_training_summary": parsed_data.get("model_training_summary", "Training summary not parsed."),
        "model_trained_pipeline_ref": parsed_data.get("trained_pipeline_ref", "default_trained_pipe.joblib") 
    }

def evaluation_node(state: MultiAgentPipelineState) -> Dict[str, any]:
    print("\n--- Evaluation Agent Node Running ---")
    trained_pipeline_ref = state.get("model_trained_pipeline_ref", "trained_model_pipeline.joblib") # Expect .joblib
    x_val_ref = state.get("fe_X_val_ref", "X_val_fe.pkl") 
    y_val_ref = state.get("fe_y_val_ref", "y_val_fe.pkl") 
    x_test_ref = state.get("fe_X_test_ref") 
    problem_type = state.get("problem_type", "classification")

    eval_tool_context_hint = (
        f"Trained pipeline ref: '{trained_pipeline_ref}'. Val X: '{x_val_ref}', Val y: '{y_val_ref}'. "
        f"Test X: '{x_test_ref if x_test_ref else 'N/A'}'. Problem: {problem_type}."
    )
    metrics_to_request = "accuracy, precision, recall, F1-score, ROC AUC" if problem_type == "classification" else "MSE, RMSE, MAE, R-squared"

    prompt_content = f"""You are an Evaluation Specialist. PythonTool takes NL instructions.
    Context: {eval_tool_context_hint}
    Tasks:
    1. Load trained pipeline '{trained_pipeline_ref}' (expected to be a .joblib file).
    2. Load validation data X_val from '{x_val_ref}' and y_val from '{y_val_ref}'.
    3. Make predictions on X_val.
    4. Calculate metrics: {metrics_to_request}. Instruct tool to report as a dictionary string within its observation.
    5. (Optional) If X_test ('{x_test_ref}') available, predict and report prediction output reference.

    ReAct Format: Action: Python, Action Input: <NL instruction>.
    "Final Answer:" MUST be a single well-formed JSON object string, enclosed in ```json ... ```.
    JSON should have keys:
    "evaluation_summary": (string) Summary.
    "validation_metrics": {{ (object of metric_name: value) e.g., "accuracy": 0.92 }}
    "test_set_prediction_status": (string) e.g., "Predictions on test set '{x_test_ref}' saved by tool to 'test_preds.csv'."
    Begin.
    """
    final_answer_json_string = run_generic_react_loop(prompt_content, state.get("max_react_iterations", 5), eval_tool_context_hint)
    parsed_data = parse_llm_json_final_answer(final_answer_json_string, "Evaluation report generation failed.")

    metrics_val = parsed_data.get("validation_metrics", {})
    if isinstance(metrics_val, str): 
        try:
            metrics_str_cleaned = re.sub(r"^[^\(\{]*", "", metrics_val) 
            if metrics_str_cleaned:
                 metrics = json.loads(metrics_str_cleaned.replace("'", "\"")) 
            else:
                metrics = {"error": "metrics string was empty after cleaning"}
        except json.JSONDecodeError:
            print(f"Warning: Could not parse metrics string from LLM's JSON: {metrics_val}")
            metrics = {"error": "failed to parse metrics string from JSON"}
    elif isinstance(metrics_val, dict):
        metrics = metrics_val
    else:
        metrics = {}

    return {
        "current_stage_completed": "Evaluation",
        "evaluation_summary": parsed_data.get("evaluation_summary", "Eval summary not parsed."),
        "evaluation_metrics": metrics,
        "test_set_prediction_status": parsed_data.get("test_set_prediction_status", "Test status not reported.")
    }

# --- 6. Construct and Compile the LangGraph ---
workflow = StateGraph(MultiAgentPipelineState)
workflow.add_node("eda_agent", eda_agent_node)
workflow.add_node("feature_engineering_agent", feature_engineering_agent_node)
workflow.add_node("modeling_agent", modeling_agent_node) 
workflow.add_node("evaluation_agent", evaluation_node) 

workflow.set_entry_point("eda_agent")
workflow.add_edge("eda_agent", "feature_engineering_agent")
workflow.add_edge("feature_engineering_agent", "modeling_agent")
workflow.add_edge("modeling_agent", "evaluation_agent")
workflow.add_edge("evaluation_agent", END)

pipeline_app = workflow.compile()

# --- 7. Example Invocation ---
if __name__ == "__main__":
    print("Starting ML Pipeline with Agent-Managed References & JSON Output (Joblib Focus)...")

    os.makedirs("dummy_pipeline_data", exist_ok=True)
    initial_data_paths = {
        "train": "dummy_pipeline_data/train_data.csv",
        "val": "dummy_pipeline_data/val_data.csv",
        "test": "dummy_pipeline_data/test_data.csv"
    }
    dummy_header = "Date,Price,Volume,FeatureA,FeatureB,Category,Target\n" 
    dummy_row_template = "2023-01-{day:02d},{price},{volume},{fA},{fB},{cat},{target}\n"
    for k, v_path in initial_data_paths.items():
        with open(v_path, "w") as f:
            f.write(dummy_header)
            for i in range(3): 
                 f.write(dummy_row_template.format(
                    day=i+1, price=100+i*10, volume=10000+i*1000, 
                    fA=0.5+i*0.1, fB=1.2-i*0.05, 
                    cat='TypeA' if i%2==0 else 'TypeB', 
                    target= (101+i*5) if k != "test" else (101+i*5) 
                ))

    initial_pipeline_state = {
        "data_paths": initial_data_paths,
        "target_column_name": "Target",
        "problem_type": "regression", 
        "max_react_iterations": 8 
    }

    config = {"configurable": {"thread_id": "ml_pipeline_agent_refs_json_006_joblib"}} # Changed thread_id

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
