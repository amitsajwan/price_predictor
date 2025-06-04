import os
import re
import json 
from typing import TypedDict, Annotated, List, Dict, Optional, Sequence
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI 
from langgraph.graph import StateGraph, END

# --- 1. Define the State for the Pipeline ---
class MultiAgentPipelineState(TypedDict):
    # Input
    data_paths: Dict[str, str] 
    target_column_name: Optional[str] 

    # Output from EdaAgentNode
    eda_comprehensive_summary: Optional[str]
    eda_identified_issues: Optional[List[str]]
    eda_fe_suggestions: Optional[List[str]] 
    eda_processed_train_ref: Optional[str] # Ref to data after EDA cleaning
    eda_processed_val_ref: Optional[str]   
    eda_processed_test_ref: Optional[str]  
    eda_plot_references: Optional[Dict[str, str]] 

    # Output from FeatureEngineeringAgentNode
    fe_applied_steps_summary: Optional[str]
    fe_final_feature_list: Optional[List[str]] 
    fe_X_train_transformed_ref: Optional[str] # Ref to X_train after FE transformations
    fe_y_train_ref: Optional[str]
    fe_X_val_transformed_ref: Optional[str]   
    fe_y_val_ref: Optional[str]
    fe_X_test_transformed_ref: Optional[str]  
    # References to saved, FITTED transformers or the UNTRAINED full pipeline
    fe_transformer_references: Optional[Dict[str, str]] # e.g., {"scaler": "scaler.pkl"}
    fe_untrained_full_pipeline_ref: Optional[str] # e.g., "untrained_model_with_preprocessing.pkl"

    # Output from ModelingNode
    model_trained_pipeline_ref: Optional[str] # Ref to the single, deployable, TRAINED pipeline PKL
    model_training_summary: Optional[str]

    # Output from Evaluation Node
    evaluation_metrics: Optional[Dict[str, float]]

    # Control and tracking
    current_stage_completed: Optional[str]
    max_react_iterations: Optional[int]


# --- 2. Interface for your Agnostic PythonTool ---
# REPLACE THIS FUNCTION WITH THE ACTUAL CALL TO YOUR AGNO_PYTHON_TOOL
# Simulation must now "understand" sklearn pipeline/transformer saving/loading instructions.
def agno_python_tool_interface(instruction: str, agent_context_hint: Optional[str] = None) -> str:
    """
    This function is the integration point for your actual agno_python_tool.
    It takes a natural language instruction.
    It should return a string observation from your tool, including
    references to new data artifacts or saved sklearn objects.
    """
    print(f"    [AGNO_PYTHON_TOOL INTERFACE] Sending Instruction:\n    '{instruction}'")
    if agent_context_hint:
        print(f"    Agent Context Hint: {agent_context_hint}")
    
    sim_observation = f"Observation: PythonTool processed: '{instruction}'. "
    instruction_lower = instruction.lower()

    # EDA Related Simulations (as before, focusing on data refs)
    if "load the dataset from 'dummy_pipeline_data/train_data.csv' and report its reference" in instruction_lower:
        sim_observation += "Dataset 'dummy_pipeline_data/train_data.csv' loaded. Tool refers to it as 'initial_train_df_ref'. Shape (3, 7)."
    elif "load the dataset from 'dummy_pipeline_data/val_data.csv' and report its reference" in instruction_lower:
        sim_observation += "Dataset 'dummy_pipeline_data/val_data.csv' loaded. Tool refers to it as 'initial_val_df_ref'. Shape (3, 7)."
    elif "load the dataset from 'dummy_pipeline_data/test_data.csv' and report its reference" in instruction_lower:
        sim_observation += "Dataset 'dummy_pipeline_data/test_data.csv' loaded. Tool refers to it as 'initial_test_df_ref'. Shape (3, 7)."
    elif "clean data referenced by 'initial_train_df_ref'" in instruction_lower:
        sim_observation += "Data 'initial_train_df_ref' cleaned. New reference is 'cleaned_train_df_eda.pkl'."
    elif "generate plot for 'price' distribution from 'cleaned_train_df_eda.pkl'" in instruction_lower:
        sim_observation += "Plot 'price_dist_plot_eda.png' generated for 'cleaned_train_df_eda.pkl'."
    
    # Feature Engineering Related Simulations (sklearn heavy)
    elif "fit a standardscaler on data from 'cleaned_train_df_eda.pkl' for column 'price', save it, and report its reference" in instruction_lower:
        sim_observation += "StandardScaler fitted on 'Price' column of 'cleaned_train_df_eda.pkl'. Saved as 'fitted_price_scaler.pkl'. This is its reference."
    elif "fit a onehotencoder on data from 'cleaned_train_df_eda.pkl' for column 'category', save it, and report its reference" in instruction_lower:
        sim_observation += "OneHotEncoder fitted on 'Category' column of 'cleaned_train_df_eda.pkl'. Saved as 'fitted_category_encoder.pkl'. This is its reference."
    elif "apply the scaler 'fitted_price_scaler.pkl' and encoder 'fitted_category_encoder.pkl' to 'cleaned_train_df_eda.pkl', 'cleaned_val_df_eda.pkl', 'cleaned_test_df_eda.pkl'. report new data references for these transformed datasets." in instruction_lower:
        sim_observation += ("Scaler and Encoder applied. "
                           "New transformed data references: "
                           "train_transformed_ref: 'train_fe_transformed.pkl', "
                           "val_transformed_ref: 'val_fe_transformed.pkl', "
                           "test_transformed_ref: 'test_fe_transformed.pkl'.")
    elif "create a scikit-learn pipeline with steps: scaler (ref: 'fitted_price_scaler.pkl'), encoder (ref: 'fitted_category_encoder.pkl'), and an untrained randomforestclassifier. save this untrained pipeline and report its reference." in instruction_lower:
        sim_observation += "Untrained Scikit-learn pipeline created with specified scaler, encoder, and RandomForestClassifier. Saved as 'untrained_full_pipeline.pkl'. This is its reference."
    elif "separate target 'target' from features in 'train_fe_transformed.pkl' and 'val_fe_transformed.pkl'. report new references for x_train, y_train, x_val, y_val, and the processed x_test from 'test_fe_transformed.pkl'." in instruction_lower:
        sim_observation += ("Target 'Target' separated. "
                           "Output data references: "
                           "X_train_ref: 'X_train_final_fe.pkl', y_train_ref: 'y_train_final_fe.pkl', "
                           "X_val_ref: 'X_val_final_fe.pkl', y_val_ref: 'y_val_final_fe.pkl', "
                           "X_test_ref: 'X_test_final_fe.pkl'. "
                           "Final feature list for X datasets: ['Price_scaled', 'Category_encoded_A', ...].")

    # Modeling Related Simulations
    elif "load the untrained pipeline from 'untrained_full_pipeline.pkl' and train it using x_train from 'x_train_final_fe.pkl' and y_train from 'y_train_final_fe.pkl'. save the trained pipeline and report its reference." in instruction_lower:
        sim_observation += "Untrained pipeline 'untrained_full_pipeline.pkl' loaded. Trained using 'X_train_final_fe.pkl' and 'y_train_final_fe.pkl'. Trained pipeline saved as 'trained_model_pipeline.pkl'. This is its reference."
    else:
        sim_observation += "Task completed. If new data artifacts or sklearn objects were created/saved, their references are included here or can be requested."
    
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
    final_answer_text = "Agent did not produce a Final Answer within iteration limit."
    
    for i in range(max_iterations):
        print(f"  [GenericReActLoop] Iteration {i+1}/{max_iterations}")
        ai_response = llm.invoke(react_messages)
        ai_content = ai_response.content.strip()
        react_messages.append(ai_response) 
        print(f"    LLM: {ai_content[:400]}...")

        final_answer_match = re.search(r"Final Answer:\s*(.+)", ai_content, re.DOTALL | re.IGNORECASE)
        action_match = re.search(r"Action:\s*Python\s*Action Input:\s*(.+)", ai_content, re.DOTALL | re.IGNORECASE) 

        if final_answer_match:
            final_answer_text = final_answer_match.group(1).strip()
            print(f"    Loop Concluded. Final Answer obtained.")
            break 
        elif action_match:
            nl_instruction_for_tool = action_match.group(1).strip()
            tool_observation = agno_python_tool_interface(nl_instruction_for_tool, agent_context_hint_for_tool)
            react_messages.append(HumanMessage(content=f"Observation: {tool_observation}"))
        else:
            react_messages.append(HumanMessage(content="System hint: Your response was not in the expected format. Please use 'Action: Python\\nAction Input: <NL_instruction>' or 'Final Answer: <summary>'."))
            if i > 1: 
                final_answer_text = "Agent failed to follow output format consistently."
                print(f"    {final_answer_text}")
                break 
        if i == max_iterations - 1: 
            print(f"    Max ReAct iterations reached.")
            if ai_content and not final_answer_match: 
                final_answer_text = f"Max iterations reached. Last AI thought: {ai_content}"
    return final_answer_text

# --- 4. Define Agent Nodes (Specialized Logic and Parsing) ---

def eda_agent_node(state: MultiAgentPipelineState) -> Dict[str, any]:
    print("\n--- EDA Agent Node Running ---")
    data_paths = state["data_paths"]
    target_col = state.get("target_column_name", "Target")
    eda_tool_context_hint = f"Initial data paths: {json.dumps(data_paths)}. Target column: '{target_col}'."

    prompt_content = f"""You are an Expert EDA Data Scientist.
    PythonTool takes NL instructions and reports data/plot references.
    Initial context for PythonTool: {eda_tool_context_hint}

    Instruct PythonTool to:
    1. Load train, val, test datasets using paths from context. Ask for their references.
    2. Using these references, check structure, quality (missing values, outliers - ask for plot refs), distributions (especially '{target_col}' - ask for plot ref), correlations (ask for plot ref).
    3. Perform initial cleaning if needed, ask for NEW references for cleaned datasets.

    ReAct Format:
    Action: Python
    Action Input: <NL instruction, e.g., "Load dataset from '{data_paths.get('train')}' and report its reference.">
    (Obs from Tool, e.g., "Loaded. Tool refers to it as 'train_df_ref_01'.")

    "Final Answer:" MUST include:
    1. EDA Comprehensive Summary: <details>
    2. Data Quality Report: (list issues)
       - Missing Values: <details, e.g., "'F3' in 'train_df_ref_01' has 10% missing.">
    3. Key Insights: (list insights)
       - Insight: <details, e.g., "'Price' is right-skewed in 'train_df_ref_01'.">
    4. Feature Engineering Suggestions: (list suggestions)
       - FE Suggestion: <details, e.g., "Log transform 'Price' from 'train_df_ref_01'.">
    5. Data References (as reported by PythonTool):
       - Processed Train Data: <tool_reported_train_ref_after_eda_cleaning>
       - Processed Val Data: <tool_reported_val_ref_after_eda_cleaning>
       - Processed Test Data: <tool_reported_test_ref_after_eda_cleaning>
       - Plot - Target Distribution: <tool_reported_plot_ref>
       - Plot - Correlation Matrix: <tool_reported_plot_ref>
    Begin.
    """
    final_answer_string = run_generic_react_loop(prompt_content, state.get("max_react_iterations", 8), eda_tool_context_hint)
    
    parsed_output = {"current_stage_completed": "EDA"}
    summary_match = re.search(r"EDA Comprehensive Summary:\s*(.*?)(?=\nData Quality Report:|$)", final_answer_string, re.DOTALL | re.IGNORECASE)
    parsed_output["eda_comprehensive_summary"] = summary_match.group(1).strip() if summary_match else "Summary not parsed."
    parsed_output["eda_identified_issues"] = re.findall(r"-\s*Missing Values:\s*(.+)", final_answer_string, re.IGNORECASE) + re.findall(r"-\s*Outliers:\s*(.+)", final_answer_string, re.IGNORECASE)
    parsed_output["eda_fe_suggestions"] = re.findall(r"-\s*FE Suggestion:\s*(.+)", final_answer_string, re.IGNORECASE)
    parsed_output["eda_key_insights"] = re.findall(r"-\s*Insight:\s*(.+)", final_answer_string, re.IGNORECASE)
    
    plot_refs = {}
    plot_matches = re.findall(r"Plot\s*-\s*([^:]+):\s*(\S+)", final_answer_string, re.IGNORECASE)
    for name, ref in plot_matches: plot_refs[name.strip().lower().replace(" ", "_")] = ref.strip("'\"")
    parsed_output["eda_plot_references"] = plot_refs

    for key, pattern_str in {
        "eda_processed_train_ref": r"Processed Train Data:\s*(\S+)",
        "eda_processed_val_ref": r"Processed Val Data:\s*(\S+)",
        "eda_processed_test_ref": r"Processed Test Data:\s*(\S+)",
    }.items():
        match = re.search(pattern_str, final_answer_string, re.IGNORECASE)
        parsed_output[key] = match.group(1).strip("'\"") if match else f"default_{key}_not_found.pkl"
    return parsed_output

def feature_engineering_agent_node(state: MultiAgentPipelineState) -> Dict[str, any]:
    print("\n--- Feature Engineering Agent Node Running ---")
    train_ref_eda = state.get("eda_processed_train_ref", "train_eda.pkl")
    val_ref_eda = state.get("eda_processed_val_ref", "val_eda.pkl")
    test_ref_eda = state.get("eda_processed_test_ref", "test_eda.pkl")
    suggestions = state.get("eda_fe_suggestions", [])
    target_col = state.get("target_column_name", "Target")

    fe_tool_context_hint = (f"Input data refs from EDA: train='{train_ref_eda}', val='{val_ref_eda}', test='{test_ref_eda}'. "
                            f"Target: '{target_col}'. EDA FE Suggestions: {suggestions}")

    prompt_content = f"""You are a Feature Engineering Specialist.
    PythonTool takes NL instructions and reports data/object references.
    Context from EDA:
    - Input Train Data Ref: {train_ref_eda} (tool should use this as 'current_train_df')
    - Input Val Data Ref: {val_ref_eda} (as 'current_val_df')
    - Input Test Data Ref: {test_ref_eda} (as 'current_test_df')
    - EDA FE Suggestions: {suggestions if suggestions else 'Perform standard best-practice FE.'}
    - Target Column: '{target_col}'

    Your tasks:
    1. Instruct PythonTool to use/load datasets using EDA references.
    2. Based on EDA suggestions, instruct tool to:
        a. Fit transformers (scalers, encoders, imputers) on 'current_train_df'. Ask tool to SAVE each fitted transformer and report its reference (e.g., 'fitted_scaler.pkl').
        b. Apply these saved, fitted transformers to 'current_train_df', 'current_val_df', 'current_test_df'. Ask tool to report NEW references for these transformed datasets.
    3. (Primary Goal) Instruct PythonTool to create a Scikit-learn `Pipeline` object. This pipeline should include all the fitted transformers (loaded from their references) AND an UNTRAINED model estimator (e.g., RandomForestClassifier). Ask tool to SAVE this untrained full pipeline and report its reference.
    4. As a fallback or for verification, also instruct tool to separate features (X) and target ('{target_col}') from the transformed train/val data, and create X_test. Ask tool to report references for X_train, y_train, X_val, y_val, X_test and the final feature list.

    ReAct Format:
    Action: Python
    Action Input: <NL instruction, e.g., "Fit a StandardScaler on 'current_train_df' for numeric columns, save it, and report its reference.">

    "Final Answer:" MUST include:
    1. FE Applied Steps Summary: <Summary of transformers fitted, pipeline created.>
    2. Final Feature List: (as reported by tool for X datasets)
       - Feature: <feature_name_1>
    3. Transformer References (as reported by PythonTool):
       - Scaler Reference: <tool_reported_scaler_ref>
       - Encoder Reference: <tool_reported_encoder_ref>
       (etc. for other transformers)
    4. Untrained Full Pipeline Reference: <tool_reported_untrained_pipeline_ref>
    5. (Optional/Verification) Transformed X/y Data References:
       - X_train Transformed Reference: <tool_reported_X_train_ref>
       - y_train Reference: <tool_reported_y_train_ref> 
       (and for X_val, y_val, X_test)
    Begin.
    """
    final_answer_string = run_generic_react_loop(prompt_content, state.get("max_react_iterations", 12), fe_tool_context_hint)

    parsed_output = {"current_stage_completed": "FeatureEngineering"}
    summary_match = re.search(r"FE Applied Steps Summary:\s*(.*?)(?=\nFinal Feature List:|\nTransformer References:|\nUntrained Full Pipeline Reference:|$)", final_answer_string, re.DOTALL | re.IGNORECASE)
    parsed_output["fe_applied_steps_summary"] = summary_match.group(1).strip() if summary_match else "FE Summary not parsed."
    parsed_output["fe_final_feature_list"] = re.findall(r"-\s*Feature:\s*(.+)", final_answer_string, re.IGNORECASE)
    
    transformer_refs = {}
    # More specific parsing for known transformer types if prompted, or generic
    tf_matches = re.findall(r"-\s*(\w+)\s*Reference:\s*(\S+)", final_answer_string, re.IGNORECASE)
    for tf_name, tf_ref in tf_matches:
        if "pipeline" not in tf_name.lower() and "train" not in tf_name.lower() and "val" not in tf_name.lower() and "test" not in tf_name.lower():
            transformer_refs[tf_name.strip().lower()] = tf_ref.strip("'\"")
    parsed_output["fe_transformer_references"] = transformer_refs

    pipeline_ref_match = re.search(r"Untrained Full Pipeline Reference:\s*(\S+)", final_answer_string, re.IGNORECASE)
    parsed_output["fe_untrained_full_pipeline_ref"] = pipeline_ref_match.group(1).strip("'\"") if pipeline_ref_match else "untrained_pipeline_not_parsed.pkl"

    for key, pattern_str in {
        "fe_X_train_transformed_ref": r"X_train Transformed Reference:\s*(\S+)",
        "fe_y_train_ref": r"y_train Reference:\s*(\S+)",
        "fe_X_val_transformed_ref": r"X_val Transformed Reference:\s*(\S+)",
        "fe_y_val_ref": r"y_val Reference:\s*(\S+)",
        "fe_X_test_transformed_ref": r"X_test Transformed Reference:\s*(\S+)",
    }.items():
        match = re.search(pattern_str, final_answer_string, re.IGNORECASE)
        parsed_output[key] = match.group(1).strip("'\"") if match else f"ref_not_parsed_for_{key.replace('fe_','').replace('_transformed_ref','').replace('_ref','')}.pkl"
    # y_test is usually not created from FE, so not explicitly parsed here unless tool reports it.
    parsed_output["fe_y_test_ref"] = None 

    return parsed_output

def modeling_agent_node(state: MultiAgentPipelineState) -> Dict[str, any]: # Renamed from modeling_node
    print("\n--- Modeling Agent Node Running ---")
    untrained_pipeline_ref = state.get("fe_untrained_full_pipeline_ref", "untrained_pipeline.pkl")
    # FE agent should ideally provide transformed X/y for training the pipeline
    # If the pipeline itself handles all transforms from raw EDA data, then X/y refs might be different.
    # For this example, assume FE provides X/y that are ready for the *model estimator* part if pipeline is just model
    # OR, if fe_untrained_full_pipeline_ref contains preprocessing, it needs X_train from EDA (or earlier FE step)
    # Let's assume the `fe_untrained_full_pipeline_ref` is a full sklearn pipeline (preproc + model)
    # and it needs to be trained on data that has had *some* FE done, like X/y separation.
    x_train_ref = state.get("fe_X_train_transformed_ref", "X_train_fe.pkl") 
    y_train_ref = state.get("fe_y_train_ref", "y_train_fe.pkl")
    
    model_tool_context_hint = (f"Untrained pipeline ref: '{untrained_pipeline_ref}'. "
                               f"Train with X_train_ref: '{x_train_ref}', y_train_ref: '{y_train_ref}'.")

    prompt_content = f"""You are a Modeling Specialist.
    PythonTool takes NL instructions.
    Context from Feature Engineering:
    - Untrained Full Scikit-learn Pipeline Reference: {untrained_pipeline_ref} (This pipeline includes preprocessing steps and an untrained model estimator)
    - X_train Reference (for training the pipeline): {x_train_ref}
    - y_train Reference (for training the pipeline): {y_train_ref}

    Your task:
    1. Instruct PythonTool to load the untrained pipeline from '{untrained_pipeline_ref}'.
    2. Instruct PythonTool to load X_train from '{x_train_ref}' and y_train from '{y_train_ref}'.
    3. Instruct PythonTool to train (fit) the loaded pipeline using this X_train and y_train.
    4. Instruct PythonTool to save the ENTIRE TRAINED PIPELINE to a new file and report its reference.

    ReAct Format:
    Action: Python
    Action Input: <NL instruction>

    "Final Answer:" MUST include:
    1. Model Training Summary: <Summary of training process.>
    2. Trained Pipeline Reference: <tool_reported_TRAINED_pipeline_ref>
    Begin.
    """
    final_answer_string = run_generic_react_loop(prompt_content, state.get("max_react_iterations", 5), model_tool_context_hint)

    parsed_output = {"current_stage_completed": "Modeling"}
    summary_match = re.search(r"Model Training Summary:\s*(.*?)(?=\nTrained Pipeline Reference:|$)", final_answer_string, re.DOTALL | re.IGNORECASE)
    parsed_output["model_training_summary"] = summary_match.group(1).strip() if summary_match else "Training summary not parsed."
    
    trained_ref_match = re.search(r"Trained Pipeline Reference:\s*(\S+)", final_answer_string, re.IGNORECASE)
    parsed_output["model_trained_pipeline_ref"] = trained_ref_match.group(1).strip("'\"") if trained_ref_match else "trained_pipeline_not_parsed.pkl"
    
    return parsed_output


def evaluation_node(state: MultiAgentPipelineState) -> Dict[str, any]:
    print("\n--- Evaluation Node Running (Placeholder) ---")
    trained_pipeline_ref = state.get("model_trained_pipeline_ref", "trained_model.pkl")
    # For evaluation, we'd use X_val_transformed_ref and y_val_ref from FE stage
    x_val_ref = state.get("fe_X_val_transformed_ref", "X_val_fe.pkl")
    y_val_ref = state.get("fe_y_val_ref", "y_val_fe.pkl")
    # And potentially X_test_transformed_ref for final test predictions (if y_test exists or for submission)
    x_test_ref = state.get("fe_X_test_transformed_ref", "X_test_fe.pkl")


    print(f"  Evaluating trained pipeline: {trained_pipeline_ref}")
    print(f"  Using X_val_ref: {x_val_ref}, y_val_ref: {y_val_ref}")
    print(f"  (And potentially X_test_ref: {x_test_ref} for final predictions)")
    # Logic: Instruct tool to load trained_pipeline_ref, load X_val_ref, y_val_ref.
    # Then instruct tool to make predictions and calculate metrics.
    return {"current_stage_completed": "Evaluation", "evaluation_metrics": {"Accuracy": 0.90, "F1_Score": 0.88}}

# --- 6. Construct and Compile the LangGraph ---
workflow = StateGraph(MultiAgentPipelineState)
workflow.add_node("eda_agent", eda_agent_node)
workflow.add_node("feature_engineering_agent", feature_engineering_agent_node)
workflow.add_node("modeling_agent", modeling_agent_node) # Renamed for clarity
workflow.add_node("evaluation", evaluation_node)

workflow.set_entry_point("eda_agent")
workflow.add_edge("eda_agent", "feature_engineering_agent")
workflow.add_edge("feature_engineering_agent", "modeling_agent")
workflow.add_edge("modeling_agent", "evaluation")
workflow.add_edge("evaluation", END)

pipeline_app = workflow.compile()

# --- 7. Example Invocation ---
if __name__ == "__main__":
    print("Starting ML Pipeline with Sklearn Pipeline Integration...")

    os.makedirs("dummy_pipeline_data", exist_ok=True)
    initial_data_paths = {
        "train": "dummy_pipeline_data/train_data.csv",
        "val": "dummy_pipeline_data/val_data.csv",
        "test": "dummy_pipeline_data/test_data.csv"
    }
    dummy_header = "Date,Price,Volume,FeatureA,FeatureB,Category,Target\n" # Added Category for encoder
    dummy_row_template = "2023-01-{day:02d},{price},{volume},{fA},{fB},{cat},{target}\n"
    for k, v_path in initial_data_paths.items():
        with open(v_path, "w") as f:
            f.write(dummy_header)
            for i in range(3): 
                 f.write(dummy_row_template.format(
                    day=i+1, price=100+i*10, volume=10000+i*1000, 
                    fA=0.5+i*0.1, fB=1.2-i*0.05, 
                    cat='TypeA' if i%2==0 else 'TypeB', 
                    target=101+i*5
                ))

    initial_pipeline_state = {
        "data_paths": initial_data_paths,
        "target_column_name": "Target",
        "max_react_iterations": 6 # Iterations per agent node
    }

    config = {"configurable": {"thread_id": "ml_pipeline_sklearn_v001"}}

    print("\nInvoking pipeline stream:")
    final_state_accumulator = {} 

    for event_key, event_value in pipeline_app.stream(initial_pipeline_state, config=config, stream_mode="updates"):
        print(f"\n<<< Update from Node: {event_key} >>>")
        if isinstance(event_value, dict):
            final_state_accumulator.update(event_value) 
            for k_item, v_item in event_value.items():
                print(f"  {k_item}: {str(v_item)[:350]}...")
        else:
            print(f"  Raw event value: {str(event_value)[:350]}...")

    print("\n\n--- Final Pipeline State (from accumulated stream) ---")
    if final_state_accumulator:
        print(json.dumps(final_state_accumulator, indent=2, default=str))
    
    # Clean up dummy files
    for v_path in initial_data_paths.values():
        if os.path.exists(v_path): os.remove(v_path)
    if os.path.exists("dummy_pipeline_data"): os.rmdir("dummy_pipeline_data")

    print("\nMulti-Agent Pipeline Finished.")
