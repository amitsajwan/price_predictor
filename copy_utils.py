import os
import re
import json 
import random 
import logging 
from typing import TypedDict, Annotated, List, Dict, Optional, Sequence, Literal
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI 

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')
logger = logging.getLogger(__name__)

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
    pipeline_status_message: Optional[str] 
    current_stage_completed: Optional[str] 
    max_react_iterations: Optional[int]


# --- 2. Interface for your Agnostic PythonTool ---
# REPLACE THIS FUNCTION WITH THE ACTUAL CALL TO YOUR AGNO_PYTHON_TOOL
SIMULATED_MODEL_PERFORMANCE_REGISTRY = {} 

def agno_python_tool_interface(instruction: str, agent_context_hint: Optional[str] = None) -> str:
    global SIMULATED_MODEL_PERFORMANCE_REGISTRY
    logger.info(f"[AGNO_PYTHON_TOOL INTERFACE] Sending Instruction to your tool:\n    '{instruction}'")
    if agent_context_hint:
        logger.info(f"    Agent Context Hint (passed to your tool): {agent_context_hint}")
    
    sim_observation = f"Observation: PythonTool processed instruction: '{instruction}'. "
    instruction_lower = instruction.lower()

    if "load the dataset from" in instruction_lower and "report its reference as" in instruction_lower:
        ref_as_match = re.search(r"report its reference as '([^']+)'", instruction_lower)
        ref_name = ref_as_match.group(1) if ref_as_match else "unknown_initial_load.pkl"
        sim_observation += f"Dataset loaded. Tool reports its reference as '{ref_name}'."
    elif "identify the 'date' column in" in instruction_lower and "parse it as datetime" in instruction_lower and "report new reference as" in instruction_lower:
        ref_in_match = re.search(r"in '([^']+)'", instruction_lower)
        ref_out_match = re.search(r"new reference as '([^']+)'", instruction_lower)
        data_ref_in = ref_in_match.group(1) if ref_in_match else "unknown_ref.pkl"
        data_ref_out = ref_out_match.group(1) if ref_out_match else f"parsed_date_{data_ref_in}"
        sim_observation += f"'Date' column in '{data_ref_in}' identified and parsed to datetime64[ns]. New data reference is '{data_ref_out}'. Confirmed."
    elif "create a temporary numeric-only version of" in instruction_lower and "excluding the original 'date' column" in instruction_lower and "report the new reference as" in instruction_lower:
        ref_in_match = re.search(r"version of '([^']+)'", instruction_lower)
        ref_out_match = re.search(r"new reference as '([^']+)'", instruction_lower)
        data_ref_in = ref_in_match.group(1) if ref_in_match else "unknown_ref.pkl"
        data_ref_out = ref_out_match.group(1) if ref_out_match else f"numeric_version_of_{data_ref_in}"
        sim_observation += f"Numeric-only version of '{data_ref_in}' created (original 'Date' column excluded). New reference reported by tool: '{data_ref_out}'."
    elif "extract year, month, day, dayofweek from the 'date' column in" in instruction_lower and "drop the original 'date' column" in instruction_lower and "report new data references for train as" in instruction_lower:
        train_ref_match = re.search(r"train as '([^']+)'", instruction_lower)
        val_ref_match = re.search(r"val as '([^']+)'", instruction_lower)
        test_ref_match = re.search(r"test as '([^']+)'", instruction_lower)
        train_ref = train_ref_match.group(1) if train_ref_match else "train_with_date_features.pkl"
        val_ref = val_ref_match.group(1) if val_ref_match else "val_with_date_features.pkl"
        test_ref = test_ref_match.group(1) if test_ref_match else "test_with_date_features.pkl"
        sim_observation += ("Date features (Year, Month, Day, DayOfWeek) extracted. Original 'Date' column dropped. "
                           f"New data references reported: train='{train_ref}', val='{val_ref}', test='{test_ref}'.")
    elif "perform final cleaning on" in instruction_lower and "save the resulting dataset and report the new reference as" in instruction_lower:
        ref_out_match = re.search(r"new reference as '([^']+)'", instruction_lower)
        data_ref_out = ref_out_match.group(1) if ref_out_match else "cleaned_final_data.pkl"
        sim_observation += f"Data cleaned. New reference reported: '{data_ref_out}'."
    elif "generate a histogram for" in instruction_lower and "save it as" in instruction_lower and "report the filename, and provide a textual description" in instruction_lower:
        filename_match = re.search(r"save it as '([^']+)'", instruction_lower)
        filename = filename_match.group(1) if filename_match else "sim_histogram.png"
        sim_observation += f"Histogram generated. Plot saved by tool as '{filename}'. Description: [Simulated detailed description of plot features]."
    elif "fit a standardscaler" in instruction_lower and "save it as a .joblib file and report its reference as" in instruction_lower:
        ref_match = re.search(r"report its reference as '([^']+)'", instruction_lower)
        ref_name = ref_match.group(1) if ref_match else "fitted_scaler.joblib"
        sim_observation += f"StandardScaler fitted. Saved as '{ref_name}'. This is its reference."
    elif "create an untrained scikit-learn pipeline" in instruction_lower and "save it as a .joblib file and report its reference as" in instruction_lower:
        ref_match = re.search(r"report its reference as '([^']+)'", instruction_lower)
        ref_name = ref_match.group(1) if ref_match else "untrained_pipeline.joblib"
        sim_observation += f"Untrained Scikit-learn pipeline created. Saved. Reference is '{ref_name}'."
    elif "load the untrained pipeline" in instruction_lower and "train it using x_train" in instruction_lower and "save the trained pipeline as a .joblib file and report its reference as" in instruction_lower:
        ref_in_match = re.search(r"untrained pipeline '([^']+)'", instruction_lower)
        ref_out_match = re.search(r"report its reference as '([^']+)'", instruction_lower)
        untrained_pipe_ref = ref_in_match.group(1) if ref_in_match else "sim_untrained.joblib"
        trained_model_ref = ref_out_match.group(1) if ref_out_match else f"trained_{untrained_pipe_ref}"
        
        params_key = "default"; model_type_sim = "DefaultModel"
        if "randomforest" in untrained_pipe_ref: model_type_sim="RandomForest" 
        if "n_estimators=100" in instruction_lower : params_key="n100_d10" 
            
        full_model_key = f"{model_type_sim}_{params_key}"
        if full_model_key not in SIMULATED_MODEL_PERFORMANCE_REGISTRY: SIMULATED_MODEL_PERFORMANCE_REGISTRY[full_model_key] = random.uniform(0.6, 2.5) 
        else: SIMULATED_MODEL_PERFORMANCE_REGISTRY[full_model_key] *= random.uniform(0.85, 0.99) 
        current_sim_rmse = SIMULATED_MODEL_PERFORMANCE_REGISTRY[full_model_key]

        sim_observation += (f"Pipeline '{untrained_pipe_ref}' trained. Trained pipeline saved. Reference is '{trained_model_ref}'.")
        if "report rmse" in instruction_lower: sim_observation += f" Validation RMSE: {current_sim_rmse:.4f}."
    elif "separate target" in instruction_lower and "save them as .pkl files and report new references for x_train as" in instruction_lower:
        sim_observation += "Target separated. Tool reports new references: X_train='X_train_final.pkl', y_train='y_train_final.pkl', X_val='X_val_final.pkl', y_val='y_val_final.pkl', X_test='X_test_final.pkl'."
    elif "calculate regression metrics" in instruction_lower: 
        sim_observation += "Metrics reported by tool: {{'rmse': {random.uniform(0.1,0.8):.4f}, 'r_squared': {random.uniform(0.6,0.9):.2f}}}."
    else:
        sim_observation += "Task completed. If specific artifacts were requested to be saved and their references reported, those details are included above."
            
    logger.info(f"[AGNO_PYTHON_TOOL INTERFACE] Returning Observation:\n    '{sim_observation}'")
    return sim_observation

# --- 3. Generic ReAct Loop Engine ---
llm = ChatOpenAI(model="gpt-4o", temperature=0.1) # Or your chosen LLM

def run_generic_react_loop(
    initial_prompt_content: str,
    max_iterations: int,
    agent_context_hint_for_tool: Optional[str] = None 
) -> str: 
    react_messages: List[BaseMessage] = [SystemMessage(content=initial_prompt_content)]
    final_answer_text = json.dumps({"error": "Agent did not produce a Final Answer within iteration limit."}) 
    
    for i in range(max_iterations):
        logger.info(f"  [GenericReActLoop] Iteration {i+1}/{max_iterations}")
        ai_response = llm.invoke(react_messages)
        ai_content = ai_response.content.strip()
        react_messages.append(ai_response) 
        logger.info(f"    LLM: {ai_content[:450]}...")

        final_answer_match = re.search(r"Final Answer:\s*(```json\s*(.*?)\s*```|{\s*.*})", ai_content, re.DOTALL | re.IGNORECASE)
        action_match = re.search(r"Action:\s*Python\s*Action Input:\s*(.+)", ai_content, re.DOTALL | re.IGNORECASE) 

        if final_answer_match:
            json_block_content = final_answer_match.group(2) if final_answer_match.group(2) else final_answer_match.group(1)
            final_answer_text = json_block_content.strip()
            logger.info(f"    Loop Concluded. Final Answer (JSON string) obtained:\n{final_answer_text}")
            break 
        elif action_match:
            nl_instruction_for_tool = action_match.group(1).strip()
            tool_observation = agno_python_tool_interface(nl_instruction_for_tool, agent_context_hint_for_tool)
            react_messages.append(HumanMessage(content=f"Observation: {tool_observation}"))
        else:
            logger.warning("    LLM did not provide Action or Final Answer in expected format.")
            react_messages.append(HumanMessage(content="System hint: Your response was not in the expected format. Please use 'Action: Python\\nAction Input: <NL_instruction>' or 'Final Answer: ```json\\n{...}\\n```'. Ensure JSON is the ONLY content in Final Answer block and all strings within it are properly escaped."))
            if i > 1: # Give it a couple of chances
                final_answer_text = json.dumps({"error": "Agent failed to follow output format consistently."})
                logger.error(f"    Agent failed to follow output format after multiple hints.")
                break 
        
        if i == max_iterations - 1: 
            logger.warning(f"    Max ReAct iterations ({max_iterations}) reached.")
            if ai_content and not final_answer_match: # If it was still thinking/acting
                final_answer_text = json.dumps({"error": f"Max iterations reached. Last AI thought/action: {ai_content}"})
            # If final_answer_text is still the default error, it means no valid Final Answer was ever produced.
    return final_answer_text

# --- 4. Helper to Parse LLM's JSON Final Answer ---
def parse_llm_json_final_answer(final_answer_json_string: str, node_name_for_logging: str) -> Dict:
    default_error_message = f"{node_name_for_logging} JSON report generation failed or JSON was invalid."
    try:
        # Attempt to extract JSON from markdown code block first
        match = re.search(r"```json\s*(.*?)\s*```", final_answer_json_string, re.DOTALL)
        if match:
            json_string_cleaned = match.group(1).strip()
        else: # Assume it's a raw JSON string or needs cleaning for {}
            json_string_cleaned = final_answer_json_string.strip()
            # Basic cleanup for raw JSON that might still be wrapped in ```
            if json_string_cleaned.startswith("```") and json_string_cleaned.endswith("```"):
                 json_string_cleaned = json_string_cleaned[3:-3].strip()
        
        # Remove trailing commas before closing curly or square brackets more robustly
        json_string_cleaned = re.sub(r",\s*([}\]])", r"\1", json_string_cleaned) 
        # Remove // comments (if any slipped through, though LLM is asked not to use them IN strings)
        json_string_cleaned = re.sub(r"//.*?\n", "\n", json_string_cleaned) 
        # Remove # comments at the beginning of lines if they are not part of a string
        json_string_cleaned = re.sub(r"^\s*#.*?\n", "\n", json_string_cleaned, flags=re.MULTILINE)

        if not json_string_cleaned: 
            logger.error(f"    ERROR in {node_name_for_logging}: JSON string became empty after cleaning. Original: '{final_answer_json_string}'")
            return {"error": "JSON string empty after cleaning", "summary_error": default_error_message, "original_string": final_answer_json_string}

        parsed_data = json.loads(json_string_cleaned)
        logger.info(f"    Successfully parsed JSON for {node_name_for_logging}.")
        return parsed_data
    except json.JSONDecodeError as e:
        logger.error(f"    ERROR in {node_name_for_logging}: Failed to decode JSON: {e}. Problematic string was: '{final_answer_json_string}'")
        return {"error": f"JSON Decode Error in {node_name_for_logging}: {e}", "summary_error": default_error_message, "original_string": final_answer_json_string} 
    except Exception as e: 
        logger.error(f"    ERROR in {node_name_for_logging}: Unexpected error parsing JSON: {e}. String: '{final_answer_json_string}'")
        return {"error": f"Unexpected Parsing Error in {node_name_for_logging}: {e}", "summary_error": default_error_message, "original_string": final_answer_json_string}
