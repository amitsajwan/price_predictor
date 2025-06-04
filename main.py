import os
import re
import json 
from typing import TypedDict, Annotated, List, Dict, Optional, Sequence, Literal
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI 
from langgraph.graph import StateGraph, END

# --- 1. Define the State for the Pipeline (Relevant Parts) ---
class MultiAgentPipelineState(TypedDict):
    # Input
    data_paths: Dict[str, str] 
    target_column_name: Optional[str] 
    problem_type: Optional[Literal["classification", "regression"]] 

    # From EDA
    eda_report: Optional[Dict[str, any]] 
    eda_model_suggestions: Optional[List[str]] 

    # From Feature Engineering
    fe_X_train_ref: Optional[str] 
    fe_y_train_ref: Optional[str]
    fe_untrained_full_pipeline_ref: Optional[str] # This is key for modeling
    fe_selected_model_type: Optional[str] 
    fe_initial_hyperparameter_hints: Optional[Dict[str, any]] 

    # Output from ModelingNode
    model_trained_pipeline_ref: Optional[str] 
    model_training_summary: Optional[str]

    # ... other state fields ...
    current_stage_completed: Optional[str]
    max_react_iterations: Optional[int]


# --- 2. Interface for your Agnostic PythonTool (Simulation - Relevant Part) ---
# REPLACE THIS FUNCTION WITH THE ACTUAL CALL TO YOUR AGNO_PYTHON_TOOL
def agno_python_tool_interface(instruction: str, agent_context_hint: Optional[str] = None) -> str:
    print(f"    [AGNO_PYTHON_TOOL INTERFACE] Sending Instruction to your tool:\n    '{instruction}'")
    if agent_context_hint:
        print(f"    Agent Context Hint (passed to your tool): {agent_context_hint}")
    
    sim_observation = f"Observation: PythonTool processed instruction: '{instruction}'. "
    instruction_lower = instruction.lower()

    # ... (other simulation cases for EDA, FE) ...

    # Modeling Related Simulations
    if "load the untrained pipeline" in instruction_lower and "train it using x_train" in instruction_lower and "y_train" in instruction_lower:
        # Extract refs for more specific simulation
        untrained_pipe_ref_match = re.search(r"untrained pipeline from '([^']+)'", instruction_lower)
        x_train_ref_match = re.search(r"x_train from '([^']+)'", instruction_lower)
        y_train_ref_match = re.search(r"y_train from '([^']+)'", instruction_lower)
        
        untrained_pipe_ref = untrained_pipe_ref_match.group(1) if untrained_pipe_ref_match else "unknown_untrained_pipe.joblib"
        x_train_ref = x_train_ref_match.group(1) if x_train_ref_match else "unknown_x_train.pkl"
        y_train_ref = y_train_ref_match.group(1) if y_train_ref_match else "unknown_y_train.pkl"

        sim_observation += (f"Untrained pipeline '{untrained_pipe_ref}' loaded. "
                           f"X_train from '{x_train_ref}' and y_train from '{y_train_ref}' loaded. "
                           f"Pipeline trained successfully. ")
        if "save the trained pipeline as a .joblib file and report its reference" in instruction_lower:
            sim_observation += "Trained pipeline saved by tool. Reference is 'trained_model_pipeline_final.joblib'."
        else:
            sim_observation += "Trained pipeline is ready but not explicitly saved in this step."
    elif "load trained pipeline" in instruction_lower and "make predictions" in instruction_lower: # For Evaluation
        sim_observation += "Trained pipeline and relevant data loaded. Predictions made."
        if "calculate" in instruction_lower and "metrics" in instruction_lower:
            if "classification" in agent_context_hint.lower() if agent_context_hint else "classification" in instruction_lower:
                 sim_observation += " Metrics calculated by tool and reported as: {{'accuracy': 0.90, 'f1_score': 0.88}}."
            else: # regression
                 sim_observation += " Metrics calculated by tool and reported as: {{'mse': 12.3, 'r_squared': 0.81}}."

    else:
        sim_observation += "Task completed. If specific artifacts were requested to be saved and their references reported, those details are included above."
            
    print(f"    [AGNO_PYTHON_TOOL INTERFACE] Returning Observation:\n    '{sim_observation}'")
    return sim_observation

# --- 3. Generic ReAct Loop Engine (Assumed to be defined as before) ---
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

# --- 4. Helper to Parse LLM's JSON Final Answer (Assumed to be defined as before) ---
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

# --- 5. Modeling Agent Node (Refined) ---
def modeling_agent_node(state: MultiAgentPipelineState) -> Dict[str, any]:
    print("\n--- Modeling Agent Node Running (EDA-Informed Training) ---")
    
    # Inputs from previous stages
    untrained_pipeline_ref = state.get("fe_untrained_full_pipeline_ref", "default_untrained_pipe.joblib")
    x_train_ref = state.get("fe_X_train_ref", "default_X_train_fe.pkl") 
    y_train_ref = state.get("fe_y_train_ref", "default_y_train_fe.pkl")
    
    # Information from FE about the model chosen based on EDA
    selected_model_type_by_fe = state.get("fe_selected_model_type", "DefaultModel (e.g., RandomForest)")
    initial_hyperparams_by_fe = state.get("fe_initial_hyperparameter_hints", {})
    
    # EDA's original model suggestions (for context, though FE already acted on them)
    eda_model_suggestions = state.get("eda_model_suggestions", [])


    model_tool_context_hint = (
        f"Untrained Scikit-learn pipeline reference: '{untrained_pipeline_ref}' (this pipeline was constructed by FE and includes a '{selected_model_type_by_fe}' estimator based on EDA insights). "
        f"X_train data reference for training: '{x_train_ref}'. "
        f"y_train data reference for training: '{y_train_ref}'. "
        f"Initial hyperparameter hints from FE (if any): {json.dumps(initial_hyperparams_by_fe)}. "
        f"Original EDA model suggestions (for broader context): {json.dumps(eda_model_suggestions)}."
    )

    prompt_content = f"""You are a Modeling Specialist. Your PythonTool takes Natural Language instructions.
    Context from Feature Engineering (which considered EDA suggestions):
    - Untrained Full Scikit-learn Pipeline Reference: {untrained_pipeline_ref} 
      (This pipeline includes preprocessing steps and an UNTRAINED '{selected_model_type_by_fe}' estimator.)
    - X_train Reference (for training this pipeline): {x_train_ref}
    - y_train Reference (for training this pipeline): {y_train_ref}
    - Initial Hyperparameter Hints from FE (if any): {json.dumps(initial_hyperparams_by_fe) if initial_hyperparams_by_fe else 'Use estimator defaults.'}
    - Broader EDA Model Suggestions (for your awareness): {json.dumps(eda_model_suggestions)}

    Your Primary Task:
    1. Instruct PythonTool to load the specific UNTRAINED pipeline from '{untrained_pipeline_ref}'.
    2. Instruct PythonTool to load X_train from '{x_train_ref}' and y_train from '{y_train_ref}'.
    3. Instruct PythonTool to train (fit) this loaded pipeline using X_train and y_train. 
       If specific initial hyperparameter hints were provided ({json.dumps(initial_hyperparams_by_fe)}), instruct the tool to consider them if it's capable of setting parameters on the untrained estimator within the pipeline before fitting. Otherwise, default parameters will be used.
    4. Instruct PythonTool to save the ENTIRE TRAINED PIPELINE as a .joblib file and report its exact reference (full filename including extension).

    ReAct Format:
    Thought: (Optional) Your reasoning.
    Action: Python
    Action Input: <Natural language instruction for PythonTool>
    (System will provide Observation:)
    Observation: <result from PythonTool>

    "Final Answer:" MUST be a single well-formed JSON object string, enclosed in ```json ... ```.
    The JSON object MUST have these top-level keys:
    "model_training_summary": (string) Summary of the training process, confirming the model type trained (should be '{selected_model_type_by_fe}') and any specific parameters used or observations from training.
    "trained_pipeline_ref": (string) "<tool_reported_TRAINED_pipeline_ref.joblib>"
    
    Begin. Focus on training the pipeline referenced by '{untrained_pipeline_ref}'.
    """
    final_answer_json_string = run_generic_react_loop(
        prompt_content,
        state.get("max_react_iterations", 5), # Training is usually a few direct steps
        model_tool_context_hint
    )
    
    parsed_data = parse_llm_json_final_answer(final_answer_json_string, "Modeling report generation failed.")

    return {
        "current_stage_completed": "Modeling",
        "model_training_summary": parsed_data.get("model_training_summary", "Training summary not parsed."),
        "model_trained_pipeline_ref": parsed_data.get("trained_pipeline_ref", "default_trained_pipeline.joblib") 
    }

# --- Other Agent Nodes (EDA, FE, Evaluation) and Workflow Setup would be here ---
# For brevity, I'm only showing the modified modeling_agent_node and necessary context.
# The full pipeline code would include all nodes and the graph definition.

# Example of how it would fit into the full pipeline (assuming other nodes are defined)
if __name__ == "__main__":
    # This is a conceptual execution. You'd need the full pipeline code.
    print("Demonstrating the refined modeling_agent_node concept.")

    # Dummy state that would be built by previous EDA and FE nodes
    # In a real run, these values come from the actual execution of eda_agent_node and feature_engineering_agent_node
    dummy_state_after_fe = MultiAgentPipelineState(
        data_paths={"train": "dummy/train.csv", "val": "dummy/val.csv", "test": "dummy/test.csv"},
        target_column_name="Target",
        problem_type="regression",
        eda_report={ # Simplified EDA report
            "comprehensive_summary": "EDA found linear trends and some skewness.",
            "model_suggestions": ["Consider LinearRegression due to trends.", "Try RandomForestRegressor for non-linearities."],
            "fe_suggestions": ["Log transform 'Price'.", "Create day-of-week features."],
            "artifact_references": {
                "processed_train_data": "eda_cleaned_train.pkl",
                "processed_val_data": "eda_cleaned_val.pkl",
                "processed_test_data": "eda_cleaned_test.pkl",
            }
        },
        eda_model_suggestions=["Consider LinearRegression due to trends.", "Try RandomForestRegressor for non-linearities."], # Explicitly in state
        fe_X_train_ref="fe_X_train.pkl",
        fe_y_train_ref="fe_y_train.pkl",
        fe_untrained_full_pipeline_ref="fe_untrained_pipeline_rfr.joblib", # FE decided on RFR
        fe_selected_model_type="RandomForestRegressor",
        fe_initial_hyperparameter_hints={"n_estimators": 150, "max_depth": 10},
        max_react_iterations=5
        # ... other fields would be None or populated
    )

    print(f"\n--- Calling modeling_agent_node with dummy state from FE ---")
    modeling_output = modeling_agent_node(dummy_state_after_fe)
    
    print("\n--- Output from modeling_agent_node ---")
    print(json.dumps(modeling_output, indent=2))

    # Expected output would show a model_training_summary and a model_trained_pipeline_ref
    # based on the simulated agno_python_tool_interface responses to training instructions.
```

**Key Changes in this `modeling_agent_node`:**

1.  **Input from State:** It now explicitly pulls:
    * `fe_untrained_full_pipeline_ref`: The reference to the Scikit-learn pipeline object (preprocessors + untrained estimator) that the FE agent created.
    * `fe_selected_model_type`: The type of model estimator that the FE agent decided to include in that pipeline (this decision by FE was guided by `eda_model_suggestions`).
    * `fe_initial_hyperparameter_hints`: Any basic hyperparameter ideas from FE.
    * `eda_model_suggestions`: The original list of model suggestions from EDA is passed into the prompt for broader context, even if FE has already made a primary selection.
2.  **Prompt Refinement:**
    * The system prompt for the `modeling_agent_node` now clearly states that it should work with the specific `untrained_full_pipeline_ref` provided by the FE stage.
    * It instructs the LLM to consider the `initial_hyperparameter_hints` if the `agno_python_tool` can handle setting parameters on the untrained estimator within the pipeline before fitting.
    * The primary goal is to train *this specific pipeline* that FE has already thoughtfully constructed based on EDA.
3.  **Focus on Execution:** The "intelligence" of model *selection* has largely been pushed to the FE agent (which uses EDA's suggestions). The Modeling agent is now more focused on the robust *execution* of training for that chosen pipeline structure.
4.  **"Trying few things" (Implicitly Handled by FE):**
    * If the EDA suggested multiple good model candidates (e.g., "try Linear Regression and also a RandomForest"), the FE agent's prompt could be designed to make a choice or even (in a more complex FE agent) instruct the tool to prepare *two different* untrained pipeline references, one for each model type.
    * The Modeling agent could then be called twice (or have a loop) to train each, but for "keeping it simple" in this iteration, the FE agent makes one primary choice for the `untrained_full_pipeline_ref`.
    * The current `modeling_agent_node` prompt doesn't ask the LLM to try alternative models *from scratch* within its own ReAct loop if FE already provided a specific pipeline. It focuses on the pipeline given.

**To make it even more "intelligent" regarding trying a few things in the Modeling Node (more complex):**

If you wanted the `modeling_agent_node` itself to try, say, two different models based on EDA suggestions, you would:
1.  Pass the `eda_model_suggestions` more directly to the Modeling agent's prompt.
2.  The prompt would instruct the LLM: "Based on these suggestions (e.g., `['LinearRegression', 'RandomForestRegressor']`), instruct the PythonTool to:
    a.  Create and train a pipeline with LinearRegression (using preprocessors from FE if available separately, or by building a new preproc section). Save and report its reference and metrics.
    b.  Create and train a pipeline with RandomForestRegressor. Save and report its reference and metrics.
    c.  Compare the (simulated or tool-reported) metrics.
    d.  In your Final Answer, report the reference of the *best performing* trained pipeline as `trained_pipeline_ref` and summarize why."
This would make the `modeling_agent_node`'s ReAct loop longer and require your `agno_python_tool` to be very robust in handling these sequential model building/training/evaluation tasks and reporting distinct references.

For now, the provided code makes the FE agent the primary decision-maker for the initial model architecture in the pipeline, informed by E
