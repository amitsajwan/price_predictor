import os
import json
import shutil # For cleaning up dummy directories
from typing import TypedDict, Dict, Optional, List
from langgraph.graph import StateGraph, END
import pandas as pd # For creating dummy CSVs
import joblib 
import asyncio # For async operations
import aiohttp # For making asynchronous HTTP requests

# --- 0. Configuration & Utility ---
BASE_OUTPUT_DIR = "langgraph_pipeline_outputs"
REPORTS_DIR = os.path.join(BASE_OUTPUT_DIR, "reports")
SCRIPTS_DIR = os.path.join(BASE_OUTPUT_DIR, "scripts")
DATA_DIR_RAW = os.path.join(BASE_OUTPUT_DIR, "data", "raw_cleaned")
DATA_DIR_FE = os.path.join(BASE_OUTPUT_DIR, "data", "feature_engineered")
MODELS_DIR = os.path.join(BASE_OUTPUT_DIR, "models")

GEMINI_API_KEY = "" # Canvas will provide this if empty for gemini-2.0-flash. 
                    # For local runs, set your actual API key here or use environment variables.
GEMINI_API_URL_BASE = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"


def create_dummy_csv(file_path: str, is_target_present: bool = True):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    if "train" in file_path:
        data = {'Date': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03']),
                'Open': [10, 11, 12], 'High': [10.5, 11.5, 12.5], 'Low': [9.5, 10.5, 11.5],
                'Close': [10.2, 11.2, 12.2], 'Volume': [1000, 1100, 1200]}
        if is_target_present: data['Target_Return'] = [0.01, -0.005, 0.02]
    elif "val" in file_path:
        data = {'Date': pd.to_datetime(['2023-01-04', '2023-01-05']),
                'Open': [12.3, 13], 'High': [12.8, 13.5], 'Low': [12, 12.5],
                'Close': [12.5, 13.2], 'Volume': [1300, 1400]}
        if is_target_present: data['Target_Return'] = [0.005, 0.015]
    else: # test
        data = {'Date': pd.to_datetime(['2023-01-06', '2023-01-07']),
                'Open': [13.3, 14], 'High': [13.8, 14.5], 'Low': [13, 13.5],
                'Close': [13.5, 14.2], 'Volume': [1500, 1600]}
        if is_target_present: data['Target_Return'] = [-0.01, 0.003] 
    df = pd.DataFrame(data)
    df.set_index('Date', inplace=True)
    df.to_csv(file_path)
    print(f"Created dummy CSV: {file_path}")

def setup_directories():
    if os.path.exists(BASE_OUTPUT_DIR):
        shutil.rmtree(BASE_OUTPUT_DIR) 
    os.makedirs(REPORTS_DIR, exist_ok=True)
    os.makedirs(SCRIPTS_DIR, exist_ok=True)
    os.makedirs(DATA_DIR_RAW, exist_ok=True)
    os.makedirs(DATA_DIR_FE, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    print(f"Created base directory structure at: {os.path.abspath(BASE_OUTPUT_DIR)}")

# --- 1. Define the Workflow State ---
class WorkflowState(TypedDict):
    initial_goal: str
    input_train_path: str
    input_val_path: Optional[str]
    input_test_path: str
    config_params: Optional[Dict] 
    preliminary_analysis_report: Optional[str]
    pa_report_path: Optional[str] 
    prompt_for_eda_tool: Optional[str] 
    eda_script_path: Optional[str] 
    eda_report: Optional[str] 
    eda_report_path: Optional[str] 
    prompt_for_fe_tool: Optional[str] 
    fe_script_path: Optional[str] 
    feature_engineering_report: Optional[str]
    fe_report_path: Optional[str]
    output_fe_train_X_path: Optional[str]
    output_fe_train_y_path: Optional[str]
    output_fe_val_X_path: Optional[str]
    output_fe_val_y_path: Optional[str]
    output_fe_test_X_path: Optional[str]
    output_fe_test_y_path: Optional[str]
    fe_transformers_path: Optional[str]
    prompt_for_modeling_tool: Optional[str]
    modeling_script_path: Optional[str]
    model_evaluation_report: Optional[str]
    modeling_report_path: Optional[str]
    final_pipeline_path: Optional[str] 
    current_working_directory: str 
    error_message: Optional[str]

# --- 2. Define REAL LLM Call Functions using Gemini API with aiohttp ---
async def gemini_api_call(prompt: str, stage_name_for_logging: str) -> str:
    """Generic function to call Gemini API for text generation using aiohttp."""
    print(f"\n REAL GEMINI API CALL ({stage_name_for_logging}) ".center(80, "-"))
    print(f"Prompt to Gemini (first 300 chars): {prompt[:300]}...")

    # Check for API key if not in an environment that injects it (like Canvas)
    # This is a simple check; a more robust solution would use environment variables.
    api_key_to_use = "AIzaSyD2PRQVMeGI5aYsTvO80qJEl-joX-6rW5s"
    if not api_key_to_use and "google.colab" not in os.environ.get("SESSION_MANAGER", "") and "VSCODE_PID" not in os.environ: # Basic check for non-Canvas local env
        print("WARNING: GEMINI_API_KEY is empty. API calls will likely fail if not in a Canvas environment.")
        # You might want to raise an error or return a specific message here if running locally without a key.
        # For now, we'll let it proceed so Canvas can inject the key if it's running there.
    
    api_url = f"{GEMINI_API_URL_BASE}?key={api_key_to_use}"

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(api_url, json=payload) as response:
                if response.status != 200:
                    error_body = await response.text()
                    print(f"Error: API call failed for {stage_name_for_logging} with status {response.status}: {error_body}")
                    return f"Error: API call failed for {stage_name_for_logging} with status {response.status}. Body: {error_body[:200]}"
                
                result = await response.json()

        if result.get("candidates") and \
           result["candidates"][0].get("content") and \
           result["candidates"][0]["content"].get("parts") and \
           result["candidates"][0]["content"]["parts"][0].get("text"):
            generated_text = result["candidates"][0]["content"]["parts"][0]["text"]
            print(f"Gemini Response (first 100 chars): {generated_text[:100]}...")
            print("-" * 80)
            return generated_text
        else:
            print(f"Error: Unexpected response structure from Gemini API for {stage_name_for_logging}.")
            print(f"Full API Response: {json.dumps(result, indent=2)}")
            if result.get("promptFeedback"):
                feedback = result["promptFeedback"]
                print(f"Prompt Feedback: {feedback}")
                block_reason = feedback.get("blockReason")
                if block_reason:
                    return f"Error: Prompt blocked by API for {stage_name_for_logging}. Reason: {block_reason}. Safety Ratings: {feedback.get('safetyRatings')}"
            return f"Error: Could not parse Gemini response for {stage_name_for_logging}. Check logs for details."

    except aiohttp.ClientError as e: 
        print(f"AIOHTTP ClientError during Gemini API call for {stage_name_for_logging}: {e}")
        return f"Error: Network issue during API call for {stage_name_for_logging} - {str(e)}"
    except Exception as e:
        print(f"Generic error caught in gemini_api_call for {stage_name_for_logging}:")
        print(f"Exception Type: {type(e)}")
        print(f"Exception Details: {repr(e)}")
        # Removed the specific 'fetch' NameError check as aiohttp is used.
        # If a NameError still occurs, the prints above will show it.
        return f"Error: API call failed for {stage_name_for_logging} due to an unhandled exception - {type(e).__name__}: {str(e)}"


async def gemini_generate_report_content(instruction: str, context: str, stage_name: str) -> str:
    """Generates report content using Gemini."""
    prompt = f"Instruction: {instruction}\n\nContext:\n{context}\n\nBased on the above, generate a concise and informative report for the '{stage_name}' stage. Focus on key findings, observations, and actionable insights. Behave as an expert data analyst."
    return await gemini_api_call(prompt, f"{stage_name} Report Generation")

async def gemini_generate_execution_prompt(meta_prompt_for_creator_llm: str, next_agent_name: str) -> str:
    """
    Uses Gemini as the "Prompt Creator LLM" to generate an "Execution Prompt"
    for the pythonTool.
    """
    full_instruction_for_gemini = f"{meta_prompt_for_creator_llm}\n\n" \
                                  f"Now, generate ONLY the 'Execution Prompt' string (starting with `f\"\"\"` or `\"\"\"` and ending with `\"\"\"`) " \
                                  f"that will be used by a pythonTool to create the {next_agent_name} script. " \
                                  f"Ensure all paths and configurations from Section 1 of the above instructions are embedded as literal strings in the Execution Prompt you generate."
    return await gemini_api_call(full_instruction_for_gemini, f"Execution Prompt for {next_agent_name} Tool")


def python_tool_simulation(
    execution_prompt: str, 
    script_save_path: str,
    report_save_path: str,
    output_data_paths: Optional[Dict[str, str]] = None, 
    output_model_path: Optional[str] = None,
    stage_name: str = "UnknownStage"
) -> Dict:
    """Simulates the pythonTool by creating dummy files."""
    print(f"\n SIMULATING pythonTool Execution for {stage_name} ({os.path.basename(script_save_path)}) ".center(80, "-"))
    print(f"Received Execution Prompt (first 100 chars): {execution_prompt[:100]}...")
    
    dummy_script_content = f"# Python script: {os.path.basename(script_save_path)}\n"
    dummy_script_content += f"# Automatically generated based on LLM instructions.\n"
    dummy_script_content += f"# Execution Prompt started with: {execution_prompt[:150]}...\n\n"
    dummy_script_content += "import pandas as pd\nimport numpy as np\nimport joblib\n\n"
    dummy_script_content += f"print(f'Simulating execution of {stage_name} script: {os.path.basename(script_save_path)}')\n"
    
    if output_data_paths:
        for key, path in output_data_paths.items():
            if path: 
                dummy_script_content += f"print(f'Simulating creation of {key} at {path}')\n"
                os.makedirs(os.path.dirname(path), exist_ok=True)
                pd.DataFrame({'feature_A': [1, 2, 3], 'feature_B': [4, 5, 6]}).to_csv(path, index=False)
    
    if output_model_path:
        dummy_script_content += f"print(f'Simulating saving model/pipeline to {output_model_path}')\n"
        os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
        joblib.dump({"simulated_model_type": stage_name, "status": "trained"}, output_model_path)

    dummy_script_content += f"print(f'Simulated {stage_name} script finished. Report generated at {report_save_path}')\n"

    try:
        os.makedirs(os.path.dirname(script_save_path), exist_ok=True)
        with open(script_save_path, "w") as f:
            f.write(dummy_script_content)
        print(f"Dummy Python script for {stage_name} saved to: {script_save_path}")
    except Exception as e:
        print(f"Error saving dummy script for {stage_name}: {e}")
        return {"status": "error", "error_message": str(e)}

    report_content = f"SIMULATED Report from executing {os.path.basename(script_save_path)} for {stage_name}.\n"
    report_content += f"Execution was based on LLM-generated prompt: {execution_prompt[:80]}...\n"
    if output_data_paths:
        report_content += f"Simulated data artifacts created: {[p for p in output_data_paths.values() if p]}\n"
    if output_model_path:
        report_content += f"Simulated model/pipeline artifact created: {output_model_path}\n"
    report_content += "Key findings: Process simulated successfully."

    try:
        os.makedirs(os.path.dirname(report_save_path), exist_ok=True)
        with open(report_save_path, "w") as f:
            f.write(report_content)
        print(f"Dummy report for {stage_name} saved to: {report_save_path}")
    except Exception as e:
        print(f"Error saving dummy report for {stage_name}: {e}")
        return {"status": "error", "error_message": str(e)}

    print("-" * 80)
    return {
        "script_path": script_save_path,
        "report_path": report_save_path,
        "report_content": report_content 
    }

# --- 3. Define Agent Node Functions (now async for API calls) ---

async def preliminary_analysis_node(state: WorkflowState) -> Dict:
    print(">>> EXECUTING: Preliminary Analysis Node")
    goal = state["initial_goal"]
    config = state.get("config_params", {})
    cwd = state["current_working_directory"] 
    pa_report_path = os.path.join(cwd, "reports", "preliminary_analysis_report.txt")

    context = f"""Initial project goal: {goal}.
Available data files:
Train: {state['input_train_path']}
Validation: {state.get('input_val_path', 'N/A')}
Test: {state['input_test_path']}
Configuration: {config}
"""
    report_content = await gemini_generate_report_content(
        instruction="Analyze initial data structure and project goals. Cover data source overview (paths provided), potential data quality issues based on typical data for this goal, initial hypotheses, key objectives for the project, scope definition, and potential challenges. Also, list the provided file paths.",
        context=context,
        stage_name="PreliminaryAnalysis"
    )
    if "Error:" in report_content: 
        return {"error_message": report_content}
    try:
        os.makedirs(os.path.dirname(pa_report_path), exist_ok=True)
        with open(pa_report_path, "w") as f:
            f.write(report_content)
        print(f"Preliminary Analysis report saved to: {pa_report_path}")
    except Exception as e:
        print(f"Error saving PA report: {e}")
        return {"error_message": f"PA Save Error: {e}"}

    print("<<< COMPLETED: Preliminary Analysis Node")
    return {"preliminary_analysis_report": report_content, "pa_report_path": pa_report_path}


async def quant_strategist_for_eda_node(state: WorkflowState) -> Dict:
    print(">>> EXECUTING: Quant Strategist for EDA Node")
    pa_report = state.get("preliminary_analysis_report", "No preliminary report found.")
    config = state.get("config_params", {})
    
    eda_script_target_filename = "eda_script_generated.py" 
    eda_report_target_filename = "eda_report.txt" 

    meta_prompt_for_eda_creator_llm = f"""
    **YOU ARE THE AI PROMPT CREATOR FOR PYTHON EDA SCRIPT GENERATION**
    **Your Mission:** Based on the Preliminary Analysis Report and provided data paths, generate a detailed "Execution Prompt" for an EDA pythonTool.
    This Execution Prompt must instruct the `pythonTool` to create a Python script that:
    1. Loads data from:
       - Training data: '{state['input_train_path']}'
       - Validation data: '{state.get('input_val_path', 'None')}'
       - Test data: '{state['input_test_path']}'
       - Index column: '{config.get('index_name', 'None')}'
       - Target column: '{config.get('target_column_name', 'TARGET_COLUMN_MISSING_IN_CONFIG')}'
    2. Performs comprehensive EDA and textually summarizes: .info(), .describe(), missing values, unique values, data types.
    3. Generates and textually summarizes visualizations: histograms, box plots, bar charts, correlation matrix, target distribution.
    4. Saves the comprehensive textual EDA report to '{os.path.join(REPORTS_DIR, eda_report_target_filename)}'.
    5. The EDA script itself should be saved to '{os.path.join(SCRIPTS_DIR, eda_script_target_filename)}'.

    **Preliminary Analysis Report Content to inform your strategy:**
    {pa_report}
    """
    eda_execution_prompt = await gemini_generate_execution_prompt(
        meta_prompt_for_eda_creator_llm, "EDA"
    )
    if "Error:" in eda_execution_prompt:
        return {"error_message": eda_execution_prompt}
    print("<<< COMPLETED: Quant Strategist for EDA Node")
    return {"prompt_for_eda_tool": eda_execution_prompt}


async def eda_agent_node(state: WorkflowState) -> Dict:
    print(">>> EXECUTING: EDA Agent Node")
    execution_prompt = state.get("prompt_for_eda_tool", "Error: No EDA prompt.")
    script_path = os.path.join(SCRIPTS_DIR, "eda_script_generated.py")
    report_path = os.path.join(REPORTS_DIR, "eda_report.txt")

    if not execution_prompt or "Error:" in execution_prompt :
         print(f"EDA Agent: Invalid or missing execution prompt: {execution_prompt}")
         return {"error_message": f"EDA Agent: Invalid or missing execution prompt - {execution_prompt}"}

    tool_output = python_tool_simulation(
        execution_prompt, 
        script_path, 
        report_path,
        stage_name="EDA"
        )

    if tool_output.get("status") == "error": 
        return {"error_message": f"EDA Tool Error: {tool_output.get('error_message')}"}

    print("<<< COMPLETED: EDA Agent Node")
    return {
        "eda_script_path": tool_output.get("script_path"),
        "eda_report": tool_output.get("report_content"),
        "eda_report_path": tool_output.get("report_path")
    }

async def quant_strategist_for_fe_node(state: WorkflowState) -> Dict:
    print(">>> EXECUTING: Quant Strategist for Feature Engineering Node")
    eda_report = state.get("eda_report", "")
    pa_report = state.get("preliminary_analysis_report", "")
    config = state.get("config_params", {})
    
    output_paths = {
        "output_fe_train_X_path": os.path.join(DATA_DIR_FE, "x_train_fe.csv"),
        "output_fe_train_y_path": os.path.join(DATA_DIR_FE, "y_train_fe.csv"),
        "output_fe_val_X_path": os.path.join(DATA_DIR_FE, "x_val_fe.csv") if state.get("input_val_path") else None,
        "output_fe_val_y_path": os.path.join(DATA_DIR_FE, "y_val_fe.csv") if state.get("input_val_path") else None,
        "output_fe_test_X_path": os.path.join(DATA_DIR_FE, "x_test_fe.csv"),
        "output_fe_test_y_path": os.path.join(DATA_DIR_FE, "y_test_fe.csv"),
        "fe_transformers_path": os.path.join(MODELS_DIR, "fe_transformers.pkl"),
        "fe_report_path": os.path.join(REPORTS_DIR, "fe_report.txt"),
        "fe_script_path": os.path.join(SCRIPTS_DIR, "fe_script_generated.py")
    }
    
    meta_prompt_for_fe_tool_creator = f"""
    **YOU ARE AN AI DATA SCIENCE STRATEGIST FOR PHASE 1 FEATURE ENGINEERING (GPT-4o)**
    **Your Mission:** Analyze EDA & Preliminary reports. Output a **Detailed Strategic Plan for Phase 1 Feature Engineering** (non-fittable transforms: direct calculations, date parts, lags, rolling stats, basic NaN handling, target sep). This plan will be used to generate an Execution Prompt for a pythonTool.
    **SECTION 1: INPUT CONTEXT**
    1. EDA Report: {eda_report[:1000]}... 
    2. Preliminary Analysis Report: {pa_report[:1000]}...
    3. Configuration Details:
        - Input Train Path: '{state['input_train_path']}'
        - Input Val Path: '{state.get('input_val_path', 'None')}'
        - Input Test Path: '{state['input_test_path']}'
        - Output FE Train X Path: '{output_paths['output_fe_train_X_path']}'
        - Output FE Train y Path: '{output_paths['output_fe_train_y_path']}'
        - Output FE Val X Path: '{output_paths['output_fe_val_X_path'] if output_paths['output_fe_val_X_path'] else 'None'}'
        - Output FE Val y Path: '{output_paths['output_fe_val_y_path'] if output_paths['output_fe_val_y_path'] else 'None'}'
        - Output FE Test X Path: '{output_paths['output_fe_test_X_path']}'
        - Output FE Test y Path: '{output_paths['output_fe_test_y_path']}'
        - Output Fitted Transformers Path: '{output_paths['fe_transformers_path']}' (Note: For Phase 1 FE, this might not be used if only non-fittable transforms are done)
        - Target Variable Name: '{config.get('target_column_name', 'Target_Return')}'
        - Index Column Name: '{config.get('index_name', 'Date')}'
        - OHLCV Columns: '{config.get('ohlcv_columns_str', 'Open,High,Low,Close,Volume')}'
        - Other Date Columns: '{config.get('other_date_columns_str', 'None')}'
        - Target FE Report Filename: '{os.path.basename(output_paths['fe_report_path'])}'
        - Target FE Script Filename: '{os.path.basename(output_paths['fe_script_path'])}'
    **SECTION 2: YOUR STRATEGIC PLANNING FOR PHASE 1 FEATURE ENGINEERING**
    (LLM: Based on Section 1, detail your plan for non-fittable FE: data loading, initial imputation if critical for signal gen, target derivation/separation, signal generation (lags, rolling means from OHLCV), date part extraction, NaN handling for X & y alignment, dropping original cols. Specify pandas/numpy logic. NO FITTABLE TRANSFORMERS like scalers/encoders here.)
    **SECTION 3: YOUR OUTPUT – THE DETAILED STRATEGIC PLAN (JSON-like structure preferred)**
    (LLM: Output the plan from Section 2 in a structured format, e.g., a list of steps with descriptions, target columns, and methods. This plan will then be used to generate the Execution Prompt for the pythonTool.)
        """
    execution_prompt = await gemini_generate_execution_prompt(
        meta_prompt_for_fe_tool_creator, "FeatureEngineering_Phase1"
    )
    if "Error:" in execution_prompt:
        return {"error_message": execution_prompt, **output_paths} 

    state_update = {"prompt_for_fe_tool": execution_prompt}
    state_update.update(output_paths) 
    print("<<< COMPLETED: Quant Strategist for Feature Engineering Node")
    return state_update

async def feature_engineering_agent_node(state: WorkflowState) -> Dict:
    print(">>> EXECUTING: Feature Engineering Agent Node")
    execution_prompt = state.get("prompt_for_fe_tool", "Error: No FE prompt.")
    script_path = state.get("fe_script_path", os.path.join(SCRIPTS_DIR, "fe_script_default_generated.py"))
    report_path = state.get("fe_report_path", os.path.join(REPORTS_DIR, "fe_report_default.txt"))

    if not execution_prompt or "Error:" in execution_prompt:
        print(f"FE Agent: Invalid or missing execution prompt: {execution_prompt}")
        return {"error_message": f"FE Agent: Invalid or missing execution prompt - {execution_prompt}"}

    fe_output_data_paths = {
        "X_train_fe": state.get("output_fe_train_X_path"),
        "y_train_fe": state.get("output_fe_train_y_path"),
        "X_val_fe": state.get("output_fe_val_X_path"),
        "y_val_fe": state.get("output_fe_val_y_path"),
        "X_test_fe": state.get("output_fe_test_X_path"),
        "y_test_fe": state.get("output_fe_test_y_path"),
    }
    fe_output_data_paths = {k: v for k, v in fe_output_data_paths.items() if v}

    tool_output = python_tool_simulation(
        execution_prompt,
        script_path,
        report_path,
        output_data_paths=fe_output_data_paths,
        stage_name="FeatureEngineering"
    )
    print("<<< COMPLETED: Feature Engineering Agent Node")
    update_dict = {
        "feature_engineering_report": tool_output.get("report_content"),
    }
    return update_dict


async def quant_strategist_for_modeling_node(state: WorkflowState) -> Dict:
    print(">>> EXECUTING: Quant Strategist for Modeling Node")
    fe_report = state.get("feature_engineering_report", "")
    eda_report = state.get("eda_report", "")
    pa_report = state.get("preliminary_analysis_report", "")
    config = state.get("config_params", {})
    
    model_input_x_train = state.get("output_fe_train_X_path")
    model_input_y_train = state.get("output_fe_train_y_path")
    model_input_x_val = state.get("output_fe_val_X_path")
    model_input_y_val = state.get("output_fe_val_y_path")
    model_input_x_test = state.get("output_fe_test_X_path")
    model_input_y_test = state.get("output_fe_test_y_path")

    pipeline_output_target_path = os.path.join(MODELS_DIR, "final_model_pipeline.pkl")
    modeling_script_target_path = os.path.join(SCRIPTS_DIR, "modeling_script_generated.py")
    modeling_report_target_path = os.path.join(REPORTS_DIR, "modeling_report.txt")

    meta_prompt_for_modeling_tool_creator = f"""
**YOU ARE THE AI PROMPT CREATOR FOR PYTHON MODELING SCRIPT GENERATION (PIPELINE FOCUS)**
**Your Mission:** Analyze FE report and other context. Output a detailed "Execution Prompt" for a pythonTool to generate a modeling script. The script must create a scikit-learn Pipeline (including fittable FE like scaling/encoding if not fully done by FE Phase 1, plus the regressor), tune it with GridSearchCV, evaluate, and save the single pipeline .pkl.
**SECTION 1: INPUT CONTEXT**
1. Feature Engineering Report: {fe_report[:1000]}...
2. EDA Report: {eda_report[:1000]}...
3. Preliminary Analysis Report: {pa_report[:1000]}...
4. Input Data Paths (Feature Engineered by Phase 1 FE):
    - X_train_fe: '{model_input_x_train}'
    - y_train_fe: '{model_input_y_train}'
    - X_val_fe: '{model_input_x_val if model_input_x_val else 'None'}'
    - y_val_fe: '{model_input_y_val if model_input_y_val else 'None'}'
    - X_test_fe: '{model_input_x_test}'
    - y_test_fe: '{model_input_y_test if model_input_y_test else 'None'}'
5. Output Path for Modeling Pipeline: '{pipeline_output_target_path}'
6. Target Modeling Script Path: '{modeling_script_target_path}'
7. Target Modeling Report Path: '{modeling_report_target_path}'
8. Configuration:
    - Target Variable Name: '{config.get('target_column_name', 'Target_Return')}'
    - Index Column Name: '{config.get('index_name', 'Date')}'
    - Numerical Columns for Pipeline (names in FE data if scaling/imputation needed here): '{config.get('numerical_cols_for_pipeline_str', 'ALL_NUMERICAL_FROM_FE')}'
    - Categorical Columns for Pipeline (names in FE data if encoding needed here): '{config.get('categorical_cols_for_pipeline_str', 'ALL_CATEGORICAL_FROM_FE')}'
**SECTION 2: YOUR STRATEGIC DECISION-MAKING FOR MODELING PIPELINE**
(LLM: Based on FE report, decide if scaling/encoding/final imputation is still needed for the data coming from FE Phase 1. Choose a regressor model. Define hyperparameter grid for pipeline steps.)
**SECTION 3: CONSTRUCT YOUR OUTPUT – THE "EXECUTION PROMPT" FOR `pythonTool`**
(LLM: Generate the Execution Prompt. It must instruct pythonTool to create a script that defines the sklearn.pipeline.Pipeline (with conditional fittable FE steps + model), uses GridSearchCV, evaluates, and saves the single best pipeline.)
    """
    execution_prompt = await gemini_generate_execution_prompt(
        meta_prompt_for_modeling_tool_creator, "Modeling"
    )
    if "Error:" in execution_prompt:
        return {"error_message": execution_prompt, 
                "final_pipeline_path": pipeline_output_target_path, 
                "modeling_script_path": modeling_script_target_path,
                "modeling_report_path": modeling_report_target_path
               }
                
    print("<<< COMPLETED: Quant Strategist for Modeling Node")
    return {
        "prompt_for_modeling_tool": execution_prompt,
        "final_pipeline_path": pipeline_output_target_path,
        "modeling_script_path": modeling_script_target_path,
        "modeling_report_path": modeling_report_target_path
    }

async def modeling_agent_node(state: WorkflowState) -> Dict:
    print(">>> EXECUTING: Modeling Agent Node")
    execution_prompt = state.get("prompt_for_modeling_tool", "Error: No Modeling prompt.")
    script_path = state.get("modeling_script_path", os.path.join(SCRIPTS_DIR, "modeling_script_default.py"))
    report_path = state.get("modeling_report_path", os.path.join(REPORTS_DIR, "modeling_report_default.txt"))
    pipeline_save_path = state.get("final_pipeline_path") 

    if not pipeline_save_path: 
        return {"error_message": "Modeling Agent: final_pipeline_path not set in state."}
    if not execution_prompt or "Error:" in execution_prompt:
        print(f"Modeling Agent: Invalid or missing execution prompt: {execution_prompt}")
        return {"error_message": f"Modeling Agent: Invalid or missing execution prompt - {execution_prompt}"}

    tool_output = python_tool_simulation(
        execution_prompt,
        script_path,
        report_path,
        output_model_path=pipeline_save_path,
        stage_name="Modeling"
    )
    print("<<< COMPLETED: Modeling Agent Node")
    return {
        "model_evaluation_report": tool_output.get("report_content")
    }

# --- 4. Build the Graph ---
workflow_builder = StateGraph(WorkflowState)

workflow_builder.add_node("preliminary_analyzer", preliminary_analysis_node)
workflow_builder.add_node("quant_eda_strategist", quant_strategist_for_eda_node)
workflow_builder.add_node("eda_executor", eda_agent_node)
workflow_builder.add_node("quant_fe_strategist", quant_strategist_for_fe_node)
workflow_builder.add_node("fe_executor", feature_engineering_agent_node)
workflow_builder.add_node("quant_modeling_strategist", quant_strategist_for_modeling_node)
workflow_builder.add_node("modeling_executor", modeling_agent_node)

workflow_builder.set_entry_point("preliminary_analyzer")
workflow_builder.add_edge("preliminary_analyzer", "quant_eda_strategist")
workflow_builder.add_edge("quant_eda_strategist", "eda_executor")
workflow_builder.add_edge("eda_executor", "quant_fe_strategist")
workflow_builder.add_edge("quant_fe_strategist", "fe_executor")
workflow_builder.add_edge("fe_executor", "quant_modeling_strategist")
workflow_builder.add_edge("quant_modeling_strategist", "modeling_executor")
workflow_builder.add_edge("modeling_executor", END)

app = workflow_builder.compile()

# --- 5. Run the Graph (Async) ---
async def main():
    setup_directories()

    train_csv_path = os.path.join(DATA_DIR_RAW, "train_clean.csv")
    val_csv_path = os.path.join(DATA_DIR_RAW, "val_clean.csv")
    test_csv_path = os.path.join(DATA_DIR_RAW, "test_clean.csv")
    create_dummy_csv(train_csv_path)
    create_dummy_csv(val_csv_path)
    create_dummy_csv(test_csv_path, is_target_present=True) 

    initial_state_input = {
        "initial_goal": "Develop a stock price movement prediction model for MSFT, producing a single deployable pipeline.",
        "input_train_path": train_csv_path,
        "input_val_path": val_csv_path,
        "input_test_path": test_csv_path,
        "current_working_directory": BASE_OUTPUT_DIR, 
        "config_params": {
            "index_name": "Date",
            "target_column_name": "Target_Return",
            "ohlcv_columns_str": "Open,High,Low,Close,Volume",
            "other_date_columns_str": "",
            "numerical_cols_for_pipeline_str": "Open,High,Low,Close,Volume,SMA_5_Close,Close_Lag1", 
            "categorical_cols_for_pipeline_str": "Month,DayOfWeek" 
        }
    }

    print("\n--- Running Full Workflow Stream (Async) ---")
    async for event_update_dict_stream_item in app.astream(initial_state_input, stream_mode="values"):
        print(f"\n--- State Snapshot ---") 
        print("Current Full State Contents:")
        for key, value in event_update_dict_stream_item.items(): 
             if value is not None:
                if isinstance(value, str) and len(value) > 150 and key not in ["initial_goal"]:
                    print(f"  {key}: {str(value)[:150]}... (truncated)")
                else:
                    print(f"  {key}: {value}")
    print("\n--- Full Workflow Complete ---")
    print(f"Find outputs in directory: {os.path.abspath(BASE_OUTPUT_DIR)}")

if __name__ == "__main__":
    import asyncio 
    # This try-except block is for compatibility with environments like Jupyter
    # where an event loop might already be running.
    try:
        asyncio.run(main())
    except RuntimeError as e:
        if "asyncio.run() cannot be called from a running event loop" in str(e):
            print("INFO: asyncio.run() failed, possibly in Jupyter. Trying nest_asyncio approach.")
            import nest_asyncio
            nest_asyncio.apply()
            asyncio.run(main())
        else:
            raise
