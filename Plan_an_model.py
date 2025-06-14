import json
import pandas as pd
import joblib
import numpy as np
from typing import List, Dict, TypedDict

# Import your specific LLM and sklearn models
# from langchain_openai import ChatOpenAI # Or your preferred LLM
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error

# --- Define the Graph State ---
class GraphState(TypedDict):
    feature_sets: List[Dict]
    # The final output will be directly in the results key
    results: List[Dict] 

# --- A helper to map LLM output to actual Sklearn classes ---
SKLEARN_MODELS = {
    "RandomForestRegressor": RandomForestRegressor,
    "GradientBoostingRegressor": GradientBoostingRegressor,
    "Ridge": Ridge,
    "Lasso": Lasso,
    # Add any other models you want the LLM to be able to choose from
}

# --- The Single, Combined Node ---
def plan_and_train_models(state: GraphState):
    """
    This single node performs both LLM-based planning and Python-based execution.
    """
    print("--- ENTERING SINGLE MODELING NODE ---")
    
    # =================================================================
    #  PART 1: LLM-driven Planning (The "Selector" Logic)
    # =================================================================
    print("\n[Part 1] Generating model plan with LLM...")
    
    feature_sets = state["feature_sets"]
    feature_sets_str = json.dumps(feature_sets, indent=2)

    # This prompt is focused ONLY on getting a JSON plan
    selector_prompt = f"""
    You are an expert Machine Learning engineer. Your task is to select the best models for a given regression problem.

    CONTEXT:
    - Problem Type: Regression
    - Target Column: 'Close'
    - For each provided feature set, you must analyze its characteristics and choose the two most promising Scikit-learn regression models. Provide a clear rationale.

    INPUT:
    A list of feature sets:
    {feature_sets_str}

    TASK:
    Respond with ONLY a valid JSON object. The JSON should be a list, where each item corresponds to an input feature set and contains the feature set info, model configurations, and your rationale. Do not include any other text or explanations outside of the JSON.
    """

    # --- Replace this section with your actual LLM call ---
    # llm = ChatOpenAI(model="gpt-4-turbo", temperature=0)
    # response = llm.invoke(selector_prompt)
    # model_plan_str = response.content
    
    # For demonstration, we'll use a mocked LLM response:
    mock_llm_response_content = """
    [
      {
        "feature_set_info": {
          "name": "Feature Set 1: Polynomial Features",
          "x_train_ref": "data/set1_x_train.csv", "y_train_ref": "data/set1_y_train.csv"
        },
        "model_config": {
          "top_model_configurations": [
            { "model_name": "Ridge", "hyperparameters": { "alpha": 1.0 } },
            { "model_name": "Lasso", "hyperparameters": { "alpha": 0.1 } }
          ],
          "decision_rationale": "Ridge and Lasso are excellent for regularized linear regression, suitable for polynomial features."
        }
      },
      {
        "feature_set_info": { "name": "Feature Set 2: Time-based Features",
          "x_train_ref": "data/set2_x_train.csv", "y_train_ref": "data/set2_y_train.csv"
        },
        "model_config": {
            "top_model_configurations": [
                { "model_name": "RandomForestRegressor", "hyperparameters": { "n_estimators": 100 } },
                { "model_name": "GradientBoostingRegressor", "hyperparameters": { "n_estimators": 150 } }
            ],
            "decision_rationale": "Tree-based models are powerful for capturing complex, non-linear patterns in time-series data."
        }
      }
    ]
    """
    model_plan_str = mock_llm_response_content
    # --- End of section to replace ---

    try:
        model_plan = json.loads(model_plan_str)
        print("[Part 1] Successfully parsed model plan from LLM.")
    except json.JSONDecodeError as e:
        print(f"ERROR: LLM returned invalid JSON. Cannot proceed. Error: {e}")
        # Return an empty result or raise an exception
        return {"results": [{"error": "Failed to parse LLM response", "details": str(e)}]}

    # =================================================================
    #  PART 2: Python-driven Execution (The "Trainer" Logic)
    # =================================================================
    print("\n[Part 2] Executing training plan...")
    all_results = []

    for i, plan in enumerate(model_plan):
        feature_info = plan['feature_set_info']
        print(f"\n---> Processing Plan for: {feature_info['name']}")

        # Load data (ensure your file paths are correct)
        try:
            x_train = pd.read_csv(feature_info['x_train_ref'])
            y_train = pd.read_csv(feature_info['y_train_ref'])
        except FileNotFoundError as e:
            print(f"  SKIPPING: Could not find data file {e.filename}")
            all_results.append({"error": f"Data file not found: {e.filename}", "plan": plan})
            continue

        for model_config in plan['model_config']['top_model_configurations']:
            model_name = model_config['model_name']
            hyperparams = model_config['hyperparameters']
            
            print(f"  Training model: {model_name} with params: {hyperparams}")

            if model_name not in SKLEARN_MODELS:
                print(f"  SKIPPING: Model '{model_name}' is not a supported model.")
                continue

            # Instantiate, train, and evaluate the model
            model = SKLEARN_MODELS[model_name](**hyperparams)
            model.fit(x_train, y_train.values.ravel())
            predictions = model.predict(x_train)
            rmse = np.sqrt(mean_squared_error(y_train, predictions))
            print(f"  Trained Model RMSE: {rmse:.4f}")

            # Save the trained model artifact
            model_filename = f"trained_{model_name}_set{i+1}.joblib"
            joblib.dump(model, model_filename)
            print(f"  Model saved to {model_filename}")

            all_results.append({
                "feature_set_name": feature_info['name'],
                "model_name": model_name,
                "model_artifact_path": model_filename,
                "train_rmse": rmse,
                "decision_rationale": plan['model_config'].get('decision_rationale', '')
            })

    print("\n--- FINISHED SINGLE MODELING NODE ---")
    return {"results": all_results}
