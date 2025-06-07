import os
from typing import List, Dict, Any, Optional

# --- Core Imports ---
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain.agents import AgentExecutor, create_react_agent
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

# --- Set up your API Key ---
# Make sure your OPENAI_API_KEY is set in your environment variables
# from dotenv import load_dotenv
# load_dotenv()
# if os.getenv("OPENAI_API_KEY") is None:
#     raise ValueError("OPENAI_API_KEY is not set. Please set it in your environment.")

# ==============================================================================
# 1. DEFINE THE TARGET STRUCTURE (PYDANTIC BASEMODEL)
# ==============================================================================
class EDAState(BaseModel):
    """Data model for storing EDA results and suggestions."""
    report: Optional[Dict[str, Any]] = Field(default=None, description="A summary dictionary of the EDA report, with keys like 'correlation_matrix' or 'key_findings'.")
    model_suggestions: List[str] = Field(default_factory=list, description="Suggestions for machine learning models to try.")
    fe_suggestions: List[str] = Field(default_factory=list, description="Suggestions for feature engineering.")
    final_cleaned_train_ref: Optional[str] = Field(default=None, description="Filename or path to the final cleaned training data.")
    final_cleaned_val_ref: Optional[str] = Field(default=None, description="Filename or path to the final cleaned validation data.")
    final_cleaned_test_ref: Optional[str] = Field(default=None, description="Filename or path to the final cleaned test data.")

# ==============================================================================
# 2. CREATE MOCK TOOLS FOR THE AGENT
# In a real application, these would perform actual data operations.
# ==============================================================================
@tool
def run_exploratory_data_analysis(dataset_path: str) -> str:
    """Runs a full EDA on the dataset at the given path and returns a text summary."""
    print(f"Tool running: EDA on '{dataset_path}'...")
    return """
    EDA Summary: The analysis of the provided dataset reveals strong multicollinearity between 'feature_A' and 'feature_C'.
    The target variable 'churn' is imbalanced (90% no, 10% yes).
    Key insights show that users with 'contract_type' set to 'monthly' are 5x more likely to churn.
    """

@tool
def suggest_models_and_feature_engineering(eda_summary: str) -> str:
    """Based on an EDA summary, suggests ML models and feature engineering steps."""
    print(f"Tool running: Suggesting models based on summary...")
    return """
    Model Recommendations: Given the class imbalance, start with a 'Logistic Regression' using class weights. For better performance, a 'Gradient Boosting Classifier' like LightGBM is recommended.
    Feature Engineering Ideas: Create polynomial features for 'user_tenure'. Also, apply one-hot encoding to the 'region' column.
    """

@tool
def clean_data_and_save(dataset_path: str) -> dict:
    """Cleans the dataset and saves train, validation, and test splits."""
    print(f"Tool running: Cleaning data from '{dataset_path}'...")
    # This simulates saving files and returning their names
    return {
        "status": "success",
        "train_file": "final_cleaned_train_v3.csv",
        "val_file": "final_cleaned_val_v3.csv",
        "test_file": "final_cleaned_test_v3.csv",
    }

# ==============================================================================
# 3. CONFIGURE THE AGENT
# ==============================================================================
llm = ChatOpenAI(model="gpt-4-turbo", temperature=0)
tools = [run_exploratory_data_analysis, suggest_models_and_feature_engineering, clean_data_and_save]

# The ReAct prompt guides the agent's reasoning process.
react_prompt = PromptTemplate.from_template("""
Answer the following questions as best you can. You have access to the following tools:
{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: a comprehensive summary of all the steps taken and results found.

Begin!

Question: {input}
Thought:{agent_scratchpad}
""")

agent = create_react_agent(llm, tools, react_prompt)

# The AgentExecutor is what runs the agent's reasoning loop.
# handle_parsing_errors=True is important for robustness.
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True, # Set to True to see the agent's thought process
    handle_parsing_errors=True
)


# ==============================================================================
# 4. CONFIGURE THE FORMATTING CHAIN
# This chain's only purpose is to structure the agent's final text output.
# ==============================================================================

# Set up the PydanticOutputParser with our target EDAState model
parser = PydanticOutputParser(pydantic_object=EDAState)

# Create a prompt that instructs the model on how to format the text
formatting_prompt = PromptTemplate(
    template="""
You are an expert assistant who extracts structured information from a given text.
Analyze the text and format it into the required JSON structure. Make sure to extract all relevant details.

{format_instructions}

Here is the text to parse:
---
{text_to_parse}
---
""",
    input_variables=["text_to_parse"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

# Create the chain that links the prompt, the model, and the parser
formatting_chain = formatting_prompt | llm | parser


# ==============================================================================
# 5. DEFINE THE ORCHESTRATOR FUNCTION
# This function brings everything together.
# ==============================================================================
def run_eda_and_get_structured_output(input_text: str) -> EDAState:
    """
    Runs the EDA agent to get a raw text summary, then formats that text
    into the structured EDAState model.
    """
    # STEP 1: Run the agent to perform the tasks and get a raw text output
    print("--- 🏁 STARTING AGENT EXECUTION ---")
    result = agent_executor.invoke({"input": input_text})
    raw_output = result['output']

    print("\n\n--- ✅ AGENT EXECUTION FINISHED ---")
    print("--- 📦 Raw Agent Output ---")
    print(raw_output)

    # STEP 2: Use the formatting chain to parse the raw text into a Pydantic object
    print("\n\n--- 🔄 STARTING OUTPUT FORMATTING ---")
    structured_output = formatting_chain.invoke({"text_to_parse": raw_output})
    
    return structured_output

# ==============================================================================
# 6. RUN THE FULL PROCESS
# ==============================================================================
if __name__ == "__main__":
    user_prompt = "Perform a full EDA on the dataset 'customer_data.csv', then suggest appropriate models and feature engineering steps. Finally, clean the data and tell me the names of the saved files."
    
    final_structured_data = run_eda_and_get_structured_output(user_prompt)

    print("\n\n--- ✨ FINAL STRUCTURED RESULT ✨ ---")
    # Pydantic's .model_dump_json() is great for clean printing
    print(final_structured_data.model_dump_json(indent=2))

    print("\n--- You can now work with this as a Python object ---")
    print(f"Suggested Models: {final_structured_data.model_suggestions}")
    print(f"Training File Path: {final_structured_data.final_cleaned_train_ref}")
