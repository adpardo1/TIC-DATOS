import requests
import streamlit as st
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os

# Load environment variables from the .env file
load_dotenv()

# Get the API Key for Gemini from the environment variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Get the API Key for Llama from the environment variables
LLAMA_API_KEY = os.getenv("LLAMA_API_KEY")
client = InferenceClient(api_key=LLAMA_API_KEY)

def query_llama(prompt):
    """Query the Meta-Llama model to analyze the code."""
    try:
        response = client.chat_completion(
            model="meta-llama/Meta-Llama-3-8B-Instruct",  
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
            stream=False
        )
        if response and 'choices' in response:
            return response.choices[0].message["content"]
        else:
            raise ValueError("No response returned from the Meta-Llama model.")
    except Exception as e:
        st.error(f"Error during Llama API request: {e}")
        return None

def get_gemini_recommendations(prompt):
    """Query the Gemini API to analyze the code."""
    try:
        url = f"https://generativelanguage.googleapis.com/v1/models/gemini-pro:generateContent?key={GEMINI_API_KEY}"
        headers = {'Content-Type': 'application/json'}
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": prompt}
                    ]
                }
            ]
        }

        response = requests.post(url, headers=headers, json=payload)
        if response.status_code == 200 and 'candidates' in response.json():
            return response.json()["candidates"][0]["content"]["parts"][0]["text"]
        else:
            raise ValueError(f"API request failed with status code {response.status_code}. Response: {response.text}")
    except Exception as e:
        st.error(f"Error during Gemini API request: {e}")
        return None

def generate_debug_prompt(code):
    """Generate a prompt for code analysis, debugging, and explanation."""
    prompt = (
        "You are a code programer expert who analyzes code snippets to find errors, "
        "explain their cause, and suggest clear and detailed solutions. Your task includes:\n\n"
        "1. Identify syntax or logical errors in the provided code.\n"
        "2. Explain in detail why each error or unexpected behavior occurs.\n"
        "3. Suggest specific corrections, explaining how to implement them.\n"
        "4. If no errors are found, analyze the code to propose improvements in structure, readability, or performance.\n\n"
        "### Code to analyze:\n"
        f"```python\n{code}\n```\n"
    )
    return prompt

def execute_debug_analysis(code, model_choice):
    """Execute the code analysis based on the selected model."""
    prompt = generate_debug_prompt(code)
    try:
        if model_choice == "Meta-Llama 3-8B-Instruct":
            response = query_llama(prompt)
        elif model_choice == "Gemini Pro":
            response = get_gemini_recommendations(prompt)
        if response:
            return response
        else:
            return "No valid response received from the model."
    except Exception as e:
        return f"Error analyzing the code: {e}"

def main():
    st.title("Code Analysis, Debugging, and Explanation with AI")
    st.write("Provide a code snippet to analyze errors, get explanations, and suggestions for improvement.")

    # Input for the code
    code_input = st.text_area("Enter the code here:")

    # Model selection (Llama or Gemini)
    model_choice = st.selectbox(
        "Choose the model to analyze the code:",
        ["Meta-Llama 3-8B-Instruct-v1", "Gemini Pro"]
    )

    if st.button("Analyze Code"):
        if code_input.strip():
            st.write("Processing code analysis...")
            analysis_result = execute_debug_analysis(code_input, model_choice)
            st.subheader("Analysis Results:")
            st.write(analysis_result)
        else:
            st.error("Please enter a code snippet before proceeding.")

if __name__ == "__main__":
    main()
