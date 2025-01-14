import streamlit as st
import requests
import pandas as pd
from huggingface_hub import InferenceClient
import json
import os
from dotenv import load_dotenv 

load_dotenv()

LLAMA_API_KEY = os.getenv("LLAMA_API_KEY")
GEMMA_API_URL = os.getenv("GEMMA_API_URL")
GEMMA_API_KEY = os.getenv("GEMMA_API_KEY")

if not LLAMA_API_KEY or not GEMMA_API_URL or not GEMMA_API_KEY:
    st.error("Missing API keys in the environment. Please check your .env file.")

llama_client = InferenceClient(api_key=LLAMA_API_KEY)

headers = {"Authorization": f"Bearer {GEMMA_API_KEY}"}

# Functions
def query_gemma(payload):
    try:
        response = requests.post(GEMMA_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API request error: {e}")
        return None

def query_llama(prompt):
    try:
        messages = [{"role": "user", "content": prompt}]
        response = llama_client.chat_completion(
            model="meta-llama/Meta-Llama-3-8B-Instruct",
            messages=messages,
            max_tokens=500,
            stream=False
        )
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        st.error(f"Llama API request error: {e}")
        return None

def generate_synthetic_data_prompt(theme, variable_names, num_items):
    variables_str = ', '.join(variable_names)
    prompt = (f"Generate {num_items} synthetic data entries based on the theme '{theme}'. "
              f"Each entry should contain the following variables in the format: "
              f"({variables_str}). Provide the data entries in the same format without additional descriptions.")
    return prompt

def parse_variable_names(response_text):
    lines = response_text.split('\n')
    variable_names = []
    for line in lines:
        stripped_line = line.strip().replace('*', '').replace('"', '').replace("'", "")
        if stripped_line and stripped_line[0].isdigit() and '.' in stripped_line:
            variable_name = stripped_line.split('. ', 1)[1]
            variable_names.append(variable_name)
    return variable_names

def parse_synthetic_data(response_text, num_variables):
    rows = []
    lines = response_text.split('\n')
    for line in lines:
        line = line.strip()
        if line.startswith(tuple(str(i) for i in range(1, 11))):
            line = line.split(' ', 1)[1]
        if line.startswith('(') and line.endswith(')'):
            row = line[1:-1].split(', ')
            cleaned_row = []
            for value in row:
                if ':' in value:
                    value_cleaned = value.split(':')[1].strip()
                else:
                    value_cleaned = value.strip()
                cleaned_row.append(value_cleaned)
            if len(cleaned_row) == num_variables:
                rows.append(cleaned_row)
            else:
                st.warning(f"Incomplete row detected and skipped: {line}")
    return rows

def clean_variable_name(variable_name):
    cleaned_name = variable_name.replace('"', '').replace("'", "")
    if ':' in cleaned_name:
        return cleaned_name.split(':')[0].strip()
    return cleaned_name.strip()

# Main function
def main():
    st.title("Synthetic Data Generator")

    menu_options = ["Generate Synthetic Data", "Upload Dataset Fragment"]
    menu_selection = st.sidebar.selectbox("Select an option:", menu_options)

    if menu_selection == "Generate Synthetic Data":
        model_choice = st.selectbox("Select a model:", ("Gemma-1.1-2b-it", "Meta-Llama-3-8B-Instruct"))

        theme = st.text_input("Enter a theme for generating variables:", "")
        num_variables = st.number_input("Enter the number of variables to generate:", min_value=1, value=4)
        num_items = st.number_input("Enter the number of data entries to generate:", min_value=1, value=10)

        if st.button("Generate variables"):
            variable_names_prompt = (f"Generate a list of {num_variables} variable names that could be used in a synthetic dataset related to the theme '{theme}'. "
                                     "Make sure the variables are relevant and specific to this theme.")
            if model_choice == "Gemma-1.1-2b-it":
                variable_names_output = query_gemma({"inputs": variable_names_prompt})
                if variable_names_output:
                    variable_names_text = variable_names_output[0].get('generated_text', '')
                    st.session_state.variable_names = parse_variable_names(variable_names_text)
            elif model_choice == "Meta-Llama-3-8B-Instruct":
                variable_names_text = query_llama(variable_names_prompt)
                if variable_names_text:
                    st.session_state.variable_names = parse_variable_names(variable_names_text)

        if 'variable_names' in st.session_state:
            st.subheader("Generated Variables")
            edited_variable_names = []
            for i, var in enumerate(st.session_state.variable_names):
                edited_variable_name = st.text_input(f"Variable {i+1}:", var, key=f"var_{i}")
                edited_variable_names.append(edited_variable_name)

            new_variable = st.text_input("Add a new variable:")
            if new_variable:
                edited_variable_names.append(new_variable)

            st.session_state.variable_names = edited_variable_names

            synthetic_prompt = generate_synthetic_data_prompt(theme, edited_variable_names, num_items)

            if st.button("Generate synthetic data"):
                synthetic_output = query_llama(synthetic_prompt)
                if synthetic_output:
                    all_data_rows = parse_synthetic_data(synthetic_output, len(edited_variable_names))
                    cleaned_variable_names = [clean_variable_name(var) for var in edited_variable_names]
                    df = pd.DataFrame(all_data_rows, columns=[var.replace(' ', '_').upper() for var in cleaned_variable_names])
                    st.subheader("Synthetic Dataset")
                    st.write(df)

                    st.download_button("Download as CSV", df.to_csv(index=False).encode('utf-8'), "dataset.csv", "text/csv")
                    st.download_button("Download as JSON", df.to_json(orient='records', lines=True).encode('utf-8'), "dataset.json", "application/json")

if __name__ == "__main__":
    main()
