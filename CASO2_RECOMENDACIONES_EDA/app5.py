import requests
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os
import nbformat as nbf


# Cargar las variables de entorno del archivo .env
load_dotenv()

# Configurar la API Key de Gemini desde el archivo .env
API_KEY = os.getenv("GEMINI_API_KEY")

# Configuración de la API de Llama desde el archivo .env
LLAMA_API_KEY = os.getenv("LLAMA_API_KEY")
client = InferenceClient(api_key=LLAMA_API_KEY)

def query_llama(prompt):
    """Function to query the Meta-Llama model."""
    try:
        response = client.chat_completion(
            model="meta-llama/Meta-Llama-3-8B-Instruct",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            stream=False
        )
        return response.choices[0].message["content"]
    except Exception as e:
        st.error(f"Error during API request: {e}")
        return None

def get_gemini_recommendations(prompt):
    """Function to query the Gemini API and get recommendations."""
    url = f"https://generativelanguage.googleapis.com/v1/models/gemini-pro:generateContent?key={API_KEY}"
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
    if response.status_code == 200:
        return response.json()["candidates"][0]["content"]["parts"][0]["text"]
    else:
        raise Exception(f"API error: {response.status_code}, {response.text}")

def generate_eda_prompt(dataframe):
    """Generate a prompt for EDA recommendations based on the dataset columns and statistical summary."""
    columns = dataframe.columns.tolist()
    summary = dataframe.describe().to_string()  # Getting the statistical summary of the dataset
    prompt = (f"I have a dataset with the following columns: {', '.join(columns)}. "
              "Here is a statistical summary of the dataset:\n\n"
              f"{summary}\n\n"
              "Please provide detailed recommendations for exploratory data analysis (EDA), "
              "including data cleaning, transformation, handling missing values, normalization, "
              "and specific visualizations for each variable. "
              "Additionally, suggest relevant code snippets for these tasks.")
    return prompt

def generate_visualization_prompt(dataframe):
    """Generate a prompt for meaningful visualizations."""
    columns = dataframe.columns.tolist()
    prompt = (
        f"I have a dataset with the following columns: {', '.join(columns)}. "
        "Based on this dataset, suggest global visualization techniques to help analyze variable relationships, "
        "detect patterns, and understand the overall data structure. Avoid providing separate visualizations for each column. "
        "Instead, focus on aspects like:\n"
        "- Correlation heatmaps to identify relationships among numerical variables.\n"
        "- Scatter plots or scatterplot matrices for multivariate analysis.\n"
        "- Bar or line plots to analyze trends or aggregates.\n"
        "- Grouped visualizations, such as box plots, to compare distributions across categories.\n\n"
        "Provide Python code for each recommended visualization, ensuring compatibility with Streamlit. The code should include:\n"
        "- The use of matplotlib or seaborn for creating visualizations.\n"
        "- Displaying the plot in Streamlit using 'st.pyplot()'.\n"
        "- Closing the plot after rendering with 'plt.close()' to avoid overlaps.\n\n"
        "Tailor the visualizations to the dataset structure, such as numerical, categorical, or temporal variables, "
        "to provide useful insights and enhance understanding."
    )
    return prompt

def show_default_visualizations(dataframe):
    """Display basic default visualizations based on the dataset."""
    st.subheader("Suggested Visualizations")

    # Show distributions of numerical variables
    num_vars = dataframe.select_dtypes(include=["float64", "int64"]).columns.tolist()
    if num_vars:
        st.write("Distribution of numerical variables:")
        for var in num_vars:
            st.write(f"Distribution of {var}:")
            plt.figure(figsize=(8, 6))
            sns.histplot(dataframe[var], kde=True)
            st.pyplot(plt)
            plt.close()

def create_notebook(eda_recommendations, visualization_recommendations, code_to_execute, dataframe):
    """Create a Jupyter notebook with the recommendations and code."""
    nb = nbf.v4.new_notebook()

    # Add EDA recommendations in Markdown
    nb.cells.append(nbf.v4.new_markdown_cell("# EDA Recommendations"))
    nb.cells.append(nbf.v4.new_markdown_cell(eda_recommendations))

    # Add Visualization recommendations in Markdown
    nb.cells.append(nbf.v4.new_markdown_cell("# Visualization Recommendations"))
    nb.cells.append(nbf.v4.new_markdown_cell(visualization_recommendations))

    # Add the code to execute (ensure this is a code cell)
    nb.cells.append(nbf.v4.new_markdown_cell("# Code to Execute"))
    nb.cells.append(nbf.v4.new_code_cell(code_to_execute))

    # Save the notebook as a .ipynb file without adding any images
    return nb

def save_notebook(nb):
    """Save the generated notebook to a file."""
    notebook_filename = "eda_recommendations_and_code.ipynb"
    with open(notebook_filename, 'w') as f:
        nbf.write(nb, f)
    return notebook_filename

def execute_code(code, dataframe):
    """Execute user-provided code and display the result."""
    try:
        local_context = {"df": dataframe, "st": st, "plt": plt, "sns": sns, "__builtins__": __builtins__}
        exec(code, local_context)
    except Exception as e:
        st.error(f"Error executing code: {e}")

def main():
    st.title("Exploratory Data Analysis")
    st.write("Upload your dataset to get personalized analysis and visualizations.")

    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    # Model selection
    model_choice = st.selectbox(
        "Choose the model for generating recommendations:",
        ["Meta-Llama 3-8B-Instruct", "Gemini Pro"]
    )

    if uploaded_file:
        global df  # Declare df as global to make it accessible in the exec context
        df = pd.read_csv(uploaded_file)
        st.write("Dataset successfully loaded. Preview:")
        st.write(df.head())

        # Show default visualizations
        show_default_visualizations(df)

        # Generate prompts for EDA and visualizations
        if 'eda_recommendations' not in st.session_state:
            eda_prompt = generate_eda_prompt(df)
            visualization_prompt = generate_visualization_prompt(df)

            st.write("Generating EDA recommendations...")

            try:
                # Get recommendations based on the selected model
                if model_choice == "Meta-Llama 3-8B-Instruct":
                    eda_recommendations = query_llama(eda_prompt)
                    visualization_recommendations = query_llama(visualization_prompt)
                elif model_choice == "Gemini Pro":
                    eda_recommendations = get_gemini_recommendations(eda_prompt)
                    visualization_recommendations = get_gemini_recommendations(visualization_prompt)

                st.session_state.eda_recommendations = eda_recommendations
                st.session_state.visualization_recommendations = visualization_recommendations

                st.subheader("EDA Recommendations:")
                st.write(eda_recommendations)

                st.subheader("Visualization Recommendations with Code:")
                st.write(visualization_recommendations)

            except Exception as e:
                st.error(f"Error obtaining recommendations: {e}")

        else:
            st.subheader("EDA Recommendations:")
            st.write(st.session_state.eda_recommendations)
            st.subheader("Visualization Recommendations with Code:")
            st.write(st.session_state.visualization_recommendations)

        # Execute recommended code
        st.subheader("Execute Recommended Code:")
        recommended_code = st.text_area("Enter code here to execute (you can copy it from the recommendations):")

        if st.button("Execute Code"):
            if 'df' in globals():
                execute_code(recommended_code, df)
            else:
                st.error("No dataset loaded. Please upload a CSV file first.")

        # Export notebook
        if st.button("Export Notebook"):
            notebook = create_notebook(
                st.session_state.eda_recommendations,
                st.session_state.visualization_recommendations,
                recommended_code,
                df
            )
            notebook_filename = save_notebook(notebook)
            st.success(f"Notebook saved as {notebook_filename}. You can download it now.")
            with open(notebook_filename, "rb") as f:
                st.download_button(
                    label="Download Notebook",
                    data=f,
                    file_name=notebook_filename,
                    mime="application/octet-stream"
                )

if __name__ == "__main__":
    main()
