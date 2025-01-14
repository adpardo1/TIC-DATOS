Este prototipo permite a los usuarios generar datos sintéticos basados en temas específicos, utilizando LLMs, disponibles en la API de Hugging Face. Se pueden generar nombres de variables y datos sintéticos, así como cargar fragmentos de datasets para expandir la cantidad de datos.
En este caso se utilizaron los modelos de Gemma (Gemma-1.1-2b-it) y Llama 3(Meta-Llama-3-8B-Instruct).
La funcionalidad le sugiere al usuario  
1.	Requisitos
o	Python 3.8 o superior.
o	Librerías necesarias:
o	Streamlit (versión 1.38.0)
o	Requests (versión 2.32.3)
o	pandas (versión 2.2.2)
o	huggingface_hub (versión 0.25.1)
o	dotenv (versión 1.0.1)
o	Cuenta y acceso a la API de Hugging Face.
2.	Estructura del código
El código se organiza en varias funciones para manejar diferentes aspectos de la aplicación, tales como la consulta a las APIs de los modelos, el procesamiento de datos generados, y la interfaz gráfica con Streamlit. A continuación, se describen las partes del código que componen el prototipo
2.1.	Imports. Las librerías importadas son esenciales para el funcionamiento del prototipo incluyendo las de interfaz gráfica, solicitudes a la API y funciones básicas del prototipo:
o	import streamlit as st: Crea la interfaz interactiva del prototipo.
o	import requests: Permite realizar solicitudes HTTP, como consultas a APIs.
o	import pandas as pd: Ayuda a manejar y manipular datos en forma de tablas.
o	from huggingface_hub import InferenceClient: Proporciona un cliente para interactuar con modelos de Hugging Face.
o	import json: Permite trabajar con datos en formato JSON, que es común en las respuestas de APIs.
2.2.	Inicialización. Se configura el cliente para la API de Meta-Llama y se especifica la URL y el encabezado de la API de Gemma, adicional se debe incluir una API key para poder hacer uso de dichos modelos.
o	LLAMA_API  model="meta-llama/Meta-Llama-3-8B-Instruct"
o	GEMMA_API_URL = "https://api-inference.huggingface.co/models/google/gemma-1.1-2b-it"
o	headers = {"Authorization": " GEMMA_API_URL, LLAMA_API_URL "}, debe extraerse de un archivo .env, donde se encuentran todas las api key.
2.3.	Funciones. El código incluye varias funciones en específicos:
o	query_gemma: Realiza una consulta a la API de Gemma con una solicitud POST a la API con un payload en formato JSON que devuelve la respuesta JSON.
o	query_llama: Realiza una consulta al modelo Meta-Llama. Envía un prompt al modelo Meta-Llama así mismo utiliza la API de Hugging Face para obtener una respuesta generada.
o	generate_synthetic_data_prompt: Genera un prompt basado en el tema y las variables proporcionadas.
o	parse_variable_names: Extrae los nombres de variables sin incluir caracteres no deseados de la respuesta de la API.
o	parse_synthetic_data: Convierte el texto de salida en un formato estructurado, es decir, convierte la respuesta textual de la API en una lista de filas estructuradas, también limpia valores y verifica la coherencia en el número de variables.
o	clean_variable_name: Limpia los nombres de las variables para eliminar caracteres especiales.
