Este prototipo permite a los usuarios analizar, depurar y obtener explicaciones detalladas sobre fragmentos de código en Python, utilizando modelos avanzados de inteligencia artificial (IA) como Meta-Llama y Gemini. Los modelos son accesibles a través de APIs y se integran mediante una interfaz desarrollada con Streamlit. Este prototipo busca ayudar a identificar errores, comprender su causa, y recibir sugerencias para mejorar la calidad del código.

1.	Requisitos

o	Python 3.8 o superior.

o	Librerías necesarias:

o	Streamlit (versión 1.38.0)

o	Requests (versión 2.32.3)

o	pandas (versión 2.2.2)

o	huggingface_hub (versión 0.25.1)

o	dotenv (versión 1.0.1)

o	API Keys:

o	Acceso a la API de Meta-Llama mediante Hugging Face.

o	Acceso a la API de Gemini Pro.

2.	Estructura del código

El código se organiza en componentes bien definidos, cada uno responsable de manejar aspectos clave como las consultas a las APIs, la generación de prompts para los modelos, y la interacción con los usuarios a través de la interfaz. A continuación, se describen las partes del código que componen el prototipo.
2.1.	Imports. Las librerías importadas son esenciales para el funcionamiento del prototipo, incluyendo las de interfaz gráfica, solicitudes a la API y funciones básicas del prototipo:

o	import streamlit as st: Crea la interfaz interactiva del prototipo.

o	import requests: Permite realizar solicitudes HTTP, como consultas a APIs.

o	import pandas as pd: Ayuda a manejar y manipular datos en forma de tablas.

o	from huggingface_hub import InferenceClient: Proporciona un cliente para interactuar con modelos de Hugging Face.

o	import json: Permite trabajar con datos en formato JSON, que es común en las respuestas de APIs.

o	import os y from dotenv import load_dotenv: Manejan la carga de variables de entorno desde un archivo .env, como las claves de las APIs.

2.2.	Inicialización. Se configura el cliente para la API de Meta-Llama y se especifica la URL y el encabezado de la API de Gemma, adicional se debe incluir una API key para poder hacer uso de dichos modelos.

o	LLAMA_API  model="meta-llama/Meta-Llama-3-8B-Instruct"

o	GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1/models/gemini-pro:generateContent?"

o	headers = {"Authorization": " GEMINI_API_URL, LLAMA_API_URL "}, debe extraerse de un archive .env, donde se encuentran todas las api key.

2.3.	Funciones. El código incluye varias funciones que cumplen roles específicos:

o	get_gemini_recommendations: Realiza una consulta a la API de Gemini mediante la solicitud POST a la API de GEMINI con un payload en formato JSON también controla errores y devuelve la respuesta JSON.

o	query_llama: Realiza una consulta al modelo Meta-Llama. Envía un prompt al modelo Meta-Llama, así mismo, utiliza la API de Hugging Face para obtener una respuesta generada.

o	generate_debug_prompt: Implementa un prompt que permite el análisis, depuración y explicación del código cargado.

o	execute_debug_analysis: Toma el código ingresado por el usuario y el modelo seleccionado como parámetros, envía el prompt al modelo correspondiente (Meta-Llama o Gemini) y devuelve la respuesta generada al usuario.
