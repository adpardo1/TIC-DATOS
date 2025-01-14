El prototipo desarrollado en este apartado permite a los usuarios realizar análisis exploratorios de datos (EDA) mejorados, al integrar modelos de inteligencia artificial, específicamente Gemini y Llama. El prototipo genera recomendaciones adaptadas al contenido del dataset ingresado. Este sistema no solo facilita la comprensión inicial de los datos, sino que también ofrece respuestas para la limpieza, transformación, y visualización de datos, optimizando el flujo de trabajo en ciencia de datos.

1.	Requisitos

o	Python 3.8 o superior.

o	Librerías necesarias:

o	Streamlit (versión 1.38.0)

o	Requests (versión 2.32.3)

o	pandas (versión 2.2.2)

o	huggingface_hub (versión 0.25.1)

o	dotenv (versión 1.0.1)

o	Matplotlib (versión 3.9.2)

o	Seaborn (versión 0.13.2)

o	API Keys:

  o  	Acceso a la API de Meta-Llama mediante Hugging Face.

  o  	Acceso a la API de Gemini Pro.

2.	Estructura del código

El código del prototipo está organizado en funciones específicas que permiten manejar cada aspecto clave del sistema. La estructura incluye módulos para interactuar con las APIs, procesar datos, y diseñar la interfaz gráfica interactiva utilizando Streamlit. A continuación, se describen las principales partes que componen el código:

2.1.	Imports. Las librerías importadas son esenciales para el funcionamiento del prototipo:

o	Streamlit: Para el desarrollo de la interfaz gráfica interactiva.

o	Requests: Para realizar solicitudes HTTP a la API de Gemini.

o	Pandas: Para la manipulación y análisis de datos.

o	Huggingface_hub: Para la interacción con el modelo Llama disponible en Hugging Face.

o	Matplotlib y Seaborn: Para la generación de visualizaciones.

2.2.	Inicialización. Se configura el cliente para la API de Meta-Llama y se especifica la URL y el encabezado de la API de Gemma.

o	LLAMA_API  model="meta-llama/Meta-Llama-3-8B-Instruct"

o	GEMINI_API_URL= "https://generativelanguage.googleapis.com/v1/models/gemini-pro:generateContent?"

o	headers = {"Authorization": " GEMINI_API_URL, LLAMA_API_URL "}, debe extraerse de un archive .env, donde se encuentran todas las api key.

2.3.	Funciones. Aquí se configuran las credenciales y la conexión con las APIs de los modelos Meta-Llama (a través de la API de Hugging Face) y Gemini (a través de la API de Google) esto es fundamental para realizar las consultas y obtener las recomendaciones de EDA y visualización del mismo.

o	get_gemini_recommendations: Esta función realiza una consulta a la API de Gemini, proporcionando el texto de entrada (prompt) y retornando la respuesta generada.

o	query_llama: Realiza una consulta al modelo Meta-Llama. Envía un prompt al modelo Meta-Llama, así mismo, utiliza la API de Hugging Face para obtener una respuesta generada.

o	generate_eda_prompt (dataframe): Genera un prompt personalizado para obtener recomendaciones de EDA basadas en las columnas del dataset proporcionado.

o	generate_visualization_prompt (dataframe): Diseña un prompt para solicitar recomendaciones específicas de visualizaciones útiles para los datos cargados.

o	show_default_visualizations (dataframe): Genera visualizaciones básicas, como histogramas de variables numéricas, utilizando Seaborn y matplotlib.

o	clean_variable_name: Limpia los nombres de las variables para eliminar caracteres especiales.
