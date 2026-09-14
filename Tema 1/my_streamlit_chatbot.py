import streamlit as st

from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate



 
# Configurar la página de la app
st.set_page_config(page_title="Chatbot Básico", page_icon="🤖")
st.title("🤖 Chatbot Básico con LangChain")
st.markdown("Este es un *chatbot de ejemplo* construido con LangChain + Streamlit. ¡Escribe tu mensaje abajo para comenzar!")

with st.sidebar:
    st.header("Configuración del Modelo")
    modelo = st.selectbox("Selecciona el modelo de lenguaje:", ["llama3.2", "gpt-4o-mini", "gpt-3.5-turbo", "gpt-4"])
    temperatura = st.slider("Temperatura del modelo:", min_value=0.0, max_value=1.0, value=0.5, step=0.1)

    if modelo == "llama3.2":
        from langchain_ollama import ChatOllama
        from openai import chat

        chat_model = ChatOllama(model="llama3.2", temperature=0.7)
    else:
        chat_model = ChatOpenAI(model=modelo, temperature=temperatura)

    if st.button("🗑️ Nueva conversación"):
        st.session_state.mensajes = []
        st.rerun()
 
# Inicializar el historial de mensajes en session_state
if "mensajes" not in st.session_state:
    st.session_state.mensajes = []

# Crear el template de prompt con comportamiento específico
prompt_template = PromptTemplate(
    input_variables=["mensaje", "historial"],
    template="""Eres un asistente útil y amigable llamado ChatBot Pro. 

Historial de conversación:
{historial}

Responde de manera clara y concisa a la siguiente pregunta: {mensaje}"""
)

# Crear cadena usando LCEL (LangChain Expression Language)
cadena = prompt_template | chat_model

# Renderizar historial existente
for msg in st.session_state.mensajes:
    if isinstance(msg, SystemMessage):
        continue  # no mostrar mensajes del sistema al usuario
    
    role = "assistant" if isinstance(msg, AIMessage) else "user"
    with st.chat_message(role):
        st.markdown(msg.content)



# Input de usuario
pregunta = st.chat_input("Escribe tu mensaje:")

if pregunta:
    # Mostrar y almacenar mensaje del usuario
    with st.chat_message("user"):
        st.markdown(pregunta)
    
    # Generar y mostrar respuesta del asistente
    try:
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""

            # Streaming de la respuesta
            for chunk in cadena.stream({"mensaje": pregunta, "historial": st.session_state.mensajes}):
                full_response += chunk.content
                response_placeholder.markdown(full_response + "▌")
            
            response_placeholder.markdown(full_response)
        
        st.session_state.mensajes.append(HumanMessage(content=pregunta))
        st.session_state.mensajes.append(AIMessage(content=full_response))
        
    except Exception as e:
        st.error(f"Error al generar respuesta: {str(e)}")
        st.info("Verifica que tu API Key de OpenAI esté configurada correctamente.")