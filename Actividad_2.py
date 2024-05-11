#================================ Librerías ================================

import streamlit as st
from PIL import Image
from langchain.text_splitter import CharacterTextSplitter 
from langchain.docstore.document import Document 
from langchain.chains.summarize import load_summarize_chain 
from langchain_community.llms import CTransformers 


#========================== Definición de funciones ========================


# Esta función es responsable de dividir los datos en fragmentos más pequeños (chunks) y convertirlos en formato de documento
def chunks_and_document(txt):
    
    text_splitter = CharacterTextSplitter() 
    texts_chunks = text_splitter.split_text(txt)
    docs = [Document(page_content=t) for t in texts_chunks]
    
    return docs

    
# Carga del LLM. Se pasa como parámetros el número máximo de tokens y la temperatura para el muestreo.
def load_llm(tokens, temp):
  
    llm = CTransformers(
        model=r"/workspaces/actividad-2-tdaabd/Actividad_2/llama-2-7b-chat.ggmlv3.q2_K.bin",
        model_type="llama",
          max_new_tokens = tokens,
        temperature = temp )
        
    return llm
 

#=================================== Título ================================


# st.title('APLICACIÓN PARA RESUMIR TEXTOS')
st.markdown("<h1 style='text-align: center; color: black;'>APLICACIÓN PARA RESUMIR TEXTOS </h1>", unsafe_allow_html=True)
st.text("""""")

# st.subheader('***Técnicas de Desarrollo Avanzado de Aplicaciones Big Data***')
st.markdown("<h2 style='text-align: center; color: black;'>Técnicas de Desarrollo Avanzado de Aplicaciones Big Data </h2>", unsafe_allow_html=True)
st.text("""""")


#============================ Imagen del título ============================


img_title = "/workspaces/actividad-2-tdaabd/Actividad_2/title/title.png"

image = Image.open(img_title)
st.image(
	        image,
	        use_column_width=True,
	    )


#=============================== Barra lateral =============================


st.text("""""")

st.sidebar.title('***Funcionamiento de la aplicación***')
st.sidebar.write("""
## 1️⃣ Introduzca el texto que quiera resumir.
	""")
st.sidebar.write("""
    ## 2️⃣ Elija los parámetros para aplicar en el método :red[Map-Reduce]:
        - max_new_tokens: Limita la cantidad de tokens nuevos 
        que se pueden generar. Al establecer un límite máximo, 
        puede controlar la longitud de la salida generada 
        y evitar que el modelo genere respuestas 
        excesivamente largas o detalladas. 
        
        - temperature: La temperatura que se utilizará 
        para el muestreo. Cuanto menor sea, más deterministas 
        serán los resultados, es decir, siempre se elige 
        el siguiente token más probable. Aumentar la temperatura 
        producirá resultados más creativos. 
        
        
    	""")
        
st.sidebar.write("""
## 3️⃣ Presione ***Ejecutar*** para resumir el texto.
	""")


#================ Cuadro para introducción del texto ========================


txt_input = st.text_area('Introduzca texto:', '', height=200)


# Contar palabras del texto introducido
words = txt_input.split()

st.write('Longitud del texto introducido: {} palabras'.format(len(words)))


#============================ Parámetros LLM =============================


st.text("""""")


tokens = st.slider("Seleccione número máximo de ***tokens***: ", min_value=256, 
 max_value=1024, value=256, step=64)


temp = st.radio('Seleccione temperatura:', 
options=(0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1), horizontal=True)

    
st.text("""""")


#============================== Botón Ejecutar ==============================
            
            
#Creamos un formulario con un botón para ejecutar el modelo.
resultado = []
with st.form('summarize_form', clear_on_submit=True):
    submit = st.form_submit_button('Ejecutar')
    
    if submit:
        if len(words) > 0:
            with st.spinner('Procesando...'):
                docs = chunks_and_document(txt_input)
                llm = load_llm(tokens,temp)
                chain = load_summarize_chain(llm,chain_type='map_reduce')
                response = chain.run(docs)
                # response = chain.invoke(docs)
                resultado.append(response)
        else:
            st.write(':red[Por favor, introduzca texto antes de ejecutar la aplicación]')

            

if len(resultado):
    st.title('Resumen')
    st.info(response)

    
#=============================== Copy Right ===============================
st.text("""""")
st.text("""""")
st.text("""""")
st.text("""""")
st.text("""""")
st.write("""
### © Autor: Joaquín Alavés Sempere
	""")

