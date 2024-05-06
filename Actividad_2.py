# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# import streamlit as st
# from streamlit.logger import get_logger

# LOGGER = get_logger(__name__)


# def run():
    #st.set_page_config(
        #page_title="Hello",
        #page_icon="👋",
    #)

    #st.write("# Welcome to Streamlit! 👋")

    #st.sidebar.success("Select a demo above.")

    #st.markdown(
        #"""
        #Streamlit is an open-source app framework built specifically for
        #Machine Learning and Data Science projects.
        #**👈 Select a demo from the sidebar** to see some examples
        #of what Streamlit can do!
        ### Want to learn more?
        #- Check out [streamlit.io](https://streamlit.io)
        #- Jump into our [documentation](https://docs.streamlit.io)
        #- Ask a question in our [community
        #  forums](https://discuss.streamlit.io)
        ### See more complex demos
        #- Use a neural net to [analyze the Udacity Self-driving Car Image
         # Dataset](https://github.com/streamlit/demo-self-driving)
        #- Explore a [New York City rideshare dataset](https://github.com/streamlit/demo-uber-nyc-pickups)
    #"""
    #)


#if __name__ == "__main__":
    #run()

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
        #model=r"/workspaces/actividad-2-tdaabd/Actividad_2/llama-2-7b-chat.ggmlv3.q2_K.bin",
	model=f"https://github.com/jalaves/actividad-2-tdaabd/blob/main/Actividad_2/llama-2-7b-chat.ggmlv3.q2_K.bin",
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

