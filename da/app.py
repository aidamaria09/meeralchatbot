from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
import os
import base64
import base64
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
 

def main():
   my_secret_value = st.secrets["OPENAI_API_KEY"]
    
   
    st.set_page_config(
        page_title="Meeralchat",
        page_icon="logo.png", 
        layout="centered",
        initial_sidebar_state="auto",
    )

    
    script_directory = os.path.dirname(os.path.abspath(__file__))

     
    add_bg_from_local('Struct.png') 
    
    
    header_container = st.container()

    
    with header_container:
        st.header("Meeralchat")
  

    pdf_path = "data.pdf"
    
    pdf_reader = PdfReader(pdf_path)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
        
    
    char_text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000,
                                                chunk_overlap=200,length_function=len)
    text_chunks = char_text_splitter.split_text(text)
    
    
    embeddings = OpenAIEmbeddings()
    docsearch = FAISS.from_texts(text_chunks, embeddings) 
    llm = OpenAI() 
    chain = load_qa_chain(llm, chain_type="stuff")
    
    st.text("This is an online chatbot made by the MeeralRobotics team.\nIf you are having questions regarding the manual, this is the perfect place to be.\n")
    query = st.text_input("Ask a question about the Game Manual Part 2:")
    if query:
        docs = docsearch.similarity_search(query)
        response = chain.run(input_documents=docs, question=query)
           
        st.write(response)

if __name__ == '__main__':
    main()
