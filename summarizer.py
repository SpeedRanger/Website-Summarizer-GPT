import dotenv

dotenv.load_dotenv()
import streamlit as st
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import WebBaseLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

st.title("Summarizer")

def get_text():
    input_text = st.text_input("Type in the governance proposal below", key="input")
    return input_text 

user_input = get_text()

if(user_input):
    loader = WebBaseLoader(user_input)
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 0)
    all_splits = text_splitter.split_documents(data)