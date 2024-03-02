import os
from langchain.docstore.document import Document
from langchain_community.document_loaders import PyPDFLoader

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain.chains.summarize import load_summarize_chain


import streamlit as st

from env import GOOGLE_KEY

st.title('Summarize chapter')

os.environ['GOOGLE_API_KEY'] = GOOGLE_KEY

loader = PyPDFLoader("https://ncert.nic.in/textbook/pdf/hemh103.pdf")
docs = loader.load()

gemini_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

llm = ChatGoogleGenerativeAI(temperature=0, model="gemini-pro")
chain = load_summarize_chain(llm, chain_type="stuff")

st.write(chain.run(docs))

