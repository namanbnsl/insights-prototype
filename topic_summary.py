import os
from langchain import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

import streamlit as st

from env import GOOGLE_KEY

st.title('Summarize Topic')

os.environ['GOOGLE_API_KEY'] = GOOGLE_KEY 

loader = PyPDFLoader("https://ncert.nic.in/textbook/pdf/hemh103.pdf")
docs = loader.load()

gemini_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

vectorstore = Chroma.from_documents(
                     documents=docs,                 
                     embedding=gemini_embeddings,    
                     persist_directory="./chroma_db"
                     )

vectorstore_disk = Chroma(
                        persist_directory="./chroma_db",     
                        embedding_function=gemini_embeddings 
                   )

retriever = vectorstore_disk.as_retriever()


llm = ChatGoogleGenerativeAI(model="gemini-pro",
                 temperature=0, top_p=0)

llm_prompt_template = """
  Summarize the content given to you based on the topic.
  \n
  Topis: {question} \n Context: {context} \n Result: 
"""



llm_prompt = PromptTemplate.from_template(llm_prompt_template)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | llm_prompt
    | llm
    | StrOutputParser()
)

topics = st.text_input('Summarize Topics')

if topics:  
  st.write(rag_chain.invoke(topics))
