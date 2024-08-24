import streamlit as st
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
from prepare_database import prepare_db
from invoke_llm import invoke_llm

load_dotenv()
# print(f"KEY = {os.environ.get('GROQ_API_KEY')}")

llm = ChatGroq(model="llama3-8b-8192")
# Pass the directory, load the data, index and store in vector db then return the retriever of that vector store.

retriever = prepare_db('database')


# retrieved_docs = retriever.invoke("How many units in the property are sold?")
# print(retrieved_docs)

st.title("Housing Market Assistant")
usr_inp = st.text_input("Enter your question about the property/project:", "")

if st.button("Submit"):
    try:
        answer = invoke_llm(retriever=retriever,llm=llm,query=usr_inp)
        st.write(answer)
    except Exception as e:
        st.write(f"An error occurred: {e}")








