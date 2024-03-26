# Basic Streamlit app to connect to langchain

import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
import pandas as pd
import API

# create the open-source embedding function
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")



db2 = Chroma(persist_directory="./chroma_db", embedding_function=embedding_function)

API_KEY = API.API_KEY
# Creating an instance of the LangChain class
llm = ChatGoogleGenerativeAI(model="gemini-pro",google_api_key=API_KEY)
agent = create_csv_agent(llm,"academic_routes_final.csv",verbose=True)


# Title of the app
st.title("Career Navigator")


st.header("Ask about Educational Prospects")

course_recommendation = st.text_area("What are you looking out for?", "I like to code and work with data.")
slider = st.slider("How many courses would you like to see?", 1, 10, 3)
courses = db2.similarity_search(course_recommendation, k=slider)


st.write("Here are some courses that you might be interested in:")


for course in courses:
    course_name = course.page_content.split(":")[0]
    st.markdown(f'* {course_name}')

for course in courses:
    course_name = course.page_content.split(":")[0]
    st.markdown(f'## What is covered in {course_name}?')
    response = agent.run(f"Find the Description for Program Name:'{course_name}' in string format.")
    st.markdown(f"<p>{response}</p>", unsafe_allow_html=True)


data = []
# Create table using for loop in streamlit
for course in courses:
    course_name = course.page_content.split(":")[0]
    subject_areas = agent.run(f"Find the Subject Areas for Program Name: '{course_name}' in string format.")
    data.append({"Course Name": course_name, "Subject Areas": subject_areas})

df = pd.DataFrame(data)

st.table(df)


