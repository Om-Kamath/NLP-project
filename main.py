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
import ast
import time



#############################################
# Setup for API and Database

# create the open-source embedding function
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# load the vectors for similarity search
db2 = Chroma(persist_directory="./chroma_db", embedding_function=embedding_function)

# Fetching the csv for program description
df = pd.read_csv("academic_routes_final.csv")

# Gemini Pro API Key
API_KEY = API.API_KEY

# Creating an instance of the LangChain class
llm = ChatGoogleGenerativeAI(model="gemini-pro",google_api_key=API_KEY)
# agent = create_csv_agent(llm,"academic_routes_final.csv",verbose=True)

###########################################
# APP BEGINS HERE

# Title of the app
st.title("Career Navigator")
st.header("Tell us about yourself")


# User input
user_profile = st.text_input("What are you currently pursuing?", "I am doing a degree in Computer Science.")
course_recommendation = st.text_area("What are you looking out for?", "I like to code and work with data.")
slider = st.slider("How many courses would you like to see?", 1, 10, 3)
courses = db2.similarity_search(course_recommendation, k=slider)
st.divider()

# Display the courses fetched using similarity search
st.markdown("### Here are some courses that you might be interested in:")
for course in courses:
    course_name = course.page_content.split(":")[0]
    description = df[df['Program'] == course_name]['Description'].values[0]
    st.markdown(f'##### {course_name}')
    st.markdown(f'<p>{description}</p>', unsafe_allow_html=True)


# Prompt Template
full_template = """User Data: {user_profile}

User Request: {course_recommendation}

Courses: {courses}

Based on the information provided, analyse the user's profile into the following categories:
1. Technical
2. Creative
3. Analytical

You need to provide scores out of 10 for each of the categories as a array.

For eg.
[8, 5, 7]
"""

full_prompt = PromptTemplate.from_template(full_template)
prompt = full_prompt.format(user_profile=user_profile, course_recommendation=course_recommendation, courses=courses)

data = llm.invoke(prompt).content
data = ast.literal_eval(data)



technical_bar = st.progress(0, text="Technical")

for percent_complete in range(data[0]*10):
    time.sleep(0.01)
    technical_bar.progress(percent_complete + 1, text="Technical")


creative_bar = st.progress(0, text="Creative")

for percent_complete in range(data[1]*10):
    time.sleep(0.01)
    creative_bar.progress(percent_complete + 1, text="Creative")



analytical_bar = st.progress(0, text="Analytical")

for percent_complete in range(data[2]*10):
    time.sleep(0.01)
    analytical_bar.progress(percent_complete + 1, text="Analytical")

