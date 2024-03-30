# Basic Streamlit app to connect to langchain

import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain_community.document_loaders import PyPDFLoader
from duckduckgo_search import DDGS
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
import pandas as pd
import API
import ast
import time
import os



#############################################
# Templates
metrics_template = """
Current Qualification: {degree}
Course: {course}
Description: {description}

Based on the information provided, analyse the course into the following categories:
1. Technical
2. Creative
3. Analytical
4. Learning Curve 

To calculate the learning curve, consider the current qualification and then compare it with the course description.
The learning curve will be higher if the course is different from the current qualification. Like, if an engineer wants to become a gardener,
it will be a higher learning curve (around 8-9). If an engineer wants to become a data scientist, it will be a lower learning curve (around 2-3).

You need to provide scores out of 10 for each of the categories as a array in string ONLY.

For example
[Input]
Current Qualification: Mechanical Engineering
Course: Computer Science
Description: This course is about coding and working with data.

[Output]
[8, 5, 7, 5]
"""

resume_template = """
Act Like a skilled or very experience ATS(Application Tracking System)
with a deep understanding of all fields. Your task is fetch only the professional skills and the type of job the candidate is looking out 
for based on the introduction (if not provided, give it your best guess) from the text provided and format it in a structured way.
Text: {resume_text}
"""

resume_skill_gap = """
List out the skills missing by calculating the Skills Required - Skills Acquired. Provide descriptions of the same.
Skills Required and type of job: {skills_required}
Skills Acquired: {skills_provided}
"""
#############################################
# Functions

def ocr(item):
    model = ocr_predictor("db_resnet50", "crnn_vgg16_bn", pretrained=True)
    result = model(item)
    json_output = result.export()
    return result, json_output




#############################################
# Setup for API and Database

# create the open-source embedding function
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# load the vectors for similarity search
db_jobs = Chroma(persist_directory="./chroma_db_jobs", embedding_function=embedding_function)
db_skills = Chroma(persist_directory="./chroma_db_skills", embedding_function=embedding_function)

# Fetching the csv for program description
df = pd.read_excel("main_data/Occupation Data.xlsx")

# Uncomment for GEMINI
API_KEY = API.API_KEY_GEMINI
llm = ChatGoogleGenerativeAI(model="gemini-pro",google_api_key=API_KEY)

# Uncomment for OpenAI
# API_KEY = API.API_KEY_OPENAI
# llm= ChatOpenAI(api_key=API.API_OPENAI_KEY,model="gpt-3.5-turbo")

###########################################
# APP BEGINS HERE

# Sidebar
option = st.sidebar.selectbox(
    'Select an option',
    ['Course Recommendations', 'Resume Analyser']
)


if option == 'Course Recommendations':
    # Title of the app
    st.title("Career Navigator")
    st.header("Course Recommendations")


    # User input
    user_profile = st.text_input("What are you currently pursuing?", "I am doing a degree in Computer Science.")
    course_recommendation = st.text_area("What are you looking out for?", "I like to code and work with data.")
    slider = st.slider("How many courses would you like to see?", 1, 10, 3)
    submit = st.button("Submit")
    st.divider()

    courses_list = []
    if submit:
        # Display the courses fetched using similarity search with profiling using bars
        with st.spinner("Analyzing your preferences..."):
            courses = db_jobs.similarity_search(course_recommendation, k=slider)
            st.markdown("### Course Recommendations")
            for course in courses:
                course_name = course.page_content.split(":")[0]
                courses_list.append(course_name)
                description = df[df['Title'] == course_name]['Description'].values[0]
                st.markdown(f'##### {course_name}')
                st.markdown(f'<p>{description}</p>', unsafe_allow_html=True)



                full_prompt = PromptTemplate.from_template(metrics_template)
                prompt = full_prompt.format(degree=user_profile,course=course_name, description=description)

                data = llm.invoke(prompt).content
                print(data)
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

                learning_curve_bar = st.progress(0, text="Learning Curve")

                for percent_complete in range(data[3]*10):
                    time.sleep(0.01)
                    learning_curve_bar.progress(percent_complete + 1, text="Learning Curve",)
            st.divider()

            # Personalised Resources
            st.markdown("### Personalised Resources")
            resources = DDGS().text(f"Courses for {courses_list}", max_results=10)
            search = pd.DataFrame(resources)
            for i in range(5):
                href = search['href'][i].replace(" ","%20")
                st.markdown(f"[{search['title'][i]}]({href})")

            st.divider()
            # Roadmap
            st.markdown("### Roadmap")

elif option == 'Resume Analyser':
    st.title("Career Navigator")
    st.header("Resume Analyser")

    st.write("Upload your resume to get started:")
    uploaded_file = st.file_uploader("Choose a file", type=["pdf"])
    
    whole_words = []
    per_line_words = []
    if uploaded_file is not None:
        with st.spinner("Analyzing your resume..."):
            pdf = uploaded_file.read()
            single_img_doc = DocumentFile.from_pdf(pdf)
            result, json_output = ocr(single_img_doc)
            for block in json_output["pages"][0]["blocks"]:
                for line in block["lines"]:
                    line_words = []
                    for word in line["words"]:
                        whole_words.append(word["value"])
                        line_words.append(word["value"])
                    per_line_words.append(line_words)


        pdf_content = ""
        for line in per_line_words:
            pdf_content += " ".join(line) + "\n"

        with st.spinner("Churning out the results..."):
            full_prompt = PromptTemplate.from_template(resume_template)
            prompt = full_prompt.format(resume_text=pdf_content)
            data = llm.invoke(prompt).content
            # st.write(data)
            skills = db_skills.max_marginal_relevance_search(data, k=5, fetch_k=15)
            # skills = db_skills.as_retriever(search_type='mmr').get_relevant_documents(data)[:5]
            # st.write(skills)
            skill_gap = PromptTemplate.from_template(resume_skill_gap)
            skill_gap = skill_gap.format(skills_required=skills, skills_provided=data)
            data_skills = llm.invoke(skill_gap).content
            st.write(data_skills)
