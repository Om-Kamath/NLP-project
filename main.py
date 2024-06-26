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
from googlesearch import search
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
import pandas as pd
import API
# import ast
import time
# import os
import spacy
from spacy import displacy
import contractions
import specializations_lower
from spacy.matcher import PhraseMatcher
from spacy.tokens import Span

# stExpanderDetails
#############################################
# CSS styling
st.markdown(
    """
    <style>
    [data-testid="stExpander"] div[data-testid="stExpanderDetails"] {
        overflow-x: auto;
    }
    </style>
    """
,unsafe_allow_html=True
)


#############################################
# Templates

rephrase_template = """
Rephrase the following text such that it has a realistic tone
"The degrees offered are: {degrees}"
Output should be provided as is without any extra text. Headings should be bold and description should start on new line.
"""

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

You need to provide scores out of 10 for each of the categories. The datatype of response should be an array of integers only.

For example
#Input
Current Qualification: Mechanical Engineering
Course: Computer Science
Description: This course is about coding and working with data.

#Output
[8, 5, 7, 5]
"""

resume_template = """
Act Like a skilled or very experience ATS(Application Tracking System)
with a deep understanding of all fields. Your task is fetch only the 'Professional Skills' and the 'Job Type' the candidate is looking out 
for based on the introduction from the text provided and format it in a structured way. If the 'Job Type' is not specified in the text, give it your best guess based on the skills. Output should always contain 'Professional Skills' and 'Job Type'. There should be a single 'Job Type' only.
Text: {resume_text}
"""

resume_skill_gap = """
**Prompt:**

Imagine you're an adept Application Tracking System (ATS) with comprehensive knowledge across various fields. Your task is to identify missing skills required for a specific job or career by comparing the skills listed as necessary with those possessed by the candidate. You'll need to list out the missing skills by subtracting the acquired skills from the required skills and provide their names, descriptions, and examples of usage. Provide only the output.

**Example:**

*Input:*

Required skills and type of job: 
Skills: [Typescript, MongoDB, React]
Job Type: Web Developer

Acquired Skills of Candidate: [Bootstrap, CSS, Typescript]

*Output:*

| Missing Skill | Description                                               |
|---------------|-----------------------------------------------------------|
| MongoDB       | A document-oriented NoSQL database used in web development for storing and retrieving data efficiently.                                                                                                   |
| React         | A JavaScript library for building user interfaces, widely used for creating interactive UI components in web applications.                                                   |

*Input:*

Required skills: {required_skills}
Job Type: {job_type}
Acquired skills of candidate: {acquired_skills}

*Output:*

| Missing Skill | Description                                               |
|---------------|-----------------------------------------------------------|
| (Missing Skills) | (Description) |


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
db_academic = Chroma(persist_directory="./chroma_db_academic", embedding_function=embedding_function)

# Fetching the csv for program description
df = pd.read_excel("main_data/Occupation Data.xlsx")

# Uncomment for GEMINI
API_KEY = API.API_KEY_GEMINI
llm = ChatGoogleGenerativeAI(model="gemini-pro",google_api_key=API_KEY)

# Uncomment for OpenAI
# API_KEY = API.API_KEY_OPENAI
# llm= ChatOpenAI(api_key=API.API_KEY_OPENAI,model="gpt-3.5-turbo")

# Spacy Setup
nlp = spacy.load('en_core_web_sm')

# Load a blank spaCy NER model
custom_nlp = spacy.blank("en")

# Initialize a PhraseMatcher with the academic qualifications
matcher = PhraseMatcher(custom_nlp.vocab, attr="LOWER")
academic_qualifications = specializations_lower.majors
patterns = [nlp(qualification) for qualification in academic_qualifications]
print(patterns)
matcher.add("ACADEMIC_QUALIFICATION", None, *patterns)

###########################################
# APP BEGINS HERE

# Sidebar
option = st.sidebar.selectbox(
    'Select an option',
    ['Home', 'Resume Analyser']
)


if option == 'Home':
    # Title of the app
    st.title("Academic Navigator")


    # User input
    user_profile = st.text_input("What are you currently pursuing?", "I am doing a degree in Computer Science.")
    course_recommendation = st.text_area("What is your liking?", "I like to code and work with data.")
    slider = st.slider("How many courses would you like to see?", 1, 10, 3)
    submit = st.button("Submit")
    st.divider()

    courses_list = []
    if submit:
        with st.status("Preprocessing User Data..."):
            user_profile = contractions.fix(user_profile)
            st.markdown(f"""**Expanding contractions:**   
                        {user_profile}""")
            doc_user_profile = custom_nlp(user_profile)
            doc_user_profile_nlp = nlp(user_profile)
            course_recommendation = contractions.fix(course_recommendation)
            st.markdown(f"""**Expanding contractions:**   
                        {course_recommendation}""")
            doc_course_recommendation = nlp(course_recommendation)

            # Generate HTML for the named entity visualization
            html = displacy.render(doc_user_profile, style='ent')
            matches = matcher(doc_user_profile)

            for match_id, start, end in matches:
                span = Span(doc_user_profile, start, end, label=match_id)
                doc_user_profile.ents = list(doc_user_profile.ents) + [span]  # add span to doc.ents
            
            html = displacy.render(doc_user_profile, style='ent')
            # Display the HTML in the Streamlit app
            st.markdown(html, unsafe_allow_html=True)

            # Generate HTML for the named entity visualization for course_recommendation
            html = displacy.render(doc_course_recommendation, style='dep')
            st.markdown(html, unsafe_allow_html=True)

            # Perform tokenization using spacy
            tokens_user_profile = [token.text for token in doc_user_profile]
            st.write("Tokens in user profile:", tokens_user_profile)

            tokens_course_recommendation = [token.text for token in doc_course_recommendation]
            st.write("Tokens in course recommendation:", tokens_course_recommendation)

            # Perform tokenization using spacy on user_profile, remove stop words and handle whitespace
            tokens_user_profile = [token.text for token in doc_user_profile if not token.is_stop and not token.is_space]
            st.write("Removing Stopwords from user profile:", tokens_user_profile)

            # Perform tokenization using spacy on course_recommendation, remove stop words and handle whitespace
            tokens_course_recommendation = [token.text for token in doc_course_recommendation if not token.is_stop and not token.is_space]
            st.write("Removing Stopwords from course recommendation:", tokens_course_recommendation)

            #Convert to lowercase
            tokens_user_profile = [token.lower_ for token in doc_user_profile if not token.is_stop and not token.is_space]
            st.write("Lowercase tokens in user profile:", tokens_user_profile)

            # Perform tokenization using spacy on course_recommendation, remove stop words, handle whitespace, and convert to lowercase
            tokens_course_recommendation = [token.lower_ for token in doc_course_recommendation if not token.is_stop and not token.is_space]
            st.write("Lowercase tokens in course recommendation:", tokens_course_recommendation)

            # Perform lemmatization on user_profile
            lemmas_user_profile = [token.lemma_ for token in doc_user_profile_nlp if not token.is_stop and not token.is_space]
            st.write("Lemmas in user profile:", lemmas_user_profile)

            # Perform lemmatization on course_recommendation
            lemmas_course_recommendation = [token.lemma_ for token in doc_course_recommendation if not token.is_stop and not token.is_space]
            st.write("Lemmas in course recommendation:", lemmas_course_recommendation)


        # Display the courses fetched using similarity search with profiling using bars
        with st.spinner("Analyzing your preferences..."):
            courses = db_jobs.similarity_search(course_recommendation, k=slider)
            for course in courses:
                course_name = course.page_content.split(":")[0]
                # courses_list.append(course_name)
                description = df[df['Title'] == course_name]['Description'].values[0]
                st.markdown(f'##### {course_name}')
                st.markdown(f'<p>{description}</p>', unsafe_allow_html=True)



                full_prompt = PromptTemplate.from_template(metrics_template)
                prompt = full_prompt.format(degree=user_profile,course=course_name, description=description)

                data = llm.invoke(prompt).content
                print(data)
                #print type of data
                print(type(data))
                # data = ast.literal_eval(data)
                data = [int(float(n)) for n in eval(data)]

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

                with st.expander("View Personalized Courses"):
                    resources = search(f"Courses for {course}", num_results=2,advanced=True)
                    for i in resources:
                        if "geeksforgeeks" in i.url:
                            continue
                        st.markdown(f"**[{i.title}]({i.url})**")

                with st.status("View Potential Degrees"):
                    degree_query = f"What should be studied for {course_name}"
                    degrees = db_academic.similarity_search(degree_query, k=2)
                    degrees = [degree.page_content for degree in degrees]
                    rephrase_prompt = PromptTemplate.from_template(rephrase_template)
                    rephrase_text = rephrase_prompt.format(degrees=degrees)
                    data = llm.invoke(rephrase_text).content
                    st.write(data)



            

elif option == 'Resume Analyser':
    st.title("Academic Navigator")
    st.header("Resume Analyser")

    st.write("Upload your resume to get started:")
    uploaded_file = st.file_uploader("Choose a file", type=["pdf"])
    
    if uploaded_file:
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
            
            print("----------PDF CONTENT-----------")
            print(pdf_content)

            with st.spinner("Churning out the results..."):
                full_prompt = PromptTemplate.from_template(resume_template)
                prompt = full_prompt.format(resume_text=pdf_content)
                data = llm.invoke(prompt).content
                sections = data.split("Job Type:")
                skills_section = ""
                job_type_section = ""
                if len(sections) == 2:
                    skills_section, job_type_section = sections
                    acquired_skills = skills_section.strip().split("Professional Skills:")[1]
                    job_type = job_type_section.strip("** \n")
                    job_type_search = f"For roles like {job_type}"
                    st.markdown(f"""#### For roles like:    
{job_type}""")
                    required_skills = db_skills.similarity_search(job_type_search, k=10)
                    required_skills = [item.page_content for item in required_skills]
                    skill_gap = PromptTemplate.from_template(resume_skill_gap)
                    skill_gap = skill_gap.format(required_skills=required_skills, acquired_skills=acquired_skills, job_type=job_type)
                    data_skills = llm.invoke(skill_gap).content
                    st.write(data_skills)
                else:
                    st.warning("Unable to parse text, please try again.", icon="⚠️")
