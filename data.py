from langchain_community.vectorstores import Chroma
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain_community.document_loaders import TextLoader


# Load the data for academics

loader_careers = TextLoader("converted_text/ONET.txt")

documents = loader_careers.load()

text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_overlap=0,
    is_separator_regex=False,
    chunk_size=300,
)

docs = text_splitter.split_documents(documents)
print(len(docs))
print("Documents split successfully")
print(docs[0])

# create simple ids
ids = [str(i) for i in range(1, len(docs) + 1)]

embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma.from_documents(docs, embedding_function, persist_directory="./chroma_db_jobs", ids=ids)

print("Data loaded successfully")

loader_skills = TextLoader("converted_text/ONET_tech_skills.txt")
documents_skills = loader_skills.load()

docs_skills = text_splitter.split_documents(documents_skills)
print(len(docs_skills))
print("Documents split successfully")
print(docs_skills[0])

# create simple ids
ids_skills = [str(i) for i in range(1, len(docs_skills) + 1)]

db_skills = Chroma.from_documents(docs_skills, embedding_function, persist_directory="./chroma_db_skills", ids=ids_skills)

print("Data loaded successfully")


loader_academic = TextLoader("converted_text/merged_data_academic.txt")
documents_academic = loader_academic.load()

docs_academic = text_splitter.split_documents(documents_academic)
print(len(docs_academic))
print("Documents split successfully")
print(docs_academic[0])

# create simple ids
ids_academic = [str(i) for i in range(1, len(docs_academic) + 1)]

db_skills = Chroma.from_documents(docs_academic, embedding_function, persist_directory="./chroma_db_academic", ids=ids_academic)

print("Data loaded successfully")
