from langchain_community.vectorstores import Chroma
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain_community.document_loaders import TextLoader


# Load the data for Careers

# loader = TextLoader("merged_data.txt")

# documents = loader.load()

# text_splitter = CharacterTextSplitter(
#     separator="\n",
#     chunk_overlap=0,
#     is_separator_regex=False,
#     chunk_size=300,
# )

# docs = text_splitter.split_documents(documents)
# print(len(docs))
# print("Documents split successfully")
# print(docs[0])

# embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# db = Chroma.from_documents(docs, embedding_function, persist_directory="./chroma_db")
# print("Data loaded successfully")



# Load the data for academics

loader = TextLoader("merged_data_academic.txt")

documents = loader.load()

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
db = Chroma.from_documents(docs, embedding_function, persist_directory="./chroma_db", ids=ids)



print("Data loaded successfully")