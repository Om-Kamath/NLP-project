# Important Files

- data.ipynb: Converting the csv to LLM format by merging the headers.

- data.py: Converting the .txt from data.ipynb to vectors and loading them to chromadb.

- main.ipynb: Streamlit app and fetching the datastore vectors.


Step 1:
Run data.py to generate chromadb embeddings

Step 2:
streamlit run main.py