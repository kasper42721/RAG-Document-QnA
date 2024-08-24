from langchain_community.document_loaders import PyMuPDFLoader,CSVLoader,TextLoader,Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
import os

def load_documents(directory):
    documents = []
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if filename.endswith('.txt'):
            loader = TextLoader(file_path)
        elif filename.endswith('.pdf'):
            loader = PyMuPDFLoader(file_path)
        elif filename.endswith('.csv'):
            loader = CSVLoader(file_path)
        else:
            continue
        documents.extend(loader.load())

    # # Splitting into chunks
    # text_splitter = RecursiveCharacterTextSplitter(
    #     chunk_size=4000, chunk_overlap=500, add_start_index=True
    # )
    # chunked_docs = text_splitter.split_documents(documents)
    return documents


def index_to_vectordb(documents):
    # Embed the document
    embeddings = HuggingFaceEmbeddings()
    try:
        print("Loading existing Vector DB")
        # Vector Database
        vector_store = Chroma(
            collection_name="vector-db",
            embedding_function=embeddings,
            persist_directory="vectordb"
        )
        vector_store.add_documents(documents=documents)
    except:
        print("Creating New Vector Store")
        vector_store = Chroma(
            collection_name="vector-db",
            embedding_function=embeddings,
            persist_directory="vectordb"
        )
        vector_store.add_documents(documents=documents)
    
    return vector_store

def prepare_db(directory):
    docs = load_documents(directory)
    vector_store = index_to_vectordb(docs)
    return vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 6})