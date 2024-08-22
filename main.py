from langchain_groq import ChatGroq
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()
print(f"KEY = {os.environ.get('GROQ_API_KEY')}")

llm = ChatGroq(model="llama3-8b-8192")


# Loading the documents
loader = TextLoader('story.txt',encoding='UTF-8')
docs = loader.load()

# Splitting into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
chunked_docs = text_splitter.split_documents(docs)

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
except:
    print("Creating New Vector Store")
    vector_store = Chroma(
        collection_name="vector-db",
        embedding_function=embeddings,
        persist_directory="vectordb"
    )
    vector_store.add_documents(documents=chunked_docs)

# Query the docs
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 1})
# retrieved_docs = retriever.invoke("Who is the protagonist of the story?")

system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

response = rag_chain.invoke({"input": "Who is the protagonist of the story?"})
print(response["answer"])
# print(f"A = {type(out)}")
# print(f"B = {out}")
