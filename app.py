import os
import shutil
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from streamlit.runtime.uploaded_file_manager import UploadedFile
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

chatHistory = []

def list_files_in_folder(folder_path):
    """
    Lists all files in the specified folder.
    Returns a list of file names.
    """
    files = []
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        for item in os.listdir(folder_path):
            item_path = os.path.join(folder_path, item)
            if os.path.isfile(item_path):
                files.append(item)
    return files


def clean_previous_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_storage_dir = os.path.join(current_dir, "documents")
    db_dir = os.path.join(current_dir, "db")
    persistent_directory = os.path.join(db_dir, "chroma_db_for_rag")
    
    # st.info("Cleaning up previous data...")
    if os.path.exists(pdf_storage_dir):
        try:
            shutil.rmtree(pdf_storage_dir)
            # st.success(f"Removed previous PDF documents from {pdf_storage_dir}")
        except OSError as e:
            # st.error(f"Error removing documents directory {pdf_storage_dir}: {e}")
            st.error("Something went wrong")
    
    if os.path.exists(pdf_storage_dir):
        try:
            shutil.rmtree(persistent_directory)
            # st.success(f"Removed previous vector database from {persistent_directory}")
        except OSError as e:
            st.error("Something went wrong")
            # st.error(f"Error removing Chroma DB directory {persistent_directory}: {e}")
    
    
    os.makedirs(pdf_storage_dir, exist_ok=True)
    os.makedirs(persistent_directory, exist_ok=True)

def process_document(uploaded_file: UploadedFile) -> list[Document]:
    
    clean_previous_data()
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_storage_dir = os.path.join(current_dir, "documents")
    os.makedirs(pdf_storage_dir, exist_ok=True)
    saved_file_path = os.path.join(pdf_storage_dir, uploaded_file.name)
    
    with open(saved_file_path, mode='wb') as w:
        w.write(uploaded_file.read())

    # if os.path.exists(saved_file_path):
    #     st.success(f'File {uploaded_file.name} is successfully saved!')

    loader = PyPDFLoader(saved_file_path)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 200,
        separators=["\n\n", ".", "?", "!", " ", ""]
    )

    return text_splitter.split_documents(docs)

def get_vector_collection(all_splits: list[Document]) -> Chroma:
    gemini_ef = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    db_dir = os.path.join(current_dir, "db")
    persistent_directory = os.path.join(db_dir, "chroma_db_for_rag")
    
    collection_name = "rag_collection"

    chroma_client = Chroma.from_documents(
        collection_name=collection_name,
        embedding=gemini_ef,
        documents=all_splits,
        persist_directory=persistent_directory,
        collection_metadata={"hnsw:space": "cosine"}
    )
    
    st.success("Document processed successfully")
    
    return chroma_client


def retrive_relevant_doc(db_client:Chroma, query:str):
    retriver = db_client.as_retriever(
        search_type="similarity",
        search_kwargs={"k":10}
    )
    
    relevant_docs = retriver.invoke(query)
    return relevant_docs

def rag_chain(db_client:Chroma, query:str):
    combined_input = f"""
        Here are some documents that might help answer the question:
        {query}

        Relevant Documents:
        {"\n\n".join([doc.page_content for doc in retrive_relevant_doc(db_client, query)])}

        Please provide a detail answer (Please do not reference to any page) based only on the provided documents.
        Also use {chatHistory}
        If the answer is not found in documents, respond with 'I'm not sure'.
    """

    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
    messages = [
        SystemMessage(content="You are a helpful assistant"),
        HumanMessage(content=combined_input)
    ]

    chatHistory.append(messages)

    prompt_template = ChatPromptTemplate.from_messages(messages)

    chain = prompt_template | model | StrOutputParser()

    result = chain.invoke({"query":query, "chatHistory":chatHistory} )

    chatHistory.append(AIMessage(content=result))
    return result
    
    


if __name__ == '__main__':
    with st.sidebar:
        st.set_page_config(page_title= "RAG QnA BOT")
        st.header("RAG Question and Answer")
        
        uploaded_file = st.file_uploader(
            "   Upload PDF files for Chat about it (one pdf at a tine)   ",
            type=[".pdf"], 
            accept_multiple_files=False
        )

        process = st.button(
            "Process"
        )

        if process:
            with st.spinner("Processing"):
                all_splits = process_document(uploaded_file)
                st.session_state.db_client = get_vector_collection(all_splits)
      

    
    query = st.text_input("Write questions here")
    ask_question = st.button("Ask Questions")     
    
    if ask_question:
        if "db_client" in st.session_state and st.session_state.db_client is not None:
            with st.spinner("Fetching Answer"):
                st.write(rag_chain(st.session_state.db_client, query))
        
        else:
            st.error("Please process a document first by uploading a PDF and clicking 'Process'.")