import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import WebBaseLoader, CSVLoader, JSONLoader, PyPDFLoader

load_dotenv()

def WebsiteLoader(urls):
    """Load documents from websites"""
    if not urls or not any(urls):
        return []
    loader = WebBaseLoader(urls)
    return loader.load()

def CSVFileLoader(file_paths):
    """Load documents from CSV files"""
    docs = []
    for file_path in file_paths:
        if os.path.exists(file_path):
            loader = CSVLoader(file_path=file_path)
            docs.extend(loader.load())
    return docs

def PDFLoader(pdf_files):
    """Load documents from PDF files"""
    docs = []
    for pdf_file in pdf_files:
        if os.path.exists(pdf_file):
            loader = PyPDFLoader(file_path=pdf_file)
            docs.extend(loader.load())
    return docs

def JSONFileLoader(json_files):
    """Load documents from JSON files"""
    docs = []
    for json_file in json_files:
        if os.path.exists(json_file):
            loader = JSONLoader(file_path=json_file)
            docs.extend(loader.load())
    return docs

def create_vector_store_from_files(file_paths, file_types, api_key, persist_directory="./chroma_db"):
    """Create vector store from various file types"""
    all_documents = []
    
    # Load documents based on file types
    for file_path, file_type in zip(file_paths, file_types):
        if file_type.lower() == 'pdf':
            docs = PDFLoader([file_path])
        elif file_type.lower() == 'csv':
            docs = CSVFileLoader([file_path])
        elif file_type.lower() == 'json':
            docs = JSONFileLoader([file_path])
        else:
            print(f"Unsupported file type: {file_type}")
            continue
        all_documents.extend(docs)
    
    if not all_documents:
        print("No documents loaded!")
        return None
    
    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
    )
    
    split_documents = text_splitter.split_documents(all_documents)
    print(f"Split {len(all_documents)} documents into {len(split_documents)} chunks")
    
    # Create embeddings with error handling
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
    except Exception as e:
        print(f"Error creating embeddings: {str(e)}")
        return None
    
    # Create vector store
    vectorstore = Chroma.from_documents(
        documents=split_documents,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    
    # Persist the database
    vectorstore.persist()
    print("✅ Data successfully stored in ChromaDB!")
    return vectorstore

def create_vector_store_from_urls(urls, api_key, persist_directory="./chroma_db"):
    """Create vector store from URLs"""
    if not urls or not any(urls):
        print("No URLs provided!")
        return None
    
    # Load documents from URLs
    all_documents = WebsiteLoader(urls)
    
    if not all_documents:
        print("No documents loaded from URLs!")
        return None
    
    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
    )
    
    split_documents = text_splitter.split_documents(all_documents)
    print(f"Split {len(all_documents)} documents into {len(split_documents)} chunks")
    
    # Create embeddings with error handling
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
    except Exception as e:
        print(f"Error creating embeddings: {str(e)}")
        return None
    
    # Create vector store
    vectorstore = Chroma.from_documents(
        documents=split_documents,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    
    # Persist the database
    vectorstore.persist()
    print("✅ Data successfully stored in ChromaDB!")
    return vectorstore

# Example usage (commented out for modularity)
if __name__ == "__main__":
    
    print("ChromaDB utilities loaded. Use the functions to create vector stores.")