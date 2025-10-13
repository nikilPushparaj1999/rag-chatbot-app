import streamlit as st
import os
import tempfile
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import WebBaseLoader, CSVLoader, JSONLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

# ---- Global Variables ---- #
api_key = os.getenv("GOOGLE_API_KEY")
MODEL = "gemini-2.0-flash"
llm = None
embeddings = None

# ---- Helper Functions ---- #
def get_llm():
    global llm
    if llm is None and api_key:
        llm = ChatGoogleGenerativeAI(model=MODEL, google_api_key=api_key, streaming=True)
    return llm

def get_embeddings():
    global embeddings
    if embeddings is None:
        try:
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'}
            )
        except Exception as e:
            st.error(f"❌ Error initializing embeddings: {str(e)}")
            return None
    return embeddings

# ---- File Processing ---- #
def process_uploaded_files(uploaded_files):
    all_documents = []
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded_file.name}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        try:
            file_extension = uploaded_file.name.lower().split('.')[-1]
            if file_extension == 'pdf':
                loader = PyPDFLoader(file_path=tmp_file_path)
            elif file_extension == 'csv':
                loader = CSVLoader(file_path=tmp_file_path)
            elif file_extension == 'json':
                loader = JSONLoader(file_path=tmp_file_path)
            else:
                st.error(f"⚠️ Unsupported file type: {file_extension}")
                continue

            docs = loader.load()
            all_documents.extend(docs)
            st.success(f"✅ {uploaded_file.name} processed successfully")
        except Exception as e:
            st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")
        finally:
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)
    return all_documents

def create_vector_store(documents, persist_directory="./chroma_db"):
    if not documents:
        return None

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
    )

    split_documents = text_splitter.split_documents(documents)
    embeddings = get_embeddings()
    if not embeddings:
        st.error("❌ Could not initialize embeddings.")
        return None

    vectorstore = Chroma.from_documents(
        documents=split_documents,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    vectorstore.persist()
    return vectorstore

# ---- Streamlit UI ---- #
st.set_page_config(page_title="📚 RAG Chatbot", layout="wide")

# 💅 Modern Chat UI Styling
st.markdown("""
<style>
/* === General App Background === */
.stApp {
    background-color: #0B0E11; /* Deep dark gray */
    color: #EAEAEA; /* Light text */
}

/* === Sidebar === */
section[data-testid="stSidebar"] {
    background-color: #111418;
    color: #EAEAEA;
    border-right: 1px solid #1F2328;
}
section[data-testid="stSidebar"] h1, h2, h3 {
    color: #66CCFF !important;
}

/* === Chat Area === */
.chat-container {
    margin-top: 10px;
}

/* === User Message Bubble === */
.user-msg {
    background: linear-gradient(135deg, #0DB9D7, #046D8B);
    color: white;
    padding: 12px 16px;
    border-radius: 18px 18px 0px 18px;
    max-width: 80%;
    margin-left: auto;
    margin-bottom: 8px;
    text-align: right;
    box-shadow: 0px 2px 6px rgba(0,0,0,0.3);
}

/* === Bot Message Bubble === */
.bot-msg {
    background: #F7F9FA;
    color: #0B0E11;
    padding: 12px 16px;
    border-radius: 18px 18px 18px 0px;
    max-width: 80%;
    margin-right: auto;
    margin-bottom: 8px;
    text-align: left;
    box-shadow: 0px 2px 6px rgba(0,0,0,0.25);
}

/* === Chat Input === */
div[data-testid="stChatInput"] textarea {
    background-color: #161B22;
    color: #EAEAEA;
    border: 1px solid #2D333B;
    border-radius: 10px;
}

/* === Buttons === */
button {
    background-color: #0DB9D7 !important;
    color: white !important;
    font-weight: 600 !important;
    border-radius: 10px !important;
}
button:hover {
    background-color: #10D1F2 !important;
}

/* === Info Boxes === */
.stAlert {
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)





# ---- Sidebar ---- #
with st.sidebar:
    st.header("⚙️ Configuration")

    MODEL = st.selectbox("Model", ["gemini-2.0-flash", "gemini-1.5-pro", "gemini-1.5-flash"], index=0)
    MAX_HISTORY = st.number_input("Max Chat History", 1, 10, 3)
    CONTEXT_SIZE = st.number_input("Context Size", 1024, 16384, 8192, step=1024)

    st.markdown("---")
    st.header("📁 Upload Files")

    uploaded_files = st.file_uploader(
        "Choose files (PDF, CSV, JSON)",
        type=['pdf', 'csv', 'json'],
        accept_multiple_files=True,
        help="Upload documents to create your knowledge base"
    )

    if st.button("🔄 Process Files", use_container_width=True):
        if not api_key:
            st.warning("Please add GOOGLE_API_KEY in your .env file.")
        elif uploaded_files:
            with st.spinner("Processing files..."):
                documents = process_uploaded_files(uploaded_files)
                if documents:
                    vectorstore = create_vector_store(documents)
                    if vectorstore:
                        st.session_state.vectorstore = vectorstore
                        st.session_state.retriever = vectorstore.as_retriever(search_type="similarity")
                        llm = get_llm()
                        if llm:
                            st.session_state.qa = RetrievalQA.from_chain_type(
                                llm=llm,
                                chain_type="stuff",
                                retriever=st.session_state.retriever,
                                return_source_documents=True
                            )
                            st.success("✅ Knowledge base created successfully!")
                        else:
                            st.error("❌ Could not initialize LLM.")
        else:
            st.warning("Please upload files first.")

# ---- Session State Setup ---- #
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(return_messages=True)

# ---- Chat Section ---- #
col1, col2 = st.columns([2.6, 1.0], gap="medium")

with col1:
    st.subheader("💬 Chat Interface")
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)

    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f"<div class='user-msg'><b>You:</b> {msg['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='bot-msg'><b>Bot:</b> {msg['content']}</div>", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    prompt = st.chat_input("Ask something about your documents...")
    if prompt:
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        st.markdown(f"<div class='user-msg'><b>You:</b> {prompt}</div>", unsafe_allow_html=True)

        if "qa" not in st.session_state:
            st.warning("⚠️ No vector store available. Please upload and process documents first.")
        else:
            with st.spinner("🤔 Thinking..."):
                result = st.session_state.qa({"query": prompt})
                answer = result.get("result", "No response generated.")
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
                st.markdown(f"<div class='bot-msg'><b>Bot:</b> {answer}</div>", unsafe_allow_html=True)

with col2:
    st.subheader("📊 Session Info")
    st.info(f"**Total messages:** {len(st.session_state.chat_history)}")
    if "vectorstore" in st.session_state:
        st.success("✅ Vector Store Loaded")
    else:
        st.warning("⚠️ No Vector Store Yet")

    if api_key:
        st.markdown("🔐 **API Key detected**")
    else:
        st.error("❌ Missing GOOGLE_API_KEY")

    st.markdown("---")
    st.caption("Built with ❤️ using Streamlit + LangChain + Gemini")

