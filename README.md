# 🤖 RAG Chatbot with File Upload

A **Retrieval-Augmented Generation (RAG)** chatbot that lets you **upload and chat with your own files** — powered by **Streamlit**, **ChromaDB**, and **Google Gemini** models.

---

## 🚀 Features

* 📁 **File Upload** — Supports **PDF**, **CSV**, and **JSON** files
* 🔍 **Vector Search** — Uses **ChromaDB** for efficient document retrieval
* 🧠 **Google Gemini Integration** — Chat using Gemini models:

  * `gemini-2.0-flash`
  * `gemini-1.5-pro`
  * `gemini-1.5-flash`
* 💬 **Interactive Chat** — Chat UI with conversation memory
* 📄 **Source Documents** — View original sources used for responses
* 🔑 **Secure API Key Handling** — Load your Google API key via `.env` file

---

## 🛠️ Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables

1. Copy `env_example.txt` → `.env`
2. Get your Google API key from [Google AI Studio](https://aistudio.google.com/)
3. Add it to your `.env` file:

   ```
   GOOGLE_API_KEY=your_actual_api_key_here
   ```

### 3. Run the Application

```bash
streamlit run main.py
```

> 💡 *This version uses Hugging Face embeddings (Sentence Transformers) for document processing and Google Gemini for responses. This avoids async loop conflicts and delivers fast performance.*

---

## 💡 How to Use

1. **Load API Key** — The app automatically reads your `.env` file
2. **Upload Files** — Use the sidebar to upload your PDFs, CSVs, or JSONs
3. **Process Files** — Click **“🔄 Process Files”** to generate the knowledge base
4. **Start Chatting** — Ask questions about your uploaded content
5. **View Sources** — Expand **“📄 Source Documents”** to see referenced excerpts

---

## 📂 Supported File Types

| File Type | Description                        |
| --------- | ---------------------------------- |
| **PDF**   | Text documents, papers, or reports |
| **CSV**   | Structured tabular data            |
| **JSON**  | Configuration or data files        |

---

## ⚙️ Configuration Options

* **Model Selection** — Choose from Gemini models
* **Context Size** — Adjust how much context the LLM can see
* **Max History** — Control how long chat memory persists
* **API Key** — Automatically loaded from `.env`

---

## 📁 Project Structure

```
├── main.py              # Streamlit app (UI + Chat)
├── chroma_store.py      # File processing + vector database functions
├── test_upload.py       # Test script for file upload
├── requirements.txt     # Dependencies list
└── .env                 # Google API key (user-provided)
```

---

## 🧩 Troubleshooting

| Issue               | Possible Fix                                             |
| ------------------- | -------------------------------------------------------- |
| ❌ Invalid API Key   | Ensure `GOOGLE_API_KEY` is set correctly in `.env`       |
| ⚙️ Missing Packages | Run `pip install -r requirements.txt`                    |
| 📁 ChromaDB Errors  | Verify write permissions for `chroma_db/` directory      |
| 🧠 Model Errors     | Check your Google AI Studio account for Gemini access    |
| 🪵 Debugging        | Check Streamlit console logs for detailed error messages |

---

## 🏁 Summary

This **RAG Chatbot** lets you interact with your own documents using **Google Gemini**.
It combines the retrieval power of **ChromaDB** with the reasoning capability of **Gemini models**, all within a simple, interactive **Streamlit UI**.

---

**Author:** *[Your Name]*
**License:** MIT
**Powered by:** 🧠 Google Gemini • 🗂️ ChromaDB • 🦙 Hugging Face • 💬 Streamlit
