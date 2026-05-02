# 🧠 Insurance Policy RAG Chatbot

A simple **Retrieval-Augmented Generation (RAG)** chatbot built using **Streamlit** that allows users to upload an insurance policy PDF and ask questions about it.

---

## 🚀 Features

* 📄 Upload insurance policy PDFs
* ✂️ Automatic text chunking
* 🔍 Semantic search using embeddings
* ⚡ Fast retrieval with FAISS
* 🤖 AI-generated answers using an LLM
* 🌐 Interactive UI with Streamlit

---

## 🏗️ Tech Stack

* **Frontend/UI**: streamlit
* **PDF Processing**: pypdf
* **Embeddings**: sentence-transformers
* **Vector Search**: faiss-cpu
* **Similarity**: scikit-learn / numpy
* **LLM**: transformers (Flan-T5)

---

## 📂 How It Works

1. Upload a PDF
2. Extract text from the document
3. Split text into smaller chunks
4. Convert chunks into embeddings
5. Store embeddings in FAISS index
6. User asks a question
7. Convert query into embedding
8. Retrieve top relevant chunks
9. Pass context + query to LLM
10. Generate final answer

---

## ▶️ Run the App

```bash
streamlit run app.py
```

---

## 💡 Example Usage

* Upload an insurance policy PDF
* Ask:

  * "Does this policy cover flood damage?"
  * "What is included in engine protection?"
* Get AI-generated answers based on document content

---

## Screenshot

<img width="1364" height="720" alt="Capture" src="https://github.com/user-attachments/assets/a4a1ccbf-f3f0-4a24-bf24-a4f08b609eeb" />

---

## ⚙️ Key Components

### 📌 Embedding Model

* `all-MiniLM-L6-v2`
* Converts text into vectors

### 📌 Vector Store

* FAISS (`IndexFlatL2`)
* Stores and retrieves similar chunks

### 📌 LLM

* `google/flan-t5-base`
* Generates answers using retrieved context

---

## ⚠️ Limitations

* Depends on PDF text quality
* Small LLM may produce less accurate answers  
* No persistent storage (resets on reload)

---

## 🧠 One-Line Summary

**Upload → Retrieve → Generate → Answer**

---

## 🙌 Acknowledgements

* Hugging Face Transformers
* Sentence Transformers
* FAISS by Meta

---
