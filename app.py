import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import T5Tokenizer, T5ForConditionalGeneration

st.title("Insurance Policy RAG Chatbot")

# Load models once (CRITICAL for Hugging Face)
@st.cache_resource
def load_models():
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
    llm_model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")
    return embed_model, tokenizer, llm_model

model, tokenizer, model_llm = load_models()

uploaded_file = st.file_uploader(
    "Upload Insurance Policy PDF",
    type="pdf"
)

if uploaded_file:

    # Extract text safely
    reader = PdfReader(uploaded_file)
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text

    if not text.strip():
        st.error("No readable text found in PDF.")
        st.stop()

    # Chunking
    chunk_size = 500
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    chunks = [chunk for chunk in chunks if chunk.strip()]

    st.write("Total chunks:", len(chunks))

    # Cache embeddings (prevents recomputation)
    @st.cache_data
    def get_embeddings(chunks):
        embeddings = model.encode(chunks)
        return np.array(embeddings).astype('float32')

    embeddings = get_embeddings(chunks)

    # FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    # User query
    query = st.text_input("Ask a question about the policy")

    if query:

        query_embedding = model.encode([query])
        query_embedding = np.array(query_embedding).astype('float32')

        # Search
        k = 3
        distances, indices = index.search(query_embedding, k)

        # Retrieve context
        retrieved_text = ""
        for i in indices[0]:
            if i < len(chunks):
                retrieved_text += chunks[i] + "\n"

        # Prompt
        prompt = f"""
Answer the question using the context below.

Context:
{retrieved_text}

Question:
{query}

Answer:
"""

        # Lightweight generation settings
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True)

        outputs = model_llm.generate(
            **inputs,
            max_new_tokens=200
        )

        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

        st.subheader("Answer")
        st.write(answer)
