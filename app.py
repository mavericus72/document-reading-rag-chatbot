import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
from transformers import T5Tokenizer, T5ForConditionalGeneration

st.title("Insurance Policy RAG Chatbot")

uploaded_file = st.file_uploader(
    "Upload Insurance Policy PDF",
    type="pdf"
)

if uploaded_file:

    # Read PDF
    reader = PdfReader(uploaded_file)

    text = ""

    for page in reader.pages:
        text += page.extract_text()

    # Chunking
    chunk_size = 500

    chunks = [
        text[i:i + chunk_size]
        for i in range(0, len(text), chunk_size)
    ]

    # Remove empty chunks
    chunks = [chunk for chunk in chunks if chunk.strip()]

    st.write("Total chunks:", len(chunks))

    if len(chunks) == 0:
        st.error("No text could be extracted from the PDF.")
        st.stop()

    # Embeddings
    model = SentenceTransformer('all-MiniLM-L6-v2')

    embeddings = model.encode(chunks)

    embeddings = np.array(embeddings).astype('float32')

    st.write("Embedding shape:", embeddings.shape)

    # FAISS
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(embeddings)

    # LLM
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")

    model_llm = T5ForConditionalGeneration.from_pretrained(
        "google/flan-t5-base"
    )

    # User query
    query = st.text_input("Ask a question about the policy")

    if query:

        # Query embedding
        query_embedding = model.encode([query])

        query_embedding = np.array(query_embedding).astype('float32')

        # Search
        k = 3

        distances, indices = index.search(query_embedding, k)

        # Retrieved context
        retrieved_text = ""

        for i in indices[0]:
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

        # Generate response
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True)

        outputs = model_llm.generate(
            **inputs,
            max_new_tokens=1500
        )

        answer = tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )

        st.subheader("Answer")

        st.write(answer)
