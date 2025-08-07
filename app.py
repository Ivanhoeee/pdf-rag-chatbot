import streamlit as st
from transformers import pipeline
from rag_utils import extract_text_from_pdf, chunk_text, embed_chunks, create_faiss_index, retrieve_similar_chunks

st.set_page_config(page_title="📚 Chat with PDF", layout="wide")
st.title("📚 Free RAG Chatbot with Memory")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "chunks" not in st.session_state:
    st.session_state.chunks = None
    st.session_state.index = None

@st.cache_resource
def load_model():
    return pipeline(
        "text2text-generation",  # This is the correct task for T5 models
        model="google/flan-t5-small",
        tokenizer="google/flan-t5-small",
    )

llm = load_model()


uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
if uploaded_file:
    with st.spinner("Processing PDF..."):
        st.text("Extracting text...")
        text = extract_text_from_pdf(uploaded_file)
        st.text("Chunking text...")
        chunks = chunk_text(text)
        st.text(f"Created {len(chunks)} chunks")
        st.text(f"Generating embeddings...")
        embeddings = embed_chunks(chunks)
        st.text("Building search index...")
        index = create_faiss_index(embeddings)
        st.session_state.chunks = chunks
        st.session_state.index = index
    st.success("PDF processed and indexed!")

user_input = st.text_input("Ask a question:")

if user_input and st.session_state.chunks:
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).markdown(msg["content"])

    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.spinner("Thinking..."):
        retrieved = retrieve_similar_chunks(user_input, st.session_state.chunks, st.session_state.index)
        context = "\n\n".join(retrieved)
        prompt = f"""Based on the following information, please answer the question.
        Information:
        {context}

        Question: {user_input}"""

        # Add a placeholder for streaming response
        response_placeholder = st.empty()
        
        # Show that something is happening
        response_placeholder.markdown("Generating response...")
        
        # Generate response - T5 models use max_length instead of max_new_tokens
        # and they directly output the answer, not including the prompt
        response = llm(prompt, max_length=100)[0]["generated_text"].strip()
        
        # If empty response, provide a fallback
        if not response:
            response = "I couldn't generate a response based on the provided information."
        
        # Update placeholder with final response
        response_placeholder.empty()

    st.chat_message("assistant").markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})

    # Add at the end of your file
st.markdown("---")
st.markdown("### PDF RAG Chatbot | Created by [Ivan Novakovic](https://github.com/Ivanhoeee/pdf-rag-chatbot)")