import streamlit as st
import traceback
import os
from dotenv import load_dotenv
from rag_utils import extract_text_from_pdf, chunk_text, embed_chunks, create_faiss_index, retrieve_similar_chunks
from model_utils import generate_response
from ui_components import (
    display_header, display_chat_messages, display_rag_process, 
    display_footer, display_embedding_visualization_with_question, 
    display_model_selector
)

# Load environment variables from .env file
load_dotenv()


try:
    # Setup UI
    display_header()

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chunks" not in st.session_state:
        st.session_state.chunks = None
        st.session_state.index = None
    if "embeddings" not in st.session_state:
        st.session_state.embeddings = None
    if "refresh_viz" not in st.session_state:
        st.session_state.refresh_viz = False
    if "last_question" not in st.session_state:
        st.session_state.last_question = None
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = "flan-t5-small"
    if "openai_api_key" not in st.session_state:
        st.session_state.openai_api_key = ""

    # Display model selector in sidebar
    selected_model = display_model_selector()
    if selected_model != st.session_state.selected_model:
        # Clear model from cache when switching
        st.session_state.selected_model = selected_model
        st.rerun()  # Rerun to apply model change

    # Update the educational notice for API models
    if "openai" in st.session_state.selected_model:
        st.markdown(f"""
        <div style="background-color:#222f3b; padding: 10px; border-radius: 4px;">
        ⚠️ <b>API Usage Notice:</b> This app is using {selected_model} via the OpenAI API.
        API calls will be charged to your account based on token usage.
        </div>
        """, unsafe_allow_html=True)
    
    # Rest of your existing app.py code...
    # Create tabs for different app functions
    tab1, tab2 = st.tabs(["Chat with PDF", "Visualize Embeddings"])
    
    with tab1:
        # File upload and processing
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
                st.session_state.embeddings = embeddings
                st.session_state.index = index
            st.success("PDF processed and indexed!")

        # Chat interface
        with st.form("question_form"):
            user_input = st.text_input("Ask a question:")
            submitted = st.form_submit_button("Submit")

        if submitted and user_input and st.session_state.chunks:
            # Save the question for visualization
            st.session_state.last_question = user_input
            
            # Add user message
            st.session_state.messages.append({"role": "user", "content": user_input})
            
            with st.spinner("Thinking..."):
                # Retrieve relevant chunks
                retrieved = retrieve_similar_chunks(user_input, st.session_state.chunks, st.session_state.index)
                context = "\n\n".join(retrieved)
                
                # Create prompt
                prompt = f"""Based on the following information, please answer the question.
                Information:
                {context}

                Question: {user_input}"""
                
                # Show RAG process
                display_rag_process(user_input, st.session_state.chunks, st.session_state.index, retrieved, context, prompt)
                
                # Generate response using the selected model
                response_placeholder = st.empty()
                response_placeholder.markdown("Generating response...")
                response = generate_response(
                    prompt, 
                    model_name=st.session_state.selected_model,
                    temperature=st.session_state.get('temperature', 0.7)
                )
                response_placeholder.empty()
            
            # Add assistant message
            st.session_state.messages.append({"role": "assistant", "content": response})
            
            # Display messages
            display_chat_messages(st.session_state.messages)

    with tab2:
        # Your existing tab2 code...
        st.header("Embedding Visualization")
        
        if st.session_state.chunks is None or st.session_state.embeddings is None:
            st.info("Please upload and process a PDF first to visualize its embeddings.")
        else:
            # Display the embedding visualization with the last question
            display_embedding_visualization_with_question(
                st.session_state.chunks, 
                st.session_state.embeddings,
                st.session_state.last_question
            )

    # Display footer
    display_footer()

except Exception as e:
    st.error(f"Application error: {str(e)}")
    st.code(traceback.format_exc())
    st.warning("If this error persists, try using only the flan-t5-small model or contact the developer.")
