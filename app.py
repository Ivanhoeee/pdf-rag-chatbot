import streamlit as st
from rag_utils import extract_text_from_pdf, chunk_text, embed_chunks, create_faiss_index, retrieve_similar_chunks
from model_utils import generate_response
from ui_components import display_header, display_chat_messages, display_rag_process, display_footer, display_embedding_visualization_with_question, display_model_selector

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

# Display model selector in sidebar
selected_model = display_model_selector()
if selected_model != st.session_state.selected_model:
    # Clear model from cache when switching
    st.session_state.selected_model = selected_model
    st.rerun()  # Rerun to apply model change

# Update the educational notice in the header if needed
if st.session_state.selected_model != "flan-t5-small":
    st.markdown(f"""
    <div style="background-color:#222f3b; padding: 10px; border-radius: 4px;">
    ⚠️ <b>Educational Tool Notice:</b> This app uses {selected_model} to demonstrate RAG concepts.
    The focus is on learning about RAG architecture and semantic retrieval.
    </div>
    """, unsafe_allow_html=True)

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
    user_input = st.text_input("Ask a question:")

    if user_input and st.session_state.chunks:
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
            response = generate_response(prompt, model_name=st.session_state.selected_model)
            response_placeholder.empty()
        
        # Add assistant message
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Display messages
        display_chat_messages(st.session_state.messages)

with tab2:
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
        
        # Add search functionality
        st.subheader("Search Visualization")
        search_query = st.text_input("Highlight chunks containing:", key="viz_search")
        
        if search_query:
            st.write(f"Highlighting chunks containing: '{search_query}'")
            
            # Filter chunks that contain the search query
            matching_indices = [i for i, chunk in enumerate(st.session_state.chunks) 
                              if search_query.lower() in chunk.lower()]
            
            if matching_indices:
                from rag_utils import visualize_embeddings_with_highlights
                fig = visualize_embeddings_with_highlights(
                    st.session_state.chunks, 
                    st.session_state.embeddings,
                    matching_indices,
                    highlight_label="Matches"
                )
                st.pyplot(fig)
                st.write(f"Found {len(matching_indices)} chunks containing '{search_query}'")
            else:
                st.write(f"No chunks found containing '{search_query}'")
# Display footer
display_footer()