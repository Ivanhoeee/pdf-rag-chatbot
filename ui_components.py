import streamlit as st

def display_header():
    """Display the app header and disclaimer."""
    st.set_page_config(
        page_title="📚 RAG Explorer",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Custom CSS for better spacing and colors
    st.markdown("""
    <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .stButton>button {
            color: white;
            border: none;
            border-radius: 4px;
            padding: 0.5rem 1rem;
        }
        .stTextInput>div>div>input {
            border-radius: 4px;
        }
        h1 {
            color: #FF4B4B;
        }
        .stAlert {
            border-radius: 4px;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Two-column layout for title and logo
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("📚 RAG Explorer")
        st.markdown("### Understand your documents through retrieval-augmented generation")
    
    # More subtle info message
    st.markdown("""
    <div style="background-color:#222f3b; padding: 10px; border-radius: 4px;">
    ⚠️ <b>Educational Tool Notice:</b> This app uses a small language model (flan-t5-small) to demonstrate RAG concepts.
    The focus is on learning about RAG architecture and semantic retrieval.
    </div>
    """, unsafe_allow_html=True)

def display_chat_messages(messages, reverse=True):
    """
    Display chat messages in the UI with improved styling.
    
    Parameters:
    -----------
    messages : list
        List of message dictionaries with 'role' and 'content' keys
    reverse : bool, default=True
        If True, display newest messages first
    """
    # Custom CSS for messages
    st.markdown("""
    <style>
        .user-message {
            background-color: #262730;
            padding: 15px;
            border-radius: 10px;
            margin: 10px 0;
        }
        .assistant-message {
            background-color: #303239;
            padding: 15px;
            border-radius: 10px;
            margin: 10px 0;
            border-left: 4px solid #FF4B4B;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Clear previous messages
    st.empty()
    
    # Display messages with improved styling
    message_list = reversed(messages) if reverse else messages
    for msg in message_list:
        if msg["role"] == "user":
            st.markdown(f"<div class='user-message'><b>You:</b><br>{msg['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='assistant-message'><b>Assistant:</b><br>{msg['content']}</div>", unsafe_allow_html=True)


def display_rag_process(user_input, chunks, index, retrieved, context, prompt):
    """Display the RAG process details in an expandable section."""
    with st.expander("🔍 Click here for RAG Process Details", expanded=False):
        st.write("**Step 1: Converting question to embedding vector**")
        st.write("Your question is transformed into a numerical vector using the sentence transformer model.")
        
        st.write("**Step 2: Semantic search in document chunks**")
        st.write(f"Found {len(retrieved)} relevant passages in the document.")
        
        st.write("**Step 3: Building context from retrieved chunks**")
        st.write(f"Combined context length: {len(context)} characters")
        
        st.write("**Step 4: Creating prompt for the language model**")
        st.write(f"Prompt: `{prompt}`")


def display_footer():
    """Display a more professional footer."""
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div style="text-align: center;">
            <p>
                <span style="color: #FF4B4B; font-weight: bold;">RAG Explorer</span> |
                Built by <a href="https://github.com/Ivanhoeee" style="color: #FF4B4B;">Ivan Novakovic</a> |
                <a href="https://github.com/Ivanhoeee/pdf-rag-chatbot" style="color: #FF4B4B;">GitHub Repo</a>
            </p>
        </div>
        """, unsafe_allow_html=True)


def display_embedding_visualization(chunks, embeddings):
    """
    Display a visualization of document embeddings in 2D space.
    
    Parameters:
    -----------
    chunks : list of str
        The text chunks from the document
    embeddings : numpy.ndarray
        The embeddings corresponding to the chunks
    """
    from rag_utils import visualize_embeddings
    
    st.write("""
    This visualization shows how different parts of the document relate to each other semantically.
    Points that are close together contain similar content or topics.
    """)
    
    # Add visualization controls
    col1, col2 = st.columns([3, 1])
    with col2:
        st.subheader("Visualization Options")
        label_count = st.slider("Number of labeled chunks:", 0, 10, 5)
        point_size = st.slider("Point size:", 10, 100, 30)
        
        if st.button("Refresh Visualization"):
            st.session_state.refresh_viz = True
    
    with col1:
        # Create and display the visualization with user-selected parameters
        fig = visualize_embeddings_with_user_settings(chunks, embeddings, label_count, point_size)
        st.pyplot(fig)


def visualize_embeddings_with_user_settings(chunks, embeddings, label_count=5, point_size=30):
    """
    Create a customized embedding visualization based on user settings.
    
    Parameters:
    -----------
    chunks : list of str
        The text chunks
    embeddings : numpy.ndarray
        The embeddings to visualize
    label_count : int
        Number of chunks to label
    point_size : int
        Size of the scatter plot points
    
    Returns:
    --------
    matplotlib.figure.Figure
        The customized visualization
    """
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    
    # Reduce dimensions for visualization
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(embeddings)
    
    # Plot in 2D
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], s=point_size)
    
    # Add labels for selected number of points
    for i, chunk in enumerate(chunks[:label_count]):
        ax.annotate(chunk[:50] + "...", 
                  (reduced_embeddings[i, 0], reduced_embeddings[i, 1]))
    
    ax.set_title("Document Embedding Space Visualization")
    plt.tight_layout()
    return fig

def display_embedding_visualization_with_question(chunks, embeddings, question=None):
    """
    Display a visualization of document embeddings with the option to show a user question.
    
    Parameters:
    -----------
    chunks : list of str
        The text chunks from the document
    embeddings : numpy.ndarray
        The embeddings corresponding to the chunks
    question : str, default=None
        The user's question to visualize alongside chunks
    """
    from sentence_transformers import SentenceTransformer
    
    st.header("Document and Question Embedding Visualization")

    # Add the concise explanation in an info box
    st.info("""
    **How This Visualization Works:**
    1. **Document chunks** (blue dots) are converted into vectors that capture their meaning
    2. **Your question** (red star) is also converted to a vector
    3. **PCA** reduces these high-dimensional vectors to 2D points you can see
    4. **Proximity** between points indicates semantic similarity
    5. **Closest chunks** to your question (highlighted) are the ones used to generate answers
    """)

    st.write("""
    This visualization shows how different parts of the document semantically relate to each other and to the asked question.
    Points that are close together contain similar content or topics.
    """)
    
    # Add visualization controls
    col1, col2 = st.columns([3, 1])
    with col2:
        st.subheader("Visualization Options")
        label_count = st.slider("Number of labeled chunks:", 0, 10, 5)
        # point_size = st.slider("Point size:", 10, 100, 30)
        
        # Option to show last question
        show_question = st.checkbox("Show last question", value=True if question else False)
        
        if st.button("Refresh Visualization"):
            st.session_state.refresh_viz = True
    
    with col1:
        # Only include question if checkbox is selected and question exists
        active_question = question if show_question and question else None
        
        # Create and display the visualization with user-selected parameters
        fig = visualize_embeddings_with_question(
            chunks, embeddings, question=active_question, 
            label_count=label_count
        )
        st.pyplot(fig)


def visualize_embeddings_with_question(chunks, embeddings, question=None, label_count=5, point_size=30):
    """
    Create a visualization of embeddings that includes a user question.
    
    Parameters:
    -----------
    chunks : list of str
        The text chunks
    embeddings : numpy.ndarray
        The embeddings to visualize
    question : str, default=None
        User question to include in the visualization
    label_count : int, default=5
        Number of chunks to label
    point_size : int, default=30
        Size of the scatter plot points
    
    Returns:
    --------
    matplotlib.figure.Figure
        The customized visualization
    """
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    import numpy as np
    from sentence_transformers import SentenceTransformer
    
    # If we have a question, get its embedding and include it
    if question:
        embedder = SentenceTransformer("all-MiniLM-L6-v2")
        question_embedding = embedder.encode([question])
        
        # Combine document and question embeddings
        combined_embeddings = np.vstack([embeddings, question_embedding])
        
        # Reduce dimensions for all embeddings together
        pca = PCA(n_components=2)
        reduced_embeddings = pca.fit_transform(combined_embeddings)
        
        # Split back into document and question parts
        reduced_doc_embeddings = reduced_embeddings[:-1]
        reduced_question_embedding = reduced_embeddings[-1].reshape(1, -1)
    else:
        # Just use document embeddings if no question
        pca = PCA(n_components=2)
        reduced_doc_embeddings = pca.fit_transform(embeddings)
    
    # Plot in 2D
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Plot document chunks
    ax.scatter(
        reduced_doc_embeddings[:, 0], 
        reduced_doc_embeddings[:, 1], 
        s=point_size,
        color='blue',
        alpha=0.7,
        label='Document chunks'
    )
    
    # Add labels for selected document chunks
    for i, chunk in enumerate(chunks[:label_count]):
        ax.annotate(
            chunk[:30] + "...", 
            (reduced_doc_embeddings[i, 0], reduced_doc_embeddings[i, 1]),
            alpha=0.8
        )
    
    # Add question point and label if provided
    if question:
        ax.scatter(
            reduced_question_embedding[:, 0],
            reduced_question_embedding[:, 1],
            s=point_size*2,
            color='red',
            marker='*',
            label='Your question'
        )
        
        # Add question label
        ax.annotate(
            f"Q: {question[:30]}{'...' if len(question) > 50 else ''}", 
            (reduced_question_embedding[0, 0], reduced_question_embedding[0, 1]),
            color='darkred',
            weight='bold'
        )
        
        # Show the nearest chunks
        # Calculate distances in the 2D space
        distances = np.linalg.norm(reduced_doc_embeddings - reduced_question_embedding, axis=1)
        closest_indices = np.argsort(distances)[:1]  # Get 3 closest chunks
        
        # Highlight the closest chunks
        for idx in closest_indices:
            ax.scatter(
                reduced_doc_embeddings[idx, 0],
                reduced_doc_embeddings[idx, 1],
                s=point_size*2.5,
                color='blue',
                marker='o',
                alpha=0.8
            )
    # Set min and max limits for better visibility
    min_x = reduced_doc_embeddings[:, 0].min() 
    max_x = reduced_doc_embeddings[:, 0].max()
    min_y = reduced_doc_embeddings[:, 1].min()
    max_y = reduced_doc_embeddings[:, 1].max()
    x_range = max_x - min_x
    y_range = max_y - min_y

    # Add 10% padding on each side
    ax.set_xlim(min_x - 0.1 * x_range, max_x + 0.1 * x_range)
    ax.set_ylim(min_y - 0.1 * y_range, max_y + 0.1 * y_range)

    ax.set_title("Document and Question Embedding Visualization", pad=20)
    ax.legend()
    plt.tight_layout()
    return fig


def display_model_selector():
    """Display model selection in sidebar and return the selected model name."""
    st.sidebar.header("Model Settings")
    
    model_options = {
        "flan-t5-small": {
            "name": "Flan-T5 Small",
            "description": "Small but fast T5 model (~300M parameters)",
            "memory": "~500MB"
        },
        "qwen-1.5b": {
            "name": "Qwen 2.5 1.5B Instruct",
            "description": "High quality Chinese-origin model with good instruction following",
            "memory": "~1.1GB" 
        },
        "llama-3.2-1b": {
            "name": "Llama 3.2 1B Instruct",
            "description": "Meta's latest compact model with strong capabilities",
            "memory": "~1GB"
        }
    }
    
    # Model selection
    selected_model = st.sidebar.selectbox(
        "Choose Language Model:",
        options=list(model_options.keys()),
        format_func=lambda x: model_options[x]["name"],
        index=0
    )
    
    # Display model info
    st.sidebar.markdown(f"**Description:** {model_options[selected_model]['description']}")
    st.sidebar.markdown(f"**Memory Usage:** {model_options[selected_model]['memory']}")
    
    # Memory warning
    if selected_model != "flan-t5-small":
        st.sidebar.warning("⚠️ First generation with this model may take 15-20 seconds to load.")
    
    return selected_model