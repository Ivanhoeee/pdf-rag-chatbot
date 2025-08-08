import fitz  # PyMuPDF
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import os
import re
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

embedder = SentenceTransformer("all-MiniLM-L6-v2")

def extract_text_from_pdf(file):
    pdf = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in pdf:
        text += page.get_text()
    return text

def chunk_text(text, chunk_size=300, overlap=30):
    """
    Divide text into semantically coherent chunks based on paragraph structure.
    
    This function serves as the main entry point for text chunking, delegating to
    specialized chunking strategies as needed.
    
    Parameters:
    -----------
    text : str
        The input text to be chunked, typically extracted from a PDF document
    chunk_size : int, default=300
        Maximum number of words per chunk (soft limit that won't split paragraphs)
    overlap : int, default=30
        Number of words to overlap between chunks (used in word-based fallback chunking)
        
    Returns:
    --------
    list of str
        A list of text chunks
    """
    # Try paragraph-based chunking first (our preferred method)
    chunks = _chunk_by_paragraphs(text, chunk_size)
    
    # If paragraph chunking didn't work well, try sentence-based chunking
    if len(chunks) <= 1 and len(text) > 500:
        chunks = _chunk_by_sentences(text, chunk_size)
    
    # As a last resort, use word-based chunking
    if len(chunks) <= 1 and len(text.split()) > chunk_size:
        chunks = _chunk_by_words(text, chunk_size, overlap)
    
    return chunks

def _chunk_by_paragraphs(text, chunk_size):
    """
    Chunk text based on paragraph boundaries.
    
    Paragraphs are identified by double newlines. This method preserves
    the natural flow and context within paragraphs.
    
    Parameters:
    -----------
    text : str
        The input text to be chunked
    chunk_size : int
        Maximum number of words per chunk
        
    Returns:
    --------
    list of str
        A list of paragraph-based text chunks
    """
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = ""
    current_size = 0
    
    for paragraph in paragraphs:
        paragraph_size = len(paragraph.split())
        if current_size + paragraph_size > chunk_size:
            if current_chunk:  # Avoid empty chunks
                chunks.append(current_chunk.strip())
            current_chunk = paragraph
            current_size = paragraph_size
        else:
            current_chunk += "\n\n" + paragraph if current_chunk else paragraph
            current_size += paragraph_size
    
    if current_chunk:  # Add the last chunk
        chunks.append(current_chunk.strip())
    
    return chunks

def _chunk_by_sentences(text, chunk_size):
    """
    Chunk text based on sentence boundaries.
    
    Sentences are identified by common punctuation marks followed by spaces.
    This method is used when paragraph structure is insufficient.
    
    Parameters:
    -----------
    text : str
        The input text to be chunked
    chunk_size : int
        Maximum number of words per chunk
        
    Returns:
    --------
    list of str
        A list of sentence-based text chunks
    """
    
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = ""
    current_size = 0
    
    for sentence in sentences:
        sentence_size = len(sentence.split())
        if current_size + sentence_size > chunk_size:
            if current_chunk:  # Avoid empty chunks
                chunks.append(current_chunk.strip())
            current_chunk = sentence
            current_size = sentence_size
        else:
            current_chunk += " " + sentence if current_chunk else sentence
            current_size += sentence_size
    
    if current_chunk:  # Add the last chunk
        chunks.append(current_chunk.strip())
    
    return chunks

def _chunk_by_words(text, chunk_size, overlap):
    """
    Chunk text based on word count with overlap.
    
    This is a fallback method when other chunking strategies don't produce
    multiple chunks. It enforces chunk size limits by splitting on word boundaries.
    
    Parameters:
    -----------
    text : str
        The input text to be chunked
    chunk_size : int
        Maximum number of words per chunk
    overlap : int
        Number of words to overlap between chunks
        
    Returns:
    --------
    list of str
        A list of word-based text chunks with specified overlap
    """
    chunks = []
    words = text.split()
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    
    return chunks


def retrieve_similar_chunks(query, chunks, index, k=3):
    """
    Find the most semantically similar chunks to a query.
    
    This function converts the user's query to an embedding vector and uses
    the FAISS index to find the k most similar chunks based on vector similarity.
    Duplicate chunks are filtered out to ensure result diversity.
    
    Parameters:
    -----------
    query : str
        The user's question or search query
    chunks : list of str
        All available text chunks from the document
    index : faiss.IndexFlatL2
        The FAISS index built from the chunk embeddings
    k : int, default=3
        Number of similar chunks to retrieve
        
    Returns:
    --------
    list of str
        The most relevant text chunks for answering the query,
        with duplicates removed
    """
    query_embedding = embedder.encode([query])
    distances, indices = index.search(np.array(query_embedding), k)
    
    # Return unique chunks, ordered by relevance
    seen = set()
    results = []
    for i in indices[0]:
        if i not in seen and i < len(chunks):
            results.append(chunks[i])
            seen.add(i)
    return results

def embed_chunks(chunks):
    """
    Convert text chunks to numerical vector embeddings.
    
    This function transforms each text chunk into a dense vector representation
    using a pre-trained Sentence Transformer model (all-MiniLM-L6-v2).
    These embeddings capture the semantic meaning of each chunk in a
    high-dimensional space.
    
    Parameters:
    -----------
    chunks : list of str
        The text chunks to be converted into embeddings
        
    Returns:
    --------
    numpy.ndarray
        An array of embeddings, where each embedding is a dense vector
    """
    return embedder.encode(chunks, show_progress_bar=False)


def create_faiss_index(embeddings):
    """
    Create a FAISS vector index for fast similarity search.
    
    This function builds a FAISS index using the L2 (Euclidean) distance metric
    for efficient nearest-neighbor search among the document chunk embeddings.
    The index enables quick retrieval of similar chunks based on vector similarity.
    
    Parameters:
    -----------
    embeddings : numpy.ndarray
        The embeddings to add to the index
        
    Returns:
    --------
    faiss.IndexFlatL2
        A FAISS index containing the embeddings, ready for similarity search
    """
    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    return index


def visualize_embeddings(chunks, embeddings):
    """
    Visualize text chunk embeddings in a reduced 2D space using PCA.
    
    This function takes high-dimensional text embeddings and projects them onto
    a 2D plane for visualization purposes. It helps to understand the relative
    semantic relationships between different text chunks by observing their
    proximity in the embedding space.
    
    Parameters:
    -----------
    chunks : list of str
        The original text chunks corresponding to the embeddings
    embeddings : numpy.ndarray
        The high-dimensional embeddings to visualize, typically output from 
        the embed_chunks function
        
    Returns:
    --------
    matplotlib.figure.Figure
        A matplotlib figure showing the 2D projection of embeddings with
        the first few chunks labeled
        
    Notes:
    ------
    This visualization uses PCA (Principal Component Analysis) for dimensionality 
    reduction and may not perfectly preserve all semantic relationships from the 
    original high-dimensional space.
    """
    # Reduce dimensions for visualization
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(embeddings)
    
    # Plot in 2D
    plt.figure(figsize=(10, 8))
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1])
    
    # Add labels for some points
    for i, chunk in enumerate(chunks[:5]):  # Label first 5 chunks
        plt.annotate(chunk[:50] + "...", 
                    (reduced_embeddings[i, 0], reduced_embeddings[i, 1]))
    
    plt.title("Embedding Space Visualization")
    return plt
