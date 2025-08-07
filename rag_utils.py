import fitz  # PyMuPDF
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import os
import re

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


def retrieve_similar_chunks(query, chunks, index, k=3):  # Reduced from 4 to 3
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
    return embedder.encode(chunks, show_progress_bar=False)

def create_faiss_index(embeddings):
    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    return index

# def retrieve_similar_chunks(query, chunks, index, k=4):
# NOTE: old code
#     query_embedding = embedder.encode([query])
#     distances, indices = index.search(np.array(query_embedding), k)
#     return [chunks[i] for i in indices[0]]
