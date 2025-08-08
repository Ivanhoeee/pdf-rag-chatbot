from transformers import pipeline
import streamlit as st

@st.cache_resource
def load_model():
    """
    Load and cache the language model for text generation.
    
    Returns:
    --------
    transformers.Pipeline
        A text generation pipeline with the loaded model
    """
    return pipeline(
        "text2text-generation",
        model="google/flan-t5-small",
        tokenizer="google/flan-t5-small",
    )

def generate_response(prompt, max_length=100):
    """
    Generate a response using the language model.
    
    Parameters:
    -----------
    prompt : str
        The prompt to send to the model
    max_length : int, default=100
        Maximum length of the generated response
        
    Returns:
    --------
    str
        The generated response text
    """
    model = load_model()
    response = model(prompt, max_length=max_length)[0]["generated_text"].strip()
    
    if not response:
        return "I couldn't generate a response based on the provided information."
    
    return response