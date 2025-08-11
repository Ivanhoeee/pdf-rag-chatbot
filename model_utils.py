import streamlit as st
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch


@st.cache_resource
def load_model(model_name="flan-t5-small"):
    """
    Load and cache the selected language model.
    """
    if model_name == "flan-t5-small":
        return pipeline(
            "text2text-generation",
            model="google/flan-t5-small",
            tokenizer="google/flan-t5-small",
        )
    
    elif model_name == "phi-1.5":  # Replacing Qwen
        try:
            return pipeline(
                "text-generation",
                model="microsoft/phi-1_5",
                tokenizer="microsoft/phi-1_5",
                trust_remote_code=True
            )
        except Exception as e:
            st.error(f"Error loading Phi-1.5 model: {e}")
            return pipeline("text2text-generation", model="google/flan-t5-small")

    elif model_name == "tinyllama":  # Replacing Llama
        try:
            return pipeline(
                "text-generation",
                model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", 
                tokenizer="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            )
        except Exception as e:
            st.error(f"Error loading TinyLlama model: {e}")
            return pipeline("text2text-generation", model="google/flan-t5-small")

def generate_response(prompt, model_name="flan-t5-small", max_length=256):
    """
    Generate a response using the selected language model.
    
    Parameters:
    -----------
    prompt : str
        The prompt to send to the model
    model_name : str
        The model to use for generation
    max_length : int, default=256
        Maximum length of the generated response
        
    Returns:
    --------
    str
        The generated response text
    """
    loaded = load_model(model_name)
    
    try:
        if model_name == "flan-t5-small":
            # T5 models use max_length and directly return the answer
            response = loaded(prompt, max_length=max_length)[0]["generated_text"].strip()
        
        elif model_name == "phi-1.5":
            # Format prompt for Phi models
            formatted_prompt = f"Based on this context: {prompt}\n\nResponse:"
            
            outputs = loaded(
                formatted_prompt,
                max_new_tokens=max_length,
                do_sample=True,
                temperature=0.7
            )
            
            # Extract just the generated text
            full_response = outputs[0]["generated_text"]
            
            # Remove the prompt to get just the answer
            if "Response:" in full_response:
                response = full_response.split("Response:")[1].strip()
            else:
                response = full_response.replace(formatted_prompt, "").strip()
            
        elif model_name == "tinyllama":
            # Format prompt for TinyLlama
            formatted_prompt = f"<|user|>\n{prompt}\n<|assistant|>"
            
            outputs = loaded(
                formatted_prompt,
                max_new_tokens=max_length,
                do_sample=True,
                temperature=0.7
            )
            
            # Extract just the assistant's reply
            full_response = outputs[0]["generated_text"]
            
            if "<|assistant|>" in full_response:
                response = full_response.split("<|assistant|>")[1].strip()
            else:
                response = full_response.replace(formatted_prompt, "").strip()
            
        else:
            response = "Model not recognized"
            
    except Exception as e:
        response = f"Error generating response: {str(e)}"
    
    if not response:
        return "I couldn't generate a response based on the provided information."
    
    return response