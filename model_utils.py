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
    
    elif model_name == "distilgpt2":
        try:
            return pipeline(
                "text-generation",
                model="distilgpt2"  # No need for tokenizer parameter, it's automatically loaded
            )
        except Exception as e:
            st.error(f"Error loading DistilGPT-2 model: {e}")
            return pipeline("text2text-generation", model="google/flan-t5-small")

    elif model_name == "distilbert-qa":
        try:
            return pipeline(
                "question-answering",
                model="distilbert-base-cased-distilled-squad"
            )
        except Exception as e:
            st.error(f"Error loading DistilBERT model: {e}")
            return pipeline("text2text-generation", model="google/flan-t5-small")

def generate_response(prompt, model_name="flan-t5-small", max_length=256):
    """Generate a response using the selected language model."""
    loaded = load_model(model_name)
    
    try:
        if model_name == "flan-t5-small":
            # T5 models use max_length and directly return the answer
            response = loaded(prompt, max_length=max_length)[0]["generated_text"].strip()
        
        elif model_name == "distilgpt2":
            # Format prompt for GPT models
            formatted_prompt = f"Question: {prompt}\n\nAnswer:"
            
            outputs = loaded(
                formatted_prompt,
                max_new_tokens=max_length,
                do_sample=True,
                temperature=0.7
            )
            
            # Extract just the generated text
            full_response = outputs[0]["generated_text"]
            
            # Remove the prompt to get just the answer
            if "Answer:" in full_response:
                response = full_response.split("Answer:")[1].strip()
            else:
                response = full_response.replace(formatted_prompt, "").strip()
        
        elif model_name == "distilbert-qa":
            # Extract the question from the prompt
            question = ""
            if "Question:" in prompt:
                question = prompt.split("Question:")[-1].strip()
            else:
                # Get the last sentence as the question
                prompt_parts = prompt.split("\n")
                for part in reversed(prompt_parts):
                    if part.strip():
                        question = part.strip()
                        break
            
            # For QA models, we need to separate the context and question
            qa_result = loaded(
                question=question,
                context=prompt
            )
            
            response = qa_result["answer"]
            
        else:
            response = "Model not recognized"
            
    except Exception as e:
        response = f"Error generating response: {str(e)}"
    
    if not response:
        return "I couldn't generate a response based on the provided information."
    
    return response