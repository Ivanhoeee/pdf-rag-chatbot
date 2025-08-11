import streamlit as st
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch

@st.cache_resource
def load_model(model_name="flan-t5-small"):
    """
    Load and cache the selected language model.
    
    Parameters:
    -----------
    model_name : str
        The model identifier to load
        
    Returns:
    --------
    model
        The loaded model ready for inference
    """
    if model_name == "flan-t5-small":
        return pipeline(
            "text2text-generation",
            model="google/flan-t5-small",
            tokenizer="google/flan-t5-small",
        )
    
    elif model_name == "qwen-1.5b":
        try:
            # Load Qwen directly using transformers - CPU only version
            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                "Qwen/Qwen2.5-1.5B-Instruct",
                trust_remote_code=True,
                device_map=None,  # Don't use device_map
                torch_dtype=torch.float32  # Use full precision for CPU
            )
            
            # Return both model and tokenizer for Qwen
            return {"model": model, "tokenizer": tokenizer}
        
        except Exception as e:
            st.error(f"Error loading Qwen model: {e}")
            # Fall back to flan-t5-small
            return pipeline(
                "text2text-generation",
                model="google/flan-t5-small", 
                tokenizer="google/flan-t5-small"
            )

    # Also update the llama-3.2-1b section similarly:
    elif model_name == "llama-3.2-1b":
        try:
            # CPU-only version
            tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
            
            # Load model without device_map
            model = AutoModelForCausalLM.from_pretrained(
                "meta-llama/Llama-3.2-1B-Instruct",
                torch_dtype=torch.float32,
                device_map=None
            )
            return {"model": model, "tokenizer": tokenizer}
            
        except Exception as e:
            st.error(f"Error loading Llama model: {e}")
            # Fall back to flan-t5-small
            return pipeline(
                "text2text-generation",
                model="google/flan-t5-small", 
                tokenizer="google/flan-t5-small"
            )

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
        
        elif model_name == "qwen-1.5b":
            # Unpack the model and tokenizer
            model = loaded["model"]
            tokenizer = loaded["tokenizer"]
            
            # Qwen 2.5 uses a different chat format than earlier versions
            # It works with the apply_chat_template method instead of .chat()
            messages = [
                {"role": "system", "content": "You are a helpful assistant that answers based on the provided context."},
                {"role": "user", "content": prompt}
            ]
            
            # Format the prompt using the chat template
            formatted_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Generate response
            inputs = tokenizer(formatted_prompt, return_tensors="pt")
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=max_length,
                temperature=0.7,
                do_sample=True
            )
            
            # Decode the full response
            full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract just the assistant's reply by removing the formatted prompt
            if full_response.startswith(formatted_prompt):
                response = full_response[len(formatted_prompt):].strip()
            else:
                # If the response doesn't contain the exact prompt, try to find the assistant's part
                if "<|assistant|>" in full_response:
                    response = full_response.split("<|assistant|>")[1].strip()
                else:
                    response = full_response
            
        elif model_name == "llama-3.2-1b":
            # Unpack the model and tokenizer
            model = loaded["model"]
            tokenizer = loaded["tokenizer"]
            
            # Format prompt for Llama 3.2 (using the correct chat template)
            # Per documentation, Llama 3.2 uses <|begin_of_text|><|start_header_id|>system<|end_header_id|>
            chat = [
                {"role": "system", "content": "You are a helpful assistant that answers based on the provided context."},
                {"role": "user", "content": prompt}
            ]
            
            # Use tokenizer's apply_chat_template to format correctly
            formatted_prompt = tokenizer.apply_chat_template(
                chat,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Generate response
            inputs = tokenizer(formatted_prompt, return_tensors="pt")
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=max_length,
                temperature=0.7,
                do_sample=True
            )
            
            # Decode the full response and extract only the assistant's reply
            full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # The response will include the original prompt - we need to extract just the answer
            response = full_response[len(formatted_prompt.strip()):]
            
            # If using chat template correctly, the assistant response should be clean
            # If there are any leftover markers, we can clean them up
            if response.startswith("assistant"):
                response = response[len("assistant"):].strip()
            
        else:
            response = "Model not recognized"
            
    except Exception as e:
        response = f"Error generating response: {str(e)}"
    
    if not response:
        return "I couldn't generate a response based on the provided information."
    
    return response