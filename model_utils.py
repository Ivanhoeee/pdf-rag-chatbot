from transformers import pipeline
import streamlit as st
import os
import time

@st.cache_resource
def load_model(model_name="flan-t5-small"):
    """
    Load and cache the language model for text generation.
    
    Parameters:
    -----------
    model_name : str
        The model to use ("flan-t5-small", "distilgpt2", "openai-gpt35", "openai-gpt4o")
        
    Returns:
    --------
    model
        The loaded model or client
    """
    if "openai" in model_name and not st.session_state.get("openai_api_key"):
        st.error("Please enter your OpenAI API key in the sidebar to use OpenAI models.")
        return None
        
    if model_name == "flan-t5-small":
        return pipeline(
            "text2text-generation",
            model="google/flan-t5-small",
            tokenizer="google/flan-t5-small",
        )
    
    # Update the OpenAI model loading section:

    elif model_name in ["openai-gpt35", "openai-gpt4o"]:
        try:
            from openai import OpenAI
            
            # Initialize the OpenAI client
            client = OpenAI(api_key=st.session_state.get("openai_api_key"))
            
            # Test the connection - updated for newer OpenAI API
            try:
                # The newer API doesn't use limit parameter
                models = client.models.list()
                # Just check if we can access the list
                if not models.data:
                    st.error("Could not retrieve models from OpenAI API")
                    return None
            except Exception as e:
                st.error(f"Error connecting to OpenAI API: {e}")
                return None
                
            return client
        except Exception as e:
            st.error(f"Error initializing OpenAI client: {e}")
            return None

def generate_response(prompt, model_name="flan-t5-small", max_length=256):
    """
    Generate a response using the selected language model.
    
    Parameters:
    -----------
    prompt : str
        The prompt to send to the model
    model_name : str
        The model to use
    max_length : int, default=256
        Maximum length of the generated response
        
    Returns:
    --------
    str
        The generated response text
    """
    model = load_model(model_name)
    
    if model is None:
        return "Please configure your API key before using this model."
    
    try:
        if model_name == "flan-t5-small":
            response = model(prompt, max_length=max_length)[0]["generated_text"].strip()
        
        elif model_name == "openai-gpt35":
            # Track token usage for cost display
            start_time = time.time()
            
            completion = model.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_length,
                temperature=0.7
            )
            
            response = completion.choices[0].message.content
            
            # Add usage information
            elapsed = time.time() - start_time
            tokens_used = completion.usage.total_tokens
            estimated_cost = tokens_used * 0.000002  # $0.002 per 1000 tokens
            
            response += f"\n\n*API call completed in {elapsed:.2f}s using {tokens_used} tokens (est. cost: ${estimated_cost:.6f})*"
            
        elif model_name == "openai-gpt4o":
            # Track token usage for cost display
            start_time = time.time()
            
            completion = model.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_length,
                temperature=0.7
            )
            
            response = completion.choices[0].message.content
            
            # Add usage information
            elapsed = time.time() - start_time
            tokens_used = completion.usage.total_tokens
            estimated_cost = tokens_used * 0.00001  # $0.01 per 1000 tokens
            
            response += f"\n\n*API call completed in {elapsed:.2f}s using {tokens_used} tokens (est. cost: ${estimated_cost:.6f})*"
            
    except Exception as e:
        response = f"Error generating response: {str(e)}"
    
    if not response:
        return "I couldn't generate a response based on the provided information."
    
    return response