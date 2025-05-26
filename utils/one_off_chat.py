# utils/one_off_chat.py

import requests
import argparse
import os
import logging
import time

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def get_response(prompt, model_name="google/flan-t5-base", api_key=None):
    """
    Get a response from the model
    
    Args:
        prompt: The prompt to send to the model
        model_name: Name of the model to use
        api_key: API key for authentication (optional for some models)
        
    Returns:
        The model's response
    """
    # TODO: Implement the get_response function
    # Set up the API URL and headers
    # Create a payload with the prompt
    # Send the payload to the API
    # Extract and return the generated text from the response
    # Handle any errors that might occur

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    payload = {"inputs": prompt}
    api_url = f"https://api-inference.huggingface.co/models/{model_name}"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "inputs": prompt
    }

    #response = requests.post(api_url, headers=headers, json=payload)
    try: 
        input_txt = payload.get("inputs", "")

        inputs = tokenizer(input_txt, return_tensors="pt")

        outputs = model.generate(**inputs)

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        return response
        
    except Exception as e:
        logging.error(f"Error generating response: {str(e)}")
        return "I'm sorry, I couldn't generate a response due to technical difficulties."


def run_chat():
    """Run an interactive chat session"""
    print("Welcome to the Simple LLM Chat! Type 'exit' to quit.")
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
            
        # TODO: Get response from the model
        # Print the response
        response = get_response(user_input)
        print(response)
        
def main():
    parser = argparse.ArgumentParser(description="Chat with an LLM")
    # TODO: Add arguments to the parser
    parser.add_argument('--model', type=str, default="google/flan-t5-base", help='Model name or path to load')
    
    args = parser.parse_args()
    
    # TODO: Run the chat function with parsed arguments
    run_chat()
    
if __name__ == "__main__":
    main()