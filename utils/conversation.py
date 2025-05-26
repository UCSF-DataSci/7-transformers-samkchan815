# utils/conversation.py

import requests
import argparse
import os
import logging

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def get_response(prompt, history=None, model_name="google/flan-t5-base", api_key=None, history_length=3):
    """
    Get a response from the model using conversation history
    
    Args:
        prompt: The current user prompt
        history: List of previous (prompt, response) tuples
        model_name: Name of the model to use
        api_key: API key for authentication
        history_length: Number of previous exchanges to include in context
        
    Returns:
        The model's response
    """
    # TODO: Implement the contextual response function
    # Initialize history if None

    # TODO: Format a prompt that includes previous exchanges
    # Get a response from the API
    # Return the response

    try:
        # Initialize history if None
        if history is None:
            history = []

        # Format prompt with context
        context = ""
        for user_input, model_response in history[-history_length:]:
            context += f"User: {user_input}\nAssistant: {model_response}\n"
        context += f"User: {prompt}\nAssistant:"

        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        # Tokenize input and generate
        inputs = tokenizer(context, return_tensors="pt", truncation=True)
        outputs = model.generate(**inputs, max_new_tokens=200)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        return response.strip()

    except Exception as e:
        logging.error(f"Error generating response: {str(e)}")
        return "I'm sorry, I couldn't generate a response due to technical difficulties."


def run_chat():
    """Run an interactive chat session with context"""
    print("Welcome to the Contextual LLM Chat! Type 'exit' to quit.")
    
    # Initialize conversation history
    history = []
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
            
        # TODO: Get response using conversation history
        # Update history
        # Print the response
        response = get_response(user_input, history=history)
        print(response)
        print(f"history: {history}")

        # Append to history
        history.append((user_input, response))
        
def main():
    parser = argparse.ArgumentParser(description="Chat with an LLM using conversation history")
    # TODO: Add arguments to the parser
    
    args = parser.parse_args()
    
    # TODO: Run the chat function with parsed arguments

    parser.add_argument("--model", default="google/flan-t5-base", help="Model to use")
    parser.add_argument("--history-length", type=int, default=3, help="Number of exchanges in history")
    args = parser.parse_args()

    run_chat()

if __name__ == "__main__":
    main()