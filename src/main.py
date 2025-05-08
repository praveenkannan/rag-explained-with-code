"""
Main entry point for the RAG Product Assistant application.
"""

import sys
from typing import Optional

from .rag_pipeline import RAGPipeline, OpenAIConfigError
from .data_manager import ProductCatalogManager

def print_welcome_message():
    """
    Display a welcome message for the Product Assistant.
    """
    print("\n" + "="*50)
    print(" RAG Product Assistant")
    print("Welcome! I'm here to help you find the perfect workspace solutions.")
    print("Type 'quit', 'exit', or 'q' to end the conversation.")
    print("Type 'help' for usage instructions.")
    print("="*50 + "\n")

def print_help_message():
    """
    Display help and usage instructions.
    """
    print("\nHow to use the RAG Product Assistant:")
    print("- Ask questions about office ergonomics, furniture, and productivity")
    print("- Examples:")
    print("  * 'What chair is best for back pain?'")
    print("  * 'I need a desk for a small home office'")
    print("  * 'Recommend products to reduce eye strain'")
    print("- Type 'quit', 'exit', or 'q' to exit")
    print("- Type 'help' to see these instructions again\n")

def interactive_chat(rag_pipeline: RAGPipeline):
    """
    Start an interactive chat session with the RAG Product Assistant.
    
    Args:
        rag_pipeline (RAGPipeline): Initialized RAG pipeline
    """
    print_welcome_message()
    
    while True:
        try:
            # Get user input
            user_input = input(" You: ").strip()
            
            # Check for exit commands
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\nThank you for using the RAG Product Assistant. Goodbye! ")
                break
            
            # Check for help command
            if user_input.lower() == 'help':
                print_help_message()
                continue
            
            # Skip empty inputs
            if not user_input:
                continue
            
            # Generate response using RAG pipeline
            print("\n Assistant:", end=" ")
            response = rag_pipeline.answer_question(user_input)
            print(response)
            print("\n" + "-"*50)
        
        except KeyboardInterrupt:
            print("\n\nOperation cancelled by user. Type 'quit' to exit.")
        except Exception as e:
            print(f"\nAn error occurred: {e}")

def main():
    """
    Main function to run the Product Assistant.
    """
    try:
        # Load products from catalog
        catalog_manager = ProductCatalogManager()
        products = catalog_manager.get_all_products()
        
        # Initialize RAG pipeline with loaded products
        rag_pipeline = RAGPipeline(products=products)
        
        # Check if any command-line arguments are provided
        if len(sys.argv) > 1:
            # If arguments are provided, treat them as a single query
            query = " ".join(sys.argv[1:])
            response = rag_pipeline.answer_question(query)
            print(f"Query: {query}")
            print("Response:", response)
        else:
            # Start interactive chat
            interactive_chat(rag_pipeline)
    
    except OpenAIConfigError:
        # Exit the application if OpenAI configuration is invalid
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
