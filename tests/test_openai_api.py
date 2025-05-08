#!/usr/bin/env python3
"""
Script to test OpenAI API key configuration and functionality.
"""

import os
import sys
import pathlib
import re

# Add project root to Python path
project_root = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from openai import OpenAI, APIError, APIStatusError, RateLimitError
from openai.types.chat import ChatCompletion
from openai.types.embedding import Embedding
from dotenv import load_dotenv, set_key

def is_valid_openai_api_key(key: str) -> bool:
    """
    Validate the format of an OpenAI API key.
    
    Args:
        key (str): The API key to validate
    
    Returns:
        bool: True if the key appears to be valid, False otherwise
    """
    # Remove potential quotes and escape characters
    key = key.strip("'\"\\")
    
    # Check for common placeholder patterns
    invalid_patterns = [
        r'^your_openai_api_key_here$',
        r'^\s*$'  # Empty or whitespace-only string
    ]
    
    # Check if the key matches any invalid pattern
    for pattern in invalid_patterns:
        if re.match(pattern, key):
            return False
    
    # Additional basic validation
    return (
        key.startswith('sk-') and 
        len(key) > 50 and 
        len(key) < 200
    )

def find_dotenv():
    """
    Find and load .env file from project root or parent directories.
    
    Returns:
        str: Path to the .env file, or None if not found
    """
    current_dir = pathlib.Path(__file__).parent.parent
    
    # Check common locations for .env file
    possible_locations = [
        current_dir / '.env',
        current_dir / '.env.local',
        current_dir / '.env.example'
    ]
    
    for env_file in possible_locations:
        if env_file.exists():
            print(f"Loading environment variables from: {env_file}")
            return str(env_file)
    
    print("No .env file found. Please create a .env file in the project root.")
    return None

def prompt_for_api_key(env_file: str) -> str:
    """
    Prompt the user to enter a valid OpenAI API key.
    
    Args:
        env_file (str): Path to the .env file
    
    Returns:
        str: A valid OpenAI API key
    """
    while True:
        print("\n" + "="*50)
        print("OpenAI API Key Required")
        print("="*50)
        print("To use this application, you need an OpenAI API key.")
        print("You can obtain one at: https://platform.openai.com/account/api-keys")
        print("\nPlease enter your OpenAI API key:")
        
        api_key = input().strip()
        
        if is_valid_openai_api_key(api_key):
            # Update .env file with the new key
            set_key(env_file, 'OPENAI_API_KEY', f"'{api_key}'")
            print("\n✓ API Key validated and saved.")
            return api_key
        else:
            print("\n" + "="*50)
            print("ERROR: Invalid API Key")
            print("Please ensure your key:")
            print("- Starts with 'sk-'")
            print("- Is 51-100 characters long")
            print("- Is obtained from OpenAI platform")
            print("="*50 + "\n")

def test_openai_api_key():
    """
    Comprehensive test of OpenAI API key functionality.
    """
    # Find and load .env file
    env_file = find_dotenv()
    if env_file:
        # Detailed diagnostic loading
        print(f"Attempting to load environment variables from: {env_file}")
        print(f"File exists: {os.path.exists(env_file)}")
        print(f"File is readable: {os.access(env_file, os.R_OK)}")
        
        # Load environment variables
        load_dotenv(env_file)
    else:
        print("\n" + "="*50)
        print("ERROR: No .env file found.")
        print("Please create a .env file with your OpenAI API key.")
        print("="*50 + "\n")
        sys.exit(1)

    # Retrieve API key
    api_key = os.getenv('OPENAI_API_KEY', '').strip("'\"\\")

    # Detailed diagnostic information
    print("\nDiagnostic Information:")
    print(f"API Key from os.getenv(): {api_key}")
    print(f"API Key length: {len(api_key) if api_key else 'N/A'}")

    # Validate API key
    if not api_key or not is_valid_openai_api_key(api_key):
        # Prompt for a new API key
        api_key = prompt_for_api_key(env_file)

    try:
        # Initialize OpenAI client
        client = OpenAI(api_key=api_key)

        # Test embedding generation
        print("Testing embedding generation...")
        embedding_response: Embedding = client.embeddings.create(
            input="Test OpenAI API key functionality",
            model="text-embedding-ada-002"
        )
        print("✓ Embedding generation successful")
        print(f"  Embedding dimensions: {len(embedding_response.data[0].embedding)}")

        # Test chat completion
        print("Testing chat completion...")
        chat_response: ChatCompletion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello, can you confirm the API is working?"}
            ],
            max_tokens=50
        )
        print("✓ Chat completion successful")
        print(f"  Response: {chat_response.choices[0].message.content}")

        # Print additional API key information
        print("\n" + "="*50)
        print("OpenAI API Key Test Results:")
        print("Status: VALID")
        print("Embedding Model: text-embedding-ada-002")
        print("Chat Model: gpt-3.5-turbo")
        print("="*50 + "\n")

    except APIError as e:
        print("\n" + "="*50)
        print("ERROR: OpenAI API Error")
        print(f"Details: {str(e)}")
        print("\nPossible reasons:")
        print("1. Temporary API service disruption")
        print("2. Network connectivity issues")
        print("\nSuggestions:")
        print("- Check OpenAI status page: https://status.openai.com")
        print("- Retry the test later")
        print("="*50 + "\n")
        sys.exit(1)

    except APIStatusError as e:
        print("\n" + "="*50)
        print("ERROR: API Quota or Billing Issue")
        print(f"Details: {str(e)}")
        print("\nPossible reasons:")
        print("1. Exceeded current API quota")
        print("2. Billing not set up correctly")
        print("3. Account restrictions")
        print("\nImportant steps:")
        print("1. Check your OpenAI account billing:")
        print("   https://platform.openai.com/account/billing/overview")
        print("2. Verify payment method")
        print("3. Add credits or upgrade your plan")
        print("4. If issues persist, contact OpenAI support")
        print("="*50 + "\n")
        sys.exit(1)

    except RateLimitError as e:
        print("\n" + "="*50)
        print("ERROR: Rate Limit Exceeded")
        print(f"Details: {str(e)}")
        print("\nPossible reasons:")
        print("1. Too many requests in a short time")
        print("2. Exceeded monthly request quota")
        print("\nSuggestions:")
        print("- Wait and retry")
        print("- Check account usage at https://platform.openai.com/account/usage")
        print("="*50 + "\n")
        sys.exit(1)

    except Exception as e:
        print("\n" + "="*50)
        print("UNEXPECTED ERROR:")
        print(f"Details: {str(e)}")
        print("Please check your API configuration and network connection.")
        print("="*50 + "\n")
        sys.exit(1)

def main():
    """
    Main function to run the API key test.
    """
    test_openai_api_key()

if __name__ == "__main__":
    main()
