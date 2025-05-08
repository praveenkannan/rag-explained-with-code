"""
Embedding generation utilities for the RAG product assistant.
"""

from typing import List
from openai import OpenAI
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_embedding(text: str, model: str = "text-embedding-ada-002") -> List[float]:
    """
    Generate embedding for given text using OpenAI's embedding model.
    
    Args:
        text (str): Input text to embed
        model (str, optional): Embedding model to use. Defaults to OpenAI's ada model.
    
    Returns:
        List[float]: Embedding vector
    """
    try:
        response = client.embeddings.create(
            input=[text],
            model=model
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return []
