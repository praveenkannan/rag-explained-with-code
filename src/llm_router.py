"""
LLM routing and response generation for the RAG product assistant.
"""

from typing import List, Dict
from openai import OpenAI
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class LLMRouter:
    """
    Handles LLM-based routing and response generation.
    """
    
    @staticmethod
    def generate_answer(query: str, search_results: List[Dict], 
                        model: str = "gpt-3.5-turbo", 
                        temperature: float = 0.3) -> str:
        """
        Generate a contextual response based on retrieved products.
        
        Args:
            query (str): User's original query
            search_results (List[Dict]): Top matching products
            model (str, optional): LLM model to use. Defaults to GPT-3.5.
            temperature (float, optional): Creativity of the response. Defaults to 0.3.
        
        Returns:
            str: Generated answer
        """
        # Format the context from search results
        context = "\n\n".join([
            f"Product: {result['name']}\n"
            f"Description: {result['description']}\n"
            f"Similarity: {result['similarity']:.2f}"
            for result in search_results
        ])
        
        # Construct a prompt for the LLM
        prompt = f"""
Given the following products and their descriptions, answer the user's question.
Base your answer only on the provided product information.

{context}

User Question: {query}

Your response should:
1. Directly address the user's question
2. Reference specific products when relevant
3. Explain why the recommended products might meet their needs
4. Be concise and helpful
"""
        
        try:
            # Generate a response using GPT
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful product assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature
            )
            
            return response.choices[0].message.content
        
        except Exception as e:
            print(f"Error generating LLM response: {e}")
            return "I apologize, but I'm unable to generate a response at the moment."
    
    @staticmethod
    def expand_query(query: str, 
                     model: str = "gpt-3.5-turbo", 
                     temperature: float = 0.2) -> str:
        """
        Expand and refine the user's query for better retrieval.
        
        Args:
            query (str): Original user query
            model (str, optional): LLM model to use. Defaults to GPT-3.5.
            temperature (float, optional): Creativity of the expansion. Defaults to 0.2.
        
        Returns:
            str: Expanded query
        """
        prompt = f"""
Rewrite the following query to capture all semantic aspects:
"{query}"

Your expanded query should:
1. Include relevant synonyms
2. Make implicit concepts explicit
3. Be formatted as a clear, detailed question or statement
"""
        
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature
            )
            
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            print(f"Error expanding query: {e}")
            return query  # Fallback to original query
