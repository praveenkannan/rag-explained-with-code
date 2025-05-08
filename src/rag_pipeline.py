"""
RAG Pipeline for product recommendations.
"""

from typing import List, Dict
import numpy as np
import os
import sys
import openai
from dotenv import load_dotenv

from .embeddings import get_embedding
from .vector_db import VectorDatabase
from .llm_router import LLMRouter
from .data_manager import ProductCatalogManager

class OpenAIConfigError(Exception):
    """Custom exception for OpenAI configuration errors."""
    pass

class RAGPipeline:
    """
    Retrieval-Augmented Generation pipeline for product recommendations.
    """
    
    def __init__(
        self, 
        products: List[Dict] = None, 
        catalog_path: str = None,
        embedding_model: str = "text-embedding-ada-002",
        embedding_dim: int = 1536
    ):
        """
        Initialize the RAG pipeline.
        
        Args:
            products (List[Dict], optional): List of products to initialize with
            catalog_path (str, optional): Path to product catalog
            embedding_model (str, optional): OpenAI embedding model to use
            embedding_dim (int, optional): Dimensionality of embeddings. Defaults to 1536.
        
        Raises:
            OpenAIConfigError: If OpenAI API key is invalid or not configured
        """
        # Load environment variables
        load_dotenv()
        
        # Validate OpenAI API key
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key or api_key.strip() == 'your_openai_api_key_here':
            print("\n" + "="*50)
            print("ERROR: OpenAI API Key is missing or invalid.")
            print("Please set your OpenAI API key in the .env file.")
            print("You can obtain an API key at: https://platform.openai.com/account/api-keys")
            print("="*50 + "\n")
            raise OpenAIConfigError("Invalid or missing OpenAI API key")
        
        # Configure OpenAI API
        openai.api_key = api_key
        
        try:
            # Validate API key by making a minimal API call
            openai.Embedding.create(
                input="Test API key",
                model="text-embedding-ada-002"
            )
        except Exception as e:
            print("\n" + "="*50)
            print("ERROR: Failed to validate OpenAI API key.")
            print(f"Details: {str(e)}")
            print("Please check your API key and try again.")
            print("="*50 + "\n")
            raise OpenAIConfigError("API key validation failed") from e
        
        # Initialize catalog manager
        self.catalog_manager = ProductCatalogManager(catalog_path)
        
        # Use provided products or load from catalog
        if products is None:
            products = self.catalog_manager.get_all_products()
        
        # Generate embeddings for products
        self.products = products
        self.embeddings = self._generate_embeddings(embedding_model)
        
        # Initialize vector database
        self.vector_db = VectorDatabase(dimension=embedding_dim)
        self.vector_db.add_vectors(
            vectors=np.array(self.embeddings), 
            metadata=self.products
        )

    def _generate_embeddings(
        self, 
        embedding_model: str
    ):
        """
        Generate embeddings for products.
        
        Args:
            embedding_model (str): Embedding model to use
        
        Returns:
            List: List of embeddings
        """
        # Prepare embedding texts
        embedding_texts = []
        
        for product in self.products:
            # Combine name and description for rich embedding
            embedding_text = f"{product.get('name', '')} {product.get('description', '')}"
            embedding_texts.append(embedding_text)
        
        # Generate embeddings
        embeddings = [
            get_embedding(text, model=embedding_model) 
            for text in embedding_texts
        ]
        
        return embeddings

    def answer_question(
        self, 
        query: str, 
        top_k: int = 2,
        filter_params: Dict = None
    ) -> str:
        """
        Generate a product recommendation based on the query.
        
        Args:
            query (str): User's query
            top_k (int, optional): Number of top products to retrieve
            filter_params (Dict, optional): Additional filtering parameters
        
        Returns:
            str: Generated product recommendation
        """
        # Generate query embedding
        query_embedding = np.array(get_embedding(query))
        
        # Perform vector search
        search_results = self.vector_db.search(query_embedding, top_k)
        
        # Optional filtering
        if filter_params:
            search_results = [
                result for result in search_results
                if all(
                    result.get(key) == value 
                    for key, value in filter_params.items()
                )
            ]
        
        # Generate answer using LLM router
        answer = LLMRouter.generate_answer(query, search_results)
        
        return answer

    def add_product(
        self, 
        catalog_name: str, 
        product: Dict, 
        update_embeddings: bool = True
    ) -> bool:
        """
        Add a new product to the catalog and optionally update embeddings.
        
        Args:
            catalog_name (str): Name of the catalog to add product to
            product (Dict): Product details
            update_embeddings (bool, optional): Whether to update vector database
        
        Returns:
            bool: True if product was added successfully
        """
        # Add product to catalog
        added = self.catalog_manager.add_product(catalog_name, product)
        
        # Update embeddings if requested
        if added and update_embeddings:
            embedding_text = f"{product.get('name', '')} {product.get('description', '')}"
            embedding = get_embedding(embedding_text)
            self.vector_db.add_vectors(
                np.array([embedding]), 
                [product]
            )
        
        return added

    def filter_products(
        self, 
        category: str = None, 
        tags: List[str] = None,
        min_price: float = None,
        max_price: float = None,
        in_stock: bool = None
    ) -> List[Dict]:
        """
        Filter products based on various criteria.
        
        Args:
            category (str, optional): Filter by product category
            tags (List[str], optional): Filter by tags
            min_price (float, optional): Minimum price filter
            max_price (float, optional): Maximum price filter
            in_stock (bool, optional): Filter by stock availability
        
        Returns:
            List[Dict]: Filtered list of products
        """
        return self.catalog_manager.filter_products(
            category=category,
            tags=tags,
            min_price=min_price,
            max_price=max_price,
            in_stock=in_stock
        )
