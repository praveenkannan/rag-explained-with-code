"""
Test suite for RAG Product Assistant components.
"""

import os
import pytest
import numpy as np

from src.rag_pipeline import RAGPipeline
from src.embeddings import get_embedding
from src.vector_db import VectorDatabase
from src.data_manager import ProductCatalogManager

class TestRAGProductAssistant:
    @pytest.fixture
    def rag_pipeline(self):
        """
        Fixture to create a RAG pipeline for testing.
        """
        return RAGPipeline()

    def test_catalog_loading(self, rag_pipeline):
        """
        Test that product catalogs are loaded correctly.
        """
        catalogs = rag_pipeline.catalog_manager.get_catalogs()
        
        assert len(catalogs) > 0, "No catalogs were loaded"
        assert all(isinstance(catalog, str) for catalog in catalogs), "Catalog names should be strings"

    def test_product_filtering(self, rag_pipeline):
        """
        Test product filtering functionality.
        """
        # Test filtering by category
        ergonomic_products = rag_pipeline.filter_products(category="Ergonomic Furniture")
        assert len(ergonomic_products) > 0, "No ergonomic furniture products found"
        
        # Test filtering by price
        affordable_products = rag_pipeline.filter_products(max_price=500)
        assert len(affordable_products) > 0, "No products under $500 found"
        assert all(product['price'] <= 500 for product in affordable_products), "Price filter not working correctly"

    def test_embedding_generation(self):
        """
        Test embedding generation for products.
        """
        test_text = "Ergonomic chair for back pain"
        embedding = get_embedding(test_text)
        
        assert isinstance(embedding, list), "Embedding should be a list"
        assert len(embedding) > 0, "Embedding should not be empty"
        assert all(isinstance(x, float) for x in embedding), "Embedding should contain floats"

    def test_vector_database(self, rag_pipeline):
        """
        Test vector database functionality.
        """
        # Get all products
        products = rag_pipeline.catalog_manager.get_all_products()
        
        # Verify vector database initialization
        assert len(rag_pipeline.vector_db.vectors) > 0, "Vector database is empty"
        
        # Test vector search
        test_query = "ergonomic chair"
        query_embedding = get_embedding(test_query)
        search_results = rag_pipeline.vector_db.search(np.array(query_embedding), top_k=2)
        
        assert len(search_results) > 0, "No search results found"
        assert all('name' in result for result in search_results), "Search results missing product details"

    def test_answer_generation(self, rag_pipeline):
        """
        Test RAG pipeline answer generation.
        """
        test_queries = [
            "What chair is best for back pain?",
            "I need a desk for a small home office",
            "Recommend products to reduce eye strain"
        ]
        
        for query in test_queries:
            response = rag_pipeline.answer_question(query)
            
            assert isinstance(response, str), f"Response for '{query}' should be a string"
            assert len(response) > 0, f"Response for '{query}' should not be empty"

    def test_product_addition(self, rag_pipeline):
        """
        Test adding a new product to the catalog.
        """
        new_product = {
            "id": "test_prod1",
            "name": "Test Ergonomic Product",
            "description": "A test product for catalog addition",
            "category": "Test Category",
            "tags": ["test", "ergonomic"],
            "price": 99.99,
            "currency": "USD",
            "availability": {
                "in_stock": True,
                "quantity": 10,
                "shipping_time": "3-5 business days"
            }
        }
        
        # Add product to existing catalog
        initial_product_count = len(rag_pipeline.catalog_manager.get_all_products())
        success = rag_pipeline.add_product("Office Ergonomics", new_product)
        
        assert success, "Failed to add new product"
        
        # Verify product count increased
        updated_product_count = len(rag_pipeline.catalog_manager.get_all_products())
        assert updated_product_count == initial_product_count + 1, "Product count did not increase"
        
        # Verify added product exists
        added_products = rag_pipeline.filter_products(category="Test Category")
        assert len(added_products) > 0, "Added product not found in catalog"
