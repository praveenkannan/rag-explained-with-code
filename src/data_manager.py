"""
Data management module for product catalogs.

Provides functionality for:
- Loading product catalogs
- Adding new products
- Filtering products
- Managing multiple catalogs
"""

import json
from typing import List, Dict, Optional
from pathlib import Path
import os


class ProductCatalogManager:
    """
    Manages product catalogs with advanced filtering and manipulation capabilities.
    """

    def __init__(self, catalog_path: Optional[str] = None):
        """
        Initialize the catalog manager.
        
        Args:
            catalog_path (Optional[str]): Path to the product catalog JSON file.
                If None, uses the default path.
        """
        if catalog_path is None:
            # Default to project root data directory
            project_root = Path(__file__).resolve().parents[1]
            catalog_path = project_root / 'data' / 'products.json'
        
        self.catalog_path = Path(catalog_path)
        self.catalogs = self._load_catalogs()

    def _load_catalogs(self) -> List[Dict]:
        """
        Load product catalogs from JSON file.
        
        Returns:
            List[Dict]: List of catalog dictionaries
        """
        try:
            with open(self.catalog_path, 'r') as f:
                data = json.load(f)
                return data.get('catalogs', [])
        except FileNotFoundError:
            print(f"Catalog file not found at {self.catalog_path}")
            return []
        except json.JSONDecodeError:
            print(f"Invalid JSON in {self.catalog_path}")
            return []

    def get_all_products(self) -> List[Dict]:
        """
        Retrieve all products from all catalogs.
        
        Returns:
            List[Dict]: Flattened list of all products
        """
        return [
            product 
            for catalog in self.catalogs 
            for product in catalog.get('products', [])
        ]

    def filter_products(
        self, 
        category: Optional[str] = None, 
        tags: Optional[List[str]] = None,
        min_price: Optional[float] = None,
        max_price: Optional[float] = None,
        in_stock: Optional[bool] = None
    ) -> List[Dict]:
        """
        Filter products based on various criteria.
        
        Args:
            category (Optional[str]): Filter by product category
            tags (Optional[List[str]]): Filter by tags
            min_price (Optional[float]): Minimum price filter
            max_price (Optional[float]): Maximum price filter
            in_stock (Optional[bool]): Filter by stock availability
        
        Returns:
            List[Dict]: Filtered list of products
        """
        products = self.get_all_products()
        
        # Apply filters
        filtered_products = [
            product for product in products
            if (category is None or product.get('category') == category) and
               (tags is None or any(tag in product.get('tags', []) for tag in tags)) and
               (min_price is None or product.get('price', 0) >= min_price) and
               (max_price is None or product.get('price', float('inf')) <= max_price) and
               (in_stock is None or product.get('availability', {}).get('in_stock') == in_stock)
        ]
        
        return filtered_products

    def add_product(
        self, 
        catalog_name: str, 
        product: Dict
    ) -> bool:
        """
        Add a new product to a specific catalog.
        
        Args:
            catalog_name (str): Name of the catalog to add the product to
            product (Dict): Product details to add
        
        Returns:
            bool: True if product was added successfully, False otherwise
        """
        # Find the target catalog
        target_catalog = next(
            (cat for cat in self.catalogs if cat['name'] == catalog_name), 
            None
        )
        
        if not target_catalog:
            print(f"Catalog '{catalog_name}' not found")
            return False
        
        # Ensure products list exists
        if 'products' not in target_catalog:
            target_catalog['products'] = []
        
        # Add product
        target_catalog['products'].append(product)
        
        # Save updated catalogs
        self._save_catalogs()
        
        return True

    def _save_catalogs(self):
        """
        Save updated catalogs back to the JSON file.
        """
        try:
            with open(self.catalog_path, 'w') as f:
                json.dump({"catalogs": self.catalogs}, f, indent=4)
        except IOError as e:
            print(f"Error saving catalog: {e}")

    def get_catalogs(self) -> List[str]:
        """
        Get names of all available catalogs.
        
        Returns:
            List[str]: List of catalog names
        """
        return [catalog['name'] for catalog in self.catalogs]
