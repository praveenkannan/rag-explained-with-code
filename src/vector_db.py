"""
Vector database functionality using FAISS for semantic search.
"""

from typing import List, Dict, Tuple
import numpy as np
import faiss

class VectorDatabase:
    """
    A vector database implementation using FAISS for efficient similarity search.
    """
    
    def __init__(self, dimension: int, metric: str = 'l2'):
        """
        Initialize the vector database.
        
        Args:
            dimension (int): Dimensionality of the embedding vectors
            metric (str, optional): Distance metric. Defaults to 'l2' (Euclidean).
        """
        if metric == 'l2':
            self.index = faiss.IndexFlatL2(dimension)
        elif metric == 'ip':
            # Inner product (cosine similarity)
            self.index = faiss.IndexFlatIP(dimension)
        else:
            raise ValueError(f"Unsupported metric: {metric}")
        
        self.id_map = {}
        self.dimension = dimension
    
    def add_vectors(self, vectors: np.ndarray, metadata: List[Dict]):
        """
        Add vectors to the database.
        
        Args:
            vectors (np.ndarray): Array of embedding vectors
            metadata (List[Dict]): Corresponding metadata for each vector
        """
        # Ensure vectors are float32
        vectors = vectors.astype('float32')
        
        # Add vectors to index
        start_index = len(self.id_map)
        self.index.add(vectors)
        
        # Update id_map
        for i, meta in enumerate(metadata, start=start_index):
            self.id_map[i] = meta
    
    def search(self, query_vector: np.ndarray, top_k: int = 2) -> List[Dict]:
        """
        Perform similarity search.
        
        Args:
            query_vector (np.ndarray): Query embedding vector
            top_k (int, optional): Number of top results to return. Defaults to 2.
        
        Returns:
            List[Dict]: Top matching items with metadata and similarity scores
        """
        # Ensure query vector is float32 and 2D
        query_vector = query_vector.astype('float32').reshape(1, -1)
        
        # Perform search
        distances, indices = self.index.search(query_vector, top_k)
        
        # Prepare results
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            # Retrieve metadata
            item = self.id_map.get(idx, {})
            
            # Calculate similarity score (inverse of distance)
            similarity = 1 / (1 + distance) if distance > 0 else 1
            
            results.append({
                **item,
                "distance": float(distance),
                "similarity": float(similarity)
            })
        
        return results
    
    def __len__(self):
        """
        Get the number of vectors in the database.
        
        Returns:
            int: Number of vectors
        """
        return self.index.ntotal
