"""
Vector Search Service
Supports both FAISS and Pinecone for semantic search
"""

import logging
import pickle
import os
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logging.warning("FAISS not available. Install with: pip install faiss-cpu")

try:
    from pinecone import Pinecone
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False
    logging.warning("Pinecone not available. Install with: pip install pinecone-client")

from app.core.config import settings

logger = logging.getLogger(__name__)


class VectorSearchService:
    """Vector search service supporting both FAISS and Pinecone"""
    
    def __init__(self, use_pinecone: bool = False):
        """
        Initialize vector search service
        
        Args:
            use_pinecone: Whether to use Pinecone (True) or FAISS (False)
        """
        self.use_pinecone = use_pinecone and PINECONE_AVAILABLE and settings.PINECONE_API_KEY
        self.embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)
        self.embedding_dimension = settings.EMBEDDING_DIMENSION
        
        # Initialize storage
        self.faiss_index = None
        self.pinecone_index = None
        self.document_metadata = []  # Store metadata for FAISS
        
        if self.use_pinecone:
            self._init_pinecone()
        else:
            self._init_faiss()
            self._preload_faiss_index()
        
        logger.info(f"Initialized VectorSearchService with {'Pinecone' if self.use_pinecone else 'FAISS'}")
    
    def _init_pinecone(self):
        """Initialize Pinecone connection"""
        try:
            # Initialize Pinecone client
            pc = Pinecone(api_key=settings.PINECONE_API_KEY)
            
            # Create index if it doesn't exist
            index_name = settings.PINECONE_INDEX_NAME
            existing_indexes = [index.name for index in pc.list_indexes()]
            
            if index_name not in existing_indexes:
                pc.create_index(
                    name=index_name,
                    dimension=self.embedding_dimension,
                    metric="cosine"
                )
                logger.info(f"Created Pinecone index: {index_name}")
            
            self.pinecone_index = pc.Index(index_name)
            logger.info("Pinecone initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone: {str(e)}")
            logger.info("Falling back to FAISS")
            self.use_pinecone = False
            self._init_faiss()
    
    def _init_faiss(self):
        """Initialize FAISS index"""
        try:
            if FAISS_AVAILABLE:
                # Create FAISS index
                self.faiss_index = faiss.IndexFlatIP(self.embedding_dimension)  # Inner product for cosine similarity
                logger.info("FAISS initialized successfully")
            else:
                raise ImportError("FAISS not available")
                
        except Exception as e:
            logger.error(f"Failed to initialize FAISS: {str(e)}")
            raise
    
    def _preload_faiss_index(self):
        """Preload index into memory for faster search"""
        try:
            if self.faiss_index and self.faiss_index.ntotal > 0:
                # Only preload if there are vectors in the index
                self.faiss_index.reconstruct_n(0, self.faiss_index.ntotal)
                logger.info("FAISS index preloaded into memory")
        except Exception as e:
            logger.warning(f"Could not preload FAISS index: {str(e)}")
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """
        Add documents to the vector index
        
        Args:
            documents: List of document dictionaries with chunks and embeddings
            
        Returns:
            Success status
        """
        try:
            if self.use_pinecone:
                return self._add_to_pinecone(documents)
            else:
                return self._add_to_faiss(documents)
                
        except Exception as e:
            logger.error(f"Error adding documents to vector index: {str(e)}")
            return False
    
    def _add_to_pinecone(self, documents: List[Dict[str, Any]]) -> bool:
        """Add documents to Pinecone"""
        try:
            vectors_to_upsert = []
            
            for doc in documents:
                doc_id = doc.get("url", "unknown")
                for chunk in doc.get("chunks", []):
                    vector_id = f"{doc_id}_{chunk['id']}"
                    embedding = chunk.get("embedding", [])
                    
                    metadata = {
                        "document_url": doc_id,
                        "chunk_id": chunk["id"],
                        "text": chunk["text"][:1000],  # Pinecone metadata size limit
                        "file_type": doc.get("file_type", "unknown"),
                        "chunk_length": chunk.get("length", 0)
                    }
                    
                    vectors_to_upsert.append((vector_id, embedding, metadata))
            
            # Upsert in batches
            batch_size = 100
            for i in range(0, len(vectors_to_upsert), batch_size):
                batch = vectors_to_upsert[i:i + batch_size]
                self.pinecone_index.upsert(vectors=batch)
            
            logger.info(f"Added {len(vectors_to_upsert)} vectors to Pinecone")
            return True
            
        except Exception as e:
            logger.error(f"Error adding to Pinecone: {str(e)}")
            return False
    
    def _add_to_faiss(self, documents: List[Dict[str, Any]]) -> bool:
        """Add documents to FAISS"""
        try:
            embeddings = []
            
            for doc in documents:
                doc_id = doc.get("url", "unknown")
                for chunk in doc.get("chunks", []):
                    embedding = np.array(chunk.get("embedding", []), dtype=np.float32)
                    embeddings.append(embedding)
                    
                    # Store metadata separately for FAISS
                    metadata = {
                        "document_url": doc_id,
                        "chunk_id": chunk["id"],
                        "text": chunk["text"],
                        "file_type": doc.get("file_type", "unknown"),
                        "chunk_length": chunk.get("length", 0),
                        "vector_index": len(self.document_metadata)  # Index in FAISS
                    }
                    self.document_metadata.append(metadata)
            
            if embeddings:
                # Normalize embeddings for cosine similarity
                embeddings_array = np.vstack(embeddings)
                faiss.normalize_L2(embeddings_array)
                
                # Add to FAISS index
                self.faiss_index.add(embeddings_array)
                
                logger.info(f"Added {len(embeddings)} vectors to FAISS")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error adding to FAISS: {str(e)}")
            return False
    
    def search(self, query: str, top_k: int = 5, score_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """
        Search for similar documents
        
        Args:
            query: Search query
            top_k: Number of results to return
            score_threshold: Minimum similarity score threshold
            
        Returns:
            List of search results with metadata and scores
        """
        try:
            # Create query embedding
            query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)[0]
            
            if self.use_pinecone:
                return self._search_pinecone(query_embedding, top_k, score_threshold)
            else:
                return self._search_faiss(query_embedding, top_k, score_threshold)
                
        except Exception as e:
            logger.error(f"Error during search: {str(e)}")
            return []
    
    def _search_pinecone(self, query_embedding: np.ndarray, top_k: int, score_threshold: float) -> List[Dict[str, Any]]:
        """Search using Pinecone"""
        try:
            results = self.pinecone_index.query(
                vector=query_embedding.tolist(),
                top_k=top_k,
                include_metadata=True
            )
            
            search_results = []
            for match in results.matches:
                if match.score >= score_threshold:
                    result = {
                        "id": match.id,
                        "score": match.score,
                        "text": match.metadata.get("text", ""),
                        "document_url": match.metadata.get("document_url", ""),
                        "chunk_id": match.metadata.get("chunk_id", ""),
                        "file_type": match.metadata.get("file_type", ""),
                        "chunk_length": match.metadata.get("chunk_length", 0)
                    }
                    search_results.append(result)
            
            logger.info(f"Pinecone search returned {len(search_results)} results")
            return search_results
            
        except Exception as e:
            logger.error(f"Error searching Pinecone: {str(e)}")
            return []
    
    def _search_faiss(self, query_embedding: np.ndarray, top_k: int, score_threshold: float) -> List[Dict[str, Any]]:
        """Search using FAISS"""
        try:
            if self.faiss_index.ntotal == 0:
                logger.warning("FAISS index is empty")
                return []
            
            # Normalize query embedding
            query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
            faiss.normalize_L2(query_embedding)
            
            # Use IVF index if available for faster search
            if hasattr(self.faiss_index, 'nprobe'):
                self.faiss_index.nprobe = 5  # Balance speed/accuracy
            
            # Search
            scores, indices = self.faiss_index.search(query_embedding, min(top_k, self.faiss_index.ntotal))
            
            search_results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx != -1 and score >= score_threshold:
                    metadata = self.document_metadata[idx]
                    result = {
                        "id": f"{metadata['document_url']}_{metadata['chunk_id']}",
                        "score": float(score),
                        "text": metadata["text"],
                        "document_url": metadata["document_url"],
                        "chunk_id": metadata["chunk_id"],
                        "file_type": metadata["file_type"],
                        "chunk_length": metadata["chunk_length"]
                    }
                    search_results.append(result)
            
            logger.info(f"FAISS search returned {len(search_results)} results")
            return search_results
            
        except Exception as e:
            logger.error(f"Error searching FAISS: {str(e)}")
            return []
    
    def save_index(self, filepath: str) -> bool:
        """
        Save the vector index to disk
        
        Args:
            filepath: Path to save the index
            
        Returns:
            Success status
        """
        try:
            if not self.use_pinecone and self.faiss_index is not None:
                # Save FAISS index and metadata
                faiss.write_index(self.faiss_index, f"{filepath}.faiss")
                
                with open(f"{filepath}_metadata.pkl", "wb") as f:
                    pickle.dump(self.document_metadata, f)
                
                logger.info(f"Saved FAISS index to {filepath}")
                return True
            else:
                logger.info("Using Pinecone - index is stored in cloud")
                return True
                
        except Exception as e:
            logger.error(f"Error saving index: {str(e)}")
            return False
    
    def load_index(self, filepath: str) -> bool:
        """
        Load the vector index from disk
        
        Args:
            filepath: Path to load the index from
            
        Returns:
            Success status
        """
        try:
            if not self.use_pinecone and os.path.exists(f"{filepath}.faiss"):
                # Load FAISS index and metadata
                self.faiss_index = faiss.read_index(f"{filepath}.faiss")
                
                metadata_file = f"{filepath}_metadata.pkl"
                if os.path.exists(metadata_file):
                    with open(metadata_file, "rb") as f:
                        self.document_metadata = pickle.load(f)
                
                # Preload the loaded index
                self._preload_faiss_index()
                
                logger.info(f"Loaded FAISS index from {filepath}")
                return True
            else:
                logger.info("Using Pinecone - index is loaded from cloud")
                return True
                
        except Exception as e:
            logger.error(f"Error loading index: {str(e)}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector index"""
        try:
            if self.use_pinecone:
                stats = self.pinecone_index.describe_index_stats()
                return {
                    "backend": "Pinecone",
                    "total_vectors": stats.total_vector_count,
                    "dimension": stats.dimension,
                    "index_fullness": stats.index_fullness
                }
            else:
                return {
                    "backend": "FAISS",
                    "total_vectors": self.faiss_index.ntotal if self.faiss_index else 0,
                    "dimension": self.embedding_dimension,
                    "metadata_count": len(self.document_metadata)
                }
                
        except Exception as e:
            logger.error(f"Error getting stats: {str(e)}")
            return {"backend": "Unknown", "error": str(e)}