"""
Enhanced Vector Store using ChromaDB with embeddings support
Supports multiple embedding models and advanced similarity search
"""

import chromadb
from chromadb.config import Settings
import numpy as np
from typing import List, Dict, Optional, Any
import streamlit as st
import os
import hashlib
import json
from pathlib import Path


class EnhancedVectorStore:
    """ChromaDB-based vector store with advanced features"""
    
    def __init__(self, 
                 collection_name: str = "rag_documents",
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 persist_directory: str = "./chroma_db"):
        
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        
        # Ensure directory exists
        Path(persist_directory).mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize ChromaDB client with persistence"""
        try:
            # Configure ChromaDB settings
            settings = Settings(
                persist_directory=self.persist_directory,
                anonymized_telemetry=False
            )
            
            # Create client
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=settings
            )
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}  # Use cosine similarity
            )
            
            st.success(f"ChromaDB initialized with collection: {self.collection_name}")
            
        except Exception as e:
            st.error(f"Failed to initialize ChromaDB: {str(e)}")
            # Fallback to in-memory client
            self.client = chromadb.Client()
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
    
    def _generate_embedding(self, text: str) -> List[float]:
        """Generate embeddings using available models"""
        try:
            # Try to use sentence-transformers if available
            try:
                from sentence_transformers import SentenceTransformer
                
                # Cache model loading
                if not hasattr(self, '_embedding_model'):
                    self._embedding_model = SentenceTransformer(self.embedding_model)
                
                embedding = self._embedding_model.encode(text, convert_to_tensor=False)
                return embedding.tolist()
                
            except ImportError:
                # Fallback to simple TF-IDF based embedding
                return self._simple_embedding(text)
                
        except Exception as e:
            st.warning(f"Embedding generation error: {str(e)}")
            return self._simple_embedding(text)
    
    def _simple_embedding(self, text: str, dim: int = 384) -> List[float]:
        """Simple fallback embedding using text hashing"""
        # Create a simple hash-based embedding
        text_hash = hashlib.md5(text.encode()).hexdigest()
        
        # Convert to numerical representation
        embedding = []
        for i in range(0, len(text_hash), len(text_hash) // dim):
            chunk = text_hash[i:i + len(text_hash) // dim]
            if chunk:
                embedding.append(float(int(chunk, 16)) / 16**len(chunk))
        
        # Pad or truncate to desired dimension
        while len(embedding) < dim:
            embedding.append(0.0)
        
        return embedding[:dim]
    
    def add_documents(self, chunks: List[Dict[str, Any]]) -> bool:
        """Add document chunks to the vector store"""
        try:
            documents = []
            metadatas = []
            ids = []
            embeddings = []
            
            for i, chunk in enumerate(chunks):
                # Prepare document data
                doc_id = f"doc_{chunk.get('id', i)}_{hashlib.md5(chunk['content'].encode()).hexdigest()[:8]}"
                
                documents.append(chunk['content'])
                metadatas.append({
                    'chunk_id': chunk.get('id', i),
                    'word_count': chunk.get('word_count', len(chunk['content'].split())),
                    'char_count': chunk.get('char_count', len(chunk['content'])),
                    'source': chunk.get('source', 'uploaded_document')
                })
                ids.append(doc_id)
                
                # Generate embedding
                embedding = self._generate_embedding(chunk['content'])
                embeddings.append(embedding)
            
            # Add to collection
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids,
                embeddings=embeddings
            )
            
            st.success(f"Added {len(chunks)} document chunks to ChromaDB")
            return True
            
        except Exception as e:
            st.error(f"Error adding documents to vector store: {str(e)}")
            return False
    
    def similarity_search(self, 
                         query: str, 
                         k: int = 5, 
                         filter_metadata: Optional[Dict] = None) -> List[Dict]:
        """Perform similarity search with optional metadata filtering"""
        try:
            # Generate query embedding
            query_embedding = self._generate_embedding(query)
            
            # Perform search
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                where=filter_metadata,
                include=['documents', 'metadatas', 'distances']
            )
            
            # Format results
            formatted_results = []
            if results['documents'] and results['documents'][0]:
                for i, (doc, metadata, distance) in enumerate(zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                )):
                    # Convert distance to similarity score (ChromaDB uses distance, lower is better)
                    similarity_score = round(max(0.0, 1.0 - distance), 4)  # Convert distance to similarity
                    
                    formatted_results.append({
                        'content': doc,
                        'score': similarity_score,
                        'distance': distance,
                        'chunk_id': metadata.get('chunk_id'),
                        'metadata': metadata,
                        'rank': i + 1
                    })
            
            return formatted_results
            
        except Exception as e:
            st.error(f"Error during similarity search: {str(e)}")
            return []
    
    def hybrid_search(self, 
                     query: str, 
                     k: int = 5,
                     alpha: float = 0.7) -> List[Dict]:
        """
        Hybrid search combining semantic similarity and keyword matching
        alpha: weight for semantic search (1-alpha for keyword search)
        """
        try:
            # Get semantic results
            semantic_results = self.similarity_search(query, k=k*2)  # Get more for reranking
            
            # Get keyword-based results (simple contains matching)
            all_docs = self.collection.get(include=['documents', 'metadatas'])
            keyword_results = []
            
            query_words = set(query.lower().split())
            
            for i, (doc, metadata) in enumerate(zip(all_docs['documents'], all_docs['metadatas'])):
                doc_words = set(doc.lower().split())
                overlap = len(query_words.intersection(doc_words))
                
                if overlap > 0:
                    keyword_score = overlap / len(query_words)
                    keyword_results.append({
                        'content': doc,
                        'score': keyword_score,
                        'chunk_id': metadata.get('chunk_id'),
                        'metadata': metadata,
                        'type': 'keyword'
                    })
            
            # Sort keyword results
            keyword_results.sort(key=lambda x: x['score'], reverse=True)
            
            # Combine and rerank results
            combined_results = {}
            
            # Add semantic results
            for result in semantic_results:
                doc_id = result['chunk_id']
                combined_results[doc_id] = result.copy()
                combined_results[doc_id]['semantic_score'] = result['score']
                combined_results[doc_id]['keyword_score'] = 0
            
            # Add keyword scores
            for result in keyword_results[:k*2]:
                doc_id = result['chunk_id']
                if doc_id in combined_results:
                    combined_results[doc_id]['keyword_score'] = result['score']
                else:
                    combined_results[doc_id] = result.copy()
                    combined_results[doc_id]['semantic_score'] = 0
                    combined_results[doc_id]['keyword_score'] = result['score']
            
            # Calculate hybrid scores
            final_results = []
            for doc_id, result in combined_results.items():
                semantic_score = result.get('semantic_score', 0)
                keyword_score = result.get('keyword_score', 0)
                
                # Hybrid score
                hybrid_score = alpha * semantic_score + (1 - alpha) * keyword_score
                
                result['score'] = hybrid_score
                result['hybrid_breakdown'] = {
                    'semantic': semantic_score,
                    'keyword': keyword_score,
                    'alpha': alpha
                }
                
                final_results.append(result)
            
            # Sort by hybrid score and return top k
            final_results.sort(key=lambda x: x['score'], reverse=True)
            return final_results[:k]
            
        except Exception as e:
            st.error(f"Error during hybrid search: {str(e)}")
            return self.similarity_search(query, k)  # Fallback to semantic search
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection"""
        try:
            collection_info = self.collection.get()
            
            return {
                'total_documents': len(collection_info['documents']) if collection_info['documents'] else 0,
                'collection_name': self.collection_name,
                'embedding_model': self.embedding_model,
                'persist_directory': self.persist_directory,
                'embedding_dimension': len(collection_info['embeddings'][0]) if collection_info.get('embeddings') else 'Unknown',
                'metadata_fields': list(collection_info['metadatas'][0].keys()) if collection_info.get('metadatas') else []
            }
            
        except Exception as e:
            st.error(f"Error getting collection stats: {str(e)}")
            return {'error': str(e)}
    
    def clear_collection(self) -> bool:
        """Clear all documents from the collection"""
        try:
            # Delete the collection and recreate it
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            
            st.success("Collection cleared successfully")
            return True
            
        except Exception as e:
            st.error(f"Error clearing collection: {str(e)}")
            return False
    
    def export_collection(self, filepath: str) -> bool:
        """Export collection data to JSON file"""
        try:
            collection_data = self.collection.get(include=['documents', 'metadatas', 'embeddings'])
            
            export_data = {
                'collection_name': self.collection_name,
                'embedding_model': self.embedding_model,
                'data': collection_data,
                'stats': self.get_collection_stats()
            }
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            st.success(f"Collection exported to {filepath}")
            return True
            
        except Exception as e:
            st.error(f"Error exporting collection: {str(e)}")
            return False