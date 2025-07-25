"""
Enhanced RAG Pipeline with advanced retrieval and generation
Supports multiple search strategies and response optimization
"""

from typing import List, Dict, Optional, Any
import streamlit as st
import time
from .enhanced_vector_store import EnhancedVectorStore
from .advanced_llm_handler import AdvancedLLMHandler


class EnhancedRAGPipeline:
    """Advanced RAG pipeline with multi-strategy retrieval and generation"""
    
    def __init__(self, 
                 vector_store: EnhancedVectorStore,
                 llm_handler: AdvancedLLMHandler,
                 top_k: int = 5,
                 similarity_threshold: float = 0.3,
                 use_hybrid_search: bool = True):
        
        self.vector_store = vector_store
        self.llm_handler = llm_handler
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.use_hybrid_search = use_hybrid_search
        
        # Search strategy weights
        self.search_weights = {
            'semantic': 0.7,
            'keyword': 0.3
        }
        
        # Response quality metrics
        self.response_metrics = {
            'total_queries': 0,
            'successful_responses': 0,
            'avg_response_time': 0,
            'avg_context_relevance': 0
        }
    
    def retrieve_context(self, 
                        query: str, 
                        search_strategy: str = "hybrid") -> List[Dict]:
        """Advanced context retrieval with multiple strategies"""
        try:
            start_time = time.time()
            
            if search_strategy == "hybrid" and self.use_hybrid_search:
                # Use hybrid search combining semantic and keyword matching
                results = self.vector_store.hybrid_search(
                    query=query,
                    k=self.top_k,
                    alpha=self.search_weights['semantic']
                )
            elif search_strategy == "semantic":
                # Pure semantic search
                results = self.vector_store.similarity_search(
                    query=query,
                    k=self.top_k
                )
            else:
                # Fallback to similarity search
                results = self.vector_store.similarity_search(
                    query=query,
                    k=self.top_k
                )
            
            # Filter by similarity threshold
            filtered_results = [
                result for result in results 
                if result.get('score', 0) >= self.similarity_threshold
            ]
            
            # Calculate retrieval metrics
            retrieval_time = time.time() - start_time
            avg_score = sum(r.get('score', 0) for r in filtered_results) / len(filtered_results) if filtered_results else 0
            
            # Add retrieval metadata
            for result in filtered_results:
                result['retrieval_time'] = retrieval_time
                result['search_strategy'] = search_strategy
                result['query_similarity'] = self._calculate_query_similarity(query, result['content'])
            
            return filtered_results
            
        except Exception as e:
            st.error(f"Error in context retrieval: {str(e)}")
            return []
    
    def _calculate_query_similarity(self, query: str, content: str) -> float:
        """Calculate simple query-content similarity"""
        query_words = set(query.lower().split())
        content_words = set(content.lower().split())
        
        if not query_words or not content_words:
            return 0.0
        
        intersection = query_words.intersection(content_words)
        union = query_words.union(content_words)
        
        return len(intersection) / len(union) if union else 0.0
    
    def rerank_contexts(self, contexts: List[Dict], query: str) -> List[Dict]:
        """Re-rank retrieved contexts for better relevance"""
        try:
            # Multiple ranking factors
            for context in contexts:
                content = context['content']
                
                # Factor 1: Query word overlap
                query_overlap = self._calculate_query_similarity(query, content)
                
                # Factor 2: Content length relevance (prefer medium-length contexts)
                word_count = context.get('metadata', {}).get('word_count', len(content.split()))
                length_score = 1.0 - abs(word_count - 150) / 200  # Optimal around 150 words
                length_score = max(0.1, min(1.0, length_score))
                
                # Factor 3: Entity relevance (if entities are present)
                entities = context.get('metadata', {}).get('entities', [])
                entity_score = 0.1
                query_lower = query.lower()
                for entity in entities:
                    if entity.lower() in query_lower:
                        entity_score += 0.2
                entity_score = min(1.0, entity_score)
                
                # Factor 4: Readability bonus
                readability = context.get('metadata', {}).get('readability_score', 50)
                readability_score = readability / 100
                
                # Combined reranking score
                original_score = context.get('score', 0.5)
                rerank_score = (
                    0.4 * original_score +
                    0.3 * query_overlap +
                    0.1 * length_score +
                    0.1 * entity_score +
                    0.1 * readability_score
                )
                
                context['rerank_score'] = rerank_score
                context['ranking_factors'] = {
                    'original_score': original_score,
                    'query_overlap': query_overlap,
                    'length_score': length_score,
                    'entity_score': entity_score,
                    'readability_score': readability_score
                }
            
            # Sort by rerank score
            contexts.sort(key=lambda x: x.get('rerank_score', 0), reverse=True)
            
            return contexts
            
        except Exception as e:
            st.warning(f"Error in context reranking: {str(e)}")
            return contexts
    
    def optimize_context_selection(self, contexts: List[Dict], max_context_length: int = 2000) -> List[Dict]:
        """Optimize context selection to maximize information density"""
        if not contexts:
            return contexts
        
        selected_contexts = []
        total_length = 0
        seen_content = set()
        
        for context in contexts:
            content = context['content']
            content_hash = hash(content)
            
            # Skip duplicate content
            if content_hash in seen_content:
                continue
            
            content_length = len(content)
            
            # Check if adding this context exceeds limit
            if total_length + content_length > max_context_length and selected_contexts:
                break
            
            selected_contexts.append(context)
            total_length += content_length
            seen_content.add(content_hash)
        
        return selected_contexts
    
    def generate_response(self, query: str, contexts: List[Dict]) -> str:
        """Generate response with context optimization"""
        try:
            if not contexts:
                return "I couldn't find relevant information to answer your question. Please try rephrasing or asking about a different topic."
            
            # Extract context texts
            context_texts = [ctx['content'] for ctx in contexts]
            
            # Generate response using LLM
            response = self.llm_handler.generate_response(query, context_texts)
            
            # Enhance response with source tracking
            response = self._enhance_response_with_sources(response, contexts)
            
            return response
            
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")
            return "I apologize, but I encountered an error while generating a response."
    
    def _enhance_response_with_sources(self, response: str, contexts: List[Dict]) -> str:
        """Enhance response with source information"""
        try:
            # Add confidence indicators based on context quality
            avg_score = sum(ctx.get('score', 0) for ctx in contexts) / len(contexts)
            
            if avg_score > 0.8:
                confidence_note = "\n\n*High confidence response based on relevant context.*"
            elif avg_score > 0.5:
                confidence_note = "\n\n*Moderate confidence response. Some information may be inferred.*"
            else:
                confidence_note = "\n\n*Low confidence response. Please verify information independently.*"
            
            return response + confidence_note
            
        except:
            return response
    
    def get_response(self, query: str, 
                    search_strategy: str = "hybrid",
                    include_debug_info: bool = False) -> Optional[Dict]:
        """Complete RAG pipeline with comprehensive response generation"""
        try:
            start_time = time.time()
            self.response_metrics['total_queries'] += 1
            
            # Step 1: Retrieve contexts
            with st.spinner("🔍 Searching relevant information..."):
                contexts = self.retrieve_context(query, search_strategy)
            
            if not contexts:
                return {
                    "response": "I couldn't find relevant information in the knowledge base to answer your question. Please try rephrasing your query or ask about a different topic.",
                    "sources": [],
                    "debug_info": {"error": "No relevant contexts found"}
                }
            
            # Step 2: Re-rank contexts
            with st.spinner("📊 Analyzing context relevance..."):
                reranked_contexts = self.rerank_contexts(contexts, query)
            
            # Step 3: Optimize context selection
            optimized_contexts = self.optimize_context_selection(reranked_contexts)
            
            # Step 4: Generate response
            with st.spinner("🤖 Generating response..."):
                response = self.generate_response(query, optimized_contexts)
            
            # Step 5: Prepare source information
            sources = []
            for i, ctx in enumerate(optimized_contexts):
                source_info = {
                    'content': ctx['content'][:300] + "..." if len(ctx['content']) > 300 else ctx['content'],
                    'score': ctx.get('score', 0),
                    'rerank_score': ctx.get('rerank_score', 0),
                    'chunk_id': ctx.get('chunk_id'),
                    'metadata': ctx.get('metadata', {}),
                    'rank': i + 1
                }
                
                if include_debug_info:
                    source_info['debug'] = {
                        'retrieval_time': ctx.get('retrieval_time', 0),
                        'search_strategy': ctx.get('search_strategy', 'unknown'),
                        'ranking_factors': ctx.get('ranking_factors', {})
                    }
                
                sources.append(source_info)
            
            # Update metrics
            response_time = time.time() - start_time
            self.response_metrics['successful_responses'] += 1
            self.response_metrics['avg_response_time'] = (
                (self.response_metrics['avg_response_time'] * (self.response_metrics['successful_responses'] - 1) + response_time) /
                self.response_metrics['successful_responses']
            )
            
            # Calculate average context relevance
            avg_relevance = sum(ctx.get('score', 0) for ctx in optimized_contexts) / len(optimized_contexts)
            self.response_metrics['avg_context_relevance'] = (
                (self.response_metrics['avg_context_relevance'] * (self.response_metrics['successful_responses'] - 1) + avg_relevance) /
                self.response_metrics['successful_responses']
            )
            
            result = {
                "response": response,
                "sources": sources,
                "query": query,
                "search_strategy": search_strategy,
                "response_time": response_time,
                "context_count": len(optimized_contexts)
            }
            
            if include_debug_info:
                result["debug_info"] = {
                    "total_contexts_found": len(contexts),
                    "contexts_after_reranking": len(reranked_contexts),
                    "contexts_used": len(optimized_contexts),
                    "avg_context_score": avg_relevance,
                    "response_time": response_time,
                    "search_strategy": search_strategy
                }
            
            return result
            
        except Exception as e:
            st.error(f"Error in RAG pipeline: {str(e)}")
            return {
                "response": "I encountered an error while processing your query. Please try again.",
                "sources": [],
                "error": str(e)
            }
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get comprehensive pipeline statistics"""
        try:
            vector_stats = self.vector_store.get_collection_stats()
            llm_info = self.llm_handler.get_model_info()
            
            return {
                "vector_store": vector_stats,
                "llm_model":{
                    "name": "Mistral-7B-Instruct-v0.1",
                    "provider": "Hugging Face",
                    "auth": "via .env token",
                    "context_limit_tokens": 4096
                },
                "pipeline_config": {
                    "top_k_retrieval": self.top_k,
                    "similarity_threshold": self.similarity_threshold,
                    "use_hybrid_search": self.use_hybrid_search,
                    "search_weights": self.search_weights
                },
                "performance_metrics": self.response_metrics,
                "capabilities": [
                    "Hybrid search (semantic + keyword)",
                    "Context reranking",
                    "Response optimization",
                    "Multi-model LLM support",
                    "Performance tracking",
                    "Debug information"
                ]
            }
            
        except Exception as e:
            return {"error": f"Failed to get pipeline stats: {str(e)}"}
    
    def update_search_weights(self, semantic_weight: float, keyword_weight: float):
        """Update search strategy weights"""
        total = semantic_weight + keyword_weight
        if total > 0:
            self.search_weights['semantic'] = semantic_weight / total
            self.search_weights['keyword'] = keyword_weight / total
            st.success(f"Updated search weights: Semantic={self.search_weights['semantic']:.2f}, Keyword={self.search_weights['keyword']:.2f}")
    
    def reset_metrics(self):
        """Reset performance metrics"""
        self.response_metrics = {
            'total_queries': 0,
            'successful_responses': 0,
            'avg_response_time': 0,
            'avg_context_relevance': 0
        }
        st.success("Performance metrics reset")