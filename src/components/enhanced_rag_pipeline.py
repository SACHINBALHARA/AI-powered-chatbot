"""
Enhanced RAG Pipeline with advanced retrieval and generation
Supports multiple search strategies and response optimization
"""

from typing import List, Dict, Optional, Any
import streamlit as st
import time
from enhanced_vector_store import EnhancedVectorStore
from advanced_llm_handler import AdvancedLLMHandler


class EnhancedRAGPipeline:
    """Advanced RAG pipeline with multi-strategy retrieval, scoring, and optimized generation."""

    def __init__(self,
                 vector_store: EnhancedVectorStore,
                 llm_handler: AdvancedLLMHandler,
                 top_k: int = 5,
                 similarity_threshold: float = 0.3,
                 max_tokens=2048,
                 use_hybrid_search: bool = True):
        
        self.vector_store = vector_store
        self.llm_handler = llm_handler
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.use_hybrid_search = use_hybrid_search
        self.max_tokens=2048
        
        self.search_weights = {
            'semantic': 0.7,
            'keyword': 0.3
        }

        self.response_metrics = {
            'total_queries': 0,
            'successful_responses': 0,
            'avg_response_time': 0,
            'avg_context_relevance': 0
        }

        # NEW: Cache for feedback / evaluations
        self.feedback_log = []

    def retrieve_context(self, query: str, search_strategy: str = "hybrid") -> List[Dict]:
        """Improved context retrieval with hybrid strategies and semantic filtering"""
        try:
            start_time = time.time()

            if search_strategy == "hybrid" and self.use_hybrid_search:
                results = self.vector_store.hybrid_search(query=query, k=self.top_k, alpha=self.search_weights['semantic'])
            else:
                results = self.vector_store.similarity_search(query=query, k=self.top_k)

            # FILTER OUT low-score results early
            filtered = [r for r in results if r.get('score', 0) >= self.similarity_threshold]

            # Enrich with retrieval metadata
            for res in filtered:
                res['retrieval_time'] = time.time() - start_time
                res['query_similarity'] = self._calculate_query_similarity(query, res['content'])

            return filtered
        except Exception as e:
            st.error(f"[RAG] Retrieval failed: {str(e)}")
            return []

    def rerank_contexts(self, contexts: List[Dict], query: str) -> List[Dict]:
        """Improved reranking using multiple scoring signals including embedding similarity"""
        try:
            for ctx in contexts:
                content = ctx['content']
                # Add cosine similarity score if vector representations exist
                if 'embedding' in ctx:
                    ctx['cosine_sim'] = self.vector_store.calculate_cosine_similarity(query, ctx['embedding'])

                ctx['rerank_score'] = (
                    0.4 * ctx.get('score', 0.5) +
                    0.2 * self._calculate_query_similarity(query, content) +
                    0.2 * ctx.get('cosine_sim', 0.0) +
                    0.1 * ctx.get('metadata', {}).get('readability_score', 50) / 100 +
                    0.1 * self._entity_match_score(query, ctx.get('metadata', {}).get('entities', []))
                )
            return sorted(contexts, key=lambda x: x['rerank_score'], reverse=True)
        except Exception as e:
            st.warning(f"[RAG] Reranking error: {str(e)}")
            return contexts

    def _entity_match_score(self, query: str, entities: List[str]) -> float:
        """Simple entity query matching score"""
        query_lower = query.lower()
        return sum(1 for e in entities if e.lower() in query_lower) * 0.2

    def optimize_context_selection(self, contexts: List[Dict], max_context_length: int = 2000) -> List[Dict]:
        """Merge similar chunks and filter noisy ones"""
        unique_chunks = []
        total_length = 0
        seen_hashes = set()

        for ctx in contexts:
            content = ctx['content']
            if not content or hash(content) in seen_hashes:
                continue
            chunk_len = len(content)
            if total_length + chunk_len > max_context_length:
                break
            seen_hashes.add(hash(content))
            unique_chunks.append(ctx)
            total_length += chunk_len
        return unique_chunks

    def generate_response(self, query: str, contexts: List[Dict]) -> str:
        """Enhanced response generation with LLM and confidence notes"""
        try:
            if not contexts:
                return "No relevant information found to answer the query."

            context_texts = [c['content'] for c in contexts]
            response = self.llm_handler.generate_response(query, context_texts)
            return self._enhance_response_with_sources(response, contexts)
        except Exception as e:
            return f"Error generating response: {e}"

    def _enhance_response_with_sources(self, response: str, contexts: List[Dict]) -> str:
        try:
            avg_score = sum(c.get('score', 0) for c in contexts) / len(contexts)
            if avg_score > 0.8:
                note = "\n\n*🔍 High-confidence response.*"
            elif avg_score > 0.5:
                note = "\n\n*ℹ️ Moderate confidence – review recommended.*"
            else:
                note = "\n\n*⚠️ Low confidence – verify manually.*"
            return response + note
        except:
            return response

    def get_response(self, query: str, search_strategy: str = "hybrid", include_debug_info: bool = False) -> Optional[Dict]:
        try:
            start_time = time.time()
            self.response_metrics['total_queries'] += 1

            with st.spinner("🔎 Retrieving..."):
                raw_contexts = self.retrieve_context(query, search_strategy)

            if not raw_contexts:
                return {"response": "No information found.", "sources": []}

            with st.spinner("📊 Reranking..."):
                reranked = self.rerank_contexts(raw_contexts, query)

            optimized = self.optimize_context_selection(reranked)
            with st.spinner("🤖 Generating..."):
                final_response = self.generate_response(query, optimized)

            response_time = time.time() - start_time
            self.response_metrics['successful_responses'] += 1
            self.response_metrics['avg_response_time'] = (
                (self.response_metrics['avg_response_time'] * (self.response_metrics['successful_responses'] - 1) + response_time) /
                self.response_metrics['successful_responses']
            )

            return {
                "response": final_response,
                "query": query,
                "sources": optimized,
                "response_time": round(response_time, 2),
                "context_used": len(optimized),
                "search_strategy": search_strategy
            }
        except Exception as e:
            return {"response": f"Pipeline error: {e}"}

    def submit_feedback(self, query: str, response: str, is_helpful: bool):
        """Collect binary feedback for tuning"""
        self.feedback_log.append({
            "query": query,
            "response": response,
            "helpful": is_helpful
        })

    def get_pipeline_stats(self):
        """Get overall pipeline performance and model settings"""
        return {
            "vector_store": self.vector_store.get_collection_stats(),
            "llm_model": self.llm_handler.get_model_info(),
            "pipeline_config": {
                "top_k": self.top_k,
                "threshold": self.similarity_threshold,
                "weights": self.search_weights
            },
            "metrics": self.response_metrics,
            "feedback_collected": len(self.feedback_log),
            "features": [
                "Hybrid search (semantic + keyword)",
                "Re-ranking with embeddings and metadata",
                "Response scoring with source notes",
                "Context deduplication",
                "Feedback loop for evaluation"
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