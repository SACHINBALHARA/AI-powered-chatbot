"""
Enhanced RAG Chatbot Application
Enterprise-grade retrieval-augmented generation with advanced features
"""

import streamlit as st
import os
import sys
import time
from typing import List, Dict, Any, Optional

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.components.enhanced_document_processor import EnhancedDocumentProcessor
from src.components.enhanced_vector_store import EnhancedVectorStore
from src.components.advanced_llm_handler import AdvancedLLMHandler
from src.components.enhanced_rag_pipeline import EnhancedRAGPipeline

# Page configuration
st.set_page_config(
    page_title="Enterprise RAG Chatbot",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header { 
        font-size: 2.5rem; 
        font-weight: bold; 
        color: #1f77b4; 
        text-align: center; 
        margin-bottom: 2rem; 
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .source-container {
        background-color: #f8f9fa;
        border-left: 4px solid #007bff;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0 0.5rem 0.5rem 0;
    }
    .debug-info {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        padding: 0.5rem;
        border-radius: 0.25rem;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def initialize_session_state():
    """Initialize all session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "rag_pipeline" not in st.session_state:
        st.session_state.rag_pipeline = None
    if "document_processed" not in st.session_state:
        st.session_state.document_processed = False
    if "num_chunks" not in st.session_state:
        st.session_state.num_chunks = 0
    if "processing_stats" not in st.session_state:
        st.session_state.processing_stats = {}
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = "mistral-7b"
    if "search_strategy" not in st.session_state:
        st.session_state.search_strategy = "hybrid"
    if "debug_mode" not in st.session_state:
        st.session_state.debug_mode = False

@st.cache_resource
def initialize_components(model_name: str):
    """Initialize RAG components with caching"""
    try:
        # Initialize document processor
        doc_processor = EnhancedDocumentProcessor(
            chunk_size=300,
            overlap=50,
            min_chunk_size=100
        )
        
        # Initialize vector store with ChromaDB
        vector_store = EnhancedVectorStore(
            collection_name="enterprise_rag_docs",
            embedding_model="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Initialize LLM handler
        llm_handler = AdvancedLLMHandler(
            model_name=model_name,
            max_length=512,
            temperature=0.7
        )
        
        # Initialize RAG pipeline
        rag_pipeline = EnhancedRAGPipeline(
            vector_store=vector_store,
            llm_handler=llm_handler,
            top_k=5,
            similarity_threshold=0.3,
            use_hybrid_search=True
        )
        
        return doc_processor, vector_store, llm_handler, rag_pipeline
        
    except Exception as e:
        st.error(f"Component initialization failed: {str(e)}")
        return None, None, None, None

def process_document(uploaded_file, doc_processor, vector_store):
    """Enhanced document processing with detailed feedback"""
    try:
        # Determine file type and process accordingly
        if uploaded_file.type == "text/plain":
            content = str(uploaded_file.read(), "utf-8")
            chunks = doc_processor.process_text_file(content, uploaded_file.name)
        elif uploaded_file.type == "application/pdf":
            chunks = doc_processor.process_pdf_file(uploaded_file)
        else:
            st.error("Unsupported file type. Please upload .txt or .pdf files.")
            return False
        
        if not chunks:
            st.error("No content could be extracted from the document.")
            return False
        
        # Add to vector store
        success = vector_store.add_documents(chunks)
        
        if success:
            st.session_state.num_chunks = len(chunks)
            st.session_state.document_processed = True
            st.session_state.processing_stats = {
                'total_chunks': len(chunks),
                'avg_chunk_size': sum(c['word_count'] for c in chunks) / len(chunks),
                'total_words': sum(c['word_count'] for c in chunks),
                'avg_readability': sum(c.get('readability_score', 0) for c in chunks) / len(chunks)
            }
            
            return True
        else:
            st.error("Failed to process document into vector store.")
            return False
            
    except Exception as e:
        st.error(f"Document processing error: {str(e)}")
        return False

def display_chat_message(role: str, content: str, sources: Optional[List[Dict]] = None, debug_info: Optional[Dict] = None):
    """Enhanced chat message display with sources and debug info"""
    with st.chat_message(role):
        st.markdown(content)
        
        if sources and role == "assistant":
            with st.expander(f"📚 Sources ({len(sources)} found)", expanded=False):
                for i, source in enumerate(sources, 1):
                    st.markdown(f"**Source {i}** (Score: {source['score']:.3f})")
                    
                    if 'rerank_score' in source:
                        st.markdown(f"*Rerank Score: {source['rerank_score']:.3f}*")
                    
                    st.markdown(f"```\n{source['content']}\n```")
                    
                    # Show metadata if available
                    if source.get('metadata'):
                        metadata = source['metadata']
                        cols = st.columns(4)
                        with cols[0]:
                            st.metric("Words", metadata.get('word_count', 'N/A'))
                        with cols[1]:
                            st.metric("Readability", f"{metadata.get('readability_score', 0):.0f}")
                        with cols[2]:
                            if metadata.get('entities'):
                                st.write("**Entities:**", ", ".join(metadata['entities'][:3]))
                        with cols[3]:
                            st.write("**Source:**", metadata.get('source', 'Unknown'))
                    
                    if i < len(sources):
                        st.divider()
        
        if debug_info and st.session_state.debug_mode and role == "assistant":
            with st.expander("🔧 Debug Information", expanded=False):
                st.json(debug_info)

def stream_response(response_generator):
    """Enhanced response streaming with better visual feedback"""
    placeholder = st.empty()
    full_response = ""
    
    for chunk in response_generator:
        full_response += chunk
        placeholder.markdown(full_response + "▌")
        time.sleep(0.02)
    
    placeholder.markdown(full_response)
    return full_response

def main():
    """Main application function"""
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">🧠 Enterprise RAG Chatbot</h1>', unsafe_allow_html=True)
    st.markdown("*Advanced retrieval-augmented generation with ChromaDB, multi-model LLM support, and intelligent preprocessing*")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model selection
        model_options = {
            "Mistral 7B": "mistral-7b",
            "Llama 2 7B": "llama-7b", 
            "Zephyr 7B": "zephyr-7b",
            "Phi-2": "phi-2",
            "DialoGPT Medium": "microsoft/DialoGPT-medium"
        }
        
        selected_model_name = st.selectbox(
            "🤖 Select LLM Model",
            options=list(model_options.keys()),
            index=0
        )
        st.session_state.selected_model = model_options[selected_model_name]
        
        # Search strategy
        st.session_state.search_strategy = st.selectbox(
            "🔍 Search Strategy",
            options=["hybrid", "semantic", "keyword"],
            index=0,
            help="Hybrid combines semantic and keyword search for best results"
        )
        
        # Debug mode
        st.session_state.debug_mode = st.checkbox(
            "🔧 Debug Mode",
            help="Show detailed processing information"
        )
        
        st.divider()
        
        # Document upload section
        st.header("📁 Document Upload")
        
        uploaded_file = st.file_uploader(
            "Upload Document",
            type=['txt', 'pdf'],
            help="Upload text (.txt) or PDF (.pdf) files for analysis"
        )
        
        if uploaded_file is not None and not st.session_state.document_processed:
            if st.button("🚀 Process Document", type="primary"):
                # Initialize components
                with st.spinner("Initializing components..."):
                    doc_processor, vector_store, llm_handler, rag_pipeline = initialize_components(
                        st.session_state.selected_model
                    )
                
                if doc_processor and vector_store:
                    # Process document
                    success = process_document(uploaded_file, doc_processor, vector_store)
                    
                    if success:
                        st.session_state.vector_store = vector_store
                        st.session_state.rag_pipeline = rag_pipeline
                        st.rerun()
        
        # Document status
        if st.session_state.document_processed:
            st.success("✅ Document Ready")
            
            stats = st.session_state.processing_stats
            st.markdown("**Document Statistics:**")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Chunks", stats.get('total_chunks', 0))
                st.metric("Total Words", f"{stats.get('total_words', 0):,}")
            
            with col2:
                st.metric("Avg Chunk Size", f"{stats.get('avg_chunk_size', 0):.0f}")
                st.metric("Avg Readability", f"{stats.get('avg_readability', 0):.0f}")
        
        st.divider()
        
        # System information
        st.header("📊 System Info")
        
        if st.session_state.rag_pipeline:
            pipeline_stats = st.session_state.rag_pipeline.get_pipeline_stats()
            
            st.markdown("**Current Model:**")
            st.info(f"🤖 {selected_model_name}")
            
            st.markdown("**Performance:**")
            metrics = pipeline_stats.get('performance_metrics', {})
            if metrics.get('total_queries', 0) > 0:
                st.metric("Total Queries", metrics['total_queries'])
                st.metric("Avg Response Time", f"{metrics.get('avg_response_time', 0):.2f}s")
                st.metric("Success Rate", f"{metrics.get('successful_responses', 0)/metrics['total_queries']*100:.1f}%")
        
        # Control buttons
        st.divider()
        
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()
        
        if st.button("🔄 Reset System"):
            for key in ['messages', 'vector_store', 'rag_pipeline', 'document_processed', 
                       'num_chunks', 'processing_stats']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
        
        if st.session_state.document_processed and st.button("📤 Export Data"):
            if st.session_state.vector_store:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                export_path = f"export_rag_data_{timestamp}.json"
                
                if st.session_state.vector_store.export_collection(export_path):
                    st.success(f"Data exported to {export_path}")
    
    # Main chat interface
    if not st.session_state.document_processed:
        st.info("👆 Upload and process a document to start chatting!")
        
        # Show sample capabilities
        st.markdown("### 🌟 Advanced Features")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **🔍 Smart Search**
            - Hybrid semantic + keyword search
            - Context reranking
            - Relevance optimization
            """)
        
        with col2:
            st.markdown("""
            **🧠 Advanced LLMs**
            - Mistral, Llama, Zephyr support
            - Fine-tuning ready
            - Multi-strategy generation
            """)
        
        with col3:
            st.markdown("""
            **📄 Document Processing**
            - HTML/URL cleaning
            - Smart chunking
            - Entity extraction
            """)
        
        return
    
    # Display chat history
    for message in st.session_state.messages:
        display_chat_message(
            message["role"], 
            message["content"], 
            message.get("sources"),
            message.get("debug_info")
        )
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your document..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        display_chat_message("user", prompt)
        
        # Generate response
        if st.session_state.rag_pipeline:
            try:
                with st.chat_message("assistant"):
                    with st.spinner("Processing your question..."):
                        # Get response from RAG pipeline
                        response_data = st.session_state.rag_pipeline.get_response(
                            prompt,
                            search_strategy=st.session_state.search_strategy,
                            include_debug_info=st.session_state.debug_mode
                        )
                        
                        if response_data and response_data.get("response"):
                            response = response_data["response"]
                            sources = response_data.get("sources", [])
                            debug_info = response_data.get("debug_info")
                            
                            # Stream the response
                            def response_generator():
                                words = response.split()
                                for i, word in enumerate(words):
                                    if i == 0:
                                        yield word
                                    else:
                                        yield " " + word
                            
                            full_response = stream_response(response_generator())
                            
                            # Show performance info
                            if response_data.get('response_time'):
                                st.caption(f"⏱️ Response time: {response_data['response_time']:.2f}s | "
                                         f"📊 Contexts used: {response_data.get('context_count', 0)}")
                            
                            # Display sources
                            if sources:
                                with st.expander(f"📚 Source References ({len(sources)} found)", expanded=False):
                                    for i, source in enumerate(sources, 1):
                                        st.markdown(f"**Source {i}** (Relevance: {source['score']:.3f})")
                                        
                                        if 'rerank_score' in source:
                                            st.markdown(f"*Rerank Score: {source['rerank_score']:.3f}*")
                                        
                                        st.markdown(source['content'])
                                        
                                        # Metadata display
                                        if source.get('metadata'):
                                            metadata = source['metadata']
                                            col1, col2, col3 = st.columns(3)
                                            with col1:
                                                st.metric("Words", metadata.get('word_count', 'N/A'))
                                            with col2:
                                                st.metric("Readability", f"{metadata.get('readability_score', 0):.0f}")
                                            with col3:
                                                if metadata.get('entities'):
                                                    st.write("**Entities:**", ", ".join(metadata['entities'][:3]))
                                        
                                        if i < len(sources):
                                            st.divider()
                            
                            # Debug information
                            if debug_info and st.session_state.debug_mode:
                                with st.expander("🔧 Debug Information", expanded=False):
                                    st.json(debug_info)
                            
                            # Add message to history
                            st.session_state.messages.append({
                                "role": "assistant", 
                                "content": full_response,
                                "sources": sources,
                                "debug_info": debug_info if st.session_state.debug_mode else None
                            })
                        else:
                            error_msg = "I couldn't generate a response. Please try rephrasing your question."
                            st.error(error_msg)
                            st.session_state.messages.append({
                                "role": "assistant", 
                                "content": error_msg
                            })
                            
            except Exception as e:
                error_msg = f"An error occurred: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": error_msg
                })
        else:
            st.error("RAG pipeline not initialized. Please reset and try again.")

if __name__ == "__main__":
    main()