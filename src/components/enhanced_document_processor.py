"""
Enhanced Document Processor with advanced text preprocessing
Handles multiple document formats with intelligent chunking
"""

import re
import PyPDF2
import io
from typing import List, Dict, Any, Optional
import streamlit as st
from utils.text_preprocessor import TextPreprocessor
from typing import List
import re
from typing import List, Dict, Any


class EnhancedDocumentProcessor:
    """Advanced document processing with smart chunking and preprocessing"""
    
    def __init__(self, 
                 chunk_size: int = 300, 
                 overlap: int = 50,
                 min_chunk_size: int = 100):
        self.chunk_size = chunk_size  # Target words per chunk
        self.overlap = overlap  # Word overlap between chunks
        self.min_chunk_size = min_chunk_size  # Minimum viable chunk size
        self.preprocessor = TextPreprocessor()
        
        # Sentence boundary patterns for better chunking
        self.sentence_endings = re.compile(r'[.!?]+')
        self.paragraph_break = re.compile(r'\n\s*\n')
    
    def extract_text_from_pdf(self, pdf_file) -> str:
        """Enhanced PDF text extraction with error handling"""
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            text_parts = []
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text.strip():
                        # Add page separator for context
                        text_parts.append(f"\n--- Page {page_num + 1} ---\n{page_text}")
                except Exception as e:
                    st.warning(f"Error extracting page {page_num + 1}: {str(e)}")
                    continue
            
            raw_text = "\n".join(text_parts)
            
            if not raw_text.strip():
                raise Exception("No readable text found in PDF")
            
            return raw_text
            
        except Exception as e:
            st.error(f"PDF extraction failed: {str(e)}")
            return ""
    
    def preprocess_document(self, text: str, 
                          remove_urls: bool = True,
                          remove_html: bool = True,
                          clean_special: bool = True,
                           lowercase: bool = True,
                           normalize_unicode: bool = True,
                           remove_stopwords: bool = False,
                           lemmatize: bool = False) -> str:
        """Preprocess document text using advanced cleaning"""
        
        with st.spinner("Preprocessing document..."):
            if normalize_unicode:
                import unicodedata
                text = unicodedata.normalize("NFKD", text)
            if lowercase:
                text = text.lower()
            cleaned_text = self.preprocessor.preprocess_text(
                text,
                remove_urls=remove_urls,
                remove_html=remove_html,
                remove_emails=True,
                remove_phones=True,
                clean_special=clean_special
            )
            if remove_stopwords or lemmatize:
                import nltk
                from nltk.corpus import stopwords
                from nltk.stem import WordNetLemmatizer
                from nltk.tokenize import word_tokenize
                nltk.download('punkt', quiet=True)
                nltk.download('stopwords', quiet=True)
                nltk.download('wordnet', quiet=True)
                tokens = word_tokenize(cleaned_text)
                if remove_stopwords:
                    tokens = [word for word in tokens if word not in stopwords.words('english')]
                if lemmatize:
                    lemmatizer = WordNetLemmatizer()
                    tokens = [lemmatizer.lemmatize(word) for word in tokens]
                cleaned_text = ' '.join(tokens)

        
        
        # Show preprocessing stats
        stats = self.preprocessor.get_preprocessing_stats(text, cleaned_text)
        
        if stats['reduction_percentage'] > 5:
            with st.expander("📊 Preprocessing Statistics", expanded=False):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Original Length", f"{stats['original_length']:,} chars")
                    st.metric("Original Words", f"{stats['original_words']:,}")
                
                with col2:
                    st.metric("Processed Length", f"{stats['processed_length']:,} chars")
                    st.metric("Processed Words", f"{stats['processed_words']:,}")
                
                with col3:
                    st.metric("Size Reduction", f"{stats['reduction_percentage']:.1f}%")
                    if stats['urls_removed'] > 0:
                        st.metric("URLs Removed", stats['urls_removed'])
        
        return cleaned_text
    
    def intelligent_sentence_split(self, text: str) -> List[str]:
        """Smart sentence splitting with context preservation"""
        # Split by paragraph first
        paragraphs = self.paragraph_break.split(text)
        
        sentences = []
        for paragraph in paragraphs:
            if not paragraph.strip():
                continue
            
            # Split paragraph into sentences
            para_sentences = re.split(r'(?<=[.?!])\s+', paragraph)
            
            # Clean and filter sentences
            for sentence in para_sentences:
                sentence = sentence.strip()
                if len(sentence.split()) >= 3:  # Minimum 3 words
                    sentences.append(sentence)
        
        return sentences
    
    def semantic_chunking(self, text: str) -> List[str]:
        """Create semantically coherent chunks"""
        sentences = self.intelligent_sentence_split(text)
        
        chunks = []
        current_chunk = []
        current_word_count = 0
        
        for sentence in sentences:
            sentence_words = len(sentence.split())
            
            # Check if adding this sentence would exceed chunk size
            if current_word_count + sentence_words > self.chunk_size and current_chunk:
                # Finalize current chunk
                chunk_text = '. '.join(current_chunk) + '.'
                
                # Only add chunk if it meets minimum size
                if len(chunk_text.split()) >= self.min_chunk_size:
                    chunks.append(chunk_text)
                
                # Start new chunk with overlap
                overlap_sentences = []
                overlap_words = 0
                
                # Add sentences from end for overlap
                for s in reversed(current_chunk):
                    s_words = len(s.split())
                    if overlap_words + s_words <= self.overlap:
                        overlap_sentences.insert(0, s)
                        overlap_words += s_words
                    else:
                        break
                    
                    
                
                current_chunk = overlap_sentences[:]
                current_word_count = overlap_words
            
            # Add current sentence
            current_chunk.append(sentence)
            current_word_count += sentence_words
        
        # Add final chunk
        if current_chunk:
            chunk_text = '. '.join(current_chunk) + '.'
            if len(chunk_text.split()) >= self.min_chunk_size:
                chunks.append(chunk_text)
        
        return chunks
    
    def create_chunk_metadata(self, chunk: str, chunk_id: int, source: str = "document") -> Dict[str, Any]:
        """Create comprehensive metadata for each chunk"""
        words = chunk.split()
        
        # Extract key phrases (simple implementation)
        chunk_lower = chunk.lower()
        
        # Look for capitalized words (potential entities)
        entities = re.findall(r'\b(?:[A-Z][a-z]*\s*){1,3}', chunk)
        
        # Look for numbers and dates
        numbers = re.findall(r'\b\d+(?:\.\d+)?\b', chunk)
        dates = re.findall(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', chunk)

        sentence_count = max(1, len(self.sentence_endings.split(chunk)))
        
        return {
            'id': chunk_id,
            'content': chunk,
            'word_count': len(words),
            'char_count': len(chunk),
            'sentence_count': len(self.sentence_endings.split(chunk)),
            'source': source,
            'entities': entities[:5],  # Top 5 entities
            'numbers': numbers[:3],    # Top 3 numbers
            'dates': dates,
            'avg_word_length': sum(len(word) for word in words) / len(words) if words else 0,
            'readability_score': self._calculate_readability(chunk)
        }
    
    def _calculate_readability(self, text: str) -> float:
        """Simple readability score (Flesch Reading Ease approximation)"""
        words = text.split()
        sentence_count = max(1, len(self.sentence_endings.split(text)))

        avg_sentence_len = len(words) / sentence_count if words else 0
        
        syllable_count = sum(
            max(1, len(re.findall(r'[aeiouyAEIOUY]+', word))) for word in words
        )
        avg_syllables = syllable_count / len(words) if words else 0
        
        # Simplified Flesch Reading Ease
        score = 206.835 - (1.015 * avg_sentence_len) - (84.6 * avg_syllables)
        
        # Normalize to 0-100 scale
        return round(max(0, min(100, score)), 2)
    
    def process_document(self, 
                        text: str, 
                        source: str = "uploaded_document",
                        preprocess: bool = True) -> List[Dict[str, Any]]:
        """Complete document processing pipeline"""
        
        try:
            # Step 1: Preprocessing
            if preprocess:
                processed_text = self.preprocess_document(text)
            else:
                processed_text = text
            
            if not processed_text.strip():
                st.error("No text content found after preprocessing")
                return []
            
            # Step 2: Intelligent chunking
            with st.spinner("Creating intelligent chunks..."):
                chunks = self.semantic_chunking(processed_text)
            
            if not chunks:
                st.error("No valid chunks created from document")
                return []
            
            # Step 3: Create comprehensive metadata
            processed_chunks = []
            for i, chunk in enumerate(chunks):
                metadata = self.create_chunk_metadata(chunk, i, source)
                processed_chunks.append(metadata)
            
            # Display processing summary
            avg_readability = sum(chunk['readability_score'] for chunk in processed_chunks) / len(processed_chunks)
            avg_words = sum(chunk['word_count'] for chunk in processed_chunks) / len(processed_chunks)
            total_words = sum(c['word_count'] for c in processed_chunks)
            
            with st.success("Document processing completed!"):
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Chunks", len(processed_chunks))
                
                with col2:
                    st.metric("Avg Words/Chunk", f"{avg_words:.0f}")
                
                with col3:
                    st.metric("Avg Readability", f"{avg_readability:.0f}")
                
                with col4:
                    total_words = sum(chunk['word_count'] for chunk in processed_chunks)
                    st.metric("Total Words", f"{total_words:,}")
            
            return processed_chunks
            
        except Exception as e:
            st.error(f"Document processing failed: {str(e)}")
            return []
    
    def process_pdf_file(self, pdf_file) -> List[Dict[str, Any]]:
        """Process PDF file with extraction and chunking"""
        try:
            # Extract text from PDF
            raw_text = self.extract_text_from_pdf(pdf_file)
            
            if not raw_text:
                return []
            
            # Process the extracted text
            return self.process_document(
                raw_text, 
                source=f"PDF: {pdf_file.name}",
                preprocess=True
            )
            
        except Exception as e:
            st.error(f"PDF processing failed: {str(e)}")
            return []
    
    def process_text_file(self, text_content: str, filename: str = "text_file") -> List[Dict[str, Any]]:
        """Process plain text content"""
        return self.process_document(
            text_content,
            source=f"Text: {filename}",
            preprocess=True
        )
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processor configuration and statistics"""
        return {
            'chunk_size': self.chunk_size,
            'overlap': self.overlap,
            'min_chunk_size': self.min_chunk_size,
            'preprocessing_enabled': True,
            'supported_formats': ['txt', 'pdf'],
            'features': [
                'HTML/URL cleaning',
                'Context-aware sentence splitting',
                'Semantic chunking with overlap',
                'Named Entity Recognition (NER)',
                'Readability scoring for LLM optimization',
                'Metadata enrichment (source tagging, position indexing)'
            ]
        }