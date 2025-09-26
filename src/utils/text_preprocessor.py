"""
Advanced text preprocessing utilities for RAG chatbot
Handles URL removal, HTML cleaning, and text normalization
"""

import re
import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Optional
import streamlit as st


class TextPreprocessor:
    """Advanced text preprocessing with HTML, URL, and noise removal"""
    
    def __init__(self):
        self.url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        )
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        self.phone_pattern = re.compile(r'(\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}')
        self.html_tag_pattern = re.compile(r'<[^>]+>')
        
    def remove_urls(self, text: str) -> str:
        """Remove URLs from text"""
        return self.url_pattern.sub('', text)
    
    def remove_emails(self, text: str) -> str:
        """Remove email addresses from text"""
        return self.email_pattern.sub('[EMAIL]', text)
    
    def remove_phone_numbers(self, text: str) -> str:
        """Remove phone numbers from text"""
        return self.phone_pattern.sub('[PHONE]', text)
    
    def clean_html(self, text: str) -> str:
        """Remove HTML tags and decode HTML entities"""
        try:
            # Parse HTML content
            soup = BeautifulSoup(text, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "header", "footer", "aside"]):
                script.decompose()
            
            # Get text content
            clean_text = soup.get_text()
            
            # Clean up whitespace
            lines = (line.strip() for line in clean_text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            clean_text = ' '.join(chunk for chunk in chunks if chunk)
            
            return clean_text
            
        except Exception as e:
            st.warning(f"HTML cleaning error: {str(e)}")
            # Fallback: simple HTML tag removal
            return self.html_tag_pattern.sub('', text)
    
    def remove_extra_whitespace(self, text: str) -> str:
        """Clean excessive whitespace and normalize spacing"""
        # Replace multiple whitespace with single space
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        # Remove multiple newlines
        text = re.sub(r'\n+', '\n', text)
        
        return text
    
    def remove_special_patterns(self, text: str) -> str:
        """Remove common noise patterns in documents"""
        # Remove page numbers
        text = re.sub(r'\bPage\s+\d+\b', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\b\d+\s*/\s*\d+\b', '', text)  # Page x/y format
        
        # Remove copyright notices
        text = re.sub(r'©\s*\d{4}.*?(?:\n|$)', '', text, flags=re.IGNORECASE)
        text = re.sub(r'Copyright\s+\d{4}.*?(?:\n|$)', '', text, flags=re.IGNORECASE)
        
        # Remove excessive punctuation
        text = re.sub(r'[.]{3,}', '...', text)
        text = re.sub(r'[-]{3,}', '---', text)
        text = re.sub(r'[*]{3,}', '***', text)
        
        # Remove standalone numbers and dates in certain contexts
        text = re.sub(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', '[DATE]', text)
        
        return text
    
    def normalize_quotes(self, text: str) -> str:
        """Normalize different types of quotes"""
        # Replace smart quotes with regular quotes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        text = text.replace('„', '"').replace('‚', "'")
        
        return text
    
    def remove_redundant_info(self, text: str) -> str:
        """Remove common redundant information patterns"""
        # Remove common boilerplate text
        boilerplate_patterns = [
            r'Terms and Conditions',
            r'Privacy Policy',
            r'Cookie Policy',
            r'All rights reserved',
            r'Confidential and Proprietary',
            r'This document contains',
            r'For internal use only'
        ]
        
        for pattern in boilerplate_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        return text
    
    def preprocess_text(self, text: str, 
                       remove_urls: bool = True,
                       remove_html: bool = True,
                       remove_emails: bool = True,
                       remove_phones: bool = True,
                       clean_special: bool = True) -> str:
        """
        Comprehensive text preprocessing pipeline
        
        Args:
            text: Input text to preprocess
            remove_urls: Remove URLs from text
            remove_html: Clean HTML tags and entities
            remove_emails: Replace emails with placeholder
            remove_phones: Replace phone numbers with placeholder
            clean_special: Remove special patterns and noise
        
        Returns:
            Cleaned and preprocessed text
        """
        original_length = len(text)
        
        # HTML cleaning (should be first if HTML is present)
        if remove_html:
            text = self.clean_html(text)
        
        # URL removal
        if remove_urls:
            text = self.remove_urls(text)
        
        # Email and phone handling
        if remove_emails:
            text = self.remove_emails(text)
        
        if remove_phones:
            text = self.remove_phone_numbers(text)
        
        # Special pattern removal
        if clean_special:
            text = self.remove_special_patterns(text)
            text = self.remove_redundant_info(text)
        
        # Normalize quotes and whitespace
        text = self.normalize_quotes(text)
        text = self.remove_extra_whitespace(text)
        
        # Log preprocessing results
        final_length = len(text)
        reduction_pct = ((original_length - final_length) / original_length) * 100 if original_length > 0 else 0
        
        if reduction_pct > 10:  # Only log significant reductions
            st.info(f"Text preprocessing: {reduction_pct:.1f}% size reduction ({original_length} → {final_length} chars)")
        
        return text
    
    def extract_web_content(self, url: str) -> Optional[str]:
        """Extract clean text content from a web URL"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            # Clean and extract text
            clean_text = self.clean_html(response.text)
            clean_text = self.preprocess_text(clean_text)
            
            return clean_text
            
        except Exception as e:
            st.error(f"Error extracting content from {url}: {str(e)}")
            return None
    
    def get_preprocessing_stats(self, original_text: str, processed_text: str) -> Dict[str, any]:
        """Get detailed preprocessing statistics"""
        return {
            'original_length': len(original_text),
            'processed_length': len(processed_text),
            'reduction_percentage': ((len(original_text) - len(processed_text)) / len(original_text)) * 100,
            'original_words': len(original_text.split()),
            'processed_words': len(processed_text.split()),
            'urls_removed': len(self.url_pattern.findall(original_text)),
            'emails_found': len(self.email_pattern.findall(original_text)),
            'phones_found': len(self.phone_pattern.findall(original_text))
        }