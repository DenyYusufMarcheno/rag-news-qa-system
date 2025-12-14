"""Document preprocessing and text processing utilities."""

import re
from typing import List, Dict, Any, Optional
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)


class DocumentPreprocessor:
    """Preprocess documents for indexing and retrieval."""
    
    def __init__(self):
        """Initialize preprocessor with NLTK tools."""
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text.
        
        Args:
            text: Raw text string
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep spaces
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def tokenize(self, text: str, remove_stopwords: bool = False, 
                 apply_stemming: bool = False) -> List[str]:
        """Tokenize text with optional preprocessing.
        
        Args:
            text: Text to tokenize
            remove_stopwords: Whether to remove stopwords
            apply_stemming: Whether to apply stemming
            
        Returns:
            List of tokens
        """
        # Clean text first
        text = self.clean_text(text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords if requested
        if remove_stopwords:
            tokens = [t for t in tokens if t not in self.stop_words]
        
        # Apply stemming if requested
        if apply_stemming:
            tokens = [self.stemmer.stem(t) for t in tokens]
        
        return tokens
    
    def preprocess_document(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess a single document.
        
        Args:
            doc: Document dictionary with 'headline', 'short_description', 'category'
            
        Returns:
            Preprocessed document with additional fields
        """
        # Combine headline and description for better text representation
        headline = doc.get('headline', '')
        description = doc.get('short_description', '')
        category = doc.get('category', 'GENERAL')
        
        # Create combined text with category weighting
        # Headline is included twice to increase its importance in BM25 scoring
        # This is a simple TF-IDF boosting technique - the headline is typically
        # more informative than the description, and this helps BM25 match on
        # headline terms more strongly. Trade-off: may over-weight headline matches.
        combined_text = f"{headline} {headline} {description}"
        
        # Add category information for better context
        combined_text = f"{category.lower()} {combined_text}"
        
        # Clean the combined text
        cleaned_text = self.clean_text(combined_text)
        
        # Tokenize for BM25 (with stopwords removed and stemming)
        tokens_processed = self.tokenize(combined_text, 
                                        remove_stopwords=True, 
                                        apply_stemming=True)
        
        # Tokenize for display (minimal processing)
        tokens_display = self.tokenize(combined_text, 
                                      remove_stopwords=False, 
                                      apply_stemming=False)
        
        # Return enhanced document
        return {
            **doc,
            'combined_text': combined_text,
            'cleaned_text': cleaned_text,
            'tokens_processed': tokens_processed,
            'tokens_display': tokens_display,
            'text_for_embedding': f"{headline}. {description}"  # For FAISS embeddings
        }
    
    def preprocess_documents(self, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Preprocess a list of documents.
        
        Args:
            docs: List of document dictionaries
            
        Returns:
            List of preprocessed documents
        """
        return [self.preprocess_document(doc) for doc in docs]


def get_category_mapping() -> Dict[str, List[str]]:
    """Get category mapping for intent detection.
    
    Returns:
        Dictionary mapping intents to categories
    """
    return {
        'economy': ['BUSINESS', 'MONEY', 'POLITICS'],
        'technology': ['TECH', 'SCIENCE'],
        'sports': ['SPORTS'],
        'entertainment': ['ENTERTAINMENT', 'ARTS & CULTURE', 'ARTS'],
        'politics': ['POLITICS', 'WORLD NEWS'],
        'health': ['WELLNESS', 'HEALTHY LIVING', 'PARENTING'],
        'general': ['U.S. NEWS', 'WORLD NEWS', 'IMPACT']
    }


def normalize_category(category: str) -> str:
    """Normalize category name for matching.
    
    Args:
        category: Raw category string
        
    Returns:
        Normalized category
    """
    if not category:
        return 'GENERAL'
    
    # Convert to uppercase and handle common variations
    category = category.upper().strip()
    
    # Map common variations
    category_map = {
        'ARTS': 'ARTS & CULTURE',
        'CULTURE': 'ARTS & CULTURE',
        'US NEWS': 'U.S. NEWS',
        'WORLD': 'WORLD NEWS',
        'TECH': 'TECH',
        'TECHNOLOGY': 'TECH',
        'SCIENCE': 'SCIENCE',
    }
    
    return category_map.get(category, category)
