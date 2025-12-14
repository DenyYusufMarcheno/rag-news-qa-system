"""Query preprocessing, expansion, and intent detection."""

import re
from typing import List, Dict, Any, Optional, Tuple
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import yaml
import os


class QueryProcessor:
    """Process and enhance queries for better retrieval."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize query processor.
        
        Args:
            config_path: Path to retrieval configuration file
        """
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        
        # Load configuration
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = self._default_config()
        
        self.intent_keywords = self.config.get('intent_keywords', {})
        self.expansion_keywords = self.config.get('expansion_keywords', {})
        self.category_mappings = self.config.get('category_mappings', {})
    
    def _default_config(self) -> Dict[str, Any]:
        """Get default configuration if no config file provided."""
        return {
            'intent_keywords': {
                'economy': ['economy', 'economic', 'gdp', 'recession', 'inflation', 
                           'employment', 'jobs', 'market', 'stock', 'business'],
                'technology': ['technology', 'tech', 'software', 'ai', 'computer'],
                'sports': ['sports', 'game', 'team', 'player', 'match'],
                'entertainment': ['movie', 'music', 'celebrity', 'entertainment'],
                'politics': ['politics', 'government', 'election', 'policy'],
                'health': ['health', 'medical', 'wellness', 'doctor']
            },
            'expansion_keywords': {
                'economy': ['GDP', 'recession', 'inflation', 'employment', 'business'],
                'technology': ['innovation', 'digital', 'software'],
                'sports': ['competition', 'championship'],
                'entertainment': ['celebrity', 'show', 'performance'],
                'politics': ['government', 'policy'],
                'health': ['wellness', 'medical']
            },
            'category_mappings': {
                'economy': ['BUSINESS', 'MONEY', 'POLITICS'],
                'technology': ['TECH', 'SCIENCE'],
                'sports': ['SPORTS'],
                'entertainment': ['ENTERTAINMENT', 'ARTS & CULTURE'],
                'politics': ['POLITICS', 'WORLD NEWS'],
                'health': ['WELLNESS', 'HEALTHY LIVING']
            }
        }
    
    def clean_query(self, query: str) -> str:
        """Clean and normalize query.
        
        Args:
            query: Raw query string
            
        Returns:
            Cleaned query
        """
        if not query:
            return ""
        
        # Convert to lowercase
        query = query.lower()
        
        # Remove special characters but keep spaces
        query = re.sub(r'[^a-z0-9\s]', ' ', query)
        
        # Remove extra whitespace
        query = ' '.join(query.split())
        
        return query
    
    def extract_keywords(self, query: str, remove_stopwords: bool = True) -> List[str]:
        """Extract keywords from query.
        
        Args:
            query: Query string
            remove_stopwords: Whether to remove stopwords
            
        Returns:
            List of keywords
        """
        # Clean query
        query = self.clean_query(query)
        
        # Tokenize
        tokens = word_tokenize(query)
        
        # Remove stopwords if requested
        if remove_stopwords:
            tokens = [t for t in tokens if t not in self.stop_words and len(t) > 2]
        
        # Apply stemming
        keywords = [self.stemmer.stem(t) for t in tokens]
        
        return keywords
    
    def detect_intent(self, query: str) -> Tuple[Optional[str], float]:
        """Detect query intent/topic.
        
        Args:
            query: Query string
            
        Returns:
            Tuple of (intent, confidence_score)
        """
        query_lower = query.lower()
        query_tokens = set(word_tokenize(query_lower))
        
        intent_scores = {}
        
        # Score each intent based on keyword matches
        for intent, keywords in self.intent_keywords.items():
            # Convert keywords to set for O(1) lookups
            keyword_set = {kw.lower() for kw in keywords}
            
            # Count matching keywords efficiently
            matches = sum(1 for kw in keyword_set if kw in query_lower)
            if matches > 0:
                # Normalize by number of query tokens
                score = matches / max(len(query_tokens), 1)
                intent_scores[intent] = score
        
        # Return intent with highest score
        if intent_scores:
            best_intent = max(intent_scores.items(), key=lambda x: x[1])
            return best_intent[0], best_intent[1]
        
        return None, 0.0
    
    def expand_query(self, query: str, num_terms: int = 3) -> str:
        """Expand query with related terms.
        
        Args:
            query: Original query
            num_terms: Number of expansion terms to add
            
        Returns:
            Expanded query
        """
        # Detect intent
        intent, confidence = self.detect_intent(query)
        
        # If intent detected with reasonable confidence, add expansion terms
        if intent and confidence > 0.1:
            expansion_terms = self.expansion_keywords.get(intent, [])
            # Add up to num_terms expansion terms
            expansion = ' '.join(expansion_terms[:num_terms])
            return f"{query} {expansion}"
        
        return query
    
    def get_relevant_categories(self, query: str) -> List[str]:
        """Get relevant document categories for query.
        
        Args:
            query: Query string
            
        Returns:
            List of relevant category names
        """
        intent, confidence = self.detect_intent(query)
        
        if intent and confidence > 0.1:
            return self.category_mappings.get(intent, [])
        
        # Return empty list if no intent detected (no filtering)
        return []
    
    def process_query(self, query: str, 
                     expand: bool = True,
                     extract_kw: bool = True) -> Dict[str, Any]:
        """Process query with all enhancements.
        
        Args:
            query: Raw query string
            expand: Whether to expand query
            extract_kw: Whether to extract keywords
            
        Returns:
            Dictionary with processed query information
        """
        # Clean query
        cleaned = self.clean_query(query)
        
        # Detect intent
        intent, confidence = self.detect_intent(query)
        
        # Expand query if requested
        expanded = self.expand_query(query) if expand else query
        
        # Extract keywords if requested
        keywords = self.extract_keywords(query) if extract_kw else []
        
        # Get relevant categories
        categories = self.get_relevant_categories(query)
        
        return {
            'original': query,
            'cleaned': cleaned,
            'expanded': expanded,
            'keywords': keywords,
            'intent': intent,
            'confidence': confidence,
            'categories': categories
        }
