"""Query preprocessing and expansion module."""

import re
from typing import List, Dict

class QueryProcessor:
    """Process and enhance queries for better retrieval."""
    
    def __init__(self):
        # Topic-specific keywords for expansion
        self.topic_keywords = {
            'economy': ['GDP', 'recession', 'inflation', 'employment', 'economic', 
                       'unemployment', 'market', 'business', 'financial', 'fiscal'],
            'technology': ['tech', 'software', 'hardware', 'AI', 'digital', 
                          'innovation', 'startup', 'computing', 'internet'],
            'politics': ['government', 'election', 'policy', 'legislation', 
                        'congress', 'senate', 'political', 'vote', 'president'],
            'sports': ['game', 'team', 'player', 'championship', 'tournament', 
                      'match', 'league', 'athletic', 'competition'],
            'health': ['medical', 'disease', 'treatment', 'hospital', 'doctor', 
                      'healthcare', 'medicine', 'COVID', 'vaccine', 'health'],
            'entertainment': ['movie', 'film', 'celebrity', 'music', 'show', 
                            'actor', 'artist', 'performance', 'entertainment']
        }
        
        # Stopwords that don't add value
        self.stopwords = {'is', 'the', 'this', 'that', 'how', 'what', 'when', 
                         'where', 'who', 'a', 'an', 'are', 'was', 'were'}
    
    def detect_topic(self, query: str) -> str:
        """Detect the main topic of the query."""
        query_lower = query.lower()
        
        # Topic detection based on keywords
        if any(word in query_lower for word in ['economy', 'economic', 'gdp', 'inflation', 'recession', 'unemployment', 'jobless']):
            return 'economy'
        elif any(word in query_lower for word in ['technology', 'tech', 'ai', 'software', 'digital', 'internet']):
            return 'technology'
        elif any(word in query_lower for word in ['politics', 'political', 'government', 'election', 'congress']):
            return 'politics'
        elif any(word in query_lower for word in ['sports', 'game', 'team', 'player', 'championship']):
            return 'sports'
        elif any(word in query_lower for word in ['health', 'medical', 'covid', 'vaccine', 'disease']):
            return 'health'
        elif any(word in query_lower for word in ['entertainment', 'movie', 'celebrity', 'music', 'show']):
            return 'entertainment'
        
        return 'general'
    
    def expand_query(self, query: str) -> str:
        """Expand query with related keywords."""
        topic = self.detect_topic(query)
        
        if topic != 'general' and topic in self.topic_keywords:
            # Add top 3 related keywords
            expansion_keywords = self.topic_keywords[topic][: 3]
            expanded = f"{query} {' '.join(expansion_keywords)}"
            return expanded
        
        return query
    
    def extract_keywords(self, query: str) -> List[str]:
        """Extract important keywords from query."""
        # Remove punctuation and convert to lowercase
        query_clean = re.sub(r'[^\w\s]', '', query.lower())
        
        # Split into words
        words = query_clean.split()
        
        # Filter out stopwords
        keywords = [word for word in words if word not in self.stopwords and len(word) > 2]
        
        return keywords
    
    def get_category_filters(self, query: str) -> List[str]:
        """Get relevant news categories based on query topic."""
        topic = self.detect_topic(query)
        
        category_mapping = {
            'economy': ['BUSINESS', 'MONEY', 'POLITICS'],
            'technology': ['TECH', 'SCIENCE', 'BUSINESS'],
            'politics': ['POLITICS', 'WORLD NEWS', 'U.S. NEWS'],
            'sports': ['SPORTS'],
            'health': ['WELLNESS', 'HEALTHY LIVING', 'PARENTS'],
            'entertainment': ['ENTERTAINMENT', 'ARTS & CULTURE', 'MEDIA']
        }
        
        return category_mapping.get(topic, [])
    
    def process(self, query: str) -> Dict:
        """Process query and return all enhancements."""
        return {
            'original':  query,
            'expanded': self.expand_query(query),
            'keywords': self. extract_keywords(query),
            'topic': self.detect_topic(query),
            'category_filters': self.get_category_filters(query)
        }