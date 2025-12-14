"""Data preprocessing module for news category dataset."""

import json
import pandas as pd
from typing import List, Dict, Any
import re


class NewsDataPreprocessor:
    """Preprocessor for News Category Dataset."""
    
    def __init__(self):
        """Initialize the preprocessor."""
        self.data = None
        
    def load_data(self, filepath: str) -> pd.DataFrame:
        """
        Load news category dataset from JSON file.
        
        Args:
            filepath: Path to the JSON file
            
        Returns:
            DataFrame containing the news data
        """
        data_list = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                data_list.append(json.loads(line))
        
        self.data = pd.DataFrame(data_list)
        return self.data
    
    def clean_text(self, text: str) -> str:
        """
        Clean text by removing special characters and extra whitespace.
        
        Args:
            text: Input text string
            
        Returns:
            Cleaned text string
        """
        if not isinstance(text, str):
            return ""
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text.strip()
    
    def preprocess(self, combine_fields: bool = True) -> pd.DataFrame:
        """
        Preprocess the loaded dataset.
        
        Args:
            combine_fields: If True, combine headline and description into document text
            
        Returns:
            Preprocessed DataFrame
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        # Clean headline and short_description
        self.data['headline_clean'] = self.data['headline'].apply(self.clean_text)
        self.data['description_clean'] = self.data['short_description'].apply(self.clean_text)
        
        # Combine fields into a single document text
        if combine_fields:
            self.data['document'] = (
                self.data['headline_clean'] + '. ' + 
                self.data['description_clean']
            )
        
        # Remove empty documents
        if combine_fields:
            self.data = self.data[self.data['document'].str.strip() != '']
        
        return self.data
    
    def get_documents(self) -> List[str]:
        """
        Get list of document texts.
        
        Returns:
            List of document strings
        """
        if 'document' not in self.data.columns:
            raise ValueError("Documents not created. Call preprocess() first.")
        
        return self.data['document'].tolist()
    
    def get_metadata(self) -> List[Dict[str, Any]]:
        """
        Get metadata for each document.
        
        Returns:
            List of metadata dictionaries
        """
        metadata = []
        for _, row in self.data.iterrows():
            metadata.append({
                'category': row.get('category', ''),
                'date': row.get('date', ''),
                'authors': row.get('authors', ''),
                'link': row.get('link', '')
            })
        return metadata
    
    def save_processed_data(self, filepath: str):
        """
        Save processed data to CSV file.
        
        Args:
            filepath: Output file path
        """
        if self.data is None:
            raise ValueError("No data to save.")
        
        self.data.to_csv(filepath, index=False)
