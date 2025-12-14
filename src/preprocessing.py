"""
Data preprocessing module for RAG News QA System.
Handles cleaning, tokenization, and preparation of news dataset.
"""

import json
import pickle
import re
from typing import List, Dict, Optional
import os


class NewsDataPreprocessor: 
    """Preprocessor for News Category Dataset."""
    
    def __init__(self):
        """Initialize the preprocessor."""
        self.documents = []
        
    def clean_text(self, text: str) -> str:
        """Clean and normalize text. 
        
        Args:
            text: Raw text string
            
        Returns:
            Cleaned text string
        """
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re. sub(r'\s+', ' ', text)
        
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\-\']', '', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def process_document(self, item: Dict) -> Dict:
        """Process a single document.
        
        Args:
            item:  Raw document dictionary
            
        Returns: 
            Processed document with metadata
        """
        # Extract fields
        headline = item.get('headline', '')
        description = item.get('short_description', '')
        category = item.get('category', 'UNKNOWN')
        date = item.get('date', '')
        link = item.get('link', '')
        authors = item.get('authors', '')
        
        # Clean texts
        headline_clean = self.clean_text(headline)
        description_clean = self.clean_text(description)
        
        # Combine for full text representation
        full_text = f"{headline_clean}. {description_clean}"
        
        # Create processed document
        processed = {
            'text': full_text,
            'headline': headline_clean,
            'description': description_clean,
            'category': category,
            'date': date,
            'link': link,
            'authors':  authors
        }
        
        return processed
    
    def load_dataset(self, input_path: str) -> List[Dict]:
        """Load dataset from JSON file.
        
        Args:
            input_path:  Path to input JSON file
            
        Returns:
            List of raw documents
        """
        print(f"📥 Loading dataset from:  {input_path}")
        
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Dataset not found at {input_path}")
        
        data = []
        
        # Try to load as JSONL (JSON Lines format)
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data.append(json.loads(line))
        except json.JSONDecodeError:
            # Try as regular JSON array
            with open(input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        
        print(f"✅ Loaded {len(data)} articles")
        
        return data
    
    def process_dataset(self, input_path: str, output_path: Optional[str] = None) -> List[Dict]:
        """Process entire dataset.
        
        Args:
            input_path: Path to input dataset
            output_path: Optional path to save processed data
            
        Returns:
            List of processed documents
        """
        # Load raw data
        raw_data = self.load_dataset(input_path)
        
        print("🔄 Processing documents...")
        
        # Process each document
        processed_docs = []
        for i, item in enumerate(raw_data):
            try:
                processed = self.process_document(item)
                processed_docs.append(processed)
                
                # Progress indicator
                if (i + 1) % 10000 == 0:
                    print(f"   Processed {i + 1}/{len(raw_data)} documents...")
                    
            except Exception as e: 
                print(f"⚠️  Error processing document {i}:  {e}")
                continue
        
        print(f"✅ Successfully processed {len(processed_docs)} documents")
        
        # Save if output path provided
        if output_path:
            self.save_processed_data(processed_docs, output_path)
        
        self.documents = processed_docs
        return processed_docs
    
    def save_processed_data(self, documents: List[Dict], output_path: str):
        """Save processed documents to file.
        
        Args:
            documents: List of processed documents
            output_path: Path to save file
        """
        # Create directory if not exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        print(f"💾 Saving processed data to: {output_path}")
        
        with open(output_path, 'wb') as f:
            pickle. dump(documents, f)
        
        print(f"✅ Saved {len(documents)} documents")
    
    def load_processed_data(self, input_path: str) -> List[Dict]:
        """Load previously processed documents.
        
        Args:
            input_path: Path to processed data file
            
        Returns:
            List of processed documents
        """
        print(f"📥 Loading processed data from: {input_path}")
        
        with open(input_path, 'rb') as f:
            documents = pickle.load(f)
        
        print(f"✅ Loaded {len(documents)} processed documents")
        
        self.documents = documents
        return documents
    
    def get_statistics(self) -> Dict:
        """Get dataset statistics.
        
        Returns:
            Dictionary with statistics
        """
        if not self.documents:
            return {}
        
        categories = {}
        total_words = 0
        
        for doc in self.documents:
            # Count categories
            cat = doc.get('category', 'UNKNOWN')
            categories[cat] = categories.get(cat, 0) + 1
            
            # Count words
            total_words += len(doc.get('text', '').split())
        
        stats = {
            'total_documents': len(self.documents),
            'categories': categories,
            'avg_words_per_doc': total_words / len(self.documents) if self.documents else 0
        }
        
        return stats
    
    def print_statistics(self):
        """Print dataset statistics."""
        stats = self.get_statistics()
        
        print("\n📊 Dataset Statistics")
        print("=" * 60)
        print(f"Total Documents: {stats. get('total_documents', 0)}")
        print(f"Average Words per Document: {stats.get('avg_words_per_doc', 0):.2f}")
        print("\nCategory Distribution:")
        
        categories = stats.get('categories', {})
        sorted_categories = sorted(categories.items(), key=lambda x: x[1], reverse=True)
        
        for cat, count in sorted_categories[: 10]:  # Show top 10
            print(f"  {cat}: {count}")
        
        print("=" * 60)


def main():
    """Main function for standalone execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Preprocess News Category Dataset')
    parser.add_argument('--input', type=str, required=True, help='Input dataset path')
    parser.add_argument('--output', type=str, default='data/processed/processed_news. pkl',
                       help='Output path for processed data')
    parser.add_argument('--stats', action='store_true', help='Show statistics')
    
    args = parser.parse_args()
    
    # Initialize preprocessor
    preprocessor = NewsDataPreprocessor()
    
    # Process dataset
    documents = preprocessor.process_dataset(args.input, args.output)
    
    # Show statistics if requested
    if args.stats:
        preprocessor.print_statistics()


if __name__ == "__main__":
    main()