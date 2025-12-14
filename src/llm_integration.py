"""
LLM Integration module for RAG system. 
Supports Groq API for fast inference.
"""

import os
from typing import List, Dict, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class GroqLLM:
    """Groq LLM integration for answer generation."""
    
    def __init__(
        self, 
        model:  str = "llama-3.1-8b-instant",
        temperature: float = 0.7,
        max_tokens: int = 512
    ):
        """Initialize Groq LLM. 
        
        Args:
            model:  Groq model name
            temperature:  Sampling temperature (0-1)
            max_tokens:  Maximum tokens in response
        """
        try:
            from groq import Groq
        except ImportError:
            raise ImportError("Groq SDK not installed. Install with: pip install groq")
        
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Initialize Groq client
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY not found in environment variables.  "
                "Please set it in .env file or export GROQ_API_KEY=your_key"
            )
        
        self.client = Groq(api_key=api_key)
        print(f"✅ Groq LLM initialized with model:  {model}")
    
    def create_prompt(self, query: str, retrieved_docs: List[Dict]) -> str:
        """Create prompt from query and retrieved documents.
        
        Args:
            query: User query
            retrieved_docs: List of retrieved documents with metadata
            
        Returns:
            Formatted prompt string
        """
        # Build context from retrieved documents
        context_parts = []
        for i, doc in enumerate(retrieved_docs, 1):
            headline = doc.get('headline', '')
            description = doc.get('description', '')
            category = doc.get('category', '')
            
            context_parts. append(
                f"[Document {i}] ({category})\n"
                f"Headline: {headline}\n"
                f"Content: {description}\n"
            )
        
        context = "\n".join(context_parts)
        
        # Create prompt with instructions
        prompt = f"""You are a helpful news assistant. Based on the retrieved news articles below, answer the user's question accurately and concisely.

Retrieved News Articles:
{context}

User Question: {query}

Instructions:
1. Provide a clear, informative answer based on the retrieved articles
2. Synthesize information from multiple sources when relevant
3. If the articles don't contain enough information, acknowledge this
4. Keep the answer concise (2-4 paragraphs)
5. Maintain a professional, journalistic tone

Answer:"""
        
        return prompt
    
    def generate(self, prompt: str) -> str:
        """Generate response using Groq API.
        
        Args:
            prompt: Input prompt
            
        Returns:
            Generated text response
        """
        try: 
            response = self.client. chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful news assistant that provides accurate, concise answers based on retrieved news articles."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=1,
                stream=False
            )
            
            answer = response.choices[0].message.content
            return answer. strip()
            
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def answer_query(self, query: str, retrieved_docs: List[Dict]) -> Dict:
        """Complete pipeline:  create prompt and generate answer. 
        
        Args:
            query: User query
            retrieved_docs: Retrieved documents with metadata
            
        Returns:
            Dictionary with answer and metadata
        """
        # Create prompt
        prompt = self. create_prompt(query, retrieved_docs)
        
        # Generate answer
        answer = self.generate(prompt)
        
        # Prepare response
        response = {
            'query': query,
            'answer':  answer,
            'num_sources': len(retrieved_docs),
            'sources': [
                {
                    'headline': doc.get('headline', ''),
                    'category': doc.get('category', ''),
                    'link': doc.get('link', '')
                }
                for doc in retrieved_docs
            ]
        }
        
        return response


class LLMFactory:
    """Factory for creating LLM instances."""
    
    @staticmethod
    def create(provider: str = "groq", **kwargs):
        """Create LLM instance. 
        
        Args:
            provider: LLM provider ('groq', 'openai', 'local')
            **kwargs: Additional arguments for LLM
            
        Returns:
            LLM instance
        """
        if provider. lower() == "groq":
            return GroqLLM(**kwargs)
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")


# Convenience function
def create_llm(provider: str = "groq", **kwargs):
    """Create LLM instance (convenience function)."""
    return LLMFactory.create(provider, **kwargs)