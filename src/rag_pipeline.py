"""RAG Pipeline combining retrieval with LLM generation."""

from typing import List, Dict, Any, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch


class RAGPipeline:
    """Retrieval-Augmented Generation pipeline."""
    
    def __init__(self, retriever, model_name: str = "facebook/opt-125m", device: Optional[str] = None):
        """
        Initialize RAG pipeline.
        
        Args:
            retriever: Document retriever (BM25, FAISS, or Hybrid)
            model_name: Name of the LLM model (default: small OPT model for demo)
            device: Device to run the model on ('cpu', 'cuda', or None for auto)
        """
        self.retriever = retriever
        
        # Auto-detect device if not specified
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        
        print(f"Loading model {model_name} on {self.device}...")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            low_cpu_mem_usage=True
        )
        self.model = self.model.to(self.device)
        
        # Create text generation pipeline
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if device == "cuda" else -1
        )
        
        print("Model loaded successfully!")
        
    def retrieve_documents(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: User query
            top_k: Number of documents to retrieve
            
        Returns:
            List of retrieved documents with metadata
        """
        results = self.retriever.retrieve(query, top_k=top_k)
        
        retrieved_docs = []
        for idx, score, text in results:
            retrieved_docs.append({
                'index': idx,
                'score': score,
                'text': text
            })
        
        return retrieved_docs
    
    def create_prompt(self, query: str, retrieved_docs: List[Dict[str, Any]]) -> str:
        """
        Create a prompt combining query and retrieved documents.
        
        Args:
            query: User query
            retrieved_docs: List of retrieved documents
            
        Returns:
            Formatted prompt string
        """
        # Build context from retrieved documents
        context = "\n\n".join([
            f"Document {i+1}: {doc['text']}"
            for i, doc in enumerate(retrieved_docs)
        ])
        
        # Create prompt
        prompt = f"""Based on the following news articles, please answer the question.

Context:
{context}

Question: {query}

Answer:"""
        
        return prompt
    
    def generate_answer(self, query: str, top_k: int = 3, max_length: int = 200) -> Dict[str, Any]:
        """
        Generate an answer for a query using RAG.
        
        Args:
            query: User query
            top_k: Number of documents to retrieve
            max_length: Maximum length of generated answer
            
        Returns:
            Dictionary containing answer and retrieved documents
        """
        # Retrieve relevant documents
        retrieved_docs = self.retrieve_documents(query, top_k=top_k)
        
        # Create prompt
        prompt = self.create_prompt(query, retrieved_docs)
        
        # Generate answer
        try:
            # Truncate prompt if too long
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
            input_length = inputs['input_ids'].shape[1]
            
            outputs = self.generator(
                prompt,
                max_new_tokens=max_length,
                num_return_sequences=1,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            # Extract generated text (remove prompt)
            generated_text = outputs[0]['generated_text']
            answer = generated_text[len(prompt):].strip()
            
        except Exception as e:
            answer = f"Error generating answer: {str(e)}"
        
        return {
            'query': query,
            'answer': answer,
            'retrieved_documents': retrieved_docs,
            'num_retrieved': len(retrieved_docs)
        }
    
    def batch_generate(self, queries: List[str], top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Generate answers for multiple queries.
        
        Args:
            queries: List of queries
            top_k: Number of documents to retrieve per query
            
        Returns:
            List of results for each query
        """
        results = []
        for query in queries:
            result = self.generate_answer(query, top_k=top_k)
            results.append(result)
        
        return results


class SimpleRAGPipeline:
    """Simplified RAG pipeline without LLM (for demonstration)."""
    
    def __init__(self, retriever):
        """
        Initialize simple RAG pipeline.
        
        Args:
            retriever: Document retriever
        """
        self.retriever = retriever
        
    def generate_answer(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        """
        Generate answer by returning retrieved documents.
        
        Args:
            query: User query
            top_k: Number of documents to retrieve
            
        Returns:
            Dictionary containing retrieved documents
        """
        results = self.retriever.retrieve(query, top_k=top_k)
        
        retrieved_docs = []
        for idx, score, text in results:
            retrieved_docs.append({
                'index': idx,
                'score': score,
                'text': text
            })
        
        # Create a simple answer by concatenating top documents
        if retrieved_docs:
            answer = "Based on the retrieved documents:\n\n"
            for i, doc in enumerate(retrieved_docs):
                answer += f"{i+1}. {doc['text'][:200]}...\n\n"
        else:
            answer = "No relevant documents found."
        
        return {
            'query': query,
            'answer': answer,
            'retrieved_documents': retrieved_docs,
            'num_retrieved': len(retrieved_docs)
        }
