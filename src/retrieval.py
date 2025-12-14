"""Retrieval systems: BM25, FAISS, and Hybrid."""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from rank_bm25 import BM25Okapi
import pickle
import os


class BM25Retriever:
    """BM25 retriever for sparse lexical retrieval."""
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """Initialize BM25 retriever.
        
        Args:
            k1: BM25 k1 parameter
            b: BM25 b parameter
        """
        self.k1 = k1
        self.b = b
        self.bm25 = None
        self.documents = []
        self.tokenized_corpus = []
    
    def index_documents(self, documents: List[Dict[str, Any]]):
        """Index documents for BM25 retrieval.
        
        Args:
            documents: List of preprocessed documents with 'tokens_processed' field
        """
        self.documents = documents
        
        # Get tokenized corpus
        self.tokenized_corpus = [doc.get('tokens_processed', []) for doc in documents]
        
        # Create BM25 index
        if self.tokenized_corpus:
            self.bm25 = BM25Okapi(self.tokenized_corpus)
    
    def search(self, query_tokens: List[str], top_k: int = 10,
              category_filter: Optional[List[str]] = None) -> List[Tuple[Dict[str, Any], float]]:
        """Search documents using BM25.
        
        Args:
            query_tokens: Tokenized and preprocessed query
            top_k: Number of top documents to retrieve
            category_filter: Optional list of categories to filter by
            
        Returns:
            List of tuples (document, score)
        """
        if not self.bm25 or not query_tokens:
            return []
        
        # Get BM25 scores
        scores = self.bm25.get_scores(query_tokens)
        
        # Combine documents with scores
        doc_scores = list(zip(self.documents, scores))
        
        # Apply category filter if provided
        if category_filter:
            doc_scores = [
                (doc, score) for doc, score in doc_scores
                if doc.get('category', '').upper() in [c.upper() for c in category_filter]
            ]
        
        # Sort by score
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return top-k with scores
        return [(doc, float(score)) for doc, score in doc_scores[:top_k]]
    
    def save(self, path: str):
        """Save BM25 index to file.
        
        Args:
            path: Path to save the index
        """
        data = {
            'documents': self.documents,
            'tokenized_corpus': self.tokenized_corpus,
            'k1': self.k1,
            'b': self.b
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
    
    def load(self, path: str):
        """Load BM25 index from file.
        
        Args:
            path: Path to load the index from
        """
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.documents = data['documents']
        self.tokenized_corpus = data['tokenized_corpus']
        self.k1 = data.get('k1', 1.5)
        self.b = data.get('b', 0.75)
        
        # Recreate BM25 index
        if self.tokenized_corpus:
            self.bm25 = BM25Okapi(self.tokenized_corpus)


class FAISSRetriever:
    """FAISS retriever for dense semantic retrieval."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 normalize_embeddings: bool = True):
        """Initialize FAISS retriever.
        
        Args:
            model_name: Name of the sentence transformer model
            normalize_embeddings: Whether to normalize embeddings
        """
        self.model_name = model_name
        self.normalize_embeddings = normalize_embeddings
        self.model = None
        self.index = None
        self.documents = []
        self.embeddings = None
    
    def _load_model(self):
        """Lazy load the embedding model."""
        if self.model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(self.model_name)
            except Exception as e:
                print(f"Error loading embedding model: {e}")
                raise
    
    def _create_index(self, embeddings: np.ndarray):
        """Create FAISS index from embeddings.
        
        Args:
            embeddings: Document embeddings
        """
        try:
            import faiss
            
            dimension = embeddings.shape[1]
            
            # Use IndexFlatIP for cosine similarity (after normalization)
            if self.normalize_embeddings:
                # Normalize embeddings for cosine similarity
                faiss.normalize_L2(embeddings)
                self.index = faiss.IndexFlatIP(dimension)
            else:
                # Use L2 distance
                self.index = faiss.IndexFlatL2(dimension)
            
            # Add embeddings to index
            self.index.add(embeddings.astype('float32'))
            
        except Exception as e:
            print(f"Error creating FAISS index: {e}")
            raise
    
    def index_documents(self, documents: List[Dict[str, Any]]):
        """Index documents for FAISS retrieval.
        
        Args:
            documents: List of preprocessed documents with 'text_for_embedding' field
        """
        self.documents = documents
        
        # Load model if not loaded
        self._load_model()
        
        # Get text for embedding
        texts = [doc.get('text_for_embedding', '') for doc in documents]
        
        # Generate embeddings
        try:
            self.embeddings = self.model.encode(texts, show_progress_bar=True,
                                               convert_to_numpy=True)
        except Exception as e:
            print(f"Error encoding documents: {e}")
            raise
        
        # Create FAISS index
        self._create_index(self.embeddings)
    
    def search(self, query: str, top_k: int = 10,
              category_filter: Optional[List[str]] = None) -> List[Tuple[Dict[str, Any], float]]:
        """Search documents using FAISS.
        
        Args:
            query: Query string
            top_k: Number of top documents to retrieve
            category_filter: Optional list of categories to filter by
            
        Returns:
            List of tuples (document, score)
        """
        if not self.index:
            return []
        
        # Load model if not loaded
        self._load_model()
        
        # Encode query
        try:
            query_embedding = self.model.encode([query], convert_to_numpy=True)
            
            if self.normalize_embeddings:
                import faiss
                faiss.normalize_L2(query_embedding)
            
        except Exception as e:
            print(f"Error encoding query: {e}")
            return []
        
        # Search in FAISS index
        # Retrieve more than top_k to allow for filtering
        search_k = min(top_k * 3, len(self.documents))
        
        try:
            distances, indices = self.index.search(query_embedding.astype('float32'), search_k)
        except Exception as e:
            print(f"Error searching FAISS index: {e}")
            return []
        
        # Convert distances to similarity scores
        # For IndexFlatIP (cosine similarity), distances are already similarity scores
        # For IndexFlatL2, we convert: similarity = 1 / (1 + distance)
        if self.normalize_embeddings:
            scores = distances[0]  # Already similarity scores
        else:
            scores = 1.0 / (1.0 + distances[0])
        
        # Get documents and scores
        results = []
        for idx, score in zip(indices[0], scores):
            if idx < len(self.documents):
                doc = self.documents[idx]
                
                # Apply category filter if provided
                if category_filter:
                    if doc.get('category', '').upper() not in [c.upper() for c in category_filter]:
                        continue
                
                results.append((doc, float(score)))
                
                if len(results) >= top_k:
                    break
        
        return results
    
    def save(self, path: str):
        """Save FAISS index and documents to file.
        
        Args:
            path: Path prefix to save the index and documents
        """
        import faiss
        
        # Save FAISS index
        faiss.write_index(self.index, f"{path}.index")
        
        # Save documents and embeddings
        data = {
            'documents': self.documents,
            'embeddings': self.embeddings,
            'model_name': self.model_name,
            'normalize_embeddings': self.normalize_embeddings
        }
        with open(f"{path}.pkl", 'wb') as f:
            pickle.dump(data, f)
    
    def load(self, path: str):
        """Load FAISS index and documents from file.
        
        Args:
            path: Path prefix to load the index and documents from
        """
        import faiss
        
        # Load FAISS index
        self.index = faiss.read_index(f"{path}.index")
        
        # Load documents and embeddings
        with open(f"{path}.pkl", 'rb') as f:
            data = pickle.load(f)
        
        self.documents = data['documents']
        self.embeddings = data['embeddings']
        self.model_name = data.get('model_name', self.model_name)
        self.normalize_embeddings = data.get('normalize_embeddings', True)


class HybridRetriever:
    """Hybrid retriever combining BM25 and FAISS."""
    
    def __init__(self, bm25_retriever: BM25Retriever, 
                 faiss_retriever: FAISSRetriever,
                 alpha: float = 0.5,
                 normalize_scores: bool = True):
        """Initialize hybrid retriever.
        
        Args:
            bm25_retriever: BM25 retriever instance
            faiss_retriever: FAISS retriever instance
            alpha: Weight for BM25 scores (1-alpha for FAISS)
            normalize_scores: Whether to normalize scores before fusion
        """
        self.bm25_retriever = bm25_retriever
        self.faiss_retriever = faiss_retriever
        self.alpha = alpha
        self.normalize_scores = normalize_scores
    
    def _normalize_score_list(self, scores: List[float]) -> List[float]:
        """Normalize list of scores to [0, 1] range.
        
        Args:
            scores: List of scores
            
        Returns:
            Normalized scores
        """
        if not scores:
            return []
        
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score - min_score == 0:
            return [1.0] * len(scores)
        
        return [(s - min_score) / (max_score - min_score) for s in scores]
    
    def search(self, query: str, query_tokens: List[str], 
              top_k: int = 10,
              category_filter: Optional[List[str]] = None) -> List[Tuple[Dict[str, Any], float]]:
        """Search using hybrid retrieval.
        
        Args:
            query: Query string (for FAISS)
            query_tokens: Tokenized query (for BM25)
            top_k: Number of top documents to retrieve
            category_filter: Optional list of categories to filter by
            
        Returns:
            List of tuples (document, score)
        """
        # Retrieve from both systems (get more for fusion)
        retrieve_k = min(top_k * 2, len(self.bm25_retriever.documents))
        
        bm25_results = self.bm25_retriever.search(query_tokens, retrieve_k, category_filter)
        faiss_results = self.faiss_retriever.search(query, retrieve_k, category_filter)
        
        # Create document ID mapping using a stable key
        # Use a combination of headline and link for stable identification
        def get_doc_key(doc):
            """Generate a stable key for document identification."""
            headline = doc.get('headline', '')
            link = doc.get('link', '')
            return f"{headline}||{link}"
        
        doc_scores = {}
        
        # Process BM25 results
        bm25_scores = [score for _, score in bm25_results]
        if self.normalize_scores and bm25_scores:
            bm25_scores = self._normalize_score_list(bm25_scores)
        
        for i, (doc, _) in enumerate(bm25_results):
            doc_key = get_doc_key(doc)
            score = bm25_scores[i] if bm25_scores else 0.0
            doc_scores[doc_key] = {'doc': doc, 'bm25': score, 'faiss': 0.0}
        
        # Process FAISS results
        faiss_scores = [score for _, score in faiss_results]
        if self.normalize_scores and faiss_scores:
            faiss_scores = self._normalize_score_list(faiss_scores)
        
        for i, (doc, _) in enumerate(faiss_results):
            doc_key = get_doc_key(doc)
            score = faiss_scores[i] if faiss_scores else 0.0
            if doc_key in doc_scores:
                doc_scores[doc_key]['faiss'] = score
            else:
                doc_scores[doc_key] = {'doc': doc, 'bm25': 0.0, 'faiss': score}
        
        # Compute hybrid scores
        results = []
        for doc_id, scores in doc_scores.items():
            hybrid_score = (self.alpha * scores['bm25'] + 
                          (1 - self.alpha) * scores['faiss'])
            results.append((scores['doc'], hybrid_score))
        
        # Sort by hybrid score
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:top_k]
