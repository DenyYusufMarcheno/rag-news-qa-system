"""Evaluation module for retrieval and generation quality."""

from typing import List, Dict, Any, Set
import numpy as np
from collections import Counter


class RetrievalEvaluator:
    """Evaluator for document retrieval quality."""
    
    @staticmethod
    def precision_at_k(retrieved_docs: List[int], relevant_docs: List[int], k: int) -> float:
        """
        Calculate Precision@K.
        
        Args:
            retrieved_docs: List of retrieved document indices
            relevant_docs: List of relevant document indices (ground truth)
            k: Number of top documents to consider
            
        Returns:
            Precision@K score
        """
        if k == 0 or not retrieved_docs:
            return 0.0
        
        retrieved_set = set(retrieved_docs[:k])
        relevant_set = set(relevant_docs)
        
        intersection = retrieved_set & relevant_set
        return len(intersection) / k
    
    @staticmethod
    def recall_at_k(retrieved_docs: List[int], relevant_docs: List[int], k: int) -> float:
        """
        Calculate Recall@K.
        
        Args:
            retrieved_docs: List of retrieved document indices
            relevant_docs: List of relevant document indices (ground truth)
            k: Number of top documents to consider
            
        Returns:
            Recall@K score
        """
        if not relevant_docs or not retrieved_docs:
            return 0.0
        
        retrieved_set = set(retrieved_docs[:k])
        relevant_set = set(relevant_docs)
        
        intersection = retrieved_set & relevant_set
        return len(intersection) / len(relevant_set)
    
    @staticmethod
    def mean_reciprocal_rank(retrieved_docs_list: List[List[int]], relevant_docs_list: List[List[int]]) -> float:
        """
        Calculate Mean Reciprocal Rank (MRR).
        
        Args:
            retrieved_docs_list: List of retrieved document lists for each query
            relevant_docs_list: List of relevant document lists for each query
            
        Returns:
            MRR score
        """
        reciprocal_ranks = []
        
        for retrieved_docs, relevant_docs in zip(retrieved_docs_list, relevant_docs_list):
            relevant_set = set(relevant_docs)
            
            # Find rank of first relevant document
            for rank, doc_id in enumerate(retrieved_docs, 1):
                if doc_id in relevant_set:
                    reciprocal_ranks.append(1.0 / rank)
                    break
            else:
                reciprocal_ranks.append(0.0)
        
        return np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0
    
    @staticmethod
    def ndcg_at_k(retrieved_docs: List[int], relevant_docs: List[int], k: int) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain (NDCG@K).
        
        Args:
            retrieved_docs: List of retrieved document indices
            relevant_docs: List of relevant document indices (ground truth)
            k: Number of top documents to consider
            
        Returns:
            NDCG@K score
        """
        def dcg(relevances: List[int], k: int) -> float:
            relevances = relevances[:k]
            return sum(rel / np.log2(idx + 2) for idx, rel in enumerate(relevances))
        
        # Binary relevance: 1 if relevant, 0 otherwise
        relevant_set = set(relevant_docs)
        retrieved_relevances = [1 if doc_id in relevant_set else 0 for doc_id in retrieved_docs[:k]]
        
        # Calculate DCG
        dcg_score = dcg(retrieved_relevances, k)
        
        # Calculate ideal DCG (all relevant docs first)
        ideal_relevances = [1] * min(len(relevant_docs), k) + [0] * max(0, k - len(relevant_docs))
        idcg_score = dcg(ideal_relevances, k)
        
        if idcg_score == 0:
            return 0.0
        
        return dcg_score / idcg_score
    
    @staticmethod
    def evaluate_retrieval(retrieved_docs_list: List[List[int]], 
                          relevant_docs_list: List[List[int]], 
                          k_values: List[int] = [1, 3, 5, 10]) -> Dict[str, Any]:
        """
        Comprehensive evaluation of retrieval results.
        
        Args:
            retrieved_docs_list: List of retrieved document lists for each query
            relevant_docs_list: List of relevant document lists for each query
            k_values: List of K values to evaluate
            
        Returns:
            Dictionary of evaluation metrics
        """
        metrics = {
            'mrr': RetrievalEvaluator.mean_reciprocal_rank(retrieved_docs_list, relevant_docs_list)
        }
        
        for k in k_values:
            precisions = []
            recalls = []
            ndcgs = []
            
            for retrieved_docs, relevant_docs in zip(retrieved_docs_list, relevant_docs_list):
                precisions.append(RetrievalEvaluator.precision_at_k(retrieved_docs, relevant_docs, k))
                recalls.append(RetrievalEvaluator.recall_at_k(retrieved_docs, relevant_docs, k))
                ndcgs.append(RetrievalEvaluator.ndcg_at_k(retrieved_docs, relevant_docs, k))
            
            metrics[f'precision@{k}'] = np.mean(precisions)
            metrics[f'recall@{k}'] = np.mean(recalls)
            metrics[f'ndcg@{k}'] = np.mean(ndcgs)
        
        return metrics


class GenerationEvaluator:
    """Evaluator for answer generation quality."""
    
    @staticmethod
    def exact_match(generated: str, reference: str) -> float:
        """
        Calculate exact match score.
        
        Args:
            generated: Generated answer
            reference: Reference answer
            
        Returns:
            1.0 if exact match, 0.0 otherwise
        """
        return 1.0 if generated.strip().lower() == reference.strip().lower() else 0.0
    
    @staticmethod
    def token_overlap(generated: str, reference: str) -> float:
        """
        Calculate token overlap (F1 score).
        
        Args:
            generated: Generated answer
            reference: Reference answer
            
        Returns:
            F1 score of token overlap
        """
        gen_tokens = set(generated.lower().split())
        ref_tokens = set(reference.lower().split())
        
        if not gen_tokens or not ref_tokens:
            return 0.0
        
        common = gen_tokens & ref_tokens
        
        if not common:
            return 0.0
        
        precision = len(common) / len(gen_tokens)
        recall = len(common) / len(ref_tokens)
        
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        return f1
    
    @staticmethod
    def evaluate_generation(generated_answers: List[str], 
                           reference_answers: List[str]) -> Dict[str, float]:
        """
        Comprehensive evaluation of generated answers.
        
        Args:
            generated_answers: List of generated answers
            reference_answers: List of reference answers
            
        Returns:
            Dictionary of evaluation metrics
        """
        if len(generated_answers) != len(reference_answers):
            raise ValueError("Number of generated and reference answers must match")
        
        exact_matches = []
        token_overlaps = []
        
        for gen, ref in zip(generated_answers, reference_answers):
            exact_matches.append(GenerationEvaluator.exact_match(gen, ref))
            token_overlaps.append(GenerationEvaluator.token_overlap(gen, ref))
        
        return {
            'exact_match': np.mean(exact_matches),
            'token_f1': np.mean(token_overlaps)
        }


class RAGEvaluator:
    """Combined evaluator for RAG system."""
    
    def __init__(self):
        """Initialize RAG evaluator."""
        self.retrieval_evaluator = RetrievalEvaluator()
        self.generation_evaluator = GenerationEvaluator()
    
    def evaluate(self, 
                 retrieved_docs_list: List[List[int]],
                 relevant_docs_list: List[List[int]],
                 generated_answers: List[str],
                 reference_answers: List[str]) -> Dict[str, Any]:
        """
        Comprehensive evaluation of RAG system.
        
        Args:
            retrieved_docs_list: Retrieved documents for each query
            relevant_docs_list: Relevant documents for each query (ground truth)
            generated_answers: Generated answers
            reference_answers: Reference answers
            
        Returns:
            Dictionary containing all evaluation metrics
        """
        retrieval_metrics = self.retrieval_evaluator.evaluate_retrieval(
            retrieved_docs_list, relevant_docs_list
        )
        
        generation_metrics = self.generation_evaluator.evaluate_generation(
            generated_answers, reference_answers
        )
        
        return {
            'retrieval': retrieval_metrics,
            'generation': generation_metrics
        }
