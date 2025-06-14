from typing import List, Dict, Any, Optional, Callable
import numpy as np
from datetime import datetime
from rapidfuzz import fuzz
import logging


logger = logging.getLogger(__name__)

class SearchReranker:
    """Re-ranks search results based on multiple criteria beyond vector similarity."""
    
    def __init__(self, 
             recency_weight: float = 0.2, 
                retrieval_count_weight: float = 0.1,
                keyword_match_weight: float = 0.2,
                text_overlap_weight: float = 0.1,
                custom_scoring_fn: Optional[Callable] = None):
        """Initialize the search reranker.
        
        Args:
            recency_weight: Weight for recency factor (0-1)
            retrieval_count_weight: Weight for retrieval count factor (0-1)
            keyword_match_weight: Weight for keyword matching factor (0-1)
            text_overlap_weight: Weight for text overlap factor (0-1)
            custom_scoring_fn: Optional custom scoring function
        """
        self.recency_weight = recency_weight
        self.retrieval_count_weight = retrieval_count_weight
        self.keyword_match_weight = keyword_match_weight
        self.text_overlap_weight = text_overlap_weight
        self.custom_scoring_fn = custom_scoring_fn
        
        # Ensure weights sum to less than 1
        total_weight = recency_weight + retrieval_count_weight + keyword_match_weight + text_overlap_weight
        if total_weight >= 1.0:
            logger.warning(f"Total reranking weights ({total_weight}) should be less than 1.0. Normalizing.")
            factor = 0.7 / total_weight  # Leave 0.3 for base similarity
            self.recency_weight *= factor
            self.retrieval_count_weight *= factor
            self.keyword_match_weight *= factor
            self.text_overlap_weight *= factor
    
    def _calculate_recency_score(self, timestamp: str) -> float:
        """Calculate recency score based on timestamp.
        
        Args:
            timestamp: ISO format timestamp string
            
        Returns:
            float: Recency score (0-1)
        """
        try:
            if not timestamp:
                return 0.0
                
            # Parse timestamp
            doc_time = datetime.fromisoformat(timestamp)
            now = datetime.now()
            
            # Calculate age in days
            age_days = (now - doc_time).total_seconds() / (24 * 3600)
            
            # Exponential decay function: score = exp(-age_days/30)
            # This gives ~0.7 for week-old content, ~0.3 for month-old
            recency_score = np.exp(-age_days/30)
            return min(1.0, max(0.0, recency_score))
        except Exception as e:
            logger.warning(f"Error calculating recency score: {e}")
            return 0.0
    
    def _calculate_retrieval_score(self, retrieval_count: int) -> float:
        """Calculate score based on how often a document has been retrieved.
        
        Args:
            retrieval_count: Number of times document has been retrieved
            
        Returns:
            float: Retrieval popularity score (0-1)
        """
        if retrieval_count is None:
            return 0.0
            
        # Logarithmic scaling to prevent popular documents from dominating
        # log(1+x)/log(1+max_count) gives a score between 0-1
        # Using log(1+100) as denominator assuming 100 is a high retrieval count
        return min(1.0, max(0.0, np.log1p(retrieval_count) / np.log1p(100)))
    
    def _calculate_keyword_match_score(self, query: str, keywords: List[str], content: str) -> float:
        """Calculate score based on keyword matches with fuzzy matching support.
        
        Args:
            query: Search query
            keywords: List of keywords associated with the document
            content: Document content
            
        Returns:
            float: Keyword match score (0-1)
        """
        if not keywords:
            return 0.0
            
        # Tokenize query into words
        query_words = set(query.lower().split())
        
        # Calculate fuzzy match scores for each keyword against query words
        match_scores = []
        for kw in keywords:
            kw_lower = kw.lower()
            # Get best match score for this keyword against any query word
            best_score = max([fuzz.ratio(qw, kw_lower) / 100.0 for qw in query_words], default=0)
            match_scores.append(best_score)
        
        # Average the match scores, with a threshold to count as a match
        threshold = 0.7  # 70% similarity threshold
        keyword_score = sum(1 for score in match_scores if score > threshold) / max(1, len(keywords))
        
        return min(1.0, max(0.0, keyword_score))
    
    def _calculate_text_overlap_score(self, query: str, content: str) -> float:
        """Calculate semantic overlap between query and document content.
        
        Args:
            query: Search query
            content: Document content
            
        Returns:
            float: Text overlap score (0-1)
        """
        # Extract n-grams from query and content
        def get_ngrams(text, n=2):
            words = text.lower().split()
            return set(' '.join(words[i:i+n]) for i in range(len(words)-n+1))
        
        # Get bigrams and trigrams
        query_bigrams = get_ngrams(query, 2)
        query_trigrams = get_ngrams(query, 3)
        content_bigrams = get_ngrams(content[:1000], 2)  # Limit to first 1000 chars for efficiency
        content_trigrams = get_ngrams(content[:1000], 3)
        
        # Calculate Jaccard similarity for bigrams and trigrams
        bigram_overlap = len(query_bigrams.intersection(content_bigrams)) / max(1, len(query_bigrams.union(content_bigrams)))
        trigram_overlap = len(query_trigrams.intersection(content_trigrams)) / max(1, len(query_trigrams.union(content_trigrams)))
        
        # Combine scores (giving more weight to trigram matches)
        return 0.4 * bigram_overlap + 0.6 * trigram_overlap
    
    def rerank(self, query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Re-rank search results based on multiple factors.
        
        Args:
            query: Original search query
            results: List of search result dictionaries with 'score', 'metadata', etc.
            
        Returns:
            List of re-ranked search results
        """
        if not results:
            return []
            
        reranked_results = []
        
        for result in results:
            # Get base similarity score
            base_score = result.get('score', 0.0)
            
            # Extract metadata
            metadata = result.get('metadata', {})
            if not metadata and 'id' in result:
                # Handle case where metadata might be at top level
                metadata = {k: v for k, v in result.items() if k not in ['id', 'content', 'score']}
            
            # Calculate component scores
            timestamp = metadata.get('timestamp', '')
            recency_score = self._calculate_recency_score(timestamp)
            
            retrieval_count = metadata.get('retrieval_count', 0)
            retrieval_score = self._calculate_retrieval_score(retrieval_count)
            
            keywords = metadata.get('keywords', [])
            content = result.get('content', '')
            keyword_score = self._calculate_keyword_match_score(query, keywords, content)
            text_overlap_score = self._calculate_text_overlap_score(query, content)
            
            # Apply custom scoring if provided
            custom_score = 0.0
            if self.custom_scoring_fn:
                try:
                    custom_score = self.custom_scoring_fn(query, result)
                except Exception as e:
                    logger.warning(f"Error in custom scoring function: {e}")
            
            # Calculate final score
            base_weight = 1.0 - (self.recency_weight + self.retrieval_count_weight + 
                                self.keyword_match_weight + self.text_overlap_weight)
            
            final_score = (base_weight * base_score +
              self.recency_weight * recency_score +
              self.retrieval_count_weight * retrieval_score +
              self.keyword_match_weight * keyword_score +
              self.text_overlap_weight * text_overlap_score +
              custom_score)
            
            # Create new result with updated score
            reranked_result = dict(result)
            reranked_result['score'] = final_score
            reranked_result['score_components'] = {
                'base_score': base_score,
                'recency_score': recency_score,
                'retrieval_score': retrieval_score,
                'keyword_score': keyword_score,
                'text_overlap_score': text_overlap_score,
                'custom_score': custom_score
            }
            
            reranked_results.append(reranked_result)
        
        # Sort by final score
        reranked_results.sort(key=lambda x: x['score'], reverse=True)
        
        return reranked_results