import numpy as np
import logging
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime

logger = logging.getLogger(__name__)

class SearchReranker:
    """Re-ranks search results based on multiple criteria beyond vector similarity."""
    
    def __init__(self, 
                 recency_weight: float = 0.2, 
                 retrieval_count_weight: float = 0.1,
                 keyword_match_weight: float = 0.2,
                 text_overlap_weight: float = 0.1,
                 link_count_weight: float = 0.1,  # Added new weight parameter
                 custom_scoring_fn: Optional[Callable] = None):
        """Initialize the search reranker with configurable weights.
        
        Args:
            recency_weight: Weight for recency score (higher = more recent notes preferred)
            retrieval_count_weight: Weight for retrieval count (higher = frequently accessed notes preferred)
            keyword_match_weight: Weight for keyword matching (higher = better keyword matches preferred)
            text_overlap_weight: Weight for text overlap (higher = better content matches preferred)
            link_count_weight: Weight for link count (higher = more connected notes preferred)
            custom_scoring_fn: Optional custom scoring function
        """
        self.recency_weight = recency_weight
        self.retrieval_count_weight = retrieval_count_weight
        self.keyword_match_weight = keyword_match_weight
        self.text_overlap_weight = text_overlap_weight
        self.link_count_weight = link_count_weight  # Store the new weight
        self.custom_scoring_fn = custom_scoring_fn
        
        # Normalize weights to ensure they sum to 0.7 (since base score gets 0.3)
        total_weight = (recency_weight + retrieval_count_weight + 
                        keyword_match_weight + text_overlap_weight + 
                        link_count_weight)
        
        if total_weight > 0:
            factor = 0.7 / total_weight
            self.recency_weight *= factor
            self.retrieval_count_weight *= factor
            self.keyword_match_weight *= factor
            self.text_overlap_weight *= factor
            self.link_count_weight *= factor  # Normalize the new weight
    
    def _calculate_recency_score(self, accessed_at: Optional[str]) -> float:
        """Calculate recency score based on last accessed timestamp.
        
        Args:
            accessed_at: ISO format timestamp of last access
            
        Returns:
            Recency score between 0-1
        """
        if not accessed_at:
            return 0.0
            
        try:
            last_accessed = datetime.fromisoformat(accessed_at)
            now = datetime.now()
            days_since = (now - last_accessed).total_seconds() / (24 * 3600)
            
            # Exponential decay: 1.0 for just accessed, 0.0 for very old
            return np.exp(-0.1 * days_since)
        except (ValueError, TypeError):
            logger.warning(f"Invalid timestamp format: {accessed_at}")
            return 0.0
    
    def _calculate_retrieval_score(self, retrieval_count: int) -> float:
        """Calculate score based on how often a note has been retrieved.
        
        Args:
            retrieval_count: Number of times the note has been retrieved
            
        Returns:
            Retrieval count score between 0-1
        """
        if not retrieval_count or retrieval_count < 0:
            return 0.0
            
        # Logarithmic scaling to prevent very popular notes from dominating
        return min(1.0, max(0.0, np.log1p(retrieval_count) / np.log1p(100)))
    
    def _calculate_keyword_match_score(self, query: str, keywords: List[str], content: str) -> float:
        """Calculate score based on keyword matches.
        
        Args:
            query: Search query
            keywords: List of keywords for the note
            content: Note content
            
        Returns:
            Keyword match score between 0-1
        """
        if not keywords or not query:
            return 0.0
            
        query_terms = query.lower().split()
        keyword_score = 0.0
        
        # Check for direct keyword matches
        for term in query_terms:
            if any(term in keyword.lower() for keyword in keywords):
                keyword_score += 0.5
                
        # Check for keyword presence in content
        for keyword in keywords:
            if keyword.lower() in content.lower():
                keyword_score += 0.1
                
        return min(1.0, max(0.0, keyword_score))
    
    def _calculate_text_overlap_score(self, query: str, content: str) -> float:
        """Calculate score based on n-gram overlap between query and content.
        
        Args:
            query: Search query
            content: Note content
            
        Returns:
            Text overlap score between 0-1
        """
        if not query or not content:
            return 0.0
            
        # Create n-grams from query and content
        def get_ngrams(text, n):
            tokens = text.lower().split()
            return [' '.join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
        
        # Calculate bigram and trigram overlap
        query_bigrams = set(get_ngrams(query, 2))
        content_bigrams = set(get_ngrams(content, 3))
        
        query_trigrams = set(get_ngrams(query, 3))
        content_trigrams = set(get_ngrams(content, 3))
        
        # Calculate Jaccard similarity for bigrams and trigrams
        bigram_overlap = 0.0
        if query_bigrams and content_bigrams:
            intersection = len(query_bigrams.intersection(content_bigrams))
            union = len(query_bigrams.union(content_bigrams))
            bigram_overlap = intersection / union if union > 0 else 0.0
            
        trigram_overlap = 0.0
        if query_trigrams and content_trigrams:
            intersection = len(query_trigrams.intersection(content_trigrams))
            union = len(query_trigrams.union(content_trigrams))
            trigram_overlap = intersection / union if union > 0 else 0.0
            
        # Weight bigrams and trigrams
        return 0.4 * bigram_overlap + 0.6 * trigram_overlap
    
    def _calculate_link_count_score(self, links: List[str]) -> float:
        """Calculate a score based on the number of links in a ZettelNote.
        
        Args:
            links: List of linked note IDs
            
        Returns:
            Score between 0.0 and 1.0 based on link count
        """
        if not links:
            return 0.0
        
        # Use logarithmic scaling to prevent notes with many links from dominating
        # Assume a reasonable maximum of 50 links for normalization
        link_count = len(links)
        return min(1.0, np.log1p(link_count) / np.log1p(50))
    
    def rerank(self, query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rerank search results based on multiple factors.
        
        Args:
            query: Original search query
            results: List of search results from retriever
            
        Returns:
            Reranked list of search results
        """
        if not results:
            return []
            
        reranked_results = []
        
        for result in results:
            note = result['note']
            base_score = result['score']
            
            # Extract note attributes safely
            accessed_at = getattr(note, 'accessed_at', None)
            retrieval_count = getattr(note, 'retrieval_count', 0)
            keywords = getattr(note['note_simple'], 'keywords', [])
            tags = getattr(note, 'tags', [])
            keywords.extend(tags)
            content = getattr(note, 'content', '')
            
            # Extract links for the new scoring component
            links = getattr(note, 'links', [])
            
            # Calculate component scores
            recency_score = self._calculate_recency_score(accessed_at)
            retrieval_score = self._calculate_retrieval_score(retrieval_count)
            keyword_score = self._calculate_keyword_match_score(query, keywords, content)
            text_overlap_score = self._calculate_text_overlap_score(query, content)
            link_count_score = self._calculate_link_count_score(links)  # Calculate the new score
            
            # Apply custom scoring if provided
            custom_score = 0.0
            if self.custom_scoring_fn:
                try:
                    custom_score = self.custom_scoring_fn(query, note)
                except Exception as e:
                    logger.error(f"Error in custom scoring function: {e}")
            
            # Calculate final score
            # Base score (vector similarity) gets at least 30% weight
            final_score = (
                0.3 * base_score +
                self.recency_weight * recency_score +
                self.retrieval_count_weight * retrieval_score +
                self.keyword_match_weight * keyword_score +
                self.text_overlap_weight * text_overlap_score +
                self.link_count_weight * link_count_score +  # Add the new score component
                custom_score
            )
            
            # Create new result with updated score
            reranked_result = result.copy()
            reranked_result['score'] = final_score
            reranked_result['component_scores'] = {
                'base_score': base_score,
                'recency_score': recency_score,
                'retrieval_score': retrieval_score,
                'keyword_score': keyword_score,
                'text_overlap_score': text_overlap_score,
                'link_count_score': link_count_score  # Include the new score in component scores
            }
            
            reranked_results.append(reranked_result)
        
        # Sort by final score in descending order
        reranked_results.sort(key=lambda x: x['score'], reverse=True)
        
        return reranked_results