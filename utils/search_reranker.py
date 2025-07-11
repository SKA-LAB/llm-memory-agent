import numpy as np
import logging
import time
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
                 link_count_weight: float = 0.1,
                 custom_scoring_fn: Optional[Callable] = None):
        """Initialize the search reranker with configurable weights."""
        logger.info("Initializing SearchReranker with weights: recency=%.2f, retrieval=%.2f, keyword=%.2f, text=%.2f, link=%.2f", 
                   recency_weight, retrieval_count_weight, keyword_match_weight, text_overlap_weight, link_count_weight)
        
        self.recency_weight = recency_weight
        self.retrieval_count_weight = retrieval_count_weight
        self.keyword_match_weight = keyword_match_weight
        self.text_overlap_weight = text_overlap_weight
        self.link_count_weight = link_count_weight
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
            self.link_count_weight *= factor
            
            logger.debug("Normalized weights: recency=%.3f, retrieval=%.3f, keyword=%.3f, text=%.3f, link=%.3f", 
                        self.recency_weight, self.retrieval_count_weight, self.keyword_match_weight, 
                        self.text_overlap_weight, self.link_count_weight)
    
    def _calculate_recency_score(self, accessed_at: Optional[str]) -> float:
        """Calculate recency score based on last accessed timestamp."""
        start_time = time.time()
        
        if not accessed_at:
            logger.debug("No accessed_at timestamp, recency score = 0.0")
            elapsed = time.time() - start_time
            logger.debug("Recency score calculation took %.6f seconds", elapsed)
            return 0.0
            
        try:
            last_accessed = datetime.fromisoformat(accessed_at)
            now = datetime.now()
            days_since = (now - last_accessed).total_seconds() / (24 * 3600)
            
            # Exponential decay: 1.0 for just accessed, 0.0 for very old
            score = np.exp(-0.1 * days_since)
            logger.debug("Recency score: %.3f (%.1f days since last access)", score, days_since)
            
            elapsed = time.time() - start_time
            logger.debug("Recency score calculation took %.6f seconds", elapsed)
            return score
        except (ValueError, TypeError):
            logger.warning(f"Invalid timestamp format: {accessed_at}")
            elapsed = time.time() - start_time
            logger.debug("Recency score calculation took %.6f seconds", elapsed)
            return 0.0
    
    def _calculate_retrieval_score(self, retrieval_count: int) -> float:
        """Calculate score based on how often a note has been retrieved."""
        start_time = time.time()
        
        if not retrieval_count or retrieval_count < 0:
            logger.debug("No valid retrieval count, score = 0.0")
            elapsed = time.time() - start_time
            logger.debug("Retrieval score calculation took %.6f seconds", elapsed)
            return 0.0
            
        # Logarithmic scaling to prevent very popular notes from dominating
        score = min(1.0, max(0.0, np.log1p(retrieval_count) / np.log1p(100)))
        logger.debug("Retrieval score: %.3f (count: %d)", score, retrieval_count)
        
        elapsed = time.time() - start_time
        logger.debug("Retrieval score calculation took %.6f seconds", elapsed)
        return score
    
    def _calculate_keyword_match_score(self, query: str, keywords: List[str], content: str) -> float:
        """Calculate score based on keyword matches."""
        start_time = time.time()
        
        if not keywords or not query:
            logger.debug("No keywords or empty query, keyword score = 0.0")
            elapsed = time.time() - start_time
            logger.debug("Keyword match score calculation took %.6f seconds", elapsed)
            return 0.0
            
        query_terms = query.lower().split()
        keyword_score = 0.0
        matched_terms = []
        
        # Check for direct keyword matches
        for term in query_terms:
            matching_keywords = [keyword for keyword in keywords if term in keyword.lower()]
            if matching_keywords:
                keyword_score += 0.5
                matched_terms.append(f"{term} -> {matching_keywords}")
                logger.debug("Query term '%s' matched keywords: %s", term, matching_keywords)
                
        # Check for keyword presence in content
        content_matches = []
        for keyword in keywords:
            if keyword.lower() in content.lower():
                keyword_score += 0.1
                content_matches.append(keyword)
                
        if content_matches:
            logger.debug("Keywords found in content: %s", content_matches)
        
        final_score = min(1.0, max(0.0, keyword_score))
        logger.debug("Keyword match score: %.3f (matched terms: %s, content matches: %d)", 
                    final_score, matched_terms, len(content_matches))
        
        elapsed = time.time() - start_time
        logger.debug("Keyword match score calculation took %.6f seconds", elapsed)
        return final_score
    
    def _calculate_text_overlap_score(self, query: str, content: str) -> float:
        """Calculate score based on n-gram overlap between query and content."""
        start_time = time.time()
        
        if not query or not content:
            logger.debug("Empty query or content, text overlap score = 0.0")
            elapsed = time.time() - start_time
            logger.debug("Text overlap score calculation took %.6f seconds", elapsed)
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
        
        logger.debug("Generated %d query bigrams, %d content bigrams", len(query_bigrams), len(content_bigrams))
        logger.debug("Generated %d query trigrams, %d content trigrams", len(query_trigrams), len(content_trigrams))
        
        # Calculate Jaccard similarity for bigrams and trigrams
        bigram_overlap = 0.0
        if query_bigrams and content_bigrams:
            intersection = len(query_bigrams.intersection(content_bigrams))
            union = len(query_bigrams.union(content_bigrams))
            bigram_overlap = intersection / union if union > 0 else 0.0
            logger.debug("Bigram overlap: %.3f (intersection: %d, union: %d)", 
                        bigram_overlap, intersection, union)
            
        trigram_overlap = 0.0
        if query_trigrams and content_trigrams:
            intersection = len(query_trigrams.intersection(content_trigrams))
            union = len(query_trigrams.union(content_trigrams))
            trigram_overlap = intersection / union if union > 0 else 0.0
            logger.debug("Trigram overlap: %.3f (intersection: %d, union: %d)", 
                        trigram_overlap, intersection, union)
            
        # Weight bigrams and trigrams
        final_score = 0.4 * bigram_overlap + 0.6 * trigram_overlap
        logger.debug("Text overlap final score: %.3f", final_score)
        
        elapsed = time.time() - start_time
        logger.debug("Text overlap score calculation took %.6f seconds", elapsed)
        return final_score
    
    def _calculate_link_count_score(self, links: List[str]) -> float:
        """Calculate a score based on the number of links in a ZettelNote."""
        start_time = time.time()
        
        if not links:
            logger.debug("No links found, link count score = 0.0")
            elapsed = time.time() - start_time
            logger.debug("Link count score calculation took %.6f seconds", elapsed)
            return 0.0
        
        # Use logarithmic scaling to prevent notes with many links from dominating
        # Assume a reasonable maximum of 50 links for normalization
        link_count = len(links)
        score = min(1.0, np.log1p(link_count) / np.log1p(50))
        logger.debug("Link count score: %.3f (links: %d)", score, link_count)
        
        elapsed = time.time() - start_time
        logger.debug("Link count score calculation took %.6f seconds", elapsed)
        return score
    
    def rerank(self, query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rerank search results based on multiple factors."""
        overall_start_time = time.time()
        
        if not results:
            logger.info("No results to rerank, returning empty list")
            return []
            
        logger.info("Reranking %d search results for query: '%s'", len(results), query[:50])
        reranked_results = []
        
        for i, result in enumerate(results):
            result_start_time = time.time()
            note = result['note']
            base_score = result['score']
            note_id = result.get('id', f"result_{i}")
            
            logger.debug("Reranking note %s (base score: %.3f)", note_id, base_score)
            
            # Extract note attributes safely
            accessed_at = getattr(note, 'accessed_at', None)
            retrieval_count = getattr(note, 'retrieval_count', 0)
            
            # Log the extraction of keywords and tags
            try:
                keywords = getattr(note.note_simple, 'keywords', [])
                logger.debug("Extracted %d keywords from note_simple", len(keywords))
            except AttributeError:
                logger.debug("Failed to extract keywords from note_simple, using empty list")
                keywords = []
                
            tags = getattr(note, 'tags', [])
            logger.debug("Extracted %d tags from note", len(tags))
            
            keywords.extend(tags)
            content = getattr(note, 'content', '')
            content_preview = content[:100] + "..." if content else "[No content]"
            logger.debug("Content preview: %s", content_preview)
            
            # Extract links for the new scoring component
            links = getattr(note, 'links', [])
            logger.debug("Extracted %d links from note", len(links))
            
            # Calculate component scores
            logger.debug("Calculating component scores for note %s", note_id)
            recency_score = self._calculate_recency_score(accessed_at)
            retrieval_score = self._calculate_retrieval_score(retrieval_count)
            keyword_score = self._calculate_keyword_match_score(query, keywords, content)
            text_overlap_score = self._calculate_text_overlap_score(query, content)
            link_count_score = self._calculate_link_count_score(links)
            
            # Apply custom scoring if provided
            custom_score = 0.0
            if self.custom_scoring_fn:
                custom_start_time = time.time()
                try:
                    custom_score = self.custom_scoring_fn(query, note)
                    logger.debug("Custom scoring function returned: %.3f", custom_score)
                except Exception as e:
                    logger.error(f"Error in custom scoring function: {e}")
                custom_elapsed = time.time() - custom_start_time
                logger.debug("Custom scoring took %.6f seconds", custom_elapsed)
            
            # Calculate final score
            # Base score (vector similarity) gets at least 30% weight
            final_score = (
                0.3 * base_score +
                self.recency_weight * recency_score +
                self.retrieval_count_weight * retrieval_score +
                self.keyword_match_weight * keyword_score +
                self.text_overlap_weight * text_overlap_score +
                self.link_count_weight * link_count_score +
                custom_score
            )
            
            logger.debug("Final score calculation for %s:", note_id)
            logger.debug("  0.3 * %.3f (base) = %.3f", base_score, 0.3 * base_score)
            logger.debug("  %.3f * %.3f (recency) = %.3f", 
                        self.recency_weight, recency_score, self.recency_weight * recency_score)
            logger.debug("  %.3f * %.3f (retrieval) = %.3f", 
                        self.retrieval_count_weight, retrieval_score, self.retrieval_count_weight * retrieval_score)
            logger.debug("  %.3f * %.3f (keyword) = %.3f", 
                        self.keyword_match_weight, keyword_score, self.keyword_match_weight * keyword_score)
            logger.debug("  %.3f * %.3f (text) = %.3f", 
                        self.text_overlap_weight, text_overlap_score, self.text_overlap_weight * text_overlap_score)
            logger.debug("  %.3f * %.3f (link) = %.3f", 
                        self.link_count_weight, link_count_score, self.link_count_weight * link_count_score)
            logger.debug("  %.3f (custom)", custom_score)
            logger.debug("  = %.3f (final score)", final_score)
            
            # Create new result with updated score
            reranked_result = result.copy()
            reranked_result['score'] = final_score
            reranked_result['component_scores'] = {
                'base_score': base_score,
                'recency_score': recency_score,
                'retrieval_score': retrieval_score,
                'keyword_score': keyword_score,
                'text_overlap_score': text_overlap_score,
                'link_count_score': link_count_score
            }
            
            reranked_results.append(reranked_result)
            result_elapsed = time.time() - result_start_time
            logger.debug("Reranking note %s took %.6f seconds", note_id, result_elapsed)
        
        # Sort by final score in descending order
        reranked_results.sort(key=lambda x: x['score'], reverse=True)
        
        overall_elapsed = time.time() - overall_start_time
        logger.debug("Total reranking took %.6f seconds", overall_elapsed)
        
        return reranked_results