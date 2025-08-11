"""AI-powered intelligent query processing for legal RAG system.

This module implements advanced query processing capabilities including:
- Query intent detection and classification
- Automatic query expansion with legal synonyms
- Context-aware query optimization
- Multi-turn conversation support
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import logging
import asyncio

logger = logging.getLogger(__name__)


class QueryIntent(Enum):
    """Legal query intent classification."""
    SEARCH = "search"
    EXPLAIN = "explain"
    COMPARE = "compare"
    ANALYZE = "analyze"
    SUMMARIZE = "summarize"
    DEFINITION = "definition"
    PRECEDENT = "precedent"
    COMPLIANCE = "compliance"


@dataclass
class QueryEnhancement:
    """Query enhancement results."""
    original_query: str
    enhanced_query: str
    intent: QueryIntent
    confidence: float
    legal_terms: List[str] = field(default_factory=list)
    synonyms_added: List[str] = field(default_factory=list)
    context_expansions: List[str] = field(default_factory=list)


class LegalTermExpander:
    """Expands legal queries with relevant synonyms and related terms."""
    
    def __init__(self):
        # Legal term synonym mapping for query expansion
        self.legal_synonyms = {
            "contract": ["agreement", "compact", "covenant", "deal", "pact"],
            "liability": ["responsibility", "obligation", "accountability", "fault"],
            "breach": ["violation", "infringement", "non-compliance", "default"],
            "damages": ["compensation", "restitution", "remedy", "reparation"],
            "indemnify": ["protect", "secure", "guarantee", "hold harmless"],
            "warranty": ["guarantee", "assurance", "promise", "representation"],
            "terminate": ["end", "cancel", "dissolve", "conclude", "cease"],
            "jurisdiction": ["authority", "power", "control", "domain"],
            "precedent": ["case law", "judicial decision", "ruling", "authority"],
            "compliance": ["conformity", "adherence", "observance", "fulfillment"],
            "statute": ["law", "regulation", "act", "code", "ordinance"],
            "negligence": ["carelessness", "fault", "dereliction", "omission"],
            "consideration": ["payment", "compensation", "quid pro quo", "exchange"],
            "fiduciary": ["trustee", "custodian", "guardian", "steward"],
            "tort": ["civil wrong", "private wrong", "injury", "harm"],
        }
        
        # Legal domain expansions
        self.domain_expansions = {
            "contract": ["formation", "performance", "enforcement", "remedies"],
            "corporate": ["governance", "compliance", "securities", "M&A"],
            "employment": ["discrimination", "benefits", "termination", "safety"],
            "intellectual property": ["patents", "trademarks", "copyrights", "trade secrets"],
            "real estate": ["property rights", "zoning", "leases", "transfers"],
            "tax": ["deductions", "credits", "penalties", "audits"],
            "environmental": ["regulations", "permits", "cleanup", "liability"],
            "healthcare": ["HIPAA", "compliance", "malpractice", "regulations"],
        }
    
    def expand_query(self, query: str) -> Tuple[str, List[str], List[str]]:
        """Expand query with legal synonyms and related terms.
        
        Returns:
            Tuple of (expanded_query, synonyms_added, context_expansions)
        """
        expanded_terms = []
        synonyms_added = []
        context_expansions = []
        
        query_lower = query.lower()
        words = re.findall(r'\b\w+\b', query_lower)
        
        # Find legal terms and add synonyms
        for term, synonyms in self.legal_synonyms.items():
            if term in query_lower:
                # Add most relevant synonym
                best_synonym = self._select_best_synonym(term, synonyms, query_lower)
                if best_synonym and best_synonym not in query_lower:
                    expanded_terms.append(best_synonym)
                    synonyms_added.append(best_synonym)
        
        # Add domain-specific expansions
        for domain, expansions in self.domain_expansions.items():
            if any(word in domain.split() for word in words):
                relevant_expansions = expansions[:2]  # Limit to prevent over-expansion
                expanded_terms.extend(relevant_expansions)
                context_expansions.extend(relevant_expansions)
        
        # Construct enhanced query
        if expanded_terms:
            # Use OR logic for better retrieval
            expansion_clause = " OR ".join(expanded_terms)
            enhanced_query = f"{query} OR ({expansion_clause})"
        else:
            enhanced_query = query
        
        return enhanced_query, synonyms_added, context_expansions
    
    def _select_best_synonym(self, term: str, synonyms: List[str], context: str) -> Optional[str]:
        """Select the most appropriate synonym based on context."""
        # Simple heuristic: prefer shorter synonyms for better matching
        # In production, this would use semantic similarity
        return min(synonyms, key=len) if synonyms else None


class QueryIntentClassifier:
    """Classifies legal query intent using pattern matching."""
    
    def __init__(self):
        # Intent detection patterns
        self.intent_patterns = {
            QueryIntent.SEARCH: [
                r'\b(find|search|locate|show|get)\b',
                r'\b(documents?|cases?|statutes?)\b.*\b(about|regarding|concerning)\b',
            ],
            QueryIntent.EXPLAIN: [
                r'\b(explain|clarify|define|what\s+does|what\s+is|how\s+does)\b',
                r'\b(meaning|interpretation|definition)\b',
            ],
            QueryIntent.COMPARE: [
                r'\b(compare|contrast|difference|versus|vs\.?)\b',
                r'\b(similar|different|alike|unlike)\b',
            ],
            QueryIntent.ANALYZE: [
                r'\b(analyz|assess|evaluat|review|examin)\b',
                r'\b(implications|consequences|effects|impact)\b',
            ],
            QueryIntent.SUMMARIZE: [
                r'\b(summar|overview|brief|key\s+points)\b',
                r'\b(main\s+points|highlights|essence)\b',
            ],
            QueryIntent.DEFINITION: [
                r'\b(define|definition|what\s+is|what\s+does.*mean)\b',
                r'\b(terminology|glossary|lexicon)\b',
            ],
            QueryIntent.PRECEDENT: [
                r'\b(precedent|case\s+law|judicial\s+decision|ruling)\b',
                r'\b(similar\s+cases|comparable\s+situations)\b',
            ],
            QueryIntent.COMPLIANCE: [
                r'\b(complian|regulatory|requirement|obligation)\b',
                r'\b(must|shall|required|mandatory)\b',
            ],
        }
    
    def classify_intent(self, query: str) -> Tuple[QueryIntent, float]:
        """Classify query intent with confidence score.
        
        Returns:
            Tuple of (intent, confidence_score)
        """
        query_lower = query.lower()
        intent_scores = {}
        
        # Calculate scores for each intent
        for intent, patterns in self.intent_patterns.items():
            score = 0
            matches = 0
            
            for pattern in patterns:
                pattern_matches = len(re.findall(pattern, query_lower))
                if pattern_matches > 0:
                    score += pattern_matches
                    matches += 1
            
            # Normalize score by number of patterns
            if matches > 0:
                intent_scores[intent] = score / len(patterns)
        
        if not intent_scores:
            return QueryIntent.SEARCH, 0.5  # Default fallback
        
        # Return intent with highest score
        best_intent = max(intent_scores.items(), key=lambda x: x[1])
        confidence = min(best_intent[1], 1.0)  # Cap at 1.0
        
        return best_intent[0], confidence


class IntelligentQueryProcessor:
    """Advanced query processor with AI-powered enhancements."""
    
    def __init__(self):
        self.term_expander = LegalTermExpander()
        self.intent_classifier = QueryIntentClassifier()
        self.query_history: List[str] = []
        self.conversation_context: Dict[str, Any] = {}
    
    async def process_query(self, query: str, conversation_id: Optional[str] = None) -> QueryEnhancement:
        """Process and enhance a legal query with AI capabilities."""
        logger.info(f"Processing query: {query}")
        
        # Update conversation context
        if conversation_id:
            self._update_conversation_context(conversation_id, query)
        
        # Extract legal terms
        legal_terms = self._extract_legal_terms(query)
        
        # Classify intent
        intent, confidence = self.intent_classifier.classify_intent(query)
        logger.debug(f"Classified intent: {intent.value} (confidence: {confidence:.2f})")
        
        # Expand query with synonyms and context
        enhanced_query, synonyms_added, context_expansions = self.term_expander.expand_query(query)
        
        # Apply conversation context if available
        if conversation_id and self.conversation_context.get(conversation_id):
            enhanced_query = self._apply_conversation_context(enhanced_query, conversation_id)
        
        # Add query to history
        self.query_history.append(query)
        if len(self.query_history) > 100:  # Limit history size
            self.query_history = self.query_history[-100:]
        
        logger.info(f"Enhanced query: {enhanced_query}")
        
        return QueryEnhancement(
            original_query=query,
            enhanced_query=enhanced_query,
            intent=intent,
            confidence=confidence,
            legal_terms=legal_terms,
            synonyms_added=synonyms_added,
            context_expansions=context_expansions
        )
    
    def _extract_legal_terms(self, query: str) -> List[str]:
        """Extract legal terms from the query."""
        legal_terms = []
        query_lower = query.lower()
        
        # Check for legal terms from our synonym dictionary
        for term in self.term_expander.legal_synonyms.keys():
            if term in query_lower:
                legal_terms.append(term)
        
        # Check for domain terms
        for domain in self.term_expander.domain_expansions.keys():
            if any(word in query_lower for word in domain.split()):
                legal_terms.append(domain)
        
        return legal_terms
    
    def _update_conversation_context(self, conversation_id: str, query: str) -> None:
        """Update conversation context for multi-turn support."""
        if conversation_id not in self.conversation_context:
            self.conversation_context[conversation_id] = {
                'queries': [],
                'topics': set(),
                'last_intent': None
            }
        
        context = self.conversation_context[conversation_id]
        context['queries'].append(query)
        
        # Extract topics from query
        legal_terms = self._extract_legal_terms(query)
        context['topics'].update(legal_terms)
        
        # Limit context size
        if len(context['queries']) > 10:
            context['queries'] = context['queries'][-10:]
    
    def _apply_conversation_context(self, query: str, conversation_id: str) -> str:
        """Apply conversation context to enhance query."""
        context = self.conversation_context.get(conversation_id, {})
        topics = context.get('topics', set())
        
        if topics:
            # Add relevant context terms
            context_terms = list(topics)[:3]  # Limit to 3 most recent topics
            context_clause = " OR ".join(context_terms)
            return f"{query} OR ({context_clause})"
        
        return query
    
    def get_query_suggestions(self, partial_query: str) -> List[str]:
        """Generate query suggestions based on partial input."""
        suggestions = []
        partial_lower = partial_query.lower()
        
        # Suggest based on legal terms
        for term in self.term_expander.legal_synonyms.keys():
            if term.startswith(partial_lower):
                suggestions.append(f"What is {term}?")
                suggestions.append(f"Find documents about {term}")
        
        # Suggest based on query history
        for past_query in self.query_history[-20:]:  # Recent queries
            if partial_lower in past_query.lower():
                suggestions.append(past_query)
        
        return list(set(suggestions))[:5]  # Return unique suggestions, max 5
    
    def analyze_query_patterns(self) -> Dict[str, Any]:
        """Analyze query patterns for optimization insights."""
        if not self.query_history:
            return {}
        
        intent_counts = {}
        term_frequency = {}
        
        for query in self.query_history:
            # Analyze intent patterns
            intent, _ = self.intent_classifier.classify_intent(query)
            intent_counts[intent.value] = intent_counts.get(intent.value, 0) + 1
            
            # Analyze term frequency
            terms = self._extract_legal_terms(query)
            for term in terms:
                term_frequency[term] = term_frequency.get(term, 0) + 1
        
        return {
            'total_queries': len(self.query_history),
            'intent_distribution': intent_counts,
            'popular_terms': dict(sorted(term_frequency.items(), 
                                       key=lambda x: x[1], reverse=True)[:10]),
            'average_query_length': sum(len(q.split()) for q in self.query_history) / len(self.query_history)
        }


# Global instance for use across the application
_global_query_processor = None

def get_query_processor() -> IntelligentQueryProcessor:
    """Get global query processor instance."""
    global _global_query_processor
    if _global_query_processor is None:
        _global_query_processor = IntelligentQueryProcessor()
    return _global_query_processor


async def enhance_query(query: str, conversation_id: Optional[str] = None) -> QueryEnhancement:
    """Convenience function to enhance a query."""
    processor = get_query_processor()
    return await processor.process_query(query, conversation_id)


def get_query_suggestions(partial_query: str) -> List[str]:
    """Convenience function to get query suggestions."""
    processor = get_query_processor()
    return processor.get_query_suggestions(partial_query)


def get_query_analytics() -> Dict[str, Any]:
    """Convenience function to get query analytics."""
    processor = get_query_processor()
    return processor.analyze_query_patterns()