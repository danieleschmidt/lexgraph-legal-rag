"""
Multi-Sensory Legal Document Processor

Integrates bioneural olfactory fusion with traditional text processing
to create a comprehensive multi-sensory analysis pipeline for legal documents.

This module bridges traditional NLP with bio-inspired sensory computing,
enabling enhanced document understanding through multiple "sensory" channels.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np

from .bioneuro_olfactory_fusion import DocumentScentProfile
from .bioneuro_olfactory_fusion import get_fusion_engine


logger = logging.getLogger(__name__)


class SensoryChannel(Enum):
    """Available sensory channels for document analysis."""

    TEXTUAL = "textual"  # Traditional text processing
    OLFACTORY = "olfactory"  # Bioneural olfactory fusion
    VISUAL = "visual"  # Document structure and formatting
    TEMPORAL = "temporal"  # Time-based pattern analysis
    SEMANTIC = "semantic"  # Deep semantic understanding


@dataclass
class SensorySignal:
    """Represents a signal from a specific sensory channel."""

    channel: SensoryChannel
    strength: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    features: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class MultiSensoryAnalysis:
    """Comprehensive multi-sensory analysis result for a legal document."""

    document_id: str
    sensory_signals: list[SensorySignal]
    fusion_vector: np.ndarray
    primary_sensory_channel: SensoryChannel
    analysis_confidence: float
    scent_profile: DocumentScentProfile | None = None

    def get_signal_by_channel(self, channel: SensoryChannel) -> SensorySignal | None:
        """Get sensory signal by channel type."""
        for signal in self.sensory_signals:
            if signal.channel == channel:
                return signal
        return None

    def get_channel_strengths(self) -> dict[SensoryChannel, float]:
        """Get strength values for all channels."""
        return {signal.channel: signal.strength for signal in self.sensory_signals}


class TextualSensoryProcessor:
    """Processes traditional textual features of legal documents."""

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.TextualProcessor")

    async def process(
        self, document_text: str, metadata: dict[str, Any]
    ) -> SensorySignal:
        """Process document through textual sensory channel."""
        try:
            # Extract textual features
            features = {
                "word_count": len(document_text.split()),
                "sentence_count": document_text.count(".")
                + document_text.count("!")
                + document_text.count("?"),
                "paragraph_count": document_text.count("\n\n") + 1,
                "avg_sentence_length": self._calculate_avg_sentence_length(
                    document_text
                ),
                "lexical_diversity": self._calculate_lexical_diversity(document_text),
                "legal_term_density": self._calculate_legal_term_density(document_text),
            }

            # Calculate overall textual strength
            strength = self._calculate_textual_strength(features)
            confidence = min(
                1.0, features["word_count"] / 100.0
            )  # Higher confidence with more text

            return SensorySignal(
                channel=SensoryChannel.TEXTUAL,
                strength=strength,
                confidence=confidence,
                features=features,
            )

        except Exception as e:
            self.logger.error(f"Textual processing failed: {e}")
            return SensorySignal(
                channel=SensoryChannel.TEXTUAL,
                strength=0.0,
                confidence=0.0,
                features={"error": str(e)},
            )

    def _calculate_avg_sentence_length(self, text: str) -> float:
        """Calculate average sentence length."""
        sentences = [
            s.strip()
            for s in text.replace("!", ".").replace("?", ".").split(".")
            if s.strip()
        ]
        if not sentences:
            return 0.0

        total_words = sum(len(sentence.split()) for sentence in sentences)
        return total_words / len(sentences)

    def _calculate_lexical_diversity(self, text: str) -> float:
        """Calculate lexical diversity (unique words / total words)."""
        words = text.lower().split()
        if not words:
            return 0.0

        unique_words = set(words)
        return len(unique_words) / len(words)

    def _calculate_legal_term_density(self, text: str) -> float:
        """Calculate density of legal terminology."""
        legal_terms = [
            "contract",
            "agreement",
            "liability",
            "damages",
            "breach",
            "statute",
            "regulation",
            "compliance",
            "violation",
            "penalty",
            "jurisdiction",
            "court",
            "judge",
            "attorney",
            "counsel",
            "plaintiff",
            "defendant",
            "evidence",
            "testimony",
            "verdict",
        ]

        word_count = len(text.split())
        if word_count == 0:
            return 0.0

        legal_term_count = sum(
            1 for term in legal_terms if term.lower() in text.lower()
        )
        return legal_term_count / word_count

    def _calculate_textual_strength(self, features: dict[str, Any]) -> float:
        """Calculate overall textual processing strength."""
        # Normalize and combine features
        word_score = min(1.0, features["word_count"] / 1000.0)
        complexity_score = min(1.0, features["avg_sentence_length"] / 30.0)
        diversity_score = features["lexical_diversity"]
        legal_score = min(1.0, features["legal_term_density"] * 10.0)

        # Weighted combination
        strength = (
            0.3 * word_score
            + 0.2 * complexity_score
            + 0.2 * diversity_score
            + 0.3 * legal_score
        )

        return float(strength)


class VisualSensoryProcessor:
    """Processes visual/structural features of legal documents."""

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.VisualProcessor")

    async def process(
        self, document_text: str, metadata: dict[str, Any]
    ) -> SensorySignal:
        """Process document through visual sensory channel."""
        try:
            features = {
                "structure_complexity": self._analyze_structure_complexity(
                    document_text
                ),
                "formatting_indicators": self._detect_formatting_indicators(
                    document_text
                ),
                "section_organization": self._analyze_section_organization(
                    document_text
                ),
                "list_structure": self._analyze_list_structure(document_text),
                "whitespace_patterns": self._analyze_whitespace_patterns(document_text),
            }

            # Calculate visual processing strength
            strength = self._calculate_visual_strength(features)
            confidence = 0.8  # Visual processing generally reliable

            return SensorySignal(
                channel=SensoryChannel.VISUAL,
                strength=strength,
                confidence=confidence,
                features=features,
            )

        except Exception as e:
            self.logger.error(f"Visual processing failed: {e}")
            return SensorySignal(
                channel=SensoryChannel.VISUAL,
                strength=0.0,
                confidence=0.0,
                features={"error": str(e)},
            )

    def _analyze_structure_complexity(self, text: str) -> float:
        """Analyze structural complexity of document."""
        # Count various structural elements
        headings = text.count("\n#") + text.count("\nSection") + text.count("\nChapter")
        subsections = text.count("\n##") + text.count("\n  ") + text.count("\n\t")
        numbered_items = len(
            [
                line
                for line in text.split("\n")
                if line.strip() and line.strip()[0].isdigit()
            ]
        )

        complexity_score = min(
            1.0, (headings * 0.1) + (subsections * 0.05) + (numbered_items * 0.02)
        )
        return complexity_score

    def _detect_formatting_indicators(self, text: str) -> dict[str, int]:
        """Detect various formatting indicators."""
        return {
            "bold_indicators": text.count("**") + text.count("__"),
            "italic_indicators": text.count("*") + text.count("_"),
            "bullet_points": text.count("• ") + text.count("- ") + text.count("* "),
            "indentation_levels": len(
                {len(line) - len(line.lstrip()) for line in text.split("\n")}
            ),
            "all_caps_words": len(
                [word for word in text.split() if word.isupper() and len(word) > 2]
            ),
        }

    def _analyze_section_organization(self, text: str) -> float:
        """Analyze how well the document is organized into sections."""
        lines = text.split("\n")
        section_markers = ["Section", "Chapter", "Part", "Article", "Subsection"]

        section_count = sum(
            1 for line in lines if any(marker in line for marker in section_markers)
        )
        total_lines = len([line for line in lines if line.strip()])

        if total_lines == 0:
            return 0.0

        organization_score = min(1.0, section_count / (total_lines / 10))
        return organization_score

    def _analyze_list_structure(self, text: str) -> dict[str, Any]:
        """Analyze list and enumeration structures."""
        lines = text.split("\n")

        numbered_lists = len(
            [
                line
                for line in lines
                if line.strip()
                and len(line.strip()) > 2
                and line.strip()[:2].replace(".", "").isdigit()
            ]
        )
        bulleted_lists = len(
            [line for line in lines if line.strip().startswith(("•", "-", "*"))]
        )
        nested_lists = len(
            [
                line
                for line in lines
                if line.startswith(("  ", "\t"))
                and any(
                    line.strip().startswith(marker)
                    for marker in ["•", "-", "*", "1.", "a."]
                )
            ]
        )

        return {
            "numbered_lists": numbered_lists,
            "bulleted_lists": bulleted_lists,
            "nested_lists": nested_lists,
            "total_list_items": numbered_lists + bulleted_lists,
        }

    def _analyze_whitespace_patterns(self, text: str) -> dict[str, float]:
        """Analyze whitespace usage patterns."""
        lines = text.split("\n")

        empty_lines = len([line for line in lines if not line.strip()])
        total_lines = len(lines)

        if total_lines == 0:
            return {"whitespace_ratio": 0.0, "paragraph_separation": 0.0}

        whitespace_ratio = empty_lines / total_lines

        # Analyze paragraph separation
        paragraph_breaks = text.count("\n\n")
        paragraph_separation = paragraph_breaks / max(1, text.count("\n") / 5)

        return {
            "whitespace_ratio": whitespace_ratio,
            "paragraph_separation": min(1.0, paragraph_separation),
        }

    def _calculate_visual_strength(self, features: dict[str, Any]) -> float:
        """Calculate overall visual processing strength."""
        structure_score = features["structure_complexity"]

        formatting_score = min(
            1.0, sum(features["formatting_indicators"].values()) / 20.0
        )
        organization_score = features["section_organization"]

        list_score = min(1.0, features["list_structure"]["total_list_items"] / 10.0)

        # Weighted combination
        strength = (
            0.4 * structure_score
            + 0.2 * formatting_score
            + 0.3 * organization_score
            + 0.1 * list_score
        )

        return float(strength)


class TemporalSensoryProcessor:
    """Processes temporal patterns and time-based features."""

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.TemporalProcessor")

    async def process(
        self, document_text: str, metadata: dict[str, Any]
    ) -> SensorySignal:
        """Process document through temporal sensory channel."""
        try:
            features = {
                "temporal_references": self._extract_temporal_references(document_text),
                "chronological_order": self._analyze_chronological_order(document_text),
                "temporal_density": self._calculate_temporal_density(document_text),
                "recency_indicators": self._detect_recency_indicators(document_text),
                "temporal_context": self._analyze_temporal_context(metadata),
            }

            strength = self._calculate_temporal_strength(features)
            confidence = min(1.0, len(features["temporal_references"]) / 5.0)

            return SensorySignal(
                channel=SensoryChannel.TEMPORAL,
                strength=strength,
                confidence=confidence,
                features=features,
            )

        except Exception as e:
            self.logger.error(f"Temporal processing failed: {e}")
            return SensorySignal(
                channel=SensoryChannel.TEMPORAL,
                strength=0.0,
                confidence=0.0,
                features={"error": str(e)},
            )

    def _extract_temporal_references(self, text: str) -> list[dict[str, Any]]:
        """Extract temporal references from text."""
        import re

        temporal_patterns = [
            (
                r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b",
                "date",
            ),
            (r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", "date"),
            (r"\b(19|20)\d{2}\b", "year"),
            (
                r"\b(effective|commencing|beginning|starting|ending|expiring|until|before|after)\s+\w+",
                "temporal_marker",
            ),
            (r"\b(annually|monthly|weekly|daily|quarterly|biannually)\b", "frequency"),
            (r"\b(within|during|throughout|since|from)\s+\w+", "duration_marker"),
        ]

        temporal_refs = []
        for pattern, ref_type in temporal_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                temporal_refs.append(
                    {
                        "type": ref_type,
                        "value": match if isinstance(match, str) else match[0],
                        "pattern": pattern,
                    }
                )

        return temporal_refs

    def _analyze_chronological_order(self, text: str) -> float:
        """Analyze how well document follows chronological order."""
        temporal_refs = self._extract_temporal_references(text)
        year_refs = [ref for ref in temporal_refs if ref["type"] == "year"]

        if len(year_refs) < 2:
            return 0.5  # Neutral if insufficient temporal data

        # Check if years generally increase through the document
        year_positions = []
        for ref in year_refs:
            try:
                year = int(ref["value"])
                position = text.find(ref["value"]) / len(text)
                year_positions.append((position, year))
            except ValueError:
                continue

        if len(year_positions) < 2:
            return 0.5

        year_positions.sort(key=lambda x: x[0])  # Sort by position in text

        # Calculate how often years increase with position
        increasing_count = 0
        for i in range(1, len(year_positions)):
            if year_positions[i][1] >= year_positions[i - 1][1]:
                increasing_count += 1

        chronological_score = (
            increasing_count / (len(year_positions) - 1)
            if len(year_positions) > 1
            else 0.5
        )
        return chronological_score

    def _calculate_temporal_density(self, text: str) -> float:
        """Calculate density of temporal references."""
        temporal_refs = self._extract_temporal_references(text)
        word_count = len(text.split())

        if word_count == 0:
            return 0.0

        temporal_density = len(temporal_refs) / (
            word_count / 100
        )  # References per 100 words
        return min(1.0, temporal_density)

    def _detect_recency_indicators(self, text: str) -> dict[str, Any]:
        """Detect indicators of temporal recency."""
        from datetime import datetime

        recency_keywords = [
            "recent",
            "current",
            "latest",
            "new",
            "updated",
            "modern",
            "contemporary",
            "today",
            "now",
            "present",
            "ongoing",
        ]

        current_year = datetime.now().year
        recency_count = sum(
            1 for keyword in recency_keywords if keyword.lower() in text.lower()
        )

        # Check for recent years
        import re

        years = [int(year) for year in re.findall(r"\b(20[0-2]\d)\b", text)]
        recent_years = [year for year in years if current_year - year <= 5]

        return {
            "recency_keyword_count": recency_count,
            "recent_years_mentioned": len(recent_years),
            "total_years_mentioned": len(years),
            "recency_score": min(1.0, (recency_count + len(recent_years)) / 10.0),
        }

    def _analyze_temporal_context(self, metadata: dict[str, Any]) -> dict[str, Any]:
        """Analyze temporal context from metadata."""
        context = {
            "creation_date": metadata.get("creation_date"),
            "modification_date": metadata.get("modification_date"),
            "document_age_days": None,
            "is_recent": False,
        }

        if context["creation_date"]:
            try:
                if isinstance(context["creation_date"], str):
                    creation_date = datetime.fromisoformat(
                        context["creation_date"].replace("Z", "+00:00")
                    )
                else:
                    creation_date = context["creation_date"]

                age_days = (datetime.now() - creation_date.replace(tzinfo=None)).days
                context["document_age_days"] = age_days
                context["is_recent"] = age_days <= 365  # Recent if less than a year old

            except (ValueError, TypeError) as e:
                logger.warning(f"Could not parse creation date: {e}")

        return context

    def _calculate_temporal_strength(self, features: dict[str, Any]) -> float:
        """Calculate overall temporal processing strength."""
        ref_count_score = min(1.0, len(features["temporal_references"]) / 10.0)
        chronological_score = features["chronological_order"]
        density_score = features["temporal_density"]
        recency_score = features["recency_indicators"]["recency_score"]

        # Weighted combination
        strength = (
            0.3 * ref_count_score
            + 0.3 * chronological_score
            + 0.2 * density_score
            + 0.2 * recency_score
        )

        return float(strength)


class MultiSensoryLegalProcessor:
    """Main processor that coordinates all sensory channels."""

    def __init__(self, enable_olfactory: bool = True):
        """Initialize multi-sensory processor."""
        self.enable_olfactory = enable_olfactory

        # Initialize sensory processors
        self.textual_processor = TextualSensoryProcessor()
        self.visual_processor = VisualSensoryProcessor()
        self.temporal_processor = TemporalSensoryProcessor()

        if enable_olfactory:
            self.olfactory_engine = get_fusion_engine()

        self.logger = logging.getLogger(__name__)

    async def process_document(
        self,
        document_text: str,
        document_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> MultiSensoryAnalysis:
        """Process document through all available sensory channels."""
        metadata = metadata or {}

        self.logger.info(
            f"Processing document {document_id} through multi-sensory pipeline"
        )

        # Process through all sensory channels in parallel
        tasks = [
            self.textual_processor.process(document_text, metadata),
            self.visual_processor.process(document_text, metadata),
            self.temporal_processor.process(document_text, metadata),
        ]

        # Add olfactory processing if enabled
        olfactory_task = None
        if self.enable_olfactory:
            olfactory_task = self.olfactory_engine.analyze_document(
                document_text, document_id, metadata
            )
            tasks.append(olfactory_task)

        # Execute all processing tasks
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Separate sensory signals from olfactory profile
        sensory_signals = []
        scent_profile = None

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(f"Sensory processing task {i} failed: {result}")
                continue

            if isinstance(result, SensorySignal):
                sensory_signals.append(result)
            elif hasattr(result, "signals"):  # DocumentScentProfile
                scent_profile = result
                # Convert olfactory profile to sensory signal
                olfactory_signal = SensorySignal(
                    channel=SensoryChannel.OLFACTORY,
                    strength=float(np.mean(result.composite_scent)),
                    confidence=float(
                        np.mean([s.confidence for s in result.signals])
                        if result.signals
                        else 0.0
                    ),
                    features={
                        "receptor_signals": len(result.signals),
                        "composite_dimensions": len(result.composite_scent),
                        "similarity_hash": result.similarity_hash,
                    },
                )
                sensory_signals.append(olfactory_signal)

        # Create fusion vector combining all sensory channels
        fusion_vector = self._create_fusion_vector(sensory_signals)

        # Determine primary sensory channel
        primary_channel = self._determine_primary_channel(sensory_signals)

        # Calculate overall analysis confidence
        analysis_confidence = self._calculate_analysis_confidence(sensory_signals)

        # Create comprehensive analysis result
        analysis = MultiSensoryAnalysis(
            document_id=document_id,
            sensory_signals=sensory_signals,
            fusion_vector=fusion_vector,
            primary_sensory_channel=primary_channel,
            analysis_confidence=analysis_confidence,
            scent_profile=scent_profile,
        )

        self.logger.info(
            f"Multi-sensory analysis completed for {document_id} with "
            f"{len(sensory_signals)} channels, primary: {primary_channel.value}"
        )

        return analysis

    def _create_fusion_vector(self, signals: list[SensorySignal]) -> np.ndarray:
        """Create unified fusion vector from all sensory signals."""
        # Create vector with 2 dimensions per channel (strength + confidence)
        vector_size = len(SensoryChannel) * 2
        fusion_vector = np.zeros(vector_size)

        for signal in signals:
            channel_idx = list(SensoryChannel).index(signal.channel)
            base_idx = channel_idx * 2

            fusion_vector[base_idx] = signal.strength
            fusion_vector[base_idx + 1] = signal.confidence

        return fusion_vector

    def _determine_primary_channel(
        self, signals: list[SensorySignal]
    ) -> SensoryChannel:
        """Determine which sensory channel provides the strongest signal."""
        if not signals:
            return SensoryChannel.TEXTUAL  # Default fallback

        # Calculate weighted strength (strength * confidence)
        channel_scores = {}
        for signal in signals:
            weighted_strength = signal.strength * signal.confidence
            channel_scores[signal.channel] = weighted_strength

        # Return channel with highest weighted strength
        primary_channel = max(channel_scores.items(), key=lambda x: x[1])[0]
        return primary_channel

    def _calculate_analysis_confidence(self, signals: list[SensorySignal]) -> float:
        """Calculate overall confidence in the multi-sensory analysis."""
        if not signals:
            return 0.0

        # Weighted average of confidence scores
        total_weighted_confidence = sum(
            signal.strength * signal.confidence for signal in signals
        )
        total_strength = sum(signal.strength for signal in signals)

        if total_strength == 0:
            return 0.0

        analysis_confidence = total_weighted_confidence / total_strength
        return float(analysis_confidence)


# Global processor instance
_multisensory_processor: MultiSensoryLegalProcessor | None = None


def get_multisensory_processor(
    enable_olfactory: bool = True,
) -> MultiSensoryLegalProcessor:
    """Get or create global multi-sensory processor instance."""
    global _multisensory_processor
    if _multisensory_processor is None:
        _multisensory_processor = MultiSensoryLegalProcessor(
            enable_olfactory=enable_olfactory
        )
    return _multisensory_processor


async def analyze_document_multisensory(
    document_text: str, document_id: str, metadata: dict[str, Any] | None = None
) -> MultiSensoryAnalysis:
    """Convenience function for multi-sensory document analysis."""
    processor = get_multisensory_processor()
    return await processor.process_document(document_text, document_id, metadata)
