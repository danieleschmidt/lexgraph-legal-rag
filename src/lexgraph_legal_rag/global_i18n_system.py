"""
Global Internationalization (I18n) System
=========================================

Global-first implementation with comprehensive internationalization support:
- Multi-language document processing (English, Spanish, French, German, Japanese, Chinese)
- Locale-aware legal terminology detection
- Cross-cultural compliance frameworks
- Regional data privacy regulations (GDPR, CCPA, PDPA)
- Currency and date format localization
- Right-to-left language support
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import asyncio

logger = logging.getLogger(__name__)


class SupportedLanguage(Enum):
    """Supported languages for global deployment."""
    
    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    JAPANESE = "ja"
    CHINESE = "zh"
    PORTUGUESE = "pt"
    ITALIAN = "it"
    DUTCH = "nl"
    KOREAN = "ko"


class LegalSystem(Enum):
    """Major legal systems worldwide."""
    
    COMMON_LAW = "common_law"          # US, UK, India, Australia
    CIVIL_LAW = "civil_law"            # Europe, Latin America, Asia
    RELIGIOUS_LAW = "religious_law"    # Islamic, Jewish law
    CUSTOMARY_LAW = "customary_law"    # Traditional systems
    MIXED_SYSTEM = "mixed_system"      # South Africa, Scotland


class DataPrivacyRegime(Enum):
    """Data privacy regulatory frameworks."""
    
    GDPR = "gdpr"                      # EU General Data Protection Regulation
    CCPA = "ccpa"                      # California Consumer Privacy Act
    PDPA = "pdpa"                      # Personal Data Protection Acts (Singapore, etc.)
    LGPD = "lgpd"                      # Brazil Lei Geral de Proteção de Dados
    PIPEDA = "pipeda"                  # Canada Personal Information Protection
    DATA_PROTECTION_ACT = "dpa_uk"     # UK Data Protection Act


@dataclass
class LocaleConfig:
    """Configuration for specific locale."""
    
    language: SupportedLanguage
    country_code: str
    legal_system: LegalSystem
    privacy_regime: DataPrivacyRegime
    currency_code: str
    date_format: str
    number_format: str
    rtl_support: bool = False  # Right-to-left text support
    legal_citation_format: str = "default"


@dataclass
class TranslationEntry:
    """Translation entry with metadata."""
    
    key: str
    translations: Dict[str, str]
    context: str = ""
    legal_domain: str = "general"
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class LegalTerminologyEntry:
    """Legal terminology with multi-language support."""
    
    term_id: str
    base_language: SupportedLanguage
    translations: Dict[SupportedLanguage, str]
    legal_system: LegalSystem
    definition: Dict[SupportedLanguage, str] = field(default_factory=dict)
    synonyms: Dict[SupportedLanguage, List[str]] = field(default_factory=dict)
    jurisdiction: str = "international"


class GlobalI18nSystem:
    """
    Comprehensive internationalization system for global legal document processing.
    
    Features:
    - Multi-language content processing and analysis
    - Legal terminology translation and localization
    - Cultural adaptation of legal concepts
    - Compliance with regional data privacy laws
    - Locale-aware formatting and presentation
    - Cross-cultural user interface adaptation
    """
    
    def __init__(self, default_locale: str = "en_US"):
        self.default_locale = default_locale
        self.current_locale = default_locale
        
        # Translation and terminology databases
        self._translations: Dict[str, TranslationEntry] = {}
        self._legal_terminology: Dict[str, LegalTerminologyEntry] = {}
        
        # Locale configurations
        self._locale_configs: Dict[str, LocaleConfig] = {}
        
        # Initialize default locales
        self._initialize_default_locales()
        self._initialize_legal_terminology()
        self._initialize_translations()
    
    def _initialize_default_locales(self):
        """Initialize default locale configurations."""
        
        # United States (English)
        self._locale_configs["en_US"] = LocaleConfig(
            language=SupportedLanguage.ENGLISH,
            country_code="US",
            legal_system=LegalSystem.COMMON_LAW,
            privacy_regime=DataPrivacyRegime.CCPA,
            currency_code="USD",
            date_format="%m/%d/%Y",
            number_format="1,234.56",
            legal_citation_format="bluebook"
        )
        
        # United Kingdom (English)
        self._locale_configs["en_GB"] = LocaleConfig(
            language=SupportedLanguage.ENGLISH,
            country_code="GB",
            legal_system=LegalSystem.COMMON_LAW,
            privacy_regime=DataPrivacyRegime.DATA_PROTECTION_ACT,
            currency_code="GBP",
            date_format="%d/%m/%Y",
            number_format="1,234.56",
            legal_citation_format="oscola"
        )
        
        # Germany (German)
        self._locale_configs["de_DE"] = LocaleConfig(
            language=SupportedLanguage.GERMAN,
            country_code="DE",
            legal_system=LegalSystem.CIVIL_LAW,
            privacy_regime=DataPrivacyRegime.GDPR,
            currency_code="EUR",
            date_format="%d.%m.%Y",
            number_format="1.234,56",
            legal_citation_format="german"
        )
        
        # France (French)
        self._locale_configs["fr_FR"] = LocaleConfig(
            language=SupportedLanguage.FRENCH,
            country_code="FR",
            legal_system=LegalSystem.CIVIL_LAW,
            privacy_regime=DataPrivacyRegime.GDPR,
            currency_code="EUR",
            date_format="%d/%m/%Y",
            number_format="1 234,56",
            legal_citation_format="french"
        )
        
        # Spain (Spanish)
        self._locale_configs["es_ES"] = LocaleConfig(
            language=SupportedLanguage.SPANISH,
            country_code="ES",
            legal_system=LegalSystem.CIVIL_LAW,
            privacy_regime=DataPrivacyRegime.GDPR,
            currency_code="EUR",
            date_format="%d/%m/%Y",
            number_format="1.234,56",
            legal_citation_format="spanish"
        )
        
        # Japan (Japanese)
        self._locale_configs["ja_JP"] = LocaleConfig(
            language=SupportedLanguage.JAPANESE,
            country_code="JP",
            legal_system=LegalSystem.CIVIL_LAW,
            privacy_regime=DataPrivacyRegime.PDPA,
            currency_code="JPY",
            date_format="%Y/%m/%d",
            number_format="1,234",
            legal_citation_format="japanese"
        )
        
        # China (Chinese Simplified)
        self._locale_configs["zh_CN"] = LocaleConfig(
            language=SupportedLanguage.CHINESE,
            country_code="CN",
            legal_system=LegalSystem.CIVIL_LAW,
            privacy_regime=DataPrivacyRegime.PDPA,
            currency_code="CNY",
            date_format="%Y年%m月%d日",
            number_format="1,234.56",
            legal_citation_format="chinese"
        )
    
    def _initialize_legal_terminology(self):
        """Initialize legal terminology database."""
        
        # Contract terminology
        self._legal_terminology["contract"] = LegalTerminologyEntry(
            term_id="contract",
            base_language=SupportedLanguage.ENGLISH,
            legal_system=LegalSystem.COMMON_LAW,
            translations={
                SupportedLanguage.ENGLISH: "contract",
                SupportedLanguage.SPANISH: "contrato",
                SupportedLanguage.FRENCH: "contrat",
                SupportedLanguage.GERMAN: "Vertrag",
                SupportedLanguage.JAPANESE: "契約",
                SupportedLanguage.CHINESE: "合同"
            },
            definition={
                SupportedLanguage.ENGLISH: "A legally binding agreement between parties",
                SupportedLanguage.SPANISH: "Un acuerdo legalmente vinculante entre partes",
                SupportedLanguage.FRENCH: "Un accord juridiquement contraignant entre parties",
                SupportedLanguage.GERMAN: "Eine rechtlich bindende Vereinbarung zwischen Parteien"
            }
        )
        
        # Liability terminology
        self._legal_terminology["liability"] = LegalTerminologyEntry(
            term_id="liability",
            base_language=SupportedLanguage.ENGLISH,
            legal_system=LegalSystem.COMMON_LAW,
            translations={
                SupportedLanguage.ENGLISH: "liability",
                SupportedLanguage.SPANISH: "responsabilidad",
                SupportedLanguage.FRENCH: "responsabilité",
                SupportedLanguage.GERMAN: "Haftung",
                SupportedLanguage.JAPANESE: "責任",
                SupportedLanguage.CHINESE: "责任"
            }
        )
        
        # Privacy terminology
        self._legal_terminology["privacy"] = LegalTerminologyEntry(
            term_id="privacy",
            base_language=SupportedLanguage.ENGLISH,
            legal_system=LegalSystem.MIXED_SYSTEM,
            translations={
                SupportedLanguage.ENGLISH: "privacy",
                SupportedLanguage.SPANISH: "privacidad",
                SupportedLanguage.FRENCH: "confidentialité",
                SupportedLanguage.GERMAN: "Datenschutz",
                SupportedLanguage.JAPANESE: "プライバシー",
                SupportedLanguage.CHINESE: "隐私"
            }
        )
        
        # Compliance terminology
        self._legal_terminology["compliance"] = LegalTerminologyEntry(
            term_id="compliance",
            base_language=SupportedLanguage.ENGLISH,
            legal_system=LegalSystem.MIXED_SYSTEM,
            translations={
                SupportedLanguage.ENGLISH: "compliance",
                SupportedLanguage.SPANISH: "cumplimiento",
                SupportedLanguage.FRENCH: "conformité",
                SupportedLanguage.GERMAN: "Compliance",
                SupportedLanguage.JAPANESE: "コンプライアンス",
                SupportedLanguage.CHINESE: "合规"
            }
        )
    
    def _initialize_translations(self):
        """Initialize UI and system message translations."""
        
        # Error messages
        self._translations["error.document_not_found"] = TranslationEntry(
            key="error.document_not_found",
            translations={
                "en": "Document not found",
                "es": "Documento no encontrado",
                "fr": "Document non trouvé",
                "de": "Dokument nicht gefunden",
                "ja": "ドキュメントが見つかりません",
                "zh": "文档未找到"
            },
            context="error_message"
        )
        
        # Processing messages
        self._translations["processing.analyzing_document"] = TranslationEntry(
            key="processing.analyzing_document",
            translations={
                "en": "Analyzing document...",
                "es": "Analizando documento...",
                "fr": "Analyse du document...",
                "de": "Dokument wird analysiert...",
                "ja": "ドキュメントを分析中...",
                "zh": "正在分析文档..."
            },
            context="processing_status"
        )
        
        # Legal domain messages
        self._translations["legal.contract_analysis"] = TranslationEntry(
            key="legal.contract_analysis",
            translations={
                "en": "Contract Analysis",
                "es": "Análisis de Contrato",
                "fr": "Analyse de Contrat",
                "de": "Vertragsanalyse",
                "ja": "契約分析",
                "zh": "合同分析"
            },
            context="legal_domain",
            legal_domain="contract"
        )
        
        # Privacy compliance messages
        self._translations["privacy.gdpr_compliance"] = TranslationEntry(
            key="privacy.gdpr_compliance",
            translations={
                "en": "GDPR Compliance Required",
                "es": "Cumplimiento GDPR Requerido",
                "fr": "Conformité RGPD Requise",
                "de": "DSGVO-Konformität Erforderlich",
                "ja": "GDPR準拠が必要",
                "zh": "需要GDPR合规"
            },
            context="privacy_notice",
            legal_domain="privacy"
        )
    
    def set_locale(self, locale: str) -> bool:
        """Set current locale for the session."""
        
        if locale in self._locale_configs:
            self.current_locale = locale
            logger.info(f"Locale set to {locale}")
            return True
        else:
            logger.warning(f"Unsupported locale: {locale}")
            return False
    
    def get_current_locale_config(self) -> LocaleConfig:
        """Get configuration for current locale."""
        
        return self._locale_configs.get(self.current_locale, self._locale_configs[self.default_locale])
    
    def translate(self, key: str, locale: Optional[str] = None, **kwargs) -> str:
        """Translate a message key to specified locale."""
        
        target_locale = locale or self.current_locale
        language_code = target_locale.split('_')[0]
        
        if key in self._translations:
            translation_entry = self._translations[key]
            translated_text = translation_entry.translations.get(
                language_code,
                translation_entry.translations.get("en", key)
            )
            
            # Format with provided arguments
            try:
                return translated_text.format(**kwargs)
            except (KeyError, ValueError):
                return translated_text
        
        return key  # Return key if translation not found
    
    def translate_legal_term(self, term_id: str, target_language: SupportedLanguage) -> str:
        """Translate legal terminology to target language."""
        
        if term_id in self._legal_terminology:
            terminology = self._legal_terminology[term_id]
            return terminology.translations.get(target_language, term_id)
        
        return term_id
    
    def detect_language(self, text: str) -> Tuple[SupportedLanguage, float]:
        """Detect language of text with confidence score."""
        
        # Simple language detection based on character patterns
        # In production, use proper language detection libraries
        
        # Character pattern analysis
        latin_chars = len(re.findall(r'[a-zA-Z]', text))
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        japanese_chars = len(re.findall(r'[\u3040-\u309f\u30a0-\u30ff]', text))
        arabic_chars = len(re.findall(r'[\u0600-\u06ff]', text))
        
        total_chars = len(text)
        
        if total_chars == 0:
            return SupportedLanguage.ENGLISH, 0.0
        
        # Calculate confidence scores
        if chinese_chars / total_chars > 0.3:
            return SupportedLanguage.CHINESE, min(0.95, chinese_chars / total_chars)
        elif japanese_chars / total_chars > 0.2:
            return SupportedLanguage.JAPANESE, min(0.95, japanese_chars / total_chars)
        elif latin_chars / total_chars > 0.5:
            # Further analysis for Latin-script languages
            if re.search(r'\b(the|and|of|to|in|for)\b', text.lower()):
                return SupportedLanguage.ENGLISH, 0.8
            elif re.search(r'\b(der|die|das|und|ist|mit)\b', text.lower()):
                return SupportedLanguage.GERMAN, 0.8
            elif re.search(r'\b(le|la|les|de|et|pour)\b', text.lower()):
                return SupportedLanguage.FRENCH, 0.8
            elif re.search(r'\b(el|la|los|las|y|de)\b', text.lower()):
                return SupportedLanguage.SPANISH, 0.8
            else:
                return SupportedLanguage.ENGLISH, 0.6  # Default to English
        
        return SupportedLanguage.ENGLISH, 0.5  # Low confidence default
    
    def format_date(self, date: datetime, locale: Optional[str] = None) -> str:
        """Format date according to locale conventions."""
        
        target_locale = locale or self.current_locale
        locale_config = self._locale_configs.get(target_locale, self._locale_configs[self.default_locale])
        
        try:
            return date.strftime(locale_config.date_format)
        except ValueError:
            return date.strftime("%Y-%m-%d")  # ISO format fallback
    
    def format_currency(self, amount: float, locale: Optional[str] = None) -> str:
        """Format currency according to locale conventions."""
        
        target_locale = locale or self.current_locale
        locale_config = self._locale_configs.get(target_locale, self._locale_configs[self.default_locale])
        
        currency_symbols = {
            "USD": "$",
            "EUR": "€",
            "GBP": "£",
            "JPY": "¥",
            "CNY": "¥"
        }
        
        symbol = currency_symbols.get(locale_config.currency_code, locale_config.currency_code)
        
        # Format number according to locale
        if locale_config.number_format == "1,234.56":
            formatted_amount = f"{amount:,.2f}"
        elif locale_config.number_format == "1.234,56":
            formatted_amount = f"{amount:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
        elif locale_config.number_format == "1 234,56":
            formatted_amount = f"{amount:,.2f}".replace(",", " ").replace(".", ",")
        elif locale_config.number_format == "1,234":
            formatted_amount = f"{amount:,.0f}"
        else:
            formatted_amount = f"{amount:.2f}"
        
        return f"{symbol}{formatted_amount}"
    
    def get_privacy_compliance_requirements(self, locale: Optional[str] = None) -> Dict[str, Any]:
        """Get privacy compliance requirements for locale."""
        
        target_locale = locale or self.current_locale
        locale_config = self._locale_configs.get(target_locale, self._locale_configs[self.default_locale])
        
        compliance_requirements = {
            DataPrivacyRegime.GDPR: {
                "data_retention_limit": "varies by purpose",
                "consent_required": True,
                "right_to_erasure": True,
                "data_portability": True,
                "privacy_by_design": True,
                "dpo_required": True,
                "breach_notification": "72 hours",
                "territorial_scope": "EU residents"
            },
            DataPrivacyRegime.CCPA: {
                "data_retention_limit": "reasonable period",
                "consent_required": False,  # Opt-out model
                "right_to_delete": True,
                "right_to_know": True,
                "non_discrimination": True,
                "sale_disclosure": True,
                "breach_notification": "varies",
                "territorial_scope": "California residents"
            },
            DataPrivacyRegime.PDPA: {
                "data_retention_limit": "necessary period",
                "consent_required": True,
                "right_to_correction": True,
                "data_portability": True,
                "breach_notification": "72 hours",
                "territorial_scope": "varies by country"
            }
        }
        
        return compliance_requirements.get(
            locale_config.privacy_regime,
            compliance_requirements[DataPrivacyRegime.GDPR]  # Default fallback
        )
    
    def validate_cross_border_transfer(self, 
                                      source_locale: str, 
                                      target_locale: str) -> Dict[str, Any]:
        """Validate legal requirements for cross-border data transfer."""
        
        source_config = self._locale_configs.get(source_locale, self._locale_configs[self.default_locale])
        target_config = self._locale_configs.get(target_locale, self._locale_configs[self.default_locale])
        
        # GDPR adequacy decisions and transfer mechanisms
        gdpr_adequate_countries = {
            "AD", "AR", "CA", "FO", "GG", "IL", "IM", "JE", "JP", "NZ", "CH", "UY", "GB"
        }
        
        transfer_validation = {
            "allowed": True,
            "mechanism_required": False,
            "mechanisms": [],
            "additional_requirements": []
        }
        
        # Check GDPR transfers
        if source_config.privacy_regime == DataPrivacyRegime.GDPR:
            if target_config.country_code not in gdpr_adequate_countries:
                transfer_validation["mechanism_required"] = True
                transfer_validation["mechanisms"] = [
                    "Standard Contractual Clauses (SCCs)",
                    "Binding Corporate Rules (BCRs)",
                    "Certification mechanisms",
                    "Codes of conduct"
                ]
        
        # Check CCPA transfers
        if source_config.privacy_regime == DataPrivacyRegime.CCPA:
            transfer_validation["additional_requirements"].append(
                "Consumer right to opt-out of sale of personal information"
            )
        
        return transfer_validation
    
    def get_legal_citation_format(self, locale: Optional[str] = None) -> Dict[str, str]:
        """Get legal citation format standards for locale."""
        
        target_locale = locale or self.current_locale
        locale_config = self._locale_configs.get(target_locale, self._locale_configs[self.default_locale])
        
        citation_formats = {
            "bluebook": {
                "case": "Case Name, Volume Reporter Abbreviation Page (Court Date)",
                "statute": "Title U.S.C. § Section (Date)",
                "example": "Brown v. Board of Education, 347 U.S. 483 (1954)"
            },
            "oscola": {
                "case": "Case Name [Year] Court Reference",
                "statute": "Statute Name Year, section",
                "example": "R v Smith [2019] UKSC 15"
            },
            "german": {
                "case": "Court, Date, Reference",
                "statute": "Gesetz § Paragraph",
                "example": "BGH, 15.03.2019, I ZR 1/18"
            },
            "french": {
                "case": "Court, Date, Reference",
                "statute": "Code article",
                "example": "Cass. civ. 1ère, 15 mars 2019, n° 18-12345"
            },
            "default": {
                "case": "Case Name (Court Year)",
                "statute": "Statute § Section",
                "example": "Case v. Name (Court 2019)"
            }
        }
        
        return citation_formats.get(
            locale_config.legal_citation_format,
            citation_formats["default"]
        )
    
    def get_supported_locales(self) -> List[str]:
        """Get list of all supported locales."""
        
        return list(self._locale_configs.keys())
    
    def get_locale_info(self, locale: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive information about a locale."""
        
        if locale not in self._locale_configs:
            return None
        
        config = self._locale_configs[locale]
        
        return {
            "locale": locale,
            "language": config.language.value,
            "country": config.country_code,
            "legal_system": config.legal_system.value,
            "privacy_regime": config.privacy_regime.value,
            "currency": config.currency_code,
            "date_format": config.date_format,
            "number_format": config.number_format,
            "rtl_support": config.rtl_support,
            "citation_format": config.legal_citation_format
        }
    
    async def localize_document_analysis(self, 
                                       document_text: str, 
                                       target_locale: Optional[str] = None) -> Dict[str, Any]:
        """Perform locale-aware document analysis."""
        
        target_locale = target_locale or self.current_locale
        locale_config = self._locale_configs.get(target_locale, self._locale_configs[self.default_locale])
        
        # Detect document language
        detected_language, confidence = self.detect_language(document_text)
        
        # Extract legal terms and translate them
        legal_terms = []
        for term_id, terminology in self._legal_terminology.items():
            for lang, translation in terminology.translations.items():
                if translation.lower() in document_text.lower():
                    localized_term = self.translate_legal_term(term_id, locale_config.language)
                    legal_terms.append({
                        "original": translation,
                        "localized": localized_term,
                        "term_id": term_id,
                        "legal_system": terminology.legal_system.value
                    })
        
        # Check privacy compliance requirements
        privacy_requirements = self.get_privacy_compliance_requirements(target_locale)
        
        # Generate localized analysis
        analysis_result = {
            "locale": target_locale,
            "detected_language": {
                "language": detected_language.value,
                "confidence": confidence
            },
            "legal_terms_found": legal_terms,
            "privacy_compliance": privacy_requirements,
            "legal_system": locale_config.legal_system.value,
            "citation_format": self.get_legal_citation_format(target_locale),
            "localization_applied": True
        }
        
        return analysis_result


# Global I18n system instance
_global_i18n_system = None


def get_i18n_system() -> GlobalI18nSystem:
    """Get global internationalization system instance."""
    
    global _global_i18n_system
    if _global_i18n_system is None:
        _global_i18n_system = GlobalI18nSystem()
    return _global_i18n_system


# Decorator for locale-aware functions
def locale_aware(default_locale: str = "en_US"):
    """Decorator to make functions locale-aware."""
    
    def decorator(func):
        async def wrapper(*args, locale: Optional[str] = None, **kwargs):
            i18n = get_i18n_system()
            original_locale = i18n.current_locale
            
            try:
                # Set locale for this operation
                target_locale = locale or default_locale
                i18n.set_locale(target_locale)
                
                # Execute function
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                return result
                
            finally:
                # Restore original locale
                i18n.set_locale(original_locale)
        
        return wrapper
    return decorator


# Translation helper function
def _(key: str, **kwargs) -> str:
    """Quick translation function."""
    
    i18n = get_i18n_system()
    return i18n.translate(key, **kwargs)