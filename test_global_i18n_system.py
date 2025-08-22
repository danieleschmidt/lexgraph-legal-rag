"""
Test Global I18n System
========================

Test comprehensive internationalization and global-first features.
"""

import asyncio
import pytest
from datetime import datetime

from src.lexgraph_legal_rag.global_i18n_system import (
    GlobalI18nSystem,
    SupportedLanguage,
    LegalSystem,
    DataPrivacyRegime,
    get_i18n_system,
    locale_aware,
    _
)


class TestGlobalI18nSystem:
    """Test global internationalization system."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.i18n = GlobalI18nSystem()
    
    def test_locale_setting(self):
        """Test locale setting functionality."""
        
        # Test valid locale
        result = self.i18n.set_locale("de_DE")
        assert result is True
        assert self.i18n.current_locale == "de_DE"
        
        # Test invalid locale
        result = self.i18n.set_locale("invalid_LOCALE")
        assert result is False
        assert self.i18n.current_locale == "de_DE"  # Should remain unchanged
    
    def test_translation_system(self):
        """Test translation functionality."""
        
        # Test English translation
        self.i18n.set_locale("en_US")
        translated = self.i18n.translate("error.document_not_found")
        assert translated == "Document not found"
        
        # Test German translation
        self.i18n.set_locale("de_DE")
        translated = self.i18n.translate("error.document_not_found")
        assert translated == "Dokument nicht gefunden"
        
        # Test Spanish translation
        self.i18n.set_locale("es_ES")
        translated = self.i18n.translate("error.document_not_found")
        assert translated == "Documento no encontrado"
        
        # Test missing translation (should return key)
        translated = self.i18n.translate("missing.translation.key")
        assert translated == "missing.translation.key"
    
    def test_legal_terminology_translation(self):
        """Test legal terminology translation."""
        
        # Test contract terminology
        english = self.i18n.translate_legal_term("contract", SupportedLanguage.ENGLISH)
        assert english == "contract"
        
        spanish = self.i18n.translate_legal_term("contract", SupportedLanguage.SPANISH)
        assert spanish == "contrato"
        
        german = self.i18n.translate_legal_term("contract", SupportedLanguage.GERMAN)
        assert german == "Vertrag"
        
        chinese = self.i18n.translate_legal_term("contract", SupportedLanguage.CHINESE)
        assert chinese == "合同"
        
        # Test liability terminology
        liability_fr = self.i18n.translate_legal_term("liability", SupportedLanguage.FRENCH)
        assert liability_fr == "responsabilité"
        
        liability_ja = self.i18n.translate_legal_term("liability", SupportedLanguage.JAPANESE)
        assert liability_ja == "責任"
    
    def test_language_detection(self):
        """Test language detection functionality."""
        
        # Test English text
        english_text = "This is a contract between the parties for the provision of legal services."
        lang, confidence = self.i18n.detect_language(english_text)
        assert lang == SupportedLanguage.ENGLISH
        assert confidence > 0.7
        
        # Test German text
        german_text = "Dies ist ein Vertrag zwischen den Parteien für die Bereitstellung von Rechtsdienstleistungen."
        lang, confidence = self.i18n.detect_language(german_text)
        assert lang == SupportedLanguage.GERMAN
        assert confidence > 0.7
        
        # Test French text
        french_text = "Ceci est un contrat entre les parties pour la fourniture de services juridiques."
        lang, confidence = self.i18n.detect_language(french_text)
        assert lang == SupportedLanguage.FRENCH
        assert confidence > 0.7
        
        # Test Chinese text
        chinese_text = "这是当事人之间提供法律服务的合同。"
        lang, confidence = self.i18n.detect_language(chinese_text)
        assert lang == SupportedLanguage.CHINESE
        assert confidence > 0.8
        
        # Test Japanese text
        japanese_text = "これは当事者間の法的サービス提供契約です。"
        lang, confidence = self.i18n.detect_language(japanese_text)
        assert lang == SupportedLanguage.JAPANESE
        assert confidence > 0.8
    
    def test_date_formatting(self):
        """Test locale-aware date formatting."""
        
        test_date = datetime(2024, 3, 15)
        
        # US format (MM/dd/yyyy)
        self.i18n.set_locale("en_US")
        us_date = self.i18n.format_date(test_date)
        assert us_date == "03/15/2024"
        
        # UK format (dd/MM/yyyy)
        self.i18n.set_locale("en_GB")
        uk_date = self.i18n.format_date(test_date)
        assert uk_date == "15/03/2024"
        
        # German format (dd.MM.yyyy)
        self.i18n.set_locale("de_DE")
        german_date = self.i18n.format_date(test_date)
        assert german_date == "15.03.2024"
        
        # Japanese format (yyyy/MM/dd)
        self.i18n.set_locale("ja_JP")
        japanese_date = self.i18n.format_date(test_date)
        assert japanese_date == "2024/03/15"
    
    def test_currency_formatting(self):
        """Test locale-aware currency formatting."""
        
        amount = 1234.56
        
        # US Dollar
        self.i18n.set_locale("en_US")
        usd = self.i18n.format_currency(amount)
        assert usd == "$1,234.56"
        
        # British Pound
        self.i18n.set_locale("en_GB")
        gbp = self.i18n.format_currency(amount)
        assert gbp == "£1,234.56"
        
        # Euro (German format)
        self.i18n.set_locale("de_DE")
        eur_de = self.i18n.format_currency(amount)
        assert eur_de == "€1.234,56"
        
        # Euro (French format)
        self.i18n.set_locale("fr_FR")
        eur_fr = self.i18n.format_currency(amount)
        assert eur_fr == "€1 234,56"
        
        # Japanese Yen (no decimals)
        self.i18n.set_locale("ja_JP")
        jpy = self.i18n.format_currency(amount)
        assert jpy == "¥1,235"  # Rounded
    
    def test_privacy_compliance(self):
        """Test privacy compliance requirements."""
        
        # Test GDPR (Germany)
        self.i18n.set_locale("de_DE")
        gdpr_req = self.i18n.get_privacy_compliance_requirements()
        assert gdpr_req["consent_required"] is True
        assert gdpr_req["right_to_erasure"] is True
        assert gdpr_req["breach_notification"] == "72 hours"
        
        # Test CCPA (US California)
        self.i18n.set_locale("en_US")
        ccpa_req = self.i18n.get_privacy_compliance_requirements()
        assert ccpa_req["consent_required"] is False  # Opt-out model
        assert ccpa_req["right_to_delete"] is True
        assert ccpa_req["non_discrimination"] is True
    
    def test_cross_border_transfer_validation(self):
        """Test cross-border data transfer validation."""
        
        # Test GDPR to US transfer (requires mechanism)
        transfer_validation = self.i18n.validate_cross_border_transfer("de_DE", "en_US")
        assert transfer_validation["mechanism_required"] is True
        assert "Standard Contractual Clauses (SCCs)" in transfer_validation["mechanisms"]
        
        # Test GDPR to UK transfer (adequate country)
        transfer_validation = self.i18n.validate_cross_border_transfer("de_DE", "en_GB")
        assert transfer_validation["mechanism_required"] is False
        
        # Test GDPR to Japan transfer (adequate country)
        transfer_validation = self.i18n.validate_cross_border_transfer("de_DE", "ja_JP")
        assert transfer_validation["mechanism_required"] is False
    
    def test_legal_citation_formats(self):
        """Test legal citation format standards."""
        
        # Test Bluebook (US)
        self.i18n.set_locale("en_US")
        us_citation = self.i18n.get_legal_citation_format()
        assert "Reporter Abbreviation" in us_citation["case"]
        assert us_citation["example"] == "Brown v. Board of Education, 347 U.S. 483 (1954)"
        
        # Test OSCOLA (UK)
        self.i18n.set_locale("en_GB")
        uk_citation = self.i18n.get_legal_citation_format()
        assert "[Year]" in uk_citation["case"]
        assert uk_citation["example"] == "R v Smith [2019] UKSC 15"
        
        # Test German citation
        self.i18n.set_locale("de_DE")
        german_citation = self.i18n.get_legal_citation_format()
        assert "Court, Date, Reference" in german_citation["case"]
        assert "BGH" in german_citation["example"]
    
    def test_supported_locales(self):
        """Test supported locales functionality."""
        
        locales = self.i18n.get_supported_locales()
        
        assert "en_US" in locales
        assert "en_GB" in locales
        assert "de_DE" in locales
        assert "fr_FR" in locales
        assert "es_ES" in locales
        assert "ja_JP" in locales
        assert "zh_CN" in locales
        
        assert len(locales) >= 7
    
    def test_locale_info(self):
        """Test locale information retrieval."""
        
        # Test German locale info
        de_info = self.i18n.get_locale_info("de_DE")
        assert de_info is not None
        assert de_info["language"] == "de"
        assert de_info["country"] == "DE"
        assert de_info["legal_system"] == "civil_law"
        assert de_info["privacy_regime"] == "gdpr"
        assert de_info["currency"] == "EUR"
        
        # Test Japanese locale info
        ja_info = self.i18n.get_locale_info("ja_JP")
        assert ja_info is not None
        assert ja_info["language"] == "ja"
        assert ja_info["legal_system"] == "civil_law"
        assert ja_info["currency"] == "JPY"
        
        # Test invalid locale
        invalid_info = self.i18n.get_locale_info("invalid_LOCALE")
        assert invalid_info is None
    
    @pytest.mark.asyncio
    async def test_localized_document_analysis(self):
        """Test locale-aware document analysis."""
        
        document_text = """
        This contract establishes liability and privacy obligations 
        between the contracting parties. The agreement must ensure 
        compliance with applicable data protection regulations.
        """
        
        # Test German localization
        self.i18n.set_locale("de_DE")
        analysis = await self.i18n.localize_document_analysis(document_text)
        
        assert analysis["locale"] == "de_DE"
        assert analysis["detected_language"]["language"] == "en"
        assert analysis["legal_system"] == "civil_law"
        assert analysis["privacy_compliance"]["consent_required"] is True
        assert len(analysis["legal_terms_found"]) > 0
        
        # Check for translated legal terms
        found_terms = {term["term_id"] for term in analysis["legal_terms_found"]}
        assert "contract" in found_terms or "liability" in found_terms or "privacy" in found_terms
        
        # Test localized terms
        for term in analysis["legal_terms_found"]:
            if term["term_id"] == "contract":
                assert term["localized"] == "Vertrag"
            elif term["term_id"] == "liability":
                assert term["localized"] == "Haftung"
            elif term["term_id"] == "privacy":
                assert term["localized"] == "Datenschutz"
    
    def test_locale_aware_decorator(self):
        """Test locale-aware decorator functionality."""
        
        @locale_aware("de_DE")
        def get_current_locale_language():
            i18n = get_i18n_system()
            config = i18n.get_current_locale_config()
            return config.language.value
        
        # Test with default locale
        result = get_current_locale_language()
        assert result == "de"
        
        # Test with explicit locale
        result = get_current_locale_language(locale="fr_FR")
        assert result == "fr"
    
    def test_translation_helper_function(self):
        """Test quick translation helper function."""
        
        # Set locale and test translation
        i18n = get_i18n_system()
        i18n.set_locale("es_ES")
        
        translated = _("error.document_not_found")
        assert translated == "Documento no encontrado"
        
        # Test with parameters (if any translations supported them)
        processing_msg = _("processing.analyzing_document")
        assert "Analizando" in processing_msg


class TestI18nIntegration:
    """Test I18n system integration with other components."""
    
    @pytest.mark.asyncio
    async def test_multilingual_legal_processing(self):
        """Test multilingual legal document processing."""
        
        i18n = GlobalI18nSystem()
        
        # Test documents in different languages
        documents = {
            "english": "This contract governs the liability and privacy obligations of both parties.",
            "german": "Dieser Vertrag regelt die Haftung und Datenschutzverpflichtungen beider Parteien.",
            "french": "Ce contrat régit les obligations de responsabilité et de confidentialité des deux parties.",
            "spanish": "Este contrato rige las obligaciones de responsabilidad y privacidad de ambas partes."
        }
        
        for lang, document in documents.items():
            # Detect language
            detected_lang, confidence = i18n.detect_language(document)
            assert confidence > 0.5
            
            # Analyze with appropriate locale
            if detected_lang == SupportedLanguage.GERMAN:
                analysis = await i18n.localize_document_analysis(document, "de_DE")
                assert analysis["legal_system"] == "civil_law"
                assert analysis["privacy_compliance"]["consent_required"] is True
            elif detected_lang == SupportedLanguage.FRENCH:
                analysis = await i18n.localize_document_analysis(document, "fr_FR")
                assert analysis["legal_system"] == "civil_law"
                assert analysis["privacy_compliance"]["consent_required"] is True
            elif detected_lang == SupportedLanguage.SPANISH:
                analysis = await i18n.localize_document_analysis(document, "es_ES")
                assert analysis["legal_system"] == "civil_law"
    
    def test_global_compliance_matrix(self):
        """Test comprehensive global compliance requirements."""
        
        i18n = GlobalI18nSystem()
        
        # Test compliance across different jurisdictions
        jurisdictions = ["en_US", "en_GB", "de_DE", "fr_FR", "ja_JP"]
        
        compliance_matrix = {}
        for jurisdiction in jurisdictions:
            requirements = i18n.get_privacy_compliance_requirements(jurisdiction)
            compliance_matrix[jurisdiction] = requirements
        
        # Verify GDPR jurisdictions have consistent requirements
        gdpr_jurisdictions = ["en_GB", "de_DE", "fr_FR"]
        for jurisdiction in gdpr_jurisdictions:
            assert compliance_matrix[jurisdiction]["consent_required"] is True
            assert compliance_matrix[jurisdiction]["right_to_erasure"] is True
        
        # Verify US has different requirements
        assert compliance_matrix["en_US"]["consent_required"] is False  # CCPA opt-out model
        assert compliance_matrix["en_US"]["right_to_delete"] is True


# Performance and scale tests
@pytest.mark.asyncio
async def test_i18n_performance():
    """Test I18n system performance."""
    
    i18n = GlobalI18nSystem()
    
    # Test translation performance
    import time
    
    start_time = time.time()
    
    # Translate many messages
    for _ in range(1000):
        i18n.translate("error.document_not_found")
        i18n.translate("processing.analyzing_document")
        i18n.translate("legal.contract_analysis")
    
    translation_time = time.time() - start_time
    
    # Test language detection performance
    start_time = time.time()
    
    test_texts = [
        "This is an English contract document.",
        "Dies ist ein deutsches Vertragsdokument.",
        "Ceci est un document de contrat français.",
        "这是一个中文合同文件。",
        "これは日本語の契約書です。"
    ]
    
    for _ in range(200):
        for text in test_texts:
            i18n.detect_language(text)
    
    detection_time = time.time() - start_time
    
    # Performance assertions
    assert translation_time < 1.0  # 1000 translations in under 1 second
    assert detection_time < 2.0   # 1000 language detections in under 2 seconds
    
    print(f"✅ Translation performance: {1000/translation_time:.0f} translations/sec")
    print(f"✅ Language detection performance: {1000/detection_time:.0f} detections/sec")


if __name__ == "__main__":
    # Run basic tests
    print("🌍 Testing Global I18n System")
    print("=" * 50)
    
    async def run_tests():
        # Test basic functionality
        i18n = GlobalI18nSystem()
        
        # Test locale setting
        i18n.set_locale("de_DE")
        print(f"✅ Current locale: {i18n.current_locale}")
        
        # Test translation
        translated = i18n.translate("error.document_not_found")
        print(f"✅ German translation: {translated}")
        
        # Test legal terminology
        contract_de = i18n.translate_legal_term("contract", SupportedLanguage.GERMAN)
        print(f"✅ Contract in German: {contract_de}")
        
        # Test language detection
        german_text = "Dies ist ein Vertrag zwischen den Parteien."
        lang, confidence = i18n.detect_language(german_text)
        print(f"✅ Language detection: {lang.value} ({confidence:.2f} confidence)")
        
        # Test date formatting
        from datetime import datetime
        test_date = datetime(2024, 3, 15)
        formatted_date = i18n.format_date(test_date)
        print(f"✅ German date format: {formatted_date}")
        
        # Test currency formatting
        formatted_currency = i18n.format_currency(1234.56)
        print(f"✅ Euro currency format: {formatted_currency}")
        
        # Test document analysis
        test_doc = "This contract establishes liability and privacy obligations."
        analysis = await i18n.localize_document_analysis(test_doc)
        print(f"✅ Document analysis: Found {len(analysis['legal_terms_found'])} legal terms")
        
        print("\n🎉 Global I18n System Tests Completed Successfully!")
    
    asyncio.run(run_tests())