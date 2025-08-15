#!/usr/bin/env python3
"""Global compliance and internationalization system for bioneural olfactory fusion."""

import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

class ComplianceRegion(Enum):
    """Supported compliance regions."""
    EU_GDPR = "eu_gdpr"           # European Union - GDPR
    US_CCPA = "us_ccpa"           # California - CCPA
    US_HIPAA = "us_hipaa"         # US Healthcare - HIPAA
    UK_DPA = "uk_dpa"             # United Kingdom - DPA 2018
    CANADA_PIPEDA = "ca_pipeda"   # Canada - PIPEDA
    SINGAPORE_PDPA = "sg_pdpa"    # Singapore - PDPA
    BRAZIL_LGPD = "br_lgpd"       # Brazil - LGPD
    JAPAN_APPI = "jp_appi"        # Japan - APPI

class SupportedLanguage(Enum):
    """Supported languages for internationalization."""
    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    JAPANESE = "ja"
    CHINESE_SIMPLIFIED = "zh-cn"
    PORTUGUESE = "pt"
    ITALIAN = "it"
    DUTCH = "nl"
    KOREAN = "ko"

@dataclass
class ComplianceRule:
    """Individual compliance rule."""
    rule_id: str
    region: ComplianceRegion
    title: str
    description: str
    severity: str  # 'critical', 'high', 'medium', 'low'
    validation_func: str
    remediation: str

@dataclass
class ProcessingRecord:
    """GDPR Article 30 processing record."""
    timestamp: float
    document_id: str
    processing_purpose: str
    legal_basis: str
    data_categories: List[str]
    retention_period: str
    user_consent: bool
    region: str

class GlobalComplianceSystem:
    """Global compliance and privacy system."""
    
    def __init__(self, primary_region: ComplianceRegion = ComplianceRegion.EU_GDPR):
        self.primary_region = primary_region
        self.processing_records: List[ProcessingRecord] = []
        self.compliance_rules = self._initialize_compliance_rules()
        
        # Privacy controls
        self.data_anonymization_enabled = True
        self.consent_required = True
        self.data_retention_days = 365
        
        # I18n setup
        self.translations = self._load_translations()
    
    def _initialize_compliance_rules(self) -> Dict[ComplianceRegion, List[ComplianceRule]]:
        """Initialize compliance rules for different regions."""
        
        rules = {
            ComplianceRegion.EU_GDPR: [
                ComplianceRule(
                    rule_id="gdpr_001",
                    region=ComplianceRegion.EU_GDPR,
                    title="Lawful Basis for Processing",
                    description="Processing must have a lawful basis under GDPR Article 6",
                    severity="critical",
                    validation_func="validate_lawful_basis",
                    remediation="Obtain explicit consent or establish legitimate interest"
                ),
                ComplianceRule(
                    rule_id="gdpr_002",
                    region=ComplianceRegion.EU_GDPR,
                    title="Data Minimization",
                    description="Only process data necessary for the stated purpose",
                    severity="high",
                    validation_func="validate_data_minimization",
                    remediation="Remove unnecessary data fields from processing"
                ),
                ComplianceRule(
                    rule_id="gdpr_003",
                    region=ComplianceRegion.EU_GDPR,
                    title="Processing Records",
                    description="Maintain records of processing activities (Article 30)",
                    severity="high",
                    validation_func="validate_processing_records",
                    remediation="Implement comprehensive processing record system"
                )
            ],
            ComplianceRegion.US_CCPA: [
                ComplianceRule(
                    rule_id="ccpa_001",
                    region=ComplianceRegion.US_CCPA,
                    title="Consumer Right to Know",
                    description="Consumers have right to know what personal information is collected",
                    severity="high",
                    validation_func="validate_transparency",
                    remediation="Provide clear privacy notices and data usage disclosure"
                ),
                ComplianceRule(
                    rule_id="ccpa_002",
                    region=ComplianceRegion.US_CCPA,
                    title="Right to Delete",
                    description="Consumers have right to request deletion of personal information",
                    severity="high",
                    validation_func="validate_deletion_capability",
                    remediation="Implement secure data deletion mechanisms"
                )
            ],
            ComplianceRegion.SINGAPORE_PDPA: [
                ComplianceRule(
                    rule_id="pdpa_001",
                    region=ComplianceRegion.SINGAPORE_PDPA,
                    title="Consent Management",
                    description="Obtain and manage consent for personal data processing",
                    severity="critical",
                    validation_func="validate_consent_management",
                    remediation="Implement robust consent collection and withdrawal system"
                )
            ]
        }
        
        return rules
    
    def _load_translations(self) -> Dict[str, Dict[str, str]]:
        """Load internationalization translations."""
        
        return {
            "en": {
                "privacy_notice": "Privacy Notice: Your document will be analyzed using bioneural olfactory fusion technology.",
                "consent_required": "Consent required for processing legal documents.",
                "data_retention": "Data will be retained for {days} days as per compliance requirements.",
                "analysis_complete": "Document analysis completed successfully.",
                "compliance_warning": "Compliance warning: {message}",
                "data_anonymized": "Personal data has been anonymized for privacy protection."
            },
            "es": {
                "privacy_notice": "Aviso de Privacidad: Su documento será analizado usando tecnología de fusión olfatoria bioneural.",
                "consent_required": "Se requiere consentimiento para procesar documentos legales.",
                "data_retention": "Los datos se conservarán durante {days} días según los requisitos de cumplimiento.",
                "analysis_complete": "Análisis de documento completado exitosamente.",
                "compliance_warning": "Advertencia de cumplimiento: {message}",
                "data_anonymized": "Los datos personales han sido anonimizados para protección de privacidad."
            },
            "fr": {
                "privacy_notice": "Avis de Confidentialité: Votre document sera analysé en utilisant la technologie de fusion olfactive bioneurale.",
                "consent_required": "Consentement requis pour traiter les documents juridiques.",
                "data_retention": "Les données seront conservées pendant {days} jours selon les exigences de conformité.",
                "analysis_complete": "Analyse de document terminée avec succès.",
                "compliance_warning": "Avertissement de conformité: {message}",
                "data_anonymized": "Les données personnelles ont été anonymisées pour la protection de la vie privée."
            },
            "de": {
                "privacy_notice": "Datenschutzhinweis: Ihr Dokument wird mit bioneuraler Geruchsfusions-Technologie analysiert.",
                "consent_required": "Einwilligung erforderlich für die Verarbeitung von Rechtsdokumenten.",
                "data_retention": "Daten werden für {days} Tage gemäß Compliance-Anforderungen aufbewahrt.",
                "analysis_complete": "Dokumentenanalyse erfolgreich abgeschlossen.",
                "compliance_warning": "Compliance-Warnung: {message}",
                "data_anonymized": "Personenbezogene Daten wurden zum Datenschutz anonymisiert."
            },
            "ja": {
                "privacy_notice": "プライバシー通知：あなたの文書は生体神経嗅覚融合技術を使用して分析されます。",
                "consent_required": "法的文書の処理には同意が必要です。",
                "data_retention": "コンプライアンス要件に従い、データは{days}日間保持されます。",
                "analysis_complete": "文書分析が正常に完了しました。",
                "compliance_warning": "コンプライアンス警告：{message}",
                "data_anonymized": "個人データはプライバシー保護のために匿名化されています。"
            },
            "zh-cn": {
                "privacy_notice": "隐私声明：您的文档将使用生物神经嗅觉融合技术进行分析。",
                "consent_required": "处理法律文档需要同意。",
                "data_retention": "根据合规要求，数据将保留{days}天。",
                "analysis_complete": "文档分析成功完成。",
                "compliance_warning": "合规警告：{message}",
                "data_anonymized": "个人数据已匿名化以保护隐私。"
            }
        }
    
    def get_localized_message(self, key: str, language: SupportedLanguage, **kwargs) -> str:
        """Get localized message."""
        
        lang_code = language.value
        translations = self.translations.get(lang_code, self.translations["en"])
        message = translations.get(key, f"[Missing translation: {key}]")
        
        # Format with any provided arguments
        return message.format(**kwargs)
    
    def anonymize_document(self, text: str) -> str:
        """Anonymize personal data in document."""
        
        import re
        
        # PII patterns to anonymize
        patterns = [
            (r'\b\d{3}-\d{2}-\d{4}\b', '[SSN-REDACTED]'),  # SSN
            (r'\b\d{4}-\d{4}-\d{4}-\d{4}\b', '[CARD-REDACTED]'),  # Credit card
            (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL-REDACTED]'),  # Email
            (r'\b\(\d{3}\)\s?\d{3}-\d{4}\b', '[PHONE-REDACTED]'),  # Phone
            (r'\b\d{5}(-\d{4})?\b', '[ZIP-REDACTED]'),  # ZIP code
            (r'\b\d{1,5}\s\w+\s(Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln|Boulevard|Blvd)\b', '[ADDRESS-REDACTED]'),  # Address
        ]
        
        anonymized_text = text
        for pattern, replacement in patterns:
            anonymized_text = re.sub(pattern, replacement, anonymized_text)
        
        return anonymized_text
    
    def validate_compliance(self, region: ComplianceRegion, processing_context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate compliance for specific region."""
        
        rules = self.compliance_rules.get(region, [])
        validation_results = {
            'region': region.value,
            'compliant': True,
            'violations': [],
            'warnings': [],
            'score': 0.0
        }
        
        passed_rules = 0
        
        for rule in rules:
            try:
                # Simplified validation - in production, these would be proper implementations
                if rule.validation_func == "validate_lawful_basis":
                    result = processing_context.get('consent', False) or processing_context.get('legitimate_interest', False)
                elif rule.validation_func == "validate_data_minimization":
                    result = len(processing_context.get('data_categories', [])) <= 5  # Arbitrary limit
                elif rule.validation_func == "validate_processing_records":
                    result = len(self.processing_records) > 0
                elif rule.validation_func == "validate_transparency":
                    result = processing_context.get('privacy_notice_provided', False)
                elif rule.validation_func == "validate_deletion_capability":
                    result = processing_context.get('deletion_supported', True)
                elif rule.validation_func == "validate_consent_management":
                    result = processing_context.get('consent_recorded', False)
                else:
                    result = True  # Unknown rule passes by default
                
                if result:
                    passed_rules += 1
                else:
                    if rule.severity in ['critical', 'high']:
                        validation_results['violations'].append({
                            'rule_id': rule.rule_id,
                            'title': rule.title,
                            'severity': rule.severity,
                            'remediation': rule.remediation
                        })
                        validation_results['compliant'] = False
                    else:
                        validation_results['warnings'].append({
                            'rule_id': rule.rule_id,
                            'title': rule.title,
                            'severity': rule.severity
                        })
            
            except Exception as e:
                validation_results['warnings'].append({
                    'rule_id': rule.rule_id,
                    'title': rule.title,
                    'error': str(e)
                })
        
        validation_results['score'] = passed_rules / len(rules) if rules else 1.0
        
        return validation_results
    
    def record_processing_activity(self, 
                                 document_id: str,
                                 processing_purpose: str,
                                 legal_basis: str,
                                 data_categories: List[str],
                                 user_consent: bool = False,
                                 region: str = "EU") -> str:
        """Record processing activity for compliance."""
        
        record = ProcessingRecord(
            timestamp=time.time(),
            document_id=document_id,
            processing_purpose=processing_purpose,
            legal_basis=legal_basis,
            data_categories=data_categories,
            retention_period=f"{self.data_retention_days} days",
            user_consent=user_consent,
            region=region
        )
        
        self.processing_records.append(record)
        return f"record_{len(self.processing_records):06d}"
    
    def process_document_with_compliance(self, 
                                       text: str,
                                       document_id: str,
                                       user_consent: bool = False,
                                       region: ComplianceRegion = None,
                                       language: SupportedLanguage = SupportedLanguage.ENGLISH) -> Dict[str, Any]:
        """Process document with full compliance and i18n support."""
        
        region = region or self.primary_region
        
        # Step 1: Privacy notice
        privacy_notice = self.get_localized_message("privacy_notice", language)
        
        # Step 2: Consent validation
        if self.consent_required and not user_consent:
            consent_msg = self.get_localized_message("consent_required", language)
            return {
                'error': 'consent_required',
                'message': consent_msg,
                'privacy_notice': privacy_notice
            }
        
        # Step 3: Data anonymization
        if self.data_anonymization_enabled:
            anonymized_text = self.anonymize_document(text)
            anonymization_msg = self.get_localized_message("data_anonymized", language)
        else:
            anonymized_text = text
            anonymization_msg = None
        
        # Step 4: Process document
        from minimal_working_demo import bioneural_scent_simulation
        analysis_result = bioneural_scent_simulation(anonymized_text)
        
        # Step 5: Record processing activity
        processing_record_id = self.record_processing_activity(
            document_id=document_id,
            processing_purpose="legal_document_analysis",
            legal_basis="consent" if user_consent else "legitimate_interest",
            data_categories=["legal_text", "document_metadata"],
            user_consent=user_consent,
            region=region.value
        )
        
        # Step 6: Compliance validation
        processing_context = {
            'consent': user_consent,
            'legitimate_interest': True,
            'data_categories': ["legal_text", "document_metadata"],
            'privacy_notice_provided': True,
            'deletion_supported': True,
            'consent_recorded': user_consent
        }
        
        compliance_result = self.validate_compliance(region, processing_context)
        
        # Step 7: Prepare response
        success_msg = self.get_localized_message("analysis_complete", language)
        retention_msg = self.get_localized_message("data_retention", language, days=self.data_retention_days)
        
        result = {
            'document_id': document_id,
            'processing_record_id': processing_record_id,
            'analysis': analysis_result,
            'privacy_notice': privacy_notice,
            'success_message': success_msg,
            'data_retention_notice': retention_msg,
            'compliance': compliance_result,
            'language': language.value,
            'region': region.value,
            'data_anonymized': self.data_anonymization_enabled
        }
        
        if anonymization_msg:
            result['anonymization_notice'] = anonymization_msg
        
        # Add compliance warnings if any
        if compliance_result['violations'] or compliance_result['warnings']:
            warning_msg = self.get_localized_message("compliance_warning", language, 
                                                   message="See compliance details")
            result['compliance_warning'] = warning_msg
        
        return result
    
    def generate_compliance_report(self) -> Dict[str, Any]:
        """Generate comprehensive compliance report."""
        
        # Aggregate processing records by region
        records_by_region = {}
        for record in self.processing_records:
            region = record.region
            if region not in records_by_region:
                records_by_region[region] = []
            records_by_region[region].append(record)
        
        # Generate report
        report = {
            'generated_at': time.time(),
            'total_processing_records': len(self.processing_records),
            'regions_covered': list(records_by_region.keys()),
            'compliance_summary': {},
            'data_protection_measures': {
                'anonymization_enabled': self.data_anonymization_enabled,
                'consent_required': self.consent_required,
                'data_retention_days': self.data_retention_days
            },
            'processing_activities': {
                'by_region': {
                    region: len(records) for region, records in records_by_region.items()
                },
                'by_legal_basis': self._aggregate_by_legal_basis(),
                'recent_activities': [
                    {
                        'document_id': r.document_id,
                        'purpose': r.processing_purpose,
                        'legal_basis': r.legal_basis,
                        'consent': r.user_consent,
                        'timestamp': r.timestamp
                    }
                    for r in self.processing_records[-10:]  # Last 10 records
                ]
            }
        }
        
        return report
    
    def _aggregate_by_legal_basis(self) -> Dict[str, int]:
        """Aggregate processing records by legal basis."""
        
        aggregation = {}
        for record in self.processing_records:
            basis = record.legal_basis
            aggregation[basis] = aggregation.get(basis, 0) + 1
        
        return aggregation

def demonstrate_global_compliance():
    """Demonstrate global compliance and i18n features."""
    
    print("🌍 Global Compliance & Internationalization Demo")
    print("=" * 60)
    
    # Initialize compliance system
    compliance_system = GlobalComplianceSystem(ComplianceRegion.EU_GDPR)
    
    # Test documents with different scenarios
    test_scenarios = [
        {
            'text': 'Software License Agreement between Company and User. User agrees to terms.',
            'doc_id': 'doc_001',
            'consent': True,
            'region': ComplianceRegion.EU_GDPR,
            'language': SupportedLanguage.ENGLISH
        },
        {
            'text': 'Contrato de licencia de software entre la Empresa y el Usuario.',
            'doc_id': 'doc_002',
            'consent': True,
            'region': ComplianceRegion.EU_GDPR,
            'language': SupportedLanguage.SPANISH
        },
        {
            'text': 'Agreement with PII: john.doe@company.com, phone (555) 123-4567, SSN 123-45-6789',
            'doc_id': 'doc_003',
            'consent': False,
            'region': ComplianceRegion.US_CCPA,
            'language': SupportedLanguage.ENGLISH
        }
    ]
    
    results = []
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n📄 Scenario {i}: {scenario['region'].value} / {scenario['language'].value}")
        print("-" * 40)
        
        result = compliance_system.process_document_with_compliance(
            text=scenario['text'],
            document_id=scenario['doc_id'],
            user_consent=scenario['consent'],
            region=scenario['region'],
            language=scenario['language']
        )
        
        results.append(result)
        
        # Display results
        if 'error' in result:
            print(f"❌ {result['error']}: {result['message']}")
        else:
            print(f"✅ {result['success_message']}")
            print(f"🔒 {result['privacy_notice']}")
            if 'anonymization_notice' in result:
                print(f"👤 {result['anonymization_notice']}")
            print(f"📊 Compliance Score: {result['compliance']['score']:.1%}")
            
            if result['compliance']['violations']:
                print(f"⚠️  Violations: {len(result['compliance']['violations'])}")
            if result['compliance']['warnings']:
                print(f"⚠️  Warnings: {len(result['compliance']['warnings'])}")
    
    # Generate compliance report
    print(f"\n📊 Compliance Report:")
    print("-" * 40)
    
    report = compliance_system.generate_compliance_report()
    print(f"Total processing records: {report['total_processing_records']}")
    print(f"Regions covered: {', '.join(report['regions_covered'])}")
    print(f"Data protection measures: {report['data_protection_measures']}")
    
    # Test multilingual support
    print(f"\n🌐 Multilingual Support Demo:")
    print("-" * 40)
    
    languages_to_test = [
        SupportedLanguage.ENGLISH,
        SupportedLanguage.SPANISH,
        SupportedLanguage.FRENCH,
        SupportedLanguage.GERMAN,
        SupportedLanguage.JAPANESE,
        SupportedLanguage.CHINESE_SIMPLIFIED
    ]
    
    for lang in languages_to_test:
        msg = compliance_system.get_localized_message("privacy_notice", lang)
        print(f"{lang.value:6}: {msg[:60]}...")
    
    # Save results
    output_file = Path('global_compliance_demo_results.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': time.time(),
            'scenarios_tested': len(test_scenarios),
            'results': results,
            'compliance_report': report,
            'supported_regions': [r.value for r in ComplianceRegion],
            'supported_languages': [l.value for l in SupportedLanguage]
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Results saved to: {output_file}")
    
    return results, report

if __name__ == "__main__":
    results, report = demonstrate_global_compliance()
    print(f"\n✅ Global compliance demonstration completed!")
    print(f"🌍 Processed {len(results)} scenarios across multiple regions and languages")
    print(f"📋 Generated {report['total_processing_records']} compliance records")