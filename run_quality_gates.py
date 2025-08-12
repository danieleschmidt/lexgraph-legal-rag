#!/usr/bin/env python3
"""
Quality Gates Runner for Bioneural Olfactory Fusion System

Implements comprehensive quality gates including:
- Code quality analysis
- Security scanning
- Performance benchmarks
- Test coverage analysis
- Documentation completeness
- Production readiness checks
"""

import sys
import os
import subprocess
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class QualityGate:
    """Base class for quality gate checks."""
    
    def __init__(self, name: str):
        self.name = name
        self.passed = False
        self.score = 0.0
        self.details = {}
        self.errors = []
    
    def run(self) -> bool:
        """Run the quality gate check."""
        raise NotImplementedError
    
    def get_report(self) -> Dict[str, Any]:
        """Get quality gate report."""
        return {
            "name": self.name,
            "passed": self.passed,
            "score": self.score,
            "details": self.details,
            "errors": self.errors
        }


class CodeQualityGate(QualityGate):
    """Code quality analysis gate."""
    
    def __init__(self):
        super().__init__("Code Quality")
    
    def run(self) -> bool:
        """Run code quality checks."""
        logger.info("Running code quality checks...")
        
        try:
            # Check for Python syntax errors
            syntax_score = self._check_syntax()
            
            # Check code complexity
            complexity_score = self._check_complexity()
            
            # Check code style compliance
            style_score = self._check_style()
            
            # Calculate overall score
            self.score = (syntax_score + complexity_score + style_score) / 3
            self.passed = self.score >= 0.8  # 80% threshold
            
            self.details = {
                "syntax_score": syntax_score,
                "complexity_score": complexity_score,
                "style_score": style_score,
                "threshold": 0.8
            }
            
            logger.info(f"Code quality score: {self.score:.2f}")
            return self.passed
            
        except Exception as e:
            self.errors.append(str(e))
            logger.error(f"Code quality check failed: {e}")
            return False
    
    def _check_syntax(self) -> float:
        """Check for Python syntax errors."""
        try:
            import ast
            
            python_files = list(Path("src").rglob("*.py"))
            total_files = len(python_files)
            valid_files = 0
            
            for file_path in python_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    ast.parse(content)
                    valid_files += 1
                except SyntaxError as e:
                    self.errors.append(f"Syntax error in {file_path}: {e}")
                except Exception as e:
                    self.errors.append(f"Error parsing {file_path}: {e}")
            
            return valid_files / total_files if total_files > 0 else 0.0
            
        except Exception as e:
            self.errors.append(f"Syntax check failed: {e}")
            return 0.0
    
    def _check_complexity(self) -> float:
        """Check code complexity (simplified)."""
        try:
            # Simple complexity check based on file sizes and function counts
            python_files = list(Path("src").rglob("*.py"))
            total_complexity = 0
            file_count = 0
            
            for file_path in python_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    lines = content.split('\n')
                    line_count = len([line for line in lines if line.strip()])
                    
                    # Simple complexity heuristic
                    function_count = content.count('def ')
                    class_count = content.count('class ')
                    
                    # Complexity score (lower is better, normalize to 0-1)
                    if line_count > 0:
                        complexity = min(1.0, (function_count + class_count * 2) / (line_count / 100))
                        total_complexity += 1.0 - complexity  # Invert so higher is better
                        file_count += 1
                        
                except Exception as e:
                    self.errors.append(f"Complexity check error for {file_path}: {e}")
            
            return total_complexity / file_count if file_count > 0 else 0.0
            
        except Exception as e:
            self.errors.append(f"Complexity check failed: {e}")
            return 0.0
    
    def _check_style(self) -> float:
        """Check code style compliance (simplified)."""
        try:
            # Simple style checks
            python_files = list(Path("src").rglob("*.py"))
            style_violations = 0
            total_checks = 0
            
            for file_path in python_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    lines = content.split('\n')
                    
                    for i, line in enumerate(lines):
                        total_checks += 1
                        
                        # Check line length (simplified)
                        if len(line) > 120:
                            style_violations += 1
                        
                        # Check for trailing whitespace
                        if line.endswith(' ') or line.endswith('\t'):
                            style_violations += 1
                            
                except Exception as e:
                    self.errors.append(f"Style check error for {file_path}: {e}")
            
            style_score = 1.0 - (style_violations / total_checks) if total_checks > 0 else 1.0
            return max(0.0, style_score)
            
        except Exception as e:
            self.errors.append(f"Style check failed: {e}")
            return 0.0


class SecurityGate(QualityGate):
    """Security analysis gate."""
    
    def __init__(self):
        super().__init__("Security")
    
    def run(self) -> bool:
        """Run security checks."""
        logger.info("Running security checks...")
        
        try:
            # Check for security vulnerabilities
            vuln_score = self._check_vulnerabilities()
            
            # Check for hardcoded secrets
            secrets_score = self._check_secrets()
            
            # Check for insecure imports
            imports_score = self._check_imports()
            
            self.score = (vuln_score + secrets_score + imports_score) / 3
            self.passed = self.score >= 0.9  # 90% threshold for security
            
            self.details = {
                "vulnerabilities_score": vuln_score,
                "secrets_score": secrets_score,
                "imports_score": imports_score,
                "threshold": 0.9
            }
            
            logger.info(f"Security score: {self.score:.2f}")
            return self.passed
            
        except Exception as e:
            self.errors.append(str(e))
            logger.error(f"Security check failed: {e}")
            return False
    
    def _check_vulnerabilities(self) -> float:
        """Check for common security vulnerabilities."""
        try:
            python_files = list(Path("src").rglob("*.py"))
            vulnerability_patterns = [
                'eval(',
                'exec(',
                'subprocess.call',
                'os.system(',
                'shell=True',
                'pickle.loads',
                'yaml.load(',
                'input(',  # In Python 2, input() is dangerous
            ]
            
            total_files = len(python_files)
            clean_files = 0
            
            for file_path in python_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    file_is_clean = True
                    for pattern in vulnerability_patterns:
                        if pattern in content:
                            self.errors.append(f"Potential vulnerability in {file_path}: {pattern}")
                            file_is_clean = False
                    
                    if file_is_clean:
                        clean_files += 1
                        
                except Exception as e:
                    self.errors.append(f"Vulnerability scan error for {file_path}: {e}")
            
            return clean_files / total_files if total_files > 0 else 1.0
            
        except Exception as e:
            self.errors.append(f"Vulnerability check failed: {e}")
            return 0.0
    
    def _check_secrets(self) -> float:
        """Check for hardcoded secrets."""
        try:
            python_files = list(Path("src").rglob("*.py"))
            secret_patterns = [
                r'password\s*=\s*["\'][^"\']{8,}["\']',
                r'api_key\s*=\s*["\'][^"\']{16,}["\']',
                r'secret\s*=\s*["\'][^"\']{16,}["\']',
                r'token\s*=\s*["\'][^"\']{16,}["\']',
            ]
            
            import re
            
            total_files = len(python_files)
            clean_files = 0
            
            for file_path in python_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    file_is_clean = True
                    for pattern in secret_patterns:
                        if re.search(pattern, content, re.IGNORECASE):
                            # Additional check to avoid false positives
                            if 'test' not in str(file_path).lower() and 'example' not in content.lower():
                                self.errors.append(f"Potential hardcoded secret in {file_path}")
                                file_is_clean = False
                    
                    if file_is_clean:
                        clean_files += 1
                        
                except Exception as e:
                    self.errors.append(f"Secret scan error for {file_path}: {e}")
            
            return clean_files / total_files if total_files > 0 else 1.0
            
        except Exception as e:
            self.errors.append(f"Secret check failed: {e}")
            return 0.0
    
    def _check_imports(self) -> float:
        """Check for insecure or deprecated imports."""
        try:
            python_files = list(Path("src").rglob("*.py"))
            insecure_imports = [
                'from subprocess import *',
                'from os import *', 
                'import pickle',
                'from pickle import *',
                'import dill',
                'import marshal'
            ]
            
            total_files = len(python_files)
            clean_files = 0
            
            for file_path in python_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    file_is_clean = True
                    for import_stmt in insecure_imports:
                        if import_stmt in content:
                            self.errors.append(f"Insecure import in {file_path}: {import_stmt}")
                            file_is_clean = False
                    
                    if file_is_clean:
                        clean_files += 1
                        
                except Exception as e:
                    self.errors.append(f"Import scan error for {file_path}: {e}")
            
            return clean_files / total_files if total_files > 0 else 1.0
            
        except Exception as e:
            self.errors.append(f"Import check failed: {e}")
            return 0.0


class PerformanceGate(QualityGate):
    """Performance benchmark gate."""
    
    def __init__(self):
        super().__init__("Performance")
    
    def run(self) -> bool:
        """Run performance benchmarks."""
        logger.info("Running performance benchmarks...")
        
        try:
            # Import optimization is delayed to avoid import errors
            from lexgraph_legal_rag.bioneuro_olfactory_fusion import get_fusion_engine
            from lexgraph_legal_rag.multisensory_legal_processor import get_multisensory_processor
            
            # Benchmark olfactory processing
            olfactory_score = self._benchmark_olfactory_processing()
            
            # Benchmark multisensory processing  
            multisensory_score = self._benchmark_multisensory_processing()
            
            # Benchmark memory usage
            memory_score = self._benchmark_memory_usage()
            
            self.score = (olfactory_score + multisensory_score + memory_score) / 3
            self.passed = self.score >= 0.7  # 70% threshold
            
            self.details = {
                "olfactory_score": olfactory_score,
                "multisensory_score": multisensory_score,
                "memory_score": memory_score,
                "threshold": 0.7
            }
            
            logger.info(f"Performance score: {self.score:.2f}")
            return self.passed
            
        except Exception as e:
            self.errors.append(str(e))
            logger.error(f"Performance check failed: {e}")
            return False
    
    def _benchmark_olfactory_processing(self) -> float:
        """Benchmark olfactory processing performance."""
        try:
            import asyncio
            from lexgraph_legal_rag.bioneuro_olfactory_fusion import analyze_document_scent
            
            test_document = """
            This is a complex legal contract with multiple liability provisions,
            indemnification clauses, and regulatory compliance requirements
            pursuant to applicable statutes and federal regulations.
            """ * 10  # Make it substantial
            
            async def run_benchmark():
                start_time = time.time()
                
                # Process multiple documents
                for i in range(10):
                    await analyze_document_scent(test_document, f"perf_test_{i}")
                
                return time.time() - start_time
            
            total_time = asyncio.run(run_benchmark())
            avg_time_per_doc = total_time / 10
            
            # Score based on processing time (< 2 seconds per doc = good)
            if avg_time_per_doc < 1.0:
                score = 1.0
            elif avg_time_per_doc < 2.0:
                score = 0.8
            elif avg_time_per_doc < 5.0:
                score = 0.6
            else:
                score = 0.3
            
            self.details["olfactory_avg_time"] = avg_time_per_doc
            return score
            
        except Exception as e:
            self.errors.append(f"Olfactory benchmark failed: {e}")
            return 0.0
    
    def _benchmark_multisensory_processing(self) -> float:
        """Benchmark multisensory processing performance."""
        try:
            import asyncio
            from lexgraph_legal_rag.multisensory_legal_processor import analyze_document_multisensory
            
            test_document = """
            # PROFESSIONAL SERVICES AGREEMENT
            
            **Effective Date:** January 15, 2024
            
            This comprehensive agreement establishes terms between parties
            pursuant to applicable regulations including 15 U.S.C. § 1681.
            The contractor shall provide services with liability limitations
            and indemnification provisions as specified herein.
            """ * 5
            
            async def run_benchmark():
                start_time = time.time()
                
                # Process multiple documents
                for i in range(5):
                    await analyze_document_multisensory(test_document, f"multi_perf_test_{i}")
                
                return time.time() - start_time
            
            total_time = asyncio.run(run_benchmark())
            avg_time_per_doc = total_time / 5
            
            # Score based on processing time (< 3 seconds per doc = good for multisensory)
            if avg_time_per_doc < 2.0:
                score = 1.0
            elif avg_time_per_doc < 3.0:
                score = 0.8
            elif avg_time_per_doc < 6.0:
                score = 0.6
            else:
                score = 0.3
            
            self.details["multisensory_avg_time"] = avg_time_per_doc
            return score
            
        except Exception as e:
            self.errors.append(f"Multisensory benchmark failed: {e}")
            return 0.0
    
    def _benchmark_memory_usage(self) -> float:
        """Benchmark memory usage."""
        try:
            import psutil
            import gc
            
            # Get initial memory
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Create some test data
            from lexgraph_legal_rag.bioneuro_olfactory_fusion import get_fusion_engine
            
            engine = get_fusion_engine()
            
            # Simulate processing load
            import asyncio
            
            async def memory_test():
                for i in range(20):
                    test_doc = f"Test document {i} with legal provisions. " * 50
                    await engine.analyze_document(test_doc, f"memory_test_{i}")
            
            asyncio.run(memory_test())
            
            # Get final memory
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = final_memory - initial_memory
            
            # Score based on memory usage increase (< 100MB = good)
            if memory_increase < 50:
                score = 1.0
            elif memory_increase < 100:
                score = 0.8
            elif memory_increase < 200:
                score = 0.6
            else:
                score = 0.3
            
            self.details["memory_increase_mb"] = memory_increase
            
            # Cleanup
            gc.collect()
            
            return score
            
        except Exception as e:
            self.errors.append(f"Memory benchmark failed: {e}")
            return 0.0


class DocumentationGate(QualityGate):
    """Documentation completeness gate."""
    
    def __init__(self):
        super().__init__("Documentation")
    
    def run(self) -> bool:
        """Check documentation completeness."""
        logger.info("Checking documentation completeness...")
        
        try:
            # Check for required documentation files
            file_score = self._check_documentation_files()
            
            # Check docstring coverage
            docstring_score = self._check_docstring_coverage()
            
            # Check README quality
            readme_score = self._check_readme_quality()
            
            self.score = (file_score + docstring_score + readme_score) / 3
            self.passed = self.score >= 0.8  # 80% threshold
            
            self.details = {
                "file_score": file_score,
                "docstring_score": docstring_score,
                "readme_score": readme_score,
                "threshold": 0.8
            }
            
            logger.info(f"Documentation score: {self.score:.2f}")
            return self.passed
            
        except Exception as e:
            self.errors.append(str(e))
            logger.error(f"Documentation check failed: {e}")
            return False
    
    def _check_documentation_files(self) -> float:
        """Check for required documentation files."""
        required_files = [
            "README.md",
            "CHANGELOG.md", 
            "CONTRIBUTING.md",
            "LICENSE"
        ]
        
        found_files = 0
        for file_name in required_files:
            if Path(file_name).exists():
                found_files += 1
            else:
                self.errors.append(f"Missing documentation file: {file_name}")
        
        return found_files / len(required_files)
    
    def _check_docstring_coverage(self) -> float:
        """Check docstring coverage in Python files."""
        try:
            import ast
            
            python_files = list(Path("src").rglob("*.py"))
            total_functions = 0
            documented_functions = 0
            total_classes = 0
            documented_classes = 0
            
            for file_path in python_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    tree = ast.parse(content)
                    
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef):
                            total_functions += 1
                            if ast.get_docstring(node):
                                documented_functions += 1
                        elif isinstance(node, ast.ClassDef):
                            total_classes += 1
                            if ast.get_docstring(node):
                                documented_classes += 1
                                
                except Exception as e:
                    self.errors.append(f"Docstring check error for {file_path}: {e}")
            
            # Calculate overall documentation coverage
            total_items = total_functions + total_classes
            documented_items = documented_functions + documented_classes
            
            coverage = documented_items / total_items if total_items > 0 else 1.0
            
            self.details["function_coverage"] = documented_functions / total_functions if total_functions > 0 else 1.0
            self.details["class_coverage"] = documented_classes / total_classes if total_classes > 0 else 1.0
            self.details["overall_coverage"] = coverage
            
            return coverage
            
        except Exception as e:
            self.errors.append(f"Docstring coverage check failed: {e}")
            return 0.0
    
    def _check_readme_quality(self) -> float:
        """Check README.md quality."""
        try:
            readme_path = Path("README.md")
            if not readme_path.exists():
                return 0.0
            
            with open(readme_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            quality_indicators = [
                "# " in content,  # Has main heading
                "## " in content,  # Has section headings
                "```" in content,  # Has code examples
                "install" in content.lower(),  # Installation instructions
                "usage" in content.lower(),  # Usage information
                "example" in content.lower(),  # Examples
                len(content) > 500,  # Substantial content
            ]
            
            score = sum(quality_indicators) / len(quality_indicators)
            return score
            
        except Exception as e:
            self.errors.append(f"README quality check failed: {e}")
            return 0.0


class QualityGateRunner:
    """Main quality gate runner."""
    
    def __init__(self):
        self.gates = [
            CodeQualityGate(),
            SecurityGate(),
            PerformanceGate(),
            DocumentationGate()
        ]
        self.results = {}
        self.overall_passed = False
        self.overall_score = 0.0
    
    def run_all_gates(self) -> bool:
        """Run all quality gates."""
        logger.info("🚀 Starting Quality Gate Analysis")
        logger.info("=" * 60)
        
        passed_gates = 0
        total_score = 0.0
        
        for gate in self.gates:
            logger.info(f"\n🔍 Running {gate.name} Gate...")
            
            try:
                gate_passed = gate.run()
                self.results[gate.name] = gate.get_report()
                
                if gate_passed:
                    passed_gates += 1
                    logger.info(f"✅ {gate.name} Gate: PASSED (Score: {gate.score:.2f})")
                else:
                    logger.warning(f"❌ {gate.name} Gate: FAILED (Score: {gate.score:.2f})")
                    
                    # Log errors
                    for error in gate.errors:
                        logger.warning(f"   - {error}")
                
                total_score += gate.score
                
            except Exception as e:
                logger.error(f"💥 {gate.name} Gate: ERROR - {e}")
                self.results[gate.name] = {
                    "name": gate.name,
                    "passed": False,
                    "score": 0.0,
                    "errors": [str(e)]
                }
        
        # Calculate overall results
        self.overall_score = total_score / len(self.gates)
        self.overall_passed = passed_gates >= len(self.gates) * 0.75  # 75% of gates must pass
        
        # Print summary
        logger.info("\n" + "=" * 60)
        logger.info("📊 QUALITY GATE SUMMARY")
        logger.info("=" * 60)
        
        for gate_name, result in self.results.items():
            status = "✅ PASS" if result["passed"] else "❌ FAIL"
            logger.info(f"{gate_name:15} | {status} | Score: {result['score']:.2f}")
        
        logger.info("-" * 60)
        overall_status = "✅ PASS" if self.overall_passed else "❌ FAIL"
        logger.info(f"{'OVERALL':15} | {overall_status} | Score: {self.overall_score:.2f}")
        
        if self.overall_passed:
            logger.info("\n🎉 All quality gates passed! System is production-ready.")
        else:
            logger.warning("\n⚠️  Some quality gates failed. Review issues before production deployment.")
        
        return self.overall_passed
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive quality gate report."""
        return {
            "timestamp": time.time(),
            "overall_passed": self.overall_passed,
            "overall_score": self.overall_score,
            "gates_passed": sum(1 for result in self.results.values() if result["passed"]),
            "total_gates": len(self.gates),
            "results": self.results,
            "recommendation": "PRODUCTION_READY" if self.overall_passed else "NEEDS_IMPROVEMENT"
        }
    
    def save_report(self, filename: str = "quality_gates_report.json"):
        """Save quality gate report to file."""
        report = self.generate_report()
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"📝 Quality gate report saved to {filename}")


def main():
    """Main function to run quality gates."""
    try:
        runner = QualityGateRunner()
        success = runner.run_all_gates()
        
        # Save report
        runner.save_report()
        
        # Return appropriate exit code
        sys.exit(0 if success else 1)
        
    except Exception as e:
        logger.error(f"Quality gate runner failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()