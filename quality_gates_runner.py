#!/usr/bin/env python3
"""Standalone quality gates runner for production readiness validation."""

import asyncio
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Any


class QualityGateValidator:
    """Standalone quality gate validator."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.src_path = project_root / "src"
        self.tests_path = project_root / "tests"
    
    async def run_basic_validation(self) -> Dict[str, Any]:
        """Run basic validation that doesn't require external dependencies."""
        results = {}
        
        # File structure validation
        results["file_structure"] = self._check_file_structure()
        
        # Configuration validation
        results["configuration"] = self._check_configuration()
        
        # Documentation validation
        results["documentation"] = self._check_documentation()
        
        # Code quality (basic)
        results["code_quality"] = await self._check_code_quality()
        
        # Security (basic)
        results["security"] = self._check_basic_security()
        
        return results
    
    def _check_file_structure(self) -> Dict[str, Any]:
        """Check essential file structure."""
        required_files = [
            "README.md",
            "requirements.txt",
            "pyproject.toml",
            "src/lexgraph_legal_rag/__init__.py",
            "src/lexgraph_legal_rag/api.py",
            "src/lexgraph_legal_rag/multi_agent.py",
            "src/lexgraph_legal_rag/document_pipeline.py",
        ]
        
        missing_files = []
        present_files = []
        
        for file_path in required_files:
            full_path = self.project_root / file_path
            if full_path.exists():
                present_files.append(file_path)
            else:
                missing_files.append(file_path)
        
        if len(missing_files) == 0:
            status = "PASS"
            message = f"All {len(required_files)} essential files present"
        elif len(missing_files) <= 2:
            status = "WARNING"
            message = f"{len(missing_files)} files missing: {', '.join(missing_files)}"
        else:
            status = "FAIL"
            message = f"{len(missing_files)} critical files missing"
        
        return {
            "status": status,
            "message": message,
            "details": {
                "present_files": len(present_files),
                "missing_files": missing_files,
                "total_required": len(required_files)
            }
        }
    
    def _check_configuration(self) -> Dict[str, Any]:
        """Check configuration completeness."""
        issues = []
        
        # Check pyproject.toml
        pyproject = self.project_root / "pyproject.toml"
        if pyproject.exists():
            try:
                content = pyproject.read_text()
                if "[project]" not in content:
                    issues.append("pyproject.toml missing [project] section")
                if "dependencies" not in content:
                    issues.append("pyproject.toml missing dependencies")
            except Exception as e:
                issues.append(f"pyproject.toml read error: {e}")
        else:
            issues.append("pyproject.toml missing")
        
        # Check requirements.txt
        requirements = self.project_root / "requirements.txt"
        if requirements.exists():
            try:
                content = requirements.read_text()
                if len(content.strip().splitlines()) < 5:
                    issues.append("requirements.txt seems incomplete")
            except Exception as e:
                issues.append(f"requirements.txt read error: {e}")
        else:
            issues.append("requirements.txt missing")
        
        # Check for Docker configuration
        dockerfile = self.project_root / "Dockerfile"
        docker_compose = self.project_root / "docker-compose.yml"
        
        if not dockerfile.exists():
            issues.append("Dockerfile missing")
        
        if len(issues) == 0:
            status = "PASS"
            message = "Configuration complete"
        elif len(issues) <= 2:
            status = "WARNING"
            message = f"Minor configuration issues: {len(issues)}"
        else:
            status = "FAIL"
            message = f"Configuration incomplete: {len(issues)} issues"
        
        return {
            "status": status,
            "message": message,
            "details": {"issues": issues}
        }
    
    def _check_documentation(self) -> Dict[str, Any]:
        """Check documentation completeness."""
        required_docs = [
            "README.md",
            "CHANGELOG.md",
            "LICENSE",
            "CONTRIBUTING.md"
        ]
        
        optional_docs = [
            "docs/DEPLOYMENT.md",
            "docs/DEVELOPMENT.md",
            "docs/API_GUIDE.md"
        ]
        
        missing_required = []
        missing_optional = []
        
        for doc in required_docs:
            if not (self.project_root / doc).exists():
                missing_required.append(doc)
        
        for doc in optional_docs:
            if not (self.project_root / doc).exists():
                missing_optional.append(doc)
        
        # Check README quality
        readme_quality_issues = []
        readme = self.project_root / "README.md"
        if readme.exists():
            try:
                content = readme.read_text()
                if len(content) < 500:
                    readme_quality_issues.append("README too short")
                if "# " not in content:
                    readme_quality_issues.append("README missing main heading")
                if "## " not in content:
                    readme_quality_issues.append("README missing sections")
                if "install" not in content.lower():
                    readme_quality_issues.append("README missing installation instructions")
            except Exception:
                readme_quality_issues.append("README read error")
        
        total_issues = len(missing_required) + len(readme_quality_issues)
        
        if total_issues == 0:
            status = "PASS"
            message = f"Documentation complete ({len(missing_optional)} optional docs missing)"
        elif total_issues <= 2:
            status = "WARNING"
            message = f"Documentation mostly complete ({total_issues} issues)"
        else:
            status = "FAIL"
            message = f"Documentation incomplete ({total_issues} issues)"
        
        return {
            "status": status,
            "message": message,
            "details": {
                "missing_required": missing_required,
                "missing_optional": missing_optional,
                "readme_issues": readme_quality_issues
            }
        }
    
    async def _check_code_quality(self) -> Dict[str, Any]:
        """Check basic code quality metrics."""
        if not self.src_path.exists():
            return {
                "status": "FAIL",
                "message": "Source directory not found",
                "details": {"error": "src/ directory missing"}
            }
        
        python_files = list(self.src_path.rglob("*.py"))
        
        if len(python_files) == 0:
            return {
                "status": "FAIL", 
                "message": "No Python files found",
                "details": {"python_files": 0}
            }
        
        # Basic quality checks
        total_lines = 0
        files_with_docstrings = 0
        files_with_type_hints = 0
        
        for py_file in python_files:
            try:
                content = py_file.read_text()
                lines = content.splitlines()
                total_lines += len(lines)
                
                # Check for docstrings
                if '"""' in content or "'''" in content:
                    files_with_docstrings += 1
                
                # Check for type hints
                if " -> " in content or ": " in content:
                    files_with_type_hints += 1
                    
            except Exception:
                continue
        
        docstring_coverage = files_with_docstrings / len(python_files)
        type_hint_coverage = files_with_type_hints / len(python_files)
        
        avg_lines_per_file = total_lines / len(python_files)
        
        quality_score = (docstring_coverage + type_hint_coverage) / 2
        
        if quality_score >= 0.8:
            status = "PASS"
            message = f"Good code quality (score: {quality_score:.2f})"
        elif quality_score >= 0.6:
            status = "WARNING"
            message = f"Adequate code quality (score: {quality_score:.2f})"
        else:
            status = "FAIL"
            message = f"Poor code quality (score: {quality_score:.2f})"
        
        return {
            "status": status,
            "message": message,
            "details": {
                "python_files": len(python_files),
                "total_lines": total_lines,
                "avg_lines_per_file": avg_lines_per_file,
                "docstring_coverage": docstring_coverage,
                "type_hint_coverage": type_hint_coverage,
                "quality_score": quality_score
            }
        }
    
    def _check_basic_security(self) -> Dict[str, Any]:
        """Check basic security patterns."""
        security_issues = []
        
        # Check for hardcoded secrets patterns
        dangerous_patterns = [
            ("password", "hardcoded password"),
            ("secret", "hardcoded secret"),
            ("api_key", "hardcoded API key"),
            ("token", "hardcoded token"),
            ("pk_", "hardcoded private key"),
            ("sk_", "hardcoded secret key")
        ]
        
        if self.src_path.exists():
            for py_file in self.src_path.rglob("*.py"):
                try:
                    content = py_file.read_text().lower()
                    
                    for pattern, description in dangerous_patterns:
                        if f'"{pattern}"' in content or f"'{pattern}'" in content:
                            # Skip if it's in comments or looks like a placeholder
                            if "todo" not in content and "example" not in content:
                                security_issues.append(f"{description} in {py_file.name}")
                                
                except Exception:
                    continue
        
        # Check Dockerfile security if present
        dockerfile = self.project_root / "Dockerfile"
        if dockerfile.exists():
            try:
                content = dockerfile.read_text()
                if "USER root" in content and content.count("USER ") == 1:
                    security_issues.append("Dockerfile runs as root")
                if "--privileged" in content:
                    security_issues.append("Dockerfile uses privileged mode")
            except Exception:
                pass
        
        # Check for .env files in git
        gitignore = self.project_root / ".gitignore"
        if gitignore.exists():
            try:
                content = gitignore.read_text()
                if ".env" not in content:
                    security_issues.append(".env not in .gitignore")
            except Exception:
                pass
        
        if len(security_issues) == 0:
            status = "PASS"
            message = "No obvious security issues found"
        elif len(security_issues) <= 2:
            status = "WARNING"
            message = f"{len(security_issues)} potential security issues"
        else:
            status = "FAIL"
            message = f"{len(security_issues)} security issues found"
        
        return {
            "status": status,
            "message": message,
            "details": {"issues": security_issues}
        }


def main():
    """Main entry point."""
    project_root = Path.cwd()
    validator = QualityGateValidator(project_root)
    
    print("🔍 Running LexGraph Legal RAG Quality Gates Validation...")
    print(f"📁 Project Root: {project_root}")
    print("=" * 60)
    
    async def run_validation():
        results = await validator.run_basic_validation()
        
        total_gates = len(results)
        passed = len([r for r in results.values() if r["status"] == "PASS"])
        failed = len([r for r in results.values() if r["status"] == "FAIL"])
        warnings = len([r for r in results.values() if r["status"] == "WARNING"])
        
        for gate_name, result in results.items():
            status_emoji = {
                "PASS": "✅",
                "WARNING": "⚠️", 
                "FAIL": "❌"
            }.get(result["status"], "❓")
            
            print(f"{status_emoji} {gate_name.replace('_', ' ').title()}: {result['message']}")
        
        print("=" * 60)
        print(f"📊 Summary: {passed} passed, {failed} failed, {warnings} warnings")
        
        # Overall assessment
        if failed == 0 and warnings <= 1:
            print("🎉 PRODUCTION READY: All critical quality gates passed!")
            overall_status = "PRODUCTION_READY"
        elif failed <= 1 and warnings <= 3:
            print("⚠️  NEEDS ATTENTION: Some quality gates need improvement")
            overall_status = "NEEDS_ATTENTION"
        else:
            print("❌ NOT PRODUCTION READY: Critical quality gates failed")
            overall_status = "NOT_READY"
        
        # Save detailed results
        detailed_results = {
            "timestamp": time.time(),
            "overall_status": overall_status,
            "summary": {
                "total_gates": total_gates,
                "passed": passed,
                "failed": failed,
                "warnings": warnings
            },
            "results": results
        }
        
        results_file = project_root / "quality_gates_report.json"
        with open(results_file, "w") as f:
            json.dump(detailed_results, f, indent=2)
        
        print(f"📄 Detailed report saved to: {results_file}")
        
        return overall_status
    
    try:
        overall_status = asyncio.run(run_validation())
        
        # Exit with appropriate code
        if overall_status == "PRODUCTION_READY":
            sys.exit(0)
        elif overall_status == "NEEDS_ATTENTION":
            sys.exit(1)
        else:
            sys.exit(2)
            
    except Exception as e:
        print(f"❌ Quality gates validation failed: {e}")
        sys.exit(3)


if __name__ == "__main__":
    main()