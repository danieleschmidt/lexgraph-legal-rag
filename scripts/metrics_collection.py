#!/usr/bin/env python3
"""
Metrics Collection Script for LexGraph Legal RAG

This script collects various project metrics and updates the project-metrics.json file.
It integrates with GitHub API, code quality tools, and monitoring systems.
"""

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional

import requests


class MetricsCollector:
    """Collects and aggregates project metrics from various sources."""
    
    def __init__(self, repo_path: str = "."):
        self.repo_path = Path(repo_path)
        self.github_token = os.getenv("GITHUB_TOKEN")
        self.repo_name = os.getenv("GITHUB_REPOSITORY", "danieleschmidt/lexgraph-legal-rag")
        
    def collect_all_metrics(self) -> Dict[str, Any]:
        """Collect all available metrics."""
        print("🔍 Collecting project metrics...")
        
        metrics = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "code_quality": self.collect_code_quality_metrics(),
            "security": self.collect_security_metrics(),
            "performance": self.collect_performance_metrics(),
            "automation": self.collect_automation_metrics(),
            "github": self.collect_github_metrics(),
            "infrastructure": self.collect_infrastructure_metrics(),
        }
        
        print("✅ Metrics collection completed")
        return metrics
    
    def collect_code_quality_metrics(self) -> Dict[str, Any]:
        """Collect code quality metrics."""
        print("  📊 Collecting code quality metrics...")
        
        metrics = {}
        
        # Test coverage
        try:
            result = subprocess.run(
                ["pytest", "--cov=lexgraph_legal_rag", "--cov-report=json"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=False
            )
            
            coverage_file = self.repo_path / "coverage.json"
            if coverage_file.exists():
                with open(coverage_file) as f:
                    coverage_data = json.load(f)
                    metrics["test_coverage"] = {
                        "percentage": coverage_data.get("totals", {}).get("percent_covered", 0),
                        "lines_covered": coverage_data.get("totals", {}).get("covered_lines", 0),
                        "lines_total": coverage_data.get("totals", {}).get("num_statements", 0),
                        "files_analyzed": len(coverage_data.get("files", {}))
                    }
        except Exception as e:
            print(f"    ⚠️  Could not collect coverage metrics: {e}")
            metrics["test_coverage"] = {"percentage": 0, "error": str(e)}
        
        # Code complexity with radon
        try:
            result = subprocess.run(
                ["radon", "cc", "src/", "-j"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=False
            )
            
            if result.returncode == 0 and result.stdout:
                complexity_data = json.loads(result.stdout)
                total_complexity = 0
                function_count = 0
                
                for file_data in complexity_data.values():
                    for item in file_data:
                        if item.get("type") == "function":
                            total_complexity += item.get("complexity", 0)
                            function_count += 1
                
                avg_complexity = total_complexity / function_count if function_count > 0 else 0
                metrics["complexity"] = {
                    "average_cyclomatic": round(avg_complexity, 2),
                    "total_functions": function_count,
                    "total_complexity": total_complexity
                }
        except Exception as e:
            print(f"    ⚠️  Could not collect complexity metrics: {e}")
        
        # Lines of code
        try:
            result = subprocess.run(
                ["find", "src/", "-name", "*.py", "-exec", "wc", "-l", "{}", "+"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=False
            )
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                total_lines = 0
                file_count = 0
                
                for line in lines[:-1]:  # Last line is total
                    if line.strip():
                        parts = line.strip().split()
                        if len(parts) >= 2:
                            total_lines += int(parts[0])
                            file_count += 1
                
                metrics["lines_of_code"] = {
                    "total": total_lines,
                    "files": file_count,
                    "average_per_file": round(total_lines / file_count, 2) if file_count > 0 else 0
                }
        except Exception as e:
            print(f"    ⚠️  Could not collect LOC metrics: {e}")
        
        return metrics
    
    def collect_security_metrics(self) -> Dict[str, Any]:
        """Collect security-related metrics."""
        print("  🔒 Collecting security metrics...")
        
        metrics = {}
        
        # Bandit security scan
        try:
            result = subprocess.run(
                ["bandit", "-r", "src/", "-f", "json"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=False
            )
            
            if result.stdout:
                bandit_data = json.loads(result.stdout)
                metrics["vulnerabilities"] = {
                    "high": len([r for r in bandit_data.get("results", []) if r.get("issue_severity") == "HIGH"]),
                    "medium": len([r for r in bandit_data.get("results", []) if r.get("issue_severity") == "MEDIUM"]),
                    "low": len([r for r in bandit_data.get("results", []) if r.get("issue_severity") == "LOW"]),
                    "total": len(bandit_data.get("results", [])),
                    "files_scanned": len(bandit_data.get("metrics", {}).get("_totals", {}).get("loc", 0))
                }
        except Exception as e:
            print(f"    ⚠️  Could not collect Bandit metrics: {e}")
        
        # Safety check for dependencies
        try:
            result = subprocess.run(
                ["safety", "check", "--json"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=False
            )
            
            if result.stdout:
                safety_data = json.loads(result.stdout)
                metrics["dependency_vulnerabilities"] = len(safety_data)
        except Exception as e:
            print(f"    ⚠️  Could not collect Safety metrics: {e}")
        
        return metrics
    
    def collect_performance_metrics(self) -> Dict[str, Any]:
        """Collect performance-related metrics."""
        print("  ⚡ Collecting performance metrics...")
        
        metrics = {}
        
        # Check if performance tests exist
        perf_test_dir = self.repo_path / "tests" / "performance"
        if perf_test_dir.exists():
            metrics["performance_tests"] = {
                "enabled": True,
                "test_files": len(list(perf_test_dir.glob("*.py")))
            }
        
        # Docker image size (if Dockerfile exists)
        try:
            dockerfile = self.repo_path / "Dockerfile"
            if dockerfile.exists():
                result = subprocess.run(
                    ["docker", "images", "--format", "table {{.Repository}}:{{.Tag}}\t{{.Size}}", "--filter", "reference=lexgraph-legal-rag"],
                    capture_output=True,
                    text=True,
                    check=False
                )
                
                if result.returncode == 0 and result.stdout:
                    lines = result.stdout.strip().split('\n')[1:]  # Skip header
                    if lines:
                        metrics["docker_image"] = {
                            "size": lines[0].split('\t')[1] if '\t' in lines[0] else "unknown",
                            "available": True
                        }
        except Exception as e:
            print(f"    ⚠️  Could not collect Docker metrics: {e}")
        
        return metrics
    
    def collect_automation_metrics(self) -> Dict[str, Any]:
        """Collect automation-related metrics."""
        print("  🤖 Collecting automation metrics...")
        
        metrics = {}
        
        # Count workflow files
        workflows_dir = self.repo_path / "docs" / "workflows"
        if workflows_dir.exists():
            workflow_files = list(workflows_dir.glob("*.yml"))
            metrics["workflows"] = {
                "count": len(workflow_files),
                "files": [f.name for f in workflow_files]
            }
        
        # Count test files
        tests_dir = self.repo_path / "tests"
        if tests_dir.exists():
            test_files = list(tests_dir.glob("test_*.py"))
            metrics["tests"] = {
                "unit_tests": len([f for f in test_files if "integration" not in f.name and "e2e" not in f.name]),
                "integration_tests": len([f for f in test_files if "integration" in f.name]),
                "e2e_tests": len([f for f in test_files if "e2e" in f.name]),
                "total": len(test_files)
            }
        
        # Check pre-commit configuration
        precommit_config = self.repo_path / ".pre-commit-config.yaml"
        if precommit_config.exists():
            metrics["pre_commit"] = {"enabled": True}
        
        return metrics
    
    def collect_github_metrics(self) -> Dict[str, Any]:
        """Collect GitHub repository metrics via API."""
        print("  🐙 Collecting GitHub metrics...")
        
        if not self.github_token:
            print("    ⚠️  GitHub token not available, skipping API metrics")
            return {}
        
        metrics = {}
        headers = {"Authorization": f"token {self.github_token}"}
        
        try:
            # Repository information
            repo_response = requests.get(
                f"https://api.github.com/repos/{self.repo_name}",
                headers=headers,
                timeout=10
            )
            
            if repo_response.status_code == 200:
                repo_data = repo_response.json()
                metrics["repository"] = {
                    "stars": repo_data.get("stargazers_count", 0),
                    "forks": repo_data.get("forks_count", 0),
                    "open_issues": repo_data.get("open_issues_count", 0),
                    "size": repo_data.get("size", 0),
                    "language": repo_data.get("language", "unknown"),
                    "default_branch": repo_data.get("default_branch", "main")
                }
            
            # Pull requests
            pr_response = requests.get(
                f"https://api.github.com/repos/{self.repo_name}/pulls?state=all&per_page=100",
                headers=headers,
                timeout=10
            )
            
            if pr_response.status_code == 200:
                prs = pr_response.json()
                open_prs = [pr for pr in prs if pr.get("state") == "open"]
                closed_prs = [pr for pr in prs if pr.get("state") == "closed"]
                
                metrics["pull_requests"] = {
                    "open": len(open_prs),
                    "closed": len(closed_prs),
                    "total": len(prs)
                }
        
        except Exception as e:
            print(f"    ⚠️  Could not collect GitHub metrics: {e}")
        
        return metrics
    
    def collect_infrastructure_metrics(self) -> Dict[str, Any]:
        """Collect infrastructure and deployment metrics."""
        print("  🏗️  Collecting infrastructure metrics...")
        
        metrics = {}
        
        # Check for infrastructure files
        docker_compose = self.repo_path / "docker-compose.yml"
        if docker_compose.exists():
            metrics["docker_compose"] = {"enabled": True}
        
        k8s_dir = self.repo_path / "k8s"
        if k8s_dir.exists():
            k8s_files = list(k8s_dir.glob("*.yaml")) + list(k8s_dir.glob("*.yml"))
            metrics["kubernetes"] = {
                "enabled": True,
                "manifests": len(k8s_files)
            }
        
        monitoring_dir = self.repo_path / "monitoring"
        if monitoring_dir.exists():
            metrics["monitoring"] = {"enabled": True}
        
        return metrics
    
    def update_project_metrics_file(self, metrics: Dict[str, Any]) -> None:
        """Update the project-metrics.json file with collected metrics."""
        print("📝 Updating project-metrics.json...")
        
        metrics_file = self.repo_path / ".github" / "project-metrics.json"
        
        # Load existing metrics if file exists
        existing_metrics = {}
        if metrics_file.exists():
            try:
                with open(metrics_file) as f:
                    existing_metrics = json.load(f)
            except Exception as e:
                print(f"⚠️  Could not load existing metrics: {e}")
        
        # Update with new metrics
        existing_metrics.update({
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "collected_metrics": metrics,
            "health_score": self.calculate_health_score(metrics)
        })
        
        # Ensure .github directory exists
        metrics_file.parent.mkdir(exist_ok=True)
        
        # Write updated metrics
        with open(metrics_file, 'w') as f:
            json.dump(existing_metrics, f, indent=2, sort_keys=True)
        
        print(f"✅ Updated {metrics_file}")
    
    def calculate_health_score(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall project health score."""
        scores = {}
        
        # Code quality score
        code_quality = metrics.get("code_quality", {})
        coverage = code_quality.get("test_coverage", {}).get("percentage", 0)
        complexity = code_quality.get("complexity", {}).get("average_cyclomatic", 10)
        
        code_score = min(100, (coverage * 0.7) + max(0, (10 - complexity) * 5))
        scores["code_quality"] = round(code_score, 1)
        
        # Security score
        security = metrics.get("security", {})
        vulns = security.get("vulnerabilities", {})
        high_vulns = vulns.get("high", 0)
        medium_vulns = vulns.get("medium", 0)
        
        security_score = max(0, 100 - (high_vulns * 20) - (medium_vulns * 5))
        scores["security"] = round(security_score, 1)
        
        # Automation score  
        automation = metrics.get("automation", {})
        workflow_count = automation.get("workflows", {}).get("count", 0)
        test_count = automation.get("tests", {}).get("total", 0)
        
        automation_score = min(100, (workflow_count * 10) + (test_count * 2))
        scores["automation"] = round(automation_score, 1)
        
        # Overall score
        overall = sum(scores.values()) / len(scores) if scores else 0
        scores["overall"] = round(overall, 1)
        
        return scores


def main():
    """Main entry point."""
    collector = MetricsCollector()
    
    try:
        metrics = collector.collect_all_metrics()
        collector.update_project_metrics_file(metrics)
        
        print("\n🎉 Metrics collection completed successfully!")
        
        # Print summary
        health_score = collector.calculate_health_score(metrics)
        print(f"\n📊 Health Score Summary:")
        for category, score in health_score.items():
            print(f"  {category.replace('_', ' ').title()}: {score}/100")
        
    except Exception as e:
        print(f"❌ Error collecting metrics: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()