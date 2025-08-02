#!/usr/bin/env python3
"""
Repository Maintenance Automation Script

This script performs automated maintenance tasks for the LexGraph Legal RAG repository.
It handles cleanup, optimization, dependency updates, and health checks.
"""

import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional

import requests


class RepositoryMaintainer:
    """Automated repository maintenance tasks."""
    
    def __init__(self, repo_path: str = "."):
        self.repo_path = Path(repo_path)
        self.github_token = os.getenv("GITHUB_TOKEN")
        self.dry_run = os.getenv("DRY_RUN", "false").lower() == "true"
        
    def run_maintenance(self) -> Dict[str, Any]:
        """Run all maintenance tasks."""
        print("🛠️  Starting repository maintenance...")
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "tasks": {}
        }
        
        tasks = [
            ("cleanup", self.cleanup_temporary_files),
            ("dependencies", self.check_dependencies),
            ("security", self.security_maintenance),
            ("documentation", self.documentation_maintenance),
            ("performance", self.performance_maintenance),
            ("git", self.git_maintenance),
        ]
        
        for task_name, task_func in tasks:
            print(f"\n📋 Running {task_name} maintenance...")
            try:
                task_result = task_func()
                results["tasks"][task_name] = {
                    "status": "success",
                    "result": task_result
                }
                print(f"✅ {task_name} maintenance completed")
            except Exception as e:
                print(f"❌ {task_name} maintenance failed: {e}")
                results["tasks"][task_name] = {
                    "status": "failed",
                    "error": str(e)
                }
        
        # Generate maintenance report
        self.generate_maintenance_report(results)
        
        print("\n🎉 Repository maintenance completed!")
        return results
    
    def cleanup_temporary_files(self) -> Dict[str, Any]:
        """Clean up temporary files and caches."""
        cleaned = []
        size_saved = 0
        
        # Python cache files
        for cache_dir in self.repo_path.rglob("__pycache__"):
            if cache_dir.is_dir():
                size_saved += self.get_dir_size(cache_dir)
                if not self.dry_run:
                    shutil.rmtree(cache_dir)
                cleaned.append(str(cache_dir))
        
        # .pyc files
        for pyc_file in self.repo_path.rglob("*.pyc"):
            size_saved += pyc_file.stat().st_size
            if not self.dry_run:
                pyc_file.unlink()
            cleaned.append(str(pyc_file))
        
        # Test artifacts
        test_artifacts = [
            ".pytest_cache",
            ".coverage",
            "htmlcov",
            ".mypy_cache",
            ".ruff_cache",
            "coverage.xml",
            "test-results"
        ]
        
        for artifact in test_artifacts:
            artifact_path = self.repo_path / artifact
            if artifact_path.exists():
                if artifact_path.is_dir():
                    size_saved += self.get_dir_size(artifact_path)
                    if not self.dry_run:
                        shutil.rmtree(artifact_path)
                else:
                    size_saved += artifact_path.stat().st_size
                    if not self.dry_run:
                        artifact_path.unlink()
                cleaned.append(str(artifact_path))
        
        # Build artifacts
        build_dirs = ["build", "dist", "*.egg-info"]
        for pattern in build_dirs:
            for path in self.repo_path.rglob(pattern):
                if path.is_dir():
                    size_saved += self.get_dir_size(path)
                    if not self.dry_run:
                        shutil.rmtree(path)
                    cleaned.append(str(path))
        
        return {
            "files_cleaned": len(cleaned),
            "size_saved_mb": round(size_saved / (1024 * 1024), 2),
            "cleaned_items": cleaned[:10],  # Show first 10 items
            "dry_run": self.dry_run
        }
    
    def check_dependencies(self) -> Dict[str, Any]:
        """Check and report on dependency status."""
        results = {}
        
        # Check for outdated Python packages
        try:
            result = subprocess.run(
                ["pip", "list", "--outdated", "--format=json"],
                capture_output=True,
                text=True,
                check=False
            )
            
            if result.returncode == 0 and result.stdout:
                outdated = json.loads(result.stdout)
                results["outdated_packages"] = {
                    "count": len(outdated),
                    "packages": [
                        {
                            "name": pkg["name"],
                            "current": pkg["version"],
                            "latest": pkg["latest_version"]
                        }
                        for pkg in outdated[:5]  # Show first 5
                    ]
                }
        except Exception as e:
            results["outdated_packages"] = {"error": str(e)}
        
        # Check requirements.txt vs installed packages
        requirements_file = self.repo_path / "requirements.txt"
        if requirements_file.exists():
            try:
                result = subprocess.run(
                    ["pip", "check"],
                    capture_output=True,
                    text=True,
                    check=False
                )
                
                results["dependency_conflicts"] = {
                    "has_conflicts": result.returncode != 0,
                    "details": result.stdout if result.returncode != 0 else "No conflicts found"
                }
            except Exception as e:
                results["dependency_conflicts"] = {"error": str(e)}
        
        # Check for security vulnerabilities
        try:
            result = subprocess.run(
                ["safety", "check", "--json"],
                capture_output=True,
                text=True,
                check=False
            )
            
            if result.stdout:
                vulnerabilities = json.loads(result.stdout)
                results["security_vulnerabilities"] = {
                    "count": len(vulnerabilities),
                    "details": [
                        {
                            "package": vuln.get("package"),
                            "installed": vuln.get("installed_version"),
                            "vulnerability": vuln.get("vulnerability_id")
                        }
                        for vuln in vulnerabilities[:3]  # Show first 3
                    ]
                }
        except Exception as e:
            results["security_vulnerabilities"] = {"error": str(e)}
        
        return results
    
    def security_maintenance(self) -> Dict[str, Any]:
        """Perform security-related maintenance."""
        results = {}
        
        # Run security scan
        try:
            result = subprocess.run(
                ["bandit", "-r", "src/", "-f", "json", "-o", "security-report.json"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=False
            )
            
            if (self.repo_path / "security-report.json").exists():
                with open(self.repo_path / "security-report.json") as f:
                    security_data = json.load(f)
                    
                results["security_scan"] = {
                    "issues_found": len(security_data.get("results", [])),
                    "high_severity": len([
                        r for r in security_data.get("results", [])
                        if r.get("issue_severity") == "HIGH"
                    ]),
                    "files_scanned": security_data.get("metrics", {}).get("_totals", {}).get("nosec", 0)
                }
        except Exception as e:
            results["security_scan"] = {"error": str(e)}
        
        # Check for secrets in code
        try:
            result = subprocess.run(
                ["detect-secrets", "scan", "--all-files"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=False
            )
            
            results["secrets_scan"] = {
                "completed": True,
                "potential_secrets_found": "secrets" in result.stdout.lower()
            }
        except Exception as e:
            results["secrets_scan"] = {"error": str(e)}
        
        return results
    
    def documentation_maintenance(self) -> Dict[str, Any]:
        """Maintain and validate documentation."""
        results = {}
        
        # Check for broken internal links
        docs_dir = self.repo_path / "docs"
        if docs_dir.exists():
            md_files = list(docs_dir.rglob("*.md"))
            results["documentation_files"] = len(md_files)
            
            # Simple check for common broken patterns
            broken_links = []
            for md_file in md_files:
                try:
                    content = md_file.read_text()
                    # Check for markdown links that might be broken
                    if "](docs/" in content or "](../docs" in content:
                        broken_links.append(str(md_file))
                except Exception:
                    continue
            
            results["potentially_broken_links"] = len(broken_links)
        
        # Check README completeness
        readme_file = self.repo_path / "README.md"
        if readme_file.exists():
            content = readme_file.read_text()
            sections = [
                "## Features",
                "## Quick Start",
                "## Installation",
                "## Usage",
                "## Contributing",
                "## License"
            ]
            
            missing_sections = [s for s in sections if s not in content]
            results["readme_completeness"] = {
                "total_sections": len(sections),
                "missing_sections": missing_sections,
                "completeness_score": (len(sections) - len(missing_sections)) / len(sections) * 100
            }
        
        return results
    
    def performance_maintenance(self) -> Dict[str, Any]:
        """Perform performance-related maintenance."""
        results = {}
        
        # Check Docker image size if available
        try:
            result = subprocess.run(
                ["docker", "images", "--format", "{{.Size}}", "--filter", "reference=lexgraph-legal-rag"],
                capture_output=True,
                text=True,
                check=False
            )
            
            if result.returncode == 0 and result.stdout.strip():
                results["docker_image_size"] = result.stdout.strip()
        except Exception:
            pass
        
        # Analyze repository size
        results["repository_size"] = {
            "total_size_mb": round(self.get_dir_size(self.repo_path) / (1024 * 1024), 2),
            "largest_directories": self.get_largest_directories()
        }
        
        # Check for large files
        large_files = []
        for file_path in self.repo_path.rglob("*"):
            if file_path.is_file() and file_path.stat().st_size > 10 * 1024 * 1024:  # > 10MB
                large_files.append({
                    "path": str(file_path.relative_to(self.repo_path)),
                    "size_mb": round(file_path.stat().st_size / (1024 * 1024), 2)
                })
        
        results["large_files"] = large_files
        
        return results
    
    def git_maintenance(self) -> Dict[str, Any]:
        """Perform Git repository maintenance."""
        results = {}
        
        try:
            # Git garbage collection
            if not self.dry_run:
                subprocess.run(["git", "gc", "--prune=now"], cwd=self.repo_path, check=True)
            results["garbage_collection"] = "completed" if not self.dry_run else "dry_run"
            
            # Check repository status
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            
            results["working_directory"] = {
                "clean": len(result.stdout.strip()) == 0,
                "untracked_files": len([line for line in result.stdout.split('\n') if line.startswith('??')])
            }
            
            # Check for large objects in Git history
            result = subprocess.run(
                ["git", "rev-list", "--objects", "--all"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=False
            )
            
            if result.returncode == 0:
                # This is a simplified check - in reality you'd want more sophisticated analysis
                results["git_objects"] = {
                    "total_objects": len(result.stdout.split('\n')),
                    "analysis": "basic_count_completed"
                }
        
        except Exception as e:
            results["git_maintenance"] = {"error": str(e)}
        
        return results
    
    def get_dir_size(self, path: Path) -> int:
        """Get total size of directory in bytes."""
        total = 0
        try:
            for item in path.rglob('*'):
                if item.is_file():
                    total += item.stat().st_size
        except (OSError, PermissionError):
            pass
        return total
    
    def get_largest_directories(self) -> List[Dict[str, Any]]:
        """Get list of largest directories."""
        dirs = []
        
        for item in self.repo_path.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                size_mb = round(self.get_dir_size(item) / (1024 * 1024), 2)
                if size_mb > 1:  # Only include dirs > 1MB
                    dirs.append({
                        "name": item.name,
                        "size_mb": size_mb
                    })
        
        return sorted(dirs, key=lambda x: x["size_mb"], reverse=True)[:5]
    
    def generate_maintenance_report(self, results: Dict[str, Any]) -> None:
        """Generate a maintenance report."""
        report_file = self.repo_path / "maintenance-report.json"
        
        with open(report_file, 'w') as f:
            json.dump(results, f, indent=2, sort_keys=True)
        
        print(f"📊 Maintenance report saved to {report_file}")
        
        # Print summary
        print("\n📋 Maintenance Summary:")
        for task_name, task_result in results["tasks"].items():
            status = task_result["status"]
            emoji = "✅" if status == "success" else "❌"
            print(f"  {emoji} {task_name.title()}: {status}")


def main():
    """Main entry point."""
    maintainer = RepositoryMaintainer()
    
    try:
        results = maintainer.run_maintenance()
        
        # Exit with error code if any task failed
        failed_tasks = [
            name for name, result in results["tasks"].items()
            if result["status"] == "failed"
        ]
        
        if failed_tasks:
            print(f"\n⚠️  Some maintenance tasks failed: {', '.join(failed_tasks)}")
            sys.exit(1)
        else:
            print("\n🎉 All maintenance tasks completed successfully!")
            sys.exit(0)
    
    except Exception as e:
        print(f"❌ Maintenance failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()