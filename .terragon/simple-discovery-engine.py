#!/usr/bin/env python3
"""
Simplified Terragon Value Discovery Engine
Works with basic system tools available in any environment
"""

import json
import os
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional


class SimpleValueDiscoveryEngine:
    """Simplified value discovery engine using basic system tools"""
    
    def __init__(self):
        self.config = self._load_config()
        self.metrics_file = Path(".terragon/value-metrics.json")
        self.backlog_file = Path("BACKLOG.md")
        self.repo_root = Path(".")
        
    def _load_config(self) -> Dict:
        """Load configuration or use defaults"""
        config_file = Path(".terragon/config.json")
        if config_file.exists():
            try:
                with open(config_file) as f:
                    return json.load(f)
            except json.JSONDecodeError:
                pass
        
        return {
            "scoring": {
                "weights": {"advanced": {"wsjf": 0.5, "ice": 0.1, "technicalDebt": 0.3, "security": 0.1}},
                "thresholds": {"minScore": 15, "maxRisk": 0.7, "securityBoost": 2.5}
            }
        }
    
    def discover_value_items(self) -> List[Dict]:
        """Discover value items using basic file analysis"""
        items = []
        
        # Analyze git history for debt markers
        items.extend(self._analyze_git_history())
        
        # Search for technical debt markers in code
        items.extend(self._find_code_markers())
        
        # Check for security-related files
        items.extend(self._check_security_files())
        
        # Analyze test coverage gaps
        items.extend(self._analyze_test_gaps())
        
        # Check for documentation gaps
        items.extend(self._check_documentation())
        
        return items
    
    def _analyze_git_history(self) -> List[Dict]:
        """Analyze git history for improvement opportunities"""
        items = []
        
        try:
            # Get recent commits
            result = subprocess.run([
                "git", "log", "--since=30 days ago", "--oneline", "-20"
            ], capture_output=True, text=True, check=True)
            
            debt_keywords = ["todo", "fixme", "hack", "temp", "quick", "workaround"]
            
            for line in result.stdout.split('\n'):
                if line.strip():
                    commit_hash = line.split()[0]
                    commit_msg = line[len(commit_hash):].strip().lower()
                    
                    for keyword in debt_keywords:
                        if keyword in commit_msg:
                            items.append({
                                "id": f"git-debt-{commit_hash}",
                                "title": f"Address technical debt from commit {commit_hash[:7]}",
                                "type": "technical-debt",
                                "priority": "medium",
                                "source": "git-history",
                                "metadata": {"commit": commit_hash, "keyword": keyword}
                            })
                            break
        except subprocess.CalledProcessError:
            pass
            
        return items
    
    def _find_code_markers(self) -> List[Dict]:
        """Find TODO, FIXME, and other debt markers in code"""
        items = []
        markers = ["TODO", "FIXME", "XXX", "HACK", "BUG", "DEPRECATED"]
        
        for marker in markers:
            try:
                # Use grep to find markers in Python files
                result = subprocess.run([
                    "grep", "-r", "-n", "-i", marker, "src/", "--include=*.py"
                ], capture_output=True, text=True)
                
                for line in result.stdout.split('\n')[:3]:  # Limit per marker
                    if line.strip():
                        try:
                            file_path, line_num, content = line.split(':', 2)
                            items.append({
                                "id": f"marker-{marker.lower()}-{hash(line) % 10000}",
                                "title": f"Address {marker} in {os.path.basename(file_path)}:{line_num}",
                                "type": "technical-debt",
                                "priority": "high" if marker in ["FIXME", "BUG"] else "medium",
                                "source": "code-markers",
                                "metadata": {
                                    "file": file_path,
                                    "line": line_num,
                                    "marker": marker,
                                    "context": content.strip()[:80]
                                }
                            })
                        except ValueError:
                            continue
            except subprocess.CalledProcessError:
                continue
                
        return items
    
    def _check_security_files(self) -> List[Dict]:
        """Check for security-related improvements"""
        items = []
        
        # Check if security reports exist and have issues
        security_files = [
            "bandit-report.json",
            "security-report.json", 
            "safety-report.json"
        ]
        
        for file_path in security_files:
            if Path(file_path).exists():
                try:
                    with open(file_path) as f:
                        data = json.load(f)
                        
                    # Different report formats
                    issues = []
                    if "results" in data:  # Bandit format
                        issues = data["results"]
                    elif "vulnerabilities" in data:  # Safety format
                        issues = data["vulnerabilities"]
                    elif isinstance(data, list):  # Generic list format
                        issues = data
                    
                    if issues:
                        items.append({
                            "id": f"security-{file_path.replace('.', '-')}",
                            "title": f"Address {len(issues)} security issues in {file_path}",
                            "type": "security-fix",
                            "priority": "high",
                            "source": "security-scan",
                            "metadata": {"file": file_path, "issue_count": len(issues)}
                        })
                        
                except (json.JSONDecodeError, KeyError):
                    pass
        
        return items
    
    def _analyze_test_gaps(self) -> List[Dict]:
        """Identify test coverage gaps"""
        items = []
        
        # Count Python files vs test files
        src_files = list(Path("src").rglob("*.py")) if Path("src").exists() else []
        test_files = list(Path("tests").rglob("test_*.py")) if Path("tests").exists() else []
        
        if src_files:
            coverage_ratio = len(test_files) / len(src_files)
            
            if coverage_ratio < 0.8:  # Less than 80% test file coverage
                items.append({
                    "id": "test-coverage-gap",
                    "title": f"Improve test coverage: {len(test_files)}/{len(src_files)} files covered",
                    "type": "test-improvement",
                    "priority": "medium",
                    "source": "test-analysis",
                    "metadata": {
                        "src_files": len(src_files),
                        "test_files": len(test_files),
                        "coverage_ratio": round(coverage_ratio, 2)
                    }
                })
        
        return items
    
    def _check_documentation(self) -> List[Dict]:
        """Check for documentation improvements"""
        items = []
        
        # Check for missing or outdated documentation
        doc_files = {
            "README.md": "project overview",
            "CONTRIBUTING.md": "contribution guidelines",
            "CHANGELOG.md": "change history",
            "docs/API.md": "API documentation"
        }
        
        missing_docs = []
        for doc_file, description in doc_files.items():
            if not Path(doc_file).exists():
                missing_docs.append((doc_file, description))
        
        if missing_docs:
            for doc_file, description in missing_docs[:2]:  # Top 2 missing docs
                items.append({
                    "id": f"doc-missing-{doc_file.replace('/', '-').replace('.', '-')}",
                    "title": f"Create missing {description}: {doc_file}",
                    "type": "documentation",
                    "priority": "low",
                    "source": "doc-analysis",
                    "metadata": {"file": doc_file, "type": description}
                })
        
        return items
    
    def calculate_composite_score(self, item: Dict) -> float:
        """Calculate composite score for prioritization"""
        base_scores = {
            "security-fix": 95,
            "technical-debt": 60,
            "test-improvement": 45,
            "documentation": 25,
            "performance": 70,
            "dependency-update": 35
        }
        
        priority_multipliers = {
            "high": 1.5,
            "medium": 1.0,
            "low": 0.7
        }
        
        base_score = base_scores.get(item["type"], 50)
        priority_mult = priority_multipliers.get(item["priority"], 1.0)
        
        return round(base_score * priority_mult, 1)
    
    def score_and_prioritize(self, items: List[Dict]) -> List[Dict]:
        """Score and prioritize all items"""
        for item in items:
            item["scores"] = {
                "composite": self.calculate_composite_score(item)
            }
        
        return sorted(items, key=lambda x: x["scores"]["composite"], reverse=True)
    
    def generate_backlog_report(self, prioritized_items: List[Dict]) -> str:
        """Generate markdown backlog report"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report = f"""# 📊 Autonomous Value Backlog

Last Updated: {timestamp}
Repository: lexgraph-legal-rag (Advanced Maturity)
Discovery Engine: Simplified Value Discovery

## 🎯 Next Best Value Item
"""
        
        if prioritized_items:
            next_item = prioritized_items[0]
            report += f"""**[{next_item['id'].upper()}] {next_item['title']}**
- **Composite Score**: {next_item['scores']['composite']}
- **Type**: {next_item['type'].replace('-', ' ').title()}
- **Priority**: {next_item['priority'].title()}
- **Source**: {next_item['source'].replace('-', ' ').title()}

"""
        else:
            report += "No high-priority items discovered at this time.\n\n"
        
        report += """## 📋 Prioritized Backlog Items

| Rank | ID | Title | Score | Type | Priority |
|------|-----|--------|--------|------|----------|
"""
        
        for i, item in enumerate(prioritized_items[:15], 1):
            title_short = (item['title'][:45] + "...") if len(item['title']) > 45 else item['title']
            report += f"| {i} | {item['id']} | {title_short} | {item['scores']['composite']} | {item['type']} | {item['priority']} |\n"
        
        # Category breakdown
        type_counts = {}
        for item in prioritized_items:
            item_type = item['type']
            type_counts[item_type] = type_counts.get(item_type, 0) + 1
        
        report += f"""

## 📈 Discovery Summary
- **Total Items Found**: {len(prioritized_items)}
- **High Priority**: {len([i for i in prioritized_items if i['priority'] == 'high'])}
- **Medium Priority**: {len([i for i in prioritized_items if i['priority'] == 'medium'])}
- **Low Priority**: {len([i for i in prioritized_items if i['priority'] == 'low'])}

## 🏷️ Items by Category
"""
        
        for item_type, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
            report += f"- **{item_type.replace('-', ' ').title()}**: {count} items\n"
        
        report += f"""

## 🔍 Discovery Sources
- **Git History Analysis**: Scanned recent commits for debt indicators
- **Code Marker Search**: Found TODO, FIXME, and other debt markers
- **Security File Analysis**: Checked existing security reports
- **Test Coverage Analysis**: Evaluated test-to-source file ratios
- **Documentation Check**: Identified missing documentation files

---
*Generated by Terragon Simplified Value Discovery Engine*
*Next recommended scan: {(datetime.now() + timedelta(hours=4)).strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        return report
    
    def save_metrics(self, items: List[Dict]):
        """Save discovery metrics"""
        metrics = {
            "lastRun": datetime.now().isoformat(),
            "itemsDiscovered": len(items),
            "breakdown": {},
            "topScore": items[0]["scores"]["composite"] if items else 0,
            "averageScore": sum(item["scores"]["composite"] for item in items) / len(items) if items else 0
        }
        
        # Category breakdown
        for item in items:
            item_type = item["type"]
            if item_type not in metrics["breakdown"]:
                metrics["breakdown"][item_type] = {"count": 0, "avgScore": 0}
            metrics["breakdown"][item_type]["count"] += 1
        
        # Calculate average scores per category
        for item_type in metrics["breakdown"]:
            type_items = [item for item in items if item["type"] == item_type]
            avg_score = sum(item["scores"]["composite"] for item in type_items) / len(type_items)
            metrics["breakdown"][item_type]["avgScore"] = round(avg_score, 1)
        
        # Ensure directory exists
        self.metrics_file.parent.mkdir(exist_ok=True)
        
        with open(self.metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
    
    def run_discovery_cycle(self) -> Dict:
        """Run complete discovery cycle"""
        print("🔍 Starting simplified value discovery cycle...")
        
        # Discover items
        items = self.discover_value_items()
        print(f"   Discovered {len(items)} potential value items")
        
        if not items:
            print("   No items found - repository may be in excellent condition!")
            return {"discoveredItems": 0, "prioritizedItems": 0, "status": "no_items"}
        
        # Score and prioritize
        prioritized_items = self.score_and_prioritize(items)
        print(f"   Prioritized {len(prioritized_items)} items")
        
        # Generate backlog report
        backlog_report = self.generate_backlog_report(prioritized_items)
        
        # Save files
        with open(self.backlog_file, 'w') as f:
            f.write(backlog_report)
        
        self.save_metrics(prioritized_items)
        
        print(f"✅ Discovery cycle complete!")
        print(f"   Top item: {prioritized_items[0]['title'][:50]}... (Score: {prioritized_items[0]['scores']['composite']})")
        print(f"   Generated: BACKLOG.md")
        print(f"   Metrics: .terragon/value-metrics.json")
        
        return {
            "discoveredItems": len(items),
            "prioritizedItems": len(prioritized_items),
            "topItem": prioritized_items[0] if prioritized_items else None,
            "status": "success"
        }


def main():
    """Main execution function"""
    engine = SimpleValueDiscoveryEngine()
    result = engine.run_discovery_cycle()
    
    if result["status"] == "success":
        print(f"\n📊 Discovery Results Summary:")
        print(f"   Items discovered: {result['discoveredItems']}")
        print(f"   Items prioritized: {result['prioritizedItems']}")
        if result["topItem"]:
            print(f"   Highest value: {result['topItem']['title']}")
            print(f"   Score: {result['topItem']['scores']['composite']}")
    elif result["status"] == "no_items":
        print("\n🎉 No improvement items found - repository is in excellent condition!")
    
    print(f"\n📋 Review the generated BACKLOG.md for detailed prioritized tasks")


if __name__ == "__main__":
    main()