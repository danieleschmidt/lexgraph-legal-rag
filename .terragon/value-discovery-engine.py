#!/usr/bin/env python3
"""
Terragon Autonomous Value Discovery Engine
Continuously discovers, scores, and prioritizes high-value work items
"""

import json
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re
# import yaml  # Simplified version without external dependencies


class ValueDiscoveryEngine:
    """Advanced value discovery and prioritization engine"""
    
    def __init__(self, config_path: str = ".terragon/config.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.metrics_file = Path(".terragon/value-metrics.json")
        self.backlog_file = Path("BACKLOG.md")
        
    def _load_config(self) -> Dict:
        """Load configuration from YAML file"""
        try:
            # Simple JSON-based config for compatibility
            if self.config_path.with_suffix('.json').exists():
                with open(self.config_path.with_suffix('.json')) as f:
                    return json.load(f)
            else:
                return self._default_config()
        except FileNotFoundError:
            return self._default_config()
    
    def _default_config(self) -> Dict:
        """Default configuration for advanced repositories"""
        return {
            "scoring": {
                "weights": {"advanced": {"wsjf": 0.5, "ice": 0.1, "technicalDebt": 0.3, "security": 0.1}},
                "thresholds": {"minScore": 15, "maxRisk": 0.7, "securityBoost": 2.5}
            },
            "discovery": {
                "sources": ["gitHistory", "staticAnalysis", "securityScans", "performanceMetrics"]
            }
        }
    
    def discover_value_items(self) -> List[Dict]:
        """Comprehensive value item discovery from multiple sources"""
        items = []
        
        # Git history analysis
        items.extend(self._analyze_git_history())
        
        # Static analysis findings  
        items.extend(self._analyze_code_quality())
        
        # Security vulnerability scanning
        items.extend(self._scan_security_issues())
        
        # Performance optimization opportunities
        items.extend(self._identify_performance_improvements())
        
        # Dependency management
        items.extend(self._check_dependency_updates())
        
        # Technical debt extraction
        items.extend(self._extract_technical_debt())
        
        return items
    
    def _analyze_git_history(self) -> List[Dict]:
        """Extract TODO, FIXME, and technical debt markers from git history"""
        items = []
        
        try:
            # Get recent commits with debt indicators
            result = subprocess.run([
                "git", "log", "--since=30 days ago", "--grep=TODO\\|FIXME\\|HACK\\|XXX", 
                "--pretty=format:%h|%s|%an|%ad", "--date=short"
            ], capture_output=True, text=True, check=True)
            
            for line in result.stdout.split('\n'):
                if line.strip():
                    hash_val, subject, author, date = line.split('|', 3)
                    items.append({
                        "id": f"git-{hash_val}",
                        "title": f"Address technical debt in commit: {subject[:50]}",
                        "type": "technical-debt",
                        "priority": "medium",
                        "source": "git-history",
                        "metadata": {"commit": hash_val, "author": author, "date": date}
                    })
        except subprocess.CalledProcessError:
            pass
            
        return items
    
    def _analyze_code_quality(self) -> List[Dict]:
        """Run static analysis to identify code quality issues"""
        items = []
        
        # Run ruff for code quality issues
        try:
            result = subprocess.run([
                "ruff", "check", "src/", "--output-format=json"
            ], capture_output=True, text=True)
            
            if result.stdout:
                issues = json.loads(result.stdout)
                for issue in issues[:10]:  # Limit to top 10 issues
                    items.append({
                        "id": f"ruff-{issue.get('code', 'unknown')}",
                        "title": f"Fix {issue.get('code')}: {issue.get('message', '')[:60]}",
                        "type": "code-quality",
                        "priority": "low" if issue.get('code', '').startswith('W') else "medium",
                        "source": "static-analysis",
                        "metadata": {"file": issue.get('filename'), "line": issue.get('location', {}).get('row')}
                    })
        except (subprocess.CalledProcessError, json.JSONDecodeError):
            pass
            
        return items
    
    def _scan_security_issues(self) -> List[Dict]:
        """Identify security vulnerabilities and compliance issues"""
        items = []
        
        # Check for security scan results
        if Path("bandit-report.json").exists():
            try:
                with open("bandit-report.json") as f:
                    report = json.load(f)
                    
                for issue in report.get("results", [])[:5]:  # Top 5 security issues
                    items.append({
                        "id": f"security-{issue.get('test_id')}",
                        "title": f"Security: {issue.get('issue_text', '')[:60]}",
                        "type": "security-fix",
                        "priority": "high" if issue.get('issue_severity') == 'HIGH' else "medium",
                        "source": "security-scan",
                        "metadata": {"severity": issue.get('issue_severity'), "confidence": issue.get('issue_confidence')}
                    })
            except (json.JSONDecodeError, KeyError):
                pass
        
        return items
    
    def _identify_performance_improvements(self) -> List[Dict]:
        """Identify performance optimization opportunities"""
        items = []
        
        # Look for performance-related TODOs in code
        try:
            result = subprocess.run([
                "grep", "-r", "-n", "-i", "performance\\|slow\\|optimize\\|bottleneck", 
                "src/", "--include=*.py"
            ], capture_output=True, text=True)
            
            for line in result.stdout.split('\n')[:5]:  # Top 5 findings
                if line.strip():
                    file_path, line_num, content = line.split(':', 2)
                    items.append({
                        "id": f"perf-{hash(line)}",
                        "title": f"Performance optimization in {Path(file_path).name}",
                        "type": "performance",
                        "priority": "medium",
                        "source": "code-analysis",
                        "metadata": {"file": file_path, "line": line_num, "context": content.strip()[:100]}
                    })
        except (subprocess.CalledProcessError, ValueError):
            pass
            
        return items
    
    def _check_dependency_updates(self) -> List[Dict]:
        """Check for outdated dependencies"""
        items = []
        
        # Check Python dependencies
        try:
            result = subprocess.run([
                "pip", "list", "--outdated", "--format=json"
            ], capture_output=True, text=True)
            
            if result.stdout:
                outdated = json.loads(result.stdout)
                for pkg in outdated[:5]:  # Top 5 outdated packages
                    items.append({
                        "id": f"deps-{pkg['name']}",
                        "title": f"Update {pkg['name']} from {pkg['version']} to {pkg['latest_version']}",
                        "type": "dependency-update",
                        "priority": "low",
                        "source": "dependency-check",
                        "metadata": {"package": pkg['name'], "current": pkg['version'], "latest": pkg['latest_version']}
                    })
        except (subprocess.CalledProcessError, json.JSONDecodeError):
            pass
            
        return items
    
    def _extract_technical_debt(self) -> List[Dict]:
        """Extract technical debt markers from code"""
        items = []
        debt_patterns = ["TODO", "FIXME", "XXX", "HACK", "DEPRECATED"]
        
        for pattern in debt_patterns:
            try:
                result = subprocess.run([
                    "grep", "-r", "-n", pattern, "src/", "--include=*.py"
                ], capture_output=True, text=True)
                
                for line in result.stdout.split('\n')[:3]:  # Top 3 per pattern
                    if line.strip():
                        file_path, line_num, content = line.split(':', 2)
                        items.append({
                            "id": f"debt-{pattern.lower()}-{hash(line)}",
                            "title": f"Address {pattern} in {Path(file_path).name}:{line_num}",
                            "type": "technical-debt",
                            "priority": "high" if pattern in ["FIXME", "XXX"] else "medium",
                            "source": "code-markers",
                            "metadata": {"file": file_path, "line": line_num, "marker": pattern, "context": content.strip()[:100]}
                        })
            except (subprocess.CalledProcessError, ValueError):
                continue
                
        return items
    
    def calculate_wsjf_score(self, item: Dict) -> float:
        """Calculate Weighted Shortest Job First score"""
        # User/Business Value (1-10)
        business_value = {
            "security-fix": 9,
            "performance": 7,
            "technical-debt": 5,
            "code-quality": 4,
            "dependency-update": 3
        }.get(item["type"], 5)
        
        # Time Criticality (1-10)
        time_criticality = {
            "high": 8,
            "medium": 5,
            "low": 2
        }.get(item["priority"], 5)
        
        # Risk Reduction/Opportunity Enablement (1-10)
        risk_reduction = {
            "security-fix": 9,
            "performance": 6,
            "technical-debt": 7,
            "code-quality": 4,
            "dependency-update": 3
        }.get(item["type"], 5)
        
        cost_of_delay = business_value + time_criticality + risk_reduction
        
        # Job Size estimation (story points, 1-13)
        job_size = {
            "security-fix": 5,
            "performance": 8,
            "technical-debt": 5,
            "code-quality": 2,
            "dependency-update": 3
        }.get(item["type"], 5)
        
        return cost_of_delay / job_size if job_size > 0 else 0
    
    def calculate_ice_score(self, item: Dict) -> float:
        """Calculate Impact, Confidence, Ease score"""
        impact = {
            "security-fix": 9,
            "performance": 8,
            "technical-debt": 6,
            "code-quality": 5,
            "dependency-update": 4
        }.get(item["type"], 5)
        
        confidence = {
            "high": 9,
            "medium": 7,
            "low": 5
        }.get(item["priority"], 7)
        
        ease = {
            "security-fix": 6,
            "performance": 4,
            "technical-debt": 7,
            "code-quality": 8,
            "dependency-update": 9
        }.get(item["type"], 6)
        
        return impact * confidence * ease
    
    def calculate_technical_debt_score(self, item: Dict) -> float:
        """Calculate technical debt impact score"""
        if item["type"] != "technical-debt":
            return 0
            
        base_score = 50
        
        # Boost for high-impact markers
        marker = item.get("metadata", {}).get("marker", "")
        if marker in ["FIXME", "XXX"]:
            base_score *= 1.5
        elif marker == "HACK":
            base_score *= 1.3
            
        return base_score
    
    def calculate_composite_score(self, item: Dict) -> float:
        """Calculate final composite score with adaptive weighting"""
        weights = self.config["scoring"]["weights"]["advanced"]
        
        wsjf = self.calculate_wsjf_score(item)
        ice = self.calculate_ice_score(item) / 100  # Normalize to 0-10 scale
        tech_debt = self.calculate_technical_debt_score(item) / 10  # Normalize
        
        composite = (
            weights["wsjf"] * wsjf +
            weights["ice"] * ice +
            weights["technicalDebt"] * tech_debt
        )
        
        # Apply security boost
        if item["type"] == "security-fix":
            composite *= self.config["scoring"]["thresholds"]["securityBoost"]
            
        return round(composite, 2)
    
    def score_and_prioritize(self, items: List[Dict]) -> List[Dict]:
        """Score all items and return prioritized list"""
        for item in items:
            item["scores"] = {
                "wsjf": self.calculate_wsjf_score(item),
                "ice": self.calculate_ice_score(item),
                "technicalDebt": self.calculate_technical_debt_score(item),
                "composite": self.calculate_composite_score(item)
            }
        
        # Sort by composite score descending
        return sorted(items, key=lambda x: x["scores"]["composite"], reverse=True)
    
    def select_next_best_value(self, scored_items: List[Dict]) -> Optional[Dict]:
        """Select the next highest value item for execution"""
        min_score = self.config["scoring"]["thresholds"]["minScore"]
        
        for item in scored_items:
            if item["scores"]["composite"] >= min_score:
                return item
                
        return None
    
    def update_metrics(self, executed_item: Dict, execution_result: Dict):
        """Update value delivery metrics"""
        metrics = self._load_metrics()
        
        execution_record = {
            "timestamp": datetime.now().isoformat(),
            "itemId": executed_item["id"],
            "title": executed_item["title"],
            "type": executed_item["type"],
            "scores": executed_item["scores"],
            "executionResult": execution_result,
            "actualEffort": execution_result.get("timeSpent", 0),
            "success": execution_result.get("success", False)
        }
        
        metrics["executionHistory"].append(execution_record)
        
        # Update aggregate metrics
        if execution_result.get("success"):
            metrics["successRate"] = self._calculate_success_rate(metrics["executionHistory"])
            metrics["averageScore"] = self._calculate_average_score(metrics["executionHistory"])
            
        self._save_metrics(metrics)
    
    def _load_metrics(self) -> Dict:
        """Load existing metrics or create new structure"""
        if self.metrics_file.exists():
            with open(self.metrics_file) as f:
                return json.load(f)
        
        return {
            "executionHistory": [],
            "successRate": 0.0,
            "averageScore": 0.0,
            "lastRun": None
        }
    
    def _save_metrics(self, metrics: Dict):
        """Save metrics to file"""
        self.metrics_file.parent.mkdir(exist_ok=True)
        with open(self.metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
    
    def _calculate_success_rate(self, history: List[Dict]) -> float:
        """Calculate success rate from execution history"""
        if not history:
            return 0.0
        
        successful = sum(1 for item in history if item.get("executionResult", {}).get("success", False))
        return round(successful / len(history), 3)
    
    def _calculate_average_score(self, history: List[Dict]) -> float:
        """Calculate average composite score"""
        if not history:
            return 0.0
            
        scores = [item["scores"]["composite"] for item in history if "scores" in item]
        return round(sum(scores) / len(scores), 2) if scores else 0.0
    
    def generate_backlog_report(self, prioritized_items: List[Dict]) -> str:
        """Generate comprehensive backlog report"""
        timestamp = datetime.now().isoformat()
        metrics = self._load_metrics()
        
        report = f"""# 📊 Autonomous Value Backlog

Last Updated: {timestamp}
Repository: lexgraph-legal-rag (Advanced Maturity)

## 🎯 Next Best Value Item
"""
        
        if prioritized_items:
            next_item = prioritized_items[0]
            report += f"""**[{next_item['id'].upper()}] {next_item['title']}**
- **Composite Score**: {next_item['scores']['composite']}
- **WSJF**: {next_item['scores']['wsjf']:.1f} | **ICE**: {next_item['scores']['ice']:.0f} | **Tech Debt**: {next_item['scores']['technicalDebt']:.0f}
- **Type**: {next_item['type'].replace('-', ' ').title()}
- **Priority**: {next_item['priority'].title()}
- **Source**: {next_item['source'].replace('-', ' ').title()}

"""
        
        report += """## 📋 Top Priority Backlog Items

| Rank | ID | Title | Score | Type | Priority | Source |
|------|-----|--------|--------|------|----------|--------|
"""
        
        for i, item in enumerate(prioritized_items[:10], 1):
            title_short = item['title'][:50] + "..." if len(item['title']) > 50 else item['title']
            report += f"| {i} | {item['id']} | {title_short} | {item['scores']['composite']} | {item['type']} | {item['priority']} | {item['source']} |\n"
        
        report += f"""

## 📈 Value Discovery Metrics
- **Total Items Discovered**: {len(prioritized_items)}
- **High Priority Items**: {len([i for i in prioritized_items if i['priority'] == 'high'])}
- **Security Items**: {len([i for i in prioritized_items if i['type'] == 'security-fix'])}
- **Technical Debt Items**: {len([i for i in prioritized_items if i['type'] == 'technical-debt'])}
- **Performance Items**: {len([i for i in prioritized_items if i['type'] == 'performance'])}

## 🔄 Execution History Summary
- **Success Rate**: {metrics.get('successRate', 0):.1%}
- **Average Score**: {metrics.get('averageScore', 0):.1f}
- **Total Executions**: {len(metrics.get('executionHistory', []))}

## 🎛️ Discovery Sources Breakdown
"""
        
        source_counts = {}
        for item in prioritized_items:
            source = item['source']
            source_counts[source] = source_counts.get(source, 0) + 1
            
        for source, count in sorted(source_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(prioritized_items)) * 100 if prioritized_items else 0
            report += f"- **{source.replace('-', ' ').title()}**: {count} items ({percentage:.1f}%)\n"
        
        report += f"""

---
*Generated by Terragon Autonomous Value Discovery Engine*
*Next scan scheduled: {(datetime.now() + timedelta(hours=1)).strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        return report
    
    def run_discovery_cycle(self) -> Dict:
        """Execute complete discovery and prioritization cycle"""
        print("🔍 Starting value discovery cycle...")
        
        # Discover items from all sources
        items = self.discover_value_items()
        print(f"   Discovered {len(items)} potential value items")
        
        # Score and prioritize
        prioritized_items = self.score_and_prioritize(items)
        print(f"   Prioritized {len(prioritized_items)} items")
        
        # Select next best value
        next_item = self.select_next_best_value(prioritized_items)
        
        # Generate backlog report
        backlog_report = self.generate_backlog_report(prioritized_items)
        
        # Save backlog to file
        with open(self.backlog_file, 'w') as f:
            f.write(backlog_report)
        
        # Update last run timestamp
        metrics = self._load_metrics()
        metrics["lastRun"] = datetime.now().isoformat()
        self._save_metrics(metrics)
        
        result = {
            "discoveredItems": len(items),
            "prioritizedItems": len(prioritized_items),
            "nextBestValue": next_item,
            "backlogGenerated": True,
            "timestamp": datetime.now().isoformat()
        }
        
        print(f"✅ Discovery cycle complete. Next best value: {next_item['id'] if next_item else 'None'}")
        return result


def main():
    """Main execution function"""
    engine = ValueDiscoveryEngine()
    result = engine.run_discovery_cycle()
    
    print(f"\n📊 Discovery Results:")
    print(f"   Items discovered: {result['discoveredItems']}")
    print(f"   Items prioritized: {result['prioritizedItems']}")
    
    if result['nextBestValue']:
        item = result['nextBestValue']
        print(f"   Next best value: {item['title']} (Score: {item['scores']['composite']})")
    else:
        print("   No items meet minimum score threshold")
        
    print(f"   Backlog updated: BACKLOG.md")
    print(f"   Metrics saved: .terragon/value-metrics.json")


if __name__ == "__main__":
    main()