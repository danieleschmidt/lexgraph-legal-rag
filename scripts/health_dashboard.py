#!/usr/bin/env python3
"""
Repository Health Dashboard Generator

This script generates a comprehensive health dashboard for the LexGraph Legal RAG project.
It creates HTML and JSON reports showing project status, metrics, and recommendations.
"""

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LexGraph Legal RAG - Project Health Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #f8fafc;
            color: #334155;
            line-height: 1.6;
        }
        .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
        .header { 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 2rem;
            border-radius: 12px;
            margin-bottom: 2rem;
            text-align: center;
        }
        .header h1 { font-size: 2.5rem; margin-bottom: 0.5rem; }
        .header p { font-size: 1.2rem; opacity: 0.9; }
        
        .metrics-grid { 
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 1.5rem;
            margin-bottom: 2rem;
        }
        
        .metric-card {
            background: white;
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            border-left: 4px solid;
        }
        
        .metric-card.excellent { border-left-color: #10b981; }
        .metric-card.good { border-left-color: #f59e0b; }
        .metric-card.warning { border-left-color: #ef4444; }
        
        .metric-value {
            font-size: 2rem;
            font-weight: bold;
            margin-bottom: 0.5rem;
        }
        
        .metric-label {
            color: #6b7280;
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        
        .progress-bar {
            background: #e5e7eb;
            border-radius: 4px;
            height: 8px;
            margin-top: 0.5rem;
            overflow: hidden;
        }
        
        .progress-fill {
            height: 100%;
            border-radius: 4px;
            transition: width 0.3s ease;
        }
        
        .progress-fill.excellent { background: #10b981; }
        .progress-fill.good { background: #f59e0b; }
        .progress-fill.warning { background: #ef4444; }
        
        .dashboard-section {
            background: white;
            border-radius: 12px;
            padding: 2rem;
            margin-bottom: 2rem;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }
        
        .section-title {
            font-size: 1.5rem;
            font-weight: bold;
            margin-bottom: 1rem;
            color: #1f2937;
        }
        
        .alerts {
            display: flex;
            flex-direction: column;
            gap: 1rem;
        }
        
        .alert {
            padding: 1rem;
            border-radius: 8px;
            border-left: 4px solid;
        }
        
        .alert.warning {
            background: #fef3c7;
            border-left-color: #f59e0b;
            color: #92400e;
        }
        
        .alert.info {
            background: #dbeafe;
            border-left-color: #3b82f6;
            color: #1e40af;
        }
        
        .alert.success {
            background: #d1fae5;
            border-left-color: #10b981;
            color: #065f46;
        }
        
        .features-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
        }
        
        .feature-item {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.5rem;
        }
        
        .feature-status {
            width: 12px;
            height: 12px;
            border-radius: 50%;
        }
        
        .feature-status.enabled { background: #10b981; }
        .feature-status.disabled { background: #ef4444; }
        
        .chart-container {
            position: relative;
            height: 300px;
            margin: 1rem 0;
        }
        
        .recommendations {
            background: #f8fafc;
            border-radius: 8px;
            padding: 1.5rem;
        }
        
        .recommendations ul {
            list-style: none;
            padding-left: 0;
        }
        
        .recommendations li {
            padding: 0.5rem 0;
            border-bottom: 1px solid #e5e7eb;
            position: relative;
            padding-left: 1.5rem;
        }
        
        .recommendations li:before {
            content: "→";
            position: absolute;
            left: 0;
            color: #6366f1;
            font-weight: bold;
        }
        
        .footer {
            text-align: center;
            color: #6b7280;
            font-size: 0.9rem;
            margin-top: 2rem;
        }
        
        @media (max-width: 768px) {
            .container { padding: 10px; }
            .header h1 { font-size: 2rem; }
            .metrics-grid { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🏛️ LexGraph Legal RAG</h1>
            <p>Project Health Dashboard</p>
            <p style="font-size: 1rem; margin-top: 1rem;">
                Last Updated: {timestamp}
            </p>
        </div>
        
        <div class="metrics-grid">
            {metrics_cards}
        </div>
        
        <div class="dashboard-section">
            <h2 class="section-title">🚨 Active Alerts</h2>
            <div class="alerts">
                {alerts}
            </div>
        </div>
        
        <div class="dashboard-section">
            <h2 class="section-title">📊 Health Trends</h2>
            <div class="chart-container">
                <canvas id="healthChart"></canvas>
            </div>
        </div>
        
        <div class="dashboard-section">
            <h2 class="section-title">🎯 Feature Status</h2>
            <div class="features-grid">
                {features}
            </div>
        </div>
        
        <div class="dashboard-section">
            <h2 class="section-title">💡 Recommendations</h2>
            <div class="recommendations">
                <ul>
                    {recommendations}
                </ul>
            </div>
        </div>
        
        <div class="footer">
            <p>Generated automatically by LexGraph Legal RAG Health Dashboard</p>
            <p>🤖 Powered by Terragon Labs SDLC Automation</p>
        </div>
    </div>
    
    <script>
        // Health trends chart
        const ctx = document.getElementById('healthChart').getContext('2d');
        new Chart(ctx, {{
            type: 'line',
            data: {{
                labels: {chart_labels},
                datasets: [{{
                    label: 'Overall Health',
                    data: {chart_data},
                    borderColor: '#667eea',
                    backgroundColor: 'rgba(102, 126, 234, 0.1)',
                    tension: 0.4,
                    fill: true
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                scales: {{
                    y: {{
                        beginAtZero: true,
                        max: 100,
                        ticks: {{
                            callback: function(value) {{
                                return value + '%';
                            }}
                        }}
                    }}
                }},
                plugins: {{
                    legend: {{
                        display: false
                    }}
                }}
            }}
        }});
    </script>
</body>
</html>
"""


class HealthDashboard:
    """Generate comprehensive project health dashboard."""
    
    def __init__(self, repo_path: str = "."):
        self.repo_path = Path(repo_path)
        
    def generate_dashboard(self) -> Dict[str, Any]:
        """Generate complete health dashboard."""
        print("🏥 Generating project health dashboard...")
        
        # Load project metrics
        metrics_file = self.repo_path / ".github" / "project-metrics.json"
        project_data = {}
        
        if metrics_file.exists():
            try:
                with open(metrics_file) as f:
                    project_data = json.load(f)
            except Exception as e:
                print(f"⚠️  Could not load project metrics: {e}")
        
        # Generate dashboard data
        dashboard_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "overall_health": self.calculate_overall_health(project_data),
            "metrics": self.get_key_metrics(project_data),
            "alerts": self.get_active_alerts(project_data),
            "features": self.get_feature_status(project_data),
            "recommendations": self.get_recommendations(project_data),
            "trends": self.get_health_trends(project_data)
        }
        
        # Generate HTML dashboard
        html_content = self.generate_html_dashboard(dashboard_data)
        
        # Save files
        html_file = self.repo_path / "dashboard.html"
        json_file = self.repo_path / "health-dashboard.json"
        
        with open(html_file, 'w') as f:
            f.write(html_content)
        
        with open(json_file, 'w') as f:
            json.dump(dashboard_data, f, indent=2, sort_keys=True)
        
        print(f"✅ Dashboard generated:")
        print(f"  📄 HTML: {html_file}")
        print(f"  📊 JSON: {json_file}")
        
        return dashboard_data
    
    def calculate_overall_health(self, project_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall project health score."""
        metrics = project_data.get("metrics", {})
        
        # Weight different aspects of health
        weights = {
            "sdlc_completeness": 0.25,
            "security_score": 0.20,
            "test_coverage": 0.15,
            "automation_coverage": 0.15,
            "code_quality": 0.10,
            "documentation_health": 0.10,
            "dependency_health": 0.05
        }
        
        weighted_score = 0
        total_weight = 0
        
        for metric, weight in weights.items():
            if metric in metrics:
                weighted_score += metrics[metric] * weight
                total_weight += weight
        
        overall_score = weighted_score / total_weight if total_weight > 0 else 0
        
        # Determine health level
        if overall_score >= 90:
            health_level = "excellent"
        elif overall_score >= 75:
            health_level = "good"
        elif overall_score >= 60:
            health_level = "warning"
        else:
            health_level = "critical"
        
        return {
            "score": round(overall_score, 1),
            "level": health_level,
            "last_calculated": datetime.now(timezone.utc).isoformat()
        }
    
    def get_key_metrics(self, project_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get key metrics for dashboard cards."""
        metrics = project_data.get("metrics", {})
        
        key_metrics = [
            {
                "name": "SDLC Completeness",
                "value": f"{metrics.get('sdlc_completeness', 0)}%",
                "score": metrics.get('sdlc_completeness', 0),
                "description": "Overall SDLC implementation status"
            },
            {
                "name": "Test Coverage",
                "value": f"{metrics.get('test_coverage', 0)}%",
                "score": metrics.get('test_coverage', 0),
                "description": "Code coverage by automated tests"
            },
            {
                "name": "Security Score",
                "value": f"{metrics.get('security_score', 0)}%",
                "score": metrics.get('security_score', 0),
                "description": "Security posture and vulnerability status"
            },
            {
                "name": "Automation",
                "value": f"{metrics.get('automation_coverage', 0)}%",
                "score": metrics.get('automation_coverage', 0),
                "description": "CI/CD and automation coverage"
            },
            {
                "name": "Code Quality",
                "value": f"{metrics.get('code_quality', 0)}%",
                "score": metrics.get('code_quality', 0),
                "description": "Code quality metrics and standards"
            },
            {
                "name": "Documentation",
                "value": f"{metrics.get('documentation_health', 0)}%",
                "score": metrics.get('documentation_health', 0),
                "description": "Documentation completeness and quality"
            }
        ]
        
        # Add health level to each metric
        for metric in key_metrics:
            score = metric["score"]
            if score >= 90:
                metric["level"] = "excellent"
            elif score >= 75:
                metric["level"] = "good"
            else:
                metric["level"] = "warning"
        
        return key_metrics
    
    def get_active_alerts(self, project_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get active alerts and issues."""
        alerts = []
        
        # Check project data for existing alerts
        project_alerts = project_data.get("alerts", [])
        for alert in project_alerts:
            alerts.append({
                "type": alert.get("type", "info"),
                "message": alert.get("message", ""),
                "priority": alert.get("priority", "low"),
                "action": alert.get("action_required", "")
            })
        
        # Generate alerts based on metrics
        metrics = project_data.get("metrics", {})
        
        if metrics.get("test_coverage", 100) < 80:
            alerts.append({
                "type": "warning",
                "message": f"Test coverage is {metrics.get('test_coverage', 0)}%, below recommended 80%",
                "priority": "medium",
                "action": "Add more unit and integration tests"
            })
        
        if metrics.get("security_score", 100) < 85:
            alerts.append({
                "type": "warning",
                "message": "Security score below recommended threshold",
                "priority": "high", 
                "action": "Review and fix security vulnerabilities"
            })
        
        if not alerts:
            alerts.append({
                "type": "success",
                "message": "All systems operational - no active alerts",
                "priority": "info",
                "action": "Continue monitoring"
            })
        
        return alerts
    
    def get_feature_status(self, project_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get status of key features."""
        features = project_data.get("features", {})
        
        feature_list = []
        
        # CI/CD Features
        ci_cd = features.get("ci_cd", {})
        feature_list.append({
            "name": "Continuous Integration",
            "enabled": ci_cd.get("enabled", False),
            "category": "CI/CD"
        })
        
        # Testing Features
        testing = features.get("testing", {})
        for test_type, enabled in testing.items():
            feature_list.append({
                "name": test_type.replace("_", " ").title(),
                "enabled": enabled,
                "category": "Testing"
            })
        
        # Security Features
        security = features.get("security", {})
        for security_type, enabled in security.items():
            feature_list.append({
                "name": security_type.replace("_", " ").title(),
                "enabled": enabled,
                "category": "Security"
            })
        
        # Development Features
        development = features.get("development", {})
        for dev_feature, enabled in development.items():
            feature_list.append({
                "name": dev_feature.replace("_", " ").title(),
                "enabled": enabled,
                "category": "Development"
            })
        
        return feature_list
    
    def get_recommendations(self, project_data: Dict[str, Any]) -> List[str]:
        """Get recommendations for improvement."""
        recommendations = []
        metrics = project_data.get("metrics", {})
        
        # Existing recommendations from project data
        existing_recs = project_data.get("recommendations", [])
        recommendations.extend(existing_recs)
        
        # Generate dynamic recommendations based on metrics
        if metrics.get("test_coverage", 0) < 90:
            recommendations.append("Increase test coverage to 90% by adding unit tests for core business logic")
        
        if metrics.get("documentation_health", 0) < 95:
            recommendations.append("Improve documentation completeness with API examples and troubleshooting guides")
        
        if metrics.get("performance_score", 100) < 85:
            recommendations.append("Optimize application performance through caching and query optimization")
        
        if not recommendations:
            recommendations.append("Project is in excellent health - continue current practices")
        
        return recommendations[:5]  # Limit to top 5 recommendations
    
    def get_health_trends(self, project_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get health trend data for charts."""
        # For now, generate sample trend data
        # In a real implementation, this would come from historical data
        
        labels = ["Week 1", "Week 2", "Week 3", "Week 4", "Current"]
        data = [85, 88, 90, 92, self.calculate_overall_health(project_data)["score"]]
        
        return {
            "labels": labels,
            "data": data
        }
    
    def generate_html_dashboard(self, dashboard_data: Dict[str, Any]) -> str:
        """Generate HTML dashboard content."""
        # Generate metrics cards
        metrics_cards = []
        for metric in dashboard_data["metrics"]:
            card_html = f"""
            <div class="metric-card {metric['level']}">
                <div class="metric-value">{metric['value']}</div>
                <div class="metric-label">{metric['name']}</div>
                <div class="progress-bar">
                    <div class="progress-fill {metric['level']}" style="width: {metric['score']}%"></div>
                </div>
                <p style="margin-top: 0.5rem; font-size: 0.8rem; color: #6b7280;">
                    {metric['description']}
                </p>
            </div>
            """
            metrics_cards.append(card_html)
        
        # Generate alerts
        alerts_html = []
        for alert in dashboard_data["alerts"]:
            alert_html = f"""
            <div class="alert {alert['type']}">
                <strong>{alert['message']}</strong>
                <br>
                <small>Action: {alert['action']}</small>
            </div>
            """
            alerts_html.append(alert_html)
        
        # Generate features
        features_html = []
        for feature in dashboard_data["features"]:
            status_class = "enabled" if feature["enabled"] else "disabled"
            features_html.append(f"""
            <div class="feature-item">
                <div class="feature-status {status_class}"></div>
                <span>{feature['name']}</span>
            </div>
            """)
        
        # Generate recommendations
        recommendations_html = [f"<li>{rec}</li>" for rec in dashboard_data["recommendations"]]
        
        # Format timestamp
        timestamp = datetime.fromisoformat(dashboard_data["timestamp"].replace('Z', '+00:00'))
        formatted_timestamp = timestamp.strftime("%Y-%m-%d %H:%M UTC")
        
        # Fill template
        return HTML_TEMPLATE.format(
            timestamp=formatted_timestamp,
            metrics_cards="\n".join(metrics_cards),
            alerts="\n".join(alerts_html),
            features="\n".join(features_html),
            recommendations="\n".join(recommendations_html),
            chart_labels=json.dumps(dashboard_data["trends"]["labels"]),
            chart_data=json.dumps(dashboard_data["trends"]["data"])
        )


def main():
    """Main entry point."""
    dashboard = HealthDashboard()
    
    try:
        result = dashboard.generate_dashboard()
        
        print(f"\n🎯 Overall Health Score: {result['overall_health']['score']}%")
        print(f"📊 Health Level: {result['overall_health']['level'].title()}")
        print(f"🚨 Active Alerts: {len(result['alerts'])}")
        print(f"💡 Recommendations: {len(result['recommendations'])}")
        
        print("\n🎉 Health dashboard generated successfully!")
        
    except Exception as e:
        print(f"❌ Dashboard generation failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())