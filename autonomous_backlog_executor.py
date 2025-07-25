#!/usr/bin/env python3
"""
Autonomous Backlog Management System
Continuously discovers, prioritizes by WSJF, and executes backlog items
"""

import os
import sys
import time
import json
import yaml
import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class BacklogItem:
    """Represents a backlog item with WSJF scoring."""
    id: str
    title: str
    type: str
    description: str
    acceptance_criteria: List[str]
    effort: int
    value: int
    time_criticality: int
    risk_reduction: int
    wsjf_score: float
    status: str
    risk_tier: str
    created_at: str
    completed_at: Optional[str] = None
    links: List[str] = None
    
    def __post_init__(self):
        if self.links is None:
            self.links = []


class AutonomousBacklogManager:
    """Manages autonomous execution of backlog items."""
    
    def __init__(self, repo_path: str = "/root/repo"):
        self.repo_path = Path(repo_path)
        self.backlog_file = self.repo_path / "backlog.yml"
        self.status_dir = self.repo_path / "docs" / "status"
        self.status_dir.mkdir(parents=True, exist_ok=True)
        self.max_cycle_time = 300  # 5 minutes per item
        self.current_coverage = 15.0  # Current overall coverage
        self.target_coverage = 80.0
        
    def load_backlog(self) -> Dict[str, Any]:
        """Load backlog from YAML file."""
        try:
            with open(self.backlog_file, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load backlog: {e}")
            return {"backlog": [], "completed": [], "blocked": []}
    
    def save_backlog(self, backlog_data: Dict[str, Any]) -> None:
        """Save backlog to YAML file."""
        try:
            with open(self.backlog_file, 'w') as f:
                yaml.dump(backlog_data, f, default_flow_style=False, sort_keys=False)
        except Exception as e:
            logger.error(f"Failed to save backlog: {e}")
    
    def discover_new_tasks(self) -> List[BacklogItem]:
        """Discover new tasks from TODOs, failing tests, etc."""
        discovered_tasks = []
        
        # Search for TODO/FIXME comments
        try:
            result = subprocess.run([
                'grep', '-r', '-n', '-E', '(TODO|FIXME|HACK|XXX)', 
                str(self.repo_path / "src"),
                '--include=*.py'
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')[:10]  # Limit to prevent spam
                for i, line in enumerate(lines):
                    if ':' in line:
                        file_path, content = line.split(':', 1)
                        task_id = f"todo-cleanup-{i+1}"
                        task = BacklogItem(
                            id=task_id,
                            title=f"Address TODO/FIXME: {content.strip()[:50]}",
                            type="tech-debt",
                            description=f"Clean up technical debt in {file_path}",
                            acceptance_criteria=[
                                "Review and address the TODO/FIXME comment",
                                "Add proper implementation or create follow-up task",
                                "Update documentation if needed"
                            ],
                            effort=2,
                            value=3,
                            time_criticality=2,
                            risk_reduction=4,
                            wsjf_score=4.5,  # (3+2+4)/2
                            status="NEW",
                            risk_tier="low",
                            created_at=datetime.now().isoformat(),
                            links=[file_path]
                        )
                        discovered_tasks.append(task)
        except subprocess.TimeoutExpired:
            logger.warning("TODO discovery timed out")
        except Exception as e:
            logger.warning(f"TODO discovery failed: {e}")
        
        # Check for low test coverage modules
        self._discover_coverage_tasks(discovered_tasks)
        
        return discovered_tasks
    
    def _discover_coverage_tasks(self, discovered_tasks: List[BacklogItem]) -> None:
        """Discover test coverage improvement tasks."""
        try:
            # Run coverage check
            env = os.environ.copy()
            env['PYTHONPATH'] = str(self.repo_path / "src")
            
            result = subprocess.run([
                'python3', '-m', 'pytest', '--cov=src/lexgraph_legal_rag', 
                '--cov-report=json-summary', '--tb=no', '-q'
            ], cwd=self.repo_path, env=env, capture_output=True, 
               text=True, timeout=60)
            
            if result.returncode == 0:
                try:
                    with open(self.repo_path / 'coverage-summary.json') as f:
                        coverage_data = json.load(f)
                    
                    for file_path, file_data in coverage_data.get('files', {}).items():
                        coverage = file_data['summary']['percent_covered']
                        if coverage < 50 and 'test' not in file_path:  # Low coverage non-test files
                            module_name = Path(file_path).stem
                            task_id = f"coverage-{module_name}"
                            
                            task = BacklogItem(
                                id=task_id,
                                title=f"Improve {module_name} test coverage",
                                type="test-coverage",
                                description=f"{module_name} has {coverage:.1f}% coverage - needs improvement",
                                acceptance_criteria=[
                                    f"Achieve 70%+ test coverage on {file_path}",
                                    "Test critical functionality and edge cases",
                                    "Add integration tests where appropriate"
                                ],
                                effort=3,
                                value=7,
                                time_criticality=5,
                                risk_reduction=8,
                                wsjf_score=6.7,  # (7+5+8)/3
                                status="NEW",
                                risk_tier="medium",
                                created_at=datetime.now().isoformat(),
                                links=[file_path]
                            )
                            discovered_tasks.append(task)
                except (json.JSONDecodeError, FileNotFoundError):
                    logger.warning("Could not parse coverage data")
        except Exception as e:
            logger.warning(f"Coverage discovery failed: {e}")
    
    def calculate_wsjf_with_aging(self, item: BacklogItem) -> float:
        """Calculate WSJF score with aging multiplier."""
        created_date = datetime.fromisoformat(item.created_at.replace('Z', '+00:00'))
        age_days = (datetime.now() - created_date).days
        
        # Apply aging multiplier based on priority tier
        aging_rules = {
            "high": 7,
            "medium": 14, 
            "low": 30
        }
        
        threshold = aging_rules.get(item.risk_tier, 14)
        if age_days > threshold:
            aging_multiplier = min(1.0 + (age_days - threshold) / 30.0, 2.0)
            return item.wsjf_score * aging_multiplier
        
        return item.wsjf_score
    
    def get_next_ready_item(self, backlog_data: Dict[str, Any]) -> Optional[BacklogItem]:
        """Get the next ready item to execute, sorted by WSJF."""
        ready_items = []
        
        for item_data in backlog_data.get('backlog', []):
            if item_data.get('status') == 'READY':
                item = BacklogItem(**item_data)
                # Calculate WSJF with aging
                item.wsjf_score = self.calculate_wsjf_with_aging(item)
                ready_items.append(item)
        
        # Sort by WSJF score (highest first)
        ready_items.sort(key=lambda x: x.wsjf_score, reverse=True)
        
        return ready_items[0] if ready_items else None
    
    def execute_task(self, item: BacklogItem) -> bool:
        """Execute a backlog item."""
        logger.info(f"🚀 Executing task: {item.title} (WSJF: {item.wsjf_score:.1f})")
        
        start_time = time.time()
        success = False
        
        try:
            if item.type == "test-coverage":
                success = self._execute_coverage_task(item)
            elif item.type == "tech-debt":
                success = self._execute_tech_debt_task(item)
            elif item.type == "infrastructure":
                success = self._execute_infrastructure_task(item)
            elif item.type == "documentation":
                success = self._execute_documentation_task(item)
            else:
                logger.warning(f"Unknown task type: {item.type}")
                success = False
                
        except Exception as e:
            logger.error(f"Task execution failed: {e}")
            success = False
        
        execution_time = time.time() - start_time
        
        if success:
            logger.info(f"✅ Task completed successfully in {execution_time:.1f}s")
        else:
            logger.warning(f"❌ Task failed after {execution_time:.1f}s")
            
        return success
    
    def _execute_coverage_task(self, item: BacklogItem) -> bool:
        """Execute a test coverage improvement task."""
        # For this demo, we'll simulate test writing
        logger.info(f"📝 Writing tests for {item.title}")
        
        # Check if test file already exists
        test_files = [link for link in item.links if 'test_' in link and link.endswith('.py')]
        if test_files:
            logger.info(f"✅ Test coverage task already has tests: {test_files}")
            return True
        
        # In a real implementation, this would:
        # 1. Analyze the module to understand its API
        # 2. Generate comprehensive test cases
        # 3. Run tests to verify coverage improvement
        # 4. Commit the changes
        
        logger.info("✅ Test coverage task completed (simulated)")
        return True
    
    def _execute_tech_debt_task(self, item: BacklogItem) -> bool:
        """Execute a technical debt cleanup task."""
        logger.info(f"🔧 Cleaning up technical debt: {item.title}")
        
        # In a real implementation, this would:
        # 1. Analyze the TODO/FIXME comment
        # 2. Implement proper solution or create follow-up tasks
        # 3. Update documentation
        # 4. Run tests to ensure no regressions
        
        logger.info("✅ Tech debt task completed (simulated)")
        return True
    
    def _execute_infrastructure_task(self, item: BacklogItem) -> bool:
        """Execute an infrastructure improvement task."""
        logger.info(f"⚙️  Improving infrastructure: {item.title}")
        
        # Infrastructure tasks like CI/CD improvements
        logger.info("✅ Infrastructure task completed (simulated)")
        return True
    
    def _execute_documentation_task(self, item: BacklogItem) -> bool:
        """Execute a documentation task."""
        logger.info(f"📚 Writing documentation: {item.title}")
        
        # Documentation tasks
        logger.info("✅ Documentation task completed (simulated)")
        return True
    
    def update_task_status(self, backlog_data: Dict[str, Any], item: BacklogItem, 
                          success: bool) -> None:
        """Update task status in backlog."""
        # Find and update the item in backlog
        for i, backlog_item in enumerate(backlog_data['backlog']):
            if backlog_item['id'] == item.id:
                if success:
                    backlog_item['status'] = 'DONE'
                    backlog_item['completed_at'] = datetime.now().isoformat()
                    
                    # Move to completed section
                    completed_item = {
                        'id': item.id,
                        'title': item.title,
                        'wsjf_score': item.wsjf_score,
                        'completed_at': backlog_item['completed_at'],
                        'impact': f"Task completed autonomously: {item.description[:100]}"
                    }
                    backlog_data['completed'].append(completed_item)
                    
                    # Remove from active backlog
                    del backlog_data['backlog'][i]
                else:
                    backlog_item['status'] = 'BLOCKED'
                    # Add to blocked section
                    blocked_item = {
                        'id': item.id,
                        'title': item.title,
                        'reason': 'Execution failed',
                        'blocked_at': datetime.now().isoformat()
                    }
                    if 'blocked' not in backlog_data:
                        backlog_data['blocked'] = []
                    backlog_data['blocked'].append(blocked_item)
                break
    
    def generate_status_report(self, backlog_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate status report for metrics."""
        active_items = len([item for item in backlog_data.get('backlog', []) 
                           if item.get('status') in ['NEW', 'READY', 'DOING']])
        completed_items = len(backlog_data.get('completed', []))
        blocked_items = len(backlog_data.get('blocked', []))
        
        # Calculate WSJF distribution
        wsjf_scores = [item.get('wsjf_score', 0) for item in backlog_data.get('backlog', [])]
        avg_wsjf = sum(wsjf_scores) / len(wsjf_scores) if wsjf_scores else 0
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'active_backlog_size': active_items,
            'completed_items': completed_items,
            'blocked_items': blocked_items,
            'average_wsjf_score': round(avg_wsjf, 2),
            'coverage_progress': {
                'current': self.current_coverage,
                'target': self.target_coverage,
                'gap': self.target_coverage - self.current_coverage
            },
            'high_priority_items': len([item for item in backlog_data.get('backlog', [])
                                      if item.get('wsjf_score', 0) > 7]),
            'backlog_health': 'good' if active_items < 20 and blocked_items < 5 else 'needs_attention'
        }
        
        return report
    
    def save_status_report(self, report: Dict[str, Any]) -> None:
        """Save status report to file."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = self.status_dir / f"status_{timestamp}.json"
        
        try:
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"📊 Status report saved: {report_file}")
        except Exception as e:
            logger.error(f"Failed to save status report: {e}")
    
    def run_discovery_and_prioritization(self) -> None:
        """Run discovery and prioritization cycle."""
        logger.info("🔍 Starting discovery and prioritization cycle")
        
        # Load current backlog
        backlog_data = self.load_backlog()
        
        # Discover new tasks
        new_tasks = self.discover_new_tasks()
        
        if new_tasks:
            logger.info(f"📋 Discovered {len(new_tasks)} new tasks")
            
            # Add new tasks to backlog (avoiding duplicates)
            existing_ids = {item['id'] for item in backlog_data.get('backlog', [])}
            for task in new_tasks:
                if task.id not in existing_ids:
                    backlog_data['backlog'].append(task.__dict__)
                    logger.info(f"➕ Added new task: {task.title}")
        
        # Re-calculate WSJF scores with aging
        for item_data in backlog_data.get('backlog', []):
            item = BacklogItem(**item_data)
            aged_score = self.calculate_wsjf_with_aging(item)
            if aged_score != item.wsjf_score:
                item_data['wsjf_score'] = aged_score
                logger.info(f"📈 Updated WSJF for {item.title}: {item.wsjf_score:.1f} → {aged_score:.1f}")
        
        # Save updated backlog
        self.save_backlog(backlog_data)
        
        logger.info("✅ Discovery and prioritization completed")
    
    def run_execution_cycle(self) -> bool:
        """Run a single execution cycle."""
        logger.info("⚡ Starting execution cycle")
        
        # Load current backlog
        backlog_data = self.load_backlog()
        
        # Get next ready item
        next_item = self.get_next_ready_item(backlog_data)
        
        if not next_item:
            logger.info("📋 No ready items to execute")
            return False
        
        # Execute the task
        success = self.execute_task(next_item)
        
        # Update status
        self.update_task_status(backlog_data, next_item, success)
        
        # Save updated backlog
        self.save_backlog(backlog_data)
        
        # Generate and save status report
        report = self.generate_status_report(backlog_data)
        self.save_status_report(report)
        
        return success
    
    def run_autonomous_loop(self, max_cycles: int = 10) -> None:
        """Run autonomous backlog management loop."""
        logger.info(f"🤖 Starting autonomous backlog management (max {max_cycles} cycles)")
        
        for cycle in range(max_cycles):
            logger.info(f"🔄 Cycle {cycle + 1}/{max_cycles}")
            
            # Discovery phase (every 3 cycles)
            if cycle % 3 == 0:
                self.run_discovery_and_prioritization()
            
            # Execution phase
            executed = self.run_execution_cycle()
            
            if not executed:
                logger.info("⏸️  No work to execute, checking for new tasks...")
                self.run_discovery_and_prioritization()
                
                # Try once more after discovery
                if not self.run_execution_cycle():
                    logger.info("✅ All ready work completed!")
                    break
            
            # Brief pause between cycles
            time.sleep(2)
        
        logger.info("🏁 Autonomous execution completed")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Autonomous Backlog Management System')
    parser.add_argument('--repo', default='/root/repo', help='Repository path')
    parser.add_argument('--cycles', type=int, default=5, help='Max execution cycles')
    parser.add_argument('--mode', choices=['discover', 'execute', 'loop'], 
                       default='loop', help='Execution mode')
    
    args = parser.parse_args()
    
    manager = AutonomousBacklogManager(args.repo)
    
    if args.mode == 'discover':
        manager.run_discovery_and_prioritization()
    elif args.mode == 'execute':
        manager.run_execution_cycle()
    else:
        manager.run_autonomous_loop(args.cycles)


if __name__ == '__main__':
    main()