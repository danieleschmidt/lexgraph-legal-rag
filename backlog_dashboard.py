#!/usr/bin/env python3
"""
Autonomous Backlog Management Dashboard
Real-time monitoring and control interface
"""

import json
import yaml
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any


class BacklogDashboard:
    """Dashboard for monitoring autonomous backlog management."""
    
    def __init__(self, repo_path: str = "/root/repo"):
        self.repo_path = Path(repo_path)
        self.backlog_file = self.repo_path / "backlog.yml"
        self.status_dir = self.repo_path / "docs" / "status"
    
    def load_backlog(self) -> Dict[str, Any]:
        """Load current backlog data."""
        try:
            with open(self.backlog_file, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            st.error(f"Failed to load backlog: {e}")
            return {"backlog": [], "completed": [], "blocked": []}
    
    def load_status_reports(self) -> List[Dict[str, Any]]:
        """Load recent status reports."""
        reports = []
        if not self.status_dir.exists():
            return reports
        
        # Get the 10 most recent status files
        status_files = sorted(
            self.status_dir.glob("status_*.json"),
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )[:10]
        
        for file_path in status_files:
            try:
                with open(file_path, 'r') as f:
                    report = json.load(f)
                    report['filename'] = file_path.name
                    reports.append(report)
            except Exception as e:
                st.warning(f"Failed to load {file_path}: {e}")
        
        return reports
    
    def render_overview(self, backlog_data: Dict[str, Any]) -> None:
        """Render backlog overview metrics."""
        st.header("🤖 Autonomous Backlog Management System")
        
        # Key metrics
        active_items = [item for item in backlog_data.get('backlog', []) 
                       if item.get('status') in ['NEW', 'READY', 'DOING']]
        completed_items = backlog_data.get('completed', [])
        blocked_items = backlog_data.get('blocked', [])
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Active Items", len(active_items))
        
        with col2:
            st.metric("Completed", len(completed_items))
        
        with col3:
            st.metric("Blocked", len(blocked_items))
        
        with col4:
            # Calculate velocity (items completed in last 7 days)
            recent_completed = [
                item for item in completed_items
                if 'completed_at' in item and
                datetime.fromisoformat(item['completed_at'].replace('Z', '+00:00')) >
                datetime.now() - timedelta(days=7)
            ]
            st.metric("Weekly Velocity", len(recent_completed))
    
    def render_wsjf_distribution(self, backlog_data: Dict[str, Any]) -> None:
        """Render WSJF score distribution."""
        st.subheader("📊 WSJF Score Distribution")
        
        active_items = [item for item in backlog_data.get('backlog', []) 
                       if item.get('status') in ['NEW', 'READY', 'DOING']]
        
        if not active_items:
            st.info("No active items to display")
            return
        
        # Create dataframe
        df = pd.DataFrame([{
            'title': item.get('title', 'Unknown')[:30] + '...',
            'wsjf_score': item.get('wsjf_score', 0),
            'status': item.get('status', 'Unknown'),
            'type': item.get('type', 'Unknown'),
            'risk_tier': item.get('risk_tier', 'Unknown')
        } for item in active_items])
        
        # WSJF distribution chart
        fig = px.bar(
            df.sort_values('wsjf_score', ascending=True).tail(15),
            x='wsjf_score',
            y='title',
            color='risk_tier',
            title="Top 15 Items by WSJF Score",
            color_discrete_map={
                'high': '#ff4444',
                'medium': '#ffaa00', 
                'low': '#44ff44'
            }
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Priority items table
        st.subheader("🔥 High Priority Items (WSJF > 7)")
        high_priority = df[df['wsjf_score'] > 7].sort_values('wsjf_score', ascending=False)
        
        if not high_priority.empty:
            st.dataframe(high_priority, use_container_width=True)
        else:
            st.info("No high priority items currently")
    
    def render_progress_tracking(self, status_reports: List[Dict[str, Any]]) -> None:
        """Render progress tracking over time."""
        st.subheader("📈 Progress Tracking")
        
        if not status_reports:
            st.info("No status reports available")
            return
        
        # Sort reports by timestamp
        reports_df = pd.DataFrame(status_reports)
        reports_df['timestamp'] = pd.to_datetime(reports_df['timestamp'])
        reports_df = reports_df.sort_values('timestamp')
        
        # Progress chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=reports_df['timestamp'],
            y=reports_df['active_backlog_size'],
            mode='lines+markers',
            name='Active Items',
            line=dict(color='#ff6b6b')
        ))
        
        fig.add_trace(go.Scatter(
            x=reports_df['timestamp'],
            y=reports_df['completed_items'],
            mode='lines+markers',
            name='Completed Items',
            line=dict(color='#51cf66')
        ))
        
        fig.add_trace(go.Scatter(
            x=reports_df['timestamp'],
            y=reports_df['blocked_items'],
            mode='lines+markers',
            name='Blocked Items',
            line=dict(color='#ffd43b')
        ))
        
        fig.update_layout(
            title="Backlog Progress Over Time",
            xaxis_title="Time",
            yaxis_title="Number of Items",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Coverage progress
        if 'coverage_progress' in reports_df.columns:
            coverage_data = reports_df['coverage_progress'].apply(pd.Series)
            if not coverage_data.empty and 'current' in coverage_data:
                fig2 = go.Figure()
                
                fig2.add_trace(go.Scatter(
                    x=reports_df['timestamp'],
                    y=coverage_data['current'],
                    mode='lines+markers',
                    name='Current Coverage',
                    line=dict(color='#339af0')
                ))
                
                fig2.add_hline(
                    y=coverage_data['target'].iloc[0] if 'target' in coverage_data else 80,
                    line_dash="dash",
                    line_color="red",
                    annotation_text="Target Coverage"
                )
                
                fig2.update_layout(
                    title="Test Coverage Progress",
                    xaxis_title="Time",
                    yaxis_title="Coverage %",
                    yaxis=dict(range=[0, 100])
                )
                
                st.plotly_chart(fig2, use_container_width=True)
    
    def render_item_details(self, backlog_data: Dict[str, Any]) -> None:
        """Render detailed item information."""
        st.subheader("🔍 Item Details")
        
        all_items = backlog_data.get('backlog', [])
        
        if not all_items:
            st.info("No items to display")
            return
        
        # Item selector
        item_titles = [f"{item.get('id', 'unknown')} - {item.get('title', 'Unknown')}" 
                      for item in all_items]
        selected_title = st.selectbox("Select an item to view details:", item_titles)
        
        if selected_title:
            selected_idx = item_titles.index(selected_title)
            selected_item = all_items[selected_idx]
            
            # Display item details
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Title:**", selected_item.get('title', 'Unknown'))
                st.write("**Type:**", selected_item.get('type', 'Unknown'))
                st.write("**Status:**", selected_item.get('status', 'Unknown'))
                st.write("**Risk Tier:**", selected_item.get('risk_tier', 'Unknown'))
                st.write("**WSJF Score:**", selected_item.get('wsjf_score', 0))
            
            with col2:
                st.write("**Effort:**", selected_item.get('effort', 0))
                st.write("**Value:**", selected_item.get('value', 0))
                st.write("**Time Criticality:**", selected_item.get('time_criticality', 0))
                st.write("**Risk Reduction:**", selected_item.get('risk_reduction', 0))
                st.write("**Created:**", selected_item.get('created_at', 'Unknown'))
            
            st.write("**Description:**")
            st.write(selected_item.get('description', 'No description available'))
            
            if selected_item.get('acceptance_criteria'):
                st.write("**Acceptance Criteria:**")
                for criteria in selected_item['acceptance_criteria']:
                    st.write(f"- {criteria}")
            
            if selected_item.get('links'):
                st.write("**Related Files:**")
                for link in selected_item['links']:
                    st.write(f"- `{link}`")
    
    def render_system_health(self, backlog_data: Dict[str, Any], 
                           status_reports: List[Dict[str, Any]]) -> None:
        """Render system health indicators."""
        st.subheader("🏥 System Health")
        
        if not status_reports:
            st.warning("No recent status reports - system may not be running")
            return
        
        latest_report = status_reports[0]
        
        # Health indicators
        col1, col2, col3 = st.columns(3)
        
        with col1:
            health_status = latest_report.get('backlog_health', 'unknown')
            health_color = {
                'good': '🟢',
                'needs_attention': '🟡',
                'critical': '🔴'
            }.get(health_status, '⚪')
            
            st.metric(
                "Backlog Health",
                f"{health_color} {health_status.replace('_', ' ').title()}"
            )
        
        with col2:
            high_priority_count = latest_report.get('high_priority_items', 0)
            priority_status = "🔥" if high_priority_count > 5 else "✅"
            st.metric("High Priority Items", f"{priority_status} {high_priority_count}")
        
        with col3:
            avg_wsjf = latest_report.get('average_wsjf_score', 0)
            wsjf_status = "📈" if avg_wsjf > 5 else "📉"
            st.metric("Avg WSJF Score", f"{wsjf_status} {avg_wsjf}")
        
        # Last update time
        last_update = datetime.fromisoformat(latest_report['timestamp'].replace('Z', '+00:00'))
        time_since_update = datetime.now() - last_update
        
        if time_since_update.total_seconds() > 3600:  # More than 1 hour
            st.warning(f"⚠️ Last update was {time_since_update} ago - check if system is running")
        else:
            st.success(f"✅ Last updated {time_since_update} ago")
    
    def render_control_panel(self) -> None:
        """Render control panel for manual actions."""
        st.subheader("🎛️ Control Panel")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔍 Run Discovery", help="Discover new tasks from codebase"):
                with st.spinner("Running discovery..."):
                    # This would trigger discovery in a real implementation
                    st.success("Discovery completed! (simulated)")
        
        with col2:
            if st.button("⚡ Execute Next Item", help="Execute the highest priority ready item"):
                with st.spinner("Executing task..."):
                    # This would trigger execution in a real implementation
                    st.success("Task executed! (simulated)")
        
        with col3:
            if st.button("📊 Generate Report", help="Generate fresh status report"):
                with st.spinner("Generating report..."):
                    # This would generate a new report in a real implementation
                    st.success("Report generated! (simulated)")
    
    def run(self) -> None:
        """Run the dashboard."""
        st.set_page_config(
            page_title="Autonomous Backlog Management",
            page_icon="🤖",
            layout="wide"
        )
        
        # Load data
        backlog_data = self.load_backlog()
        status_reports = self.load_status_reports()
        
        # Render sections
        self.render_overview(backlog_data)
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 WSJF Analysis", "📈 Progress", "🔍 Item Details", 
            "🏥 System Health", "🎛️ Control Panel"
        ])
        
        with tab1:
            self.render_wsjf_distribution(backlog_data)
        
        with tab2:
            self.render_progress_tracking(status_reports)
        
        with tab3:
            self.render_item_details(backlog_data)
        
        with tab4:
            self.render_system_health(backlog_data, status_reports)
        
        with tab5:
            self.render_control_panel()
        
        # Auto-refresh
        st.sidebar.subheader("🔄 Auto Refresh")
        auto_refresh = st.sidebar.checkbox("Enable auto-refresh (30s)")
        
        if auto_refresh:
            import time
            time.sleep(30)
            st.experimental_rerun()


def main():
    """Main entry point."""
    dashboard = BacklogDashboard()
    dashboard.run()


if __name__ == '__main__':
    main()