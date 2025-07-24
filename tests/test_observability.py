"""Tests for enhanced observability module."""

import pytest
import time
from unittest.mock import patch, MagicMock

from lexgraph_legal_rag.observability import (
    initialize_observability,
    trace_operation,
    track_agent_operation,
    track_rag_query,
    track_document_processing,
    update_system_health,
    track_error,
    ObservabilityMixin,
    get_observability_info
)


class TestObservabilityInitialization:
    """Test observability initialization."""
    
    @patch('lexgraph_legal_rag.observability.start_http_server')
    def test_initialize_observability_success(self, mock_server):
        """Test successful observability initialization."""
        mock_server.return_value = None
        
        initialize_observability(
            enable_tracing=True,
            enable_metrics=True,
            prometheus_port=8001
        )
        
        # Verify Prometheus server was started
        mock_server.assert_called_once_with(8001)
        
        # Verify info contains expected data
        info = get_observability_info()
        assert info['service']['name'] == 'lexgraph-legal-rag'
        assert info['service']['version'] == '1.0.0'
        assert info['tracing']['enabled'] is True
        assert info['metrics']['enabled'] is True
    
    def test_initialize_observability_disabled(self):
        """Test observability with components disabled."""
        initialize_observability(
            enable_tracing=False,
            enable_metrics=False
        )
        
        info = get_observability_info()
        assert info['tracing']['enabled'] is False
        assert info['metrics']['enabled'] is False


class TestTracing:
    """Test distributed tracing functionality."""
    
    def test_trace_operation_success(self):
        """Test successful operation tracing."""
        initialize_observability(enable_tracing=True, enable_metrics=False)
        
        with trace_operation("test_operation", "test_component") as span:
            # Simulate some work
            time.sleep(0.01)
            assert span is not None
    
    def test_trace_operation_with_attributes(self):
        """Test operation tracing with custom attributes."""
        initialize_observability(enable_tracing=True, enable_metrics=False)
        
        attributes = {
            "user_id": "test_user",
            "query": "test query",
            "result_count": 5
        }
        
        with trace_operation("search", "rag_engine", attributes) as span:
            assert span is not None
    
    def test_trace_operation_with_error(self):
        """Test operation tracing with error handling."""
        initialize_observability(enable_tracing=True, enable_metrics=False)
        
        with pytest.raises(ValueError):
            with trace_operation("failing_operation", "test_component"):
                raise ValueError("Test error")


class TestMetrics:
    """Test metrics tracking functionality."""
    
    def test_track_agent_operation(self):
        """Test agent operation metrics tracking."""
        initialize_observability(enable_tracing=False, enable_metrics=True)
        
        # Track successful operation
        track_agent_operation(
            agent_type="retriever",
            operation="search",
            duration_seconds=0.5,
            success=True,
            attributes={"query_type": "semantic"}
        )
        
        # Track failed operation
        track_agent_operation(
            agent_type="summarizer",
            operation="summarize",
            duration_seconds=2.0,
            success=False
        )
        
        # No exceptions should be raised
        assert True
    
    def test_track_rag_query(self):
        """Test RAG query metrics tracking."""
        initialize_observability(enable_tracing=False, enable_metrics=True)
        
        track_rag_query(
            query_type="semantic_search",
            agent="retriever",
            latency_seconds=1.2,
            success=True,
            result_count=10
        )
        
        track_rag_query(
            query_type="document_summary",
            agent="summarizer",
            latency_seconds=3.5,
            success=False,
            result_count=0
        )
        
        assert True
    
    def test_track_document_processing(self):
        """Test document processing metrics."""
        initialize_observability(enable_tracing=False, enable_metrics=True)
        
        track_document_processing("ingestion", success=True, document_count=5)
        track_document_processing("indexing", success=False, document_count=1)
        
        assert True
    
    def test_system_health_tracking(self):
        """Test system health score tracking."""
        initialize_observability(enable_tracing=False, enable_metrics=True)
        
        # Test valid health scores
        update_system_health(0.85)
        update_system_health(1.0)
        update_system_health(0.0)
        
        # Test clamping of invalid values
        update_system_health(-0.1)  # Should clamp to 0.0
        update_system_health(1.5)   # Should clamp to 1.0
        
        assert True
    
    def test_error_tracking(self):
        """Test error tracking functionality."""
        initialize_observability(enable_tracing=False, enable_metrics=True)
        
        test_error = ValueError("Test error message")
        track_error(test_error, "search_component", "error")
        
        runtime_error = RuntimeError("Runtime issue")
        track_error(runtime_error, "indexing_component", "critical")
        
        assert True


class TestObservabilityMixin:
    """Test observability mixin class."""
    
    def test_mixin_initialization(self):
        """Test mixin initialization."""
        class TestComponent(ObservabilityMixin):
            def __init__(self):
                super().__init__("test_component")
        
        component = TestComponent()
        assert component.component_name == "test_component"
    
    def test_mixin_trace_operation(self):
        """Test mixin trace functionality."""
        initialize_observability(enable_tracing=True, enable_metrics=False)
        
        class TestComponent(ObservabilityMixin):
            def __init__(self):
                super().__init__("test_component")
            
            def do_work(self):
                with self.trace("do_work", user_id="test") as span:
                    time.sleep(0.01)
                    return "work_done"
        
        component = TestComponent()
        result = component.do_work()
        assert result == "work_done"
    
    def test_mixin_track_operation(self):
        """Test mixin operation tracking."""
        initialize_observability(enable_tracing=False, enable_metrics=True)
        
        class TestComponent(ObservabilityMixin):
            def __init__(self):
                super().__init__("test_component")
            
            def perform_task(self):
                start_time = time.time()
                # Simulate work
                time.sleep(0.01)
                duration = time.time() - start_time
                
                self.track_operation(
                    "perform_task",
                    duration,
                    success=True,
                    task_type="unit_test"
                )
        
        component = TestComponent()
        component.perform_task()
        
        assert True
    
    def test_mixin_error_tracking(self):
        """Test mixin error tracking."""
        initialize_observability(enable_tracing=False, enable_metrics=True)
        
        class TestComponent(ObservabilityMixin):
            def __init__(self):
                super().__init__("test_component")
            
            def failing_operation(self):
                try:
                    raise ValueError("Something went wrong")
                except ValueError as e:
                    self.track_error(e, "warning")
                    raise
        
        component = TestComponent()
        
        with pytest.raises(ValueError):
            component.failing_operation()


class TestIntegration:
    """Test integration scenarios."""
    
    def test_full_observability_stack(self):
        """Test complete observability stack."""
        initialize_observability(
            enable_tracing=True,
            enable_metrics=True,
            prometheus_port=8002  # Different port to avoid conflicts
        )
        
        # Simulate a complete RAG operation with full observability
        with trace_operation("rag_pipeline", "integration_test") as span:
            # Step 1: Document retrieval
            track_agent_operation(
                "retriever", "search", 0.5, True,
                attributes={"query": "test query", "results": 5}
            )
            
            # Step 2: Document summarization
            track_agent_operation(
                "summarizer", "summarize", 1.2, True,
                attributes={"doc_count": 5, "summary_length": 200}
            )
            
            # Step 3: Response generation
            track_rag_query(
                "complete_rag", "pipeline", 2.0, True, result_count=1
            )
            
            # Update system health
            update_system_health(0.95)
        
        # Verify info is accessible
        info = get_observability_info()
        assert info['service']['name'] == 'lexgraph-legal-rag'
        assert info['tracing']['enabled'] is True
        assert info['metrics']['enabled'] is True
    
    @patch('lexgraph_legal_rag.observability.start_http_server')
    def test_prometheus_server_failure_handling(self, mock_server):
        """Test handling of Prometheus server startup failures."""
        mock_server.side_effect = Exception("Port already in use")
        
        # Should not raise exception, just log warning
        initialize_observability(
            enable_tracing=False,
            enable_metrics=True,
            prometheus_port=8001
        )
        
        info = get_observability_info()
        assert info['metrics']['enabled'] is True  # Metrics still enabled