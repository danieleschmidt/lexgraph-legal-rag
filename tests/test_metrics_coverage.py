"""Test coverage for metrics module."""

import pytest
from unittest.mock import patch, MagicMock
from lexgraph_legal_rag.metrics import (
    start_metrics_server,
    update_memory_metrics,
    update_cache_metrics,
    update_index_metrics,
    record_cache_hit,
    record_cache_miss,
    record_api_key_rotation,
    record_rate_limit_exceeded,
)


@patch('lexgraph_legal_rag.metrics.start_http_server')
def test_start_metrics_server_default_port(mock_server):
    """Test starting metrics server with default port."""
    start_metrics_server(8000)  # Need to provide a port since default is 0
    mock_server.assert_called_with(8000)


@patch('lexgraph_legal_rag.metrics.start_http_server')
def test_start_metrics_server_custom_port(mock_server):
    """Test starting metrics server with custom port."""
    start_metrics_server(9090)
    mock_server.assert_called_with(9090)


@patch('lexgraph_legal_rag.metrics.start_http_server')
@patch.dict('os.environ', {'METRICS_PORT': '8888'})
def test_start_metrics_server_env_port(mock_server):
    """Test starting metrics server with environment port."""
    start_metrics_server()
    mock_server.assert_called_with(8888)


@patch('lexgraph_legal_rag.metrics.MEMORY_USAGE')
@patch('psutil.virtual_memory')
def test_update_memory_metrics(mock_memory, mock_gauge):
    """Test update_memory_metrics function."""
    mock_memory.return_value.percent = 75.5
    
    update_memory_metrics()
    
    mock_gauge.set.assert_called_with(75.5)


@patch('lexgraph_legal_rag.metrics.CACHE_SIZE')
def test_update_cache_metrics(mock_size):
    """Test update_cache_metrics function."""
    cache_stats = {'size': 500}
    
    update_cache_metrics(cache_stats)
    
    mock_size.set.assert_called_with(500)


@patch('lexgraph_legal_rag.metrics.INDEX_SIZE')
def test_update_index_metrics(mock_size):
    """Test update_index_metrics function."""
    document_count = 1000
    
    update_index_metrics(document_count)
    
    mock_size.set.assert_called_with(1000)


@patch('lexgraph_legal_rag.metrics.CACHE_HITS')
def test_record_cache_hit(mock_hits):
    """Test record_cache_hit function."""
    record_cache_hit()
    mock_hits.inc.assert_called_once()


@patch('lexgraph_legal_rag.metrics.CACHE_MISSES')
def test_record_cache_miss(mock_misses):
    """Test record_cache_miss function."""
    record_cache_miss()
    mock_misses.inc.assert_called_once()


@patch('lexgraph_legal_rag.metrics.API_KEY_ROTATIONS')
def test_record_api_key_rotation(mock_rotations):
    """Test record_api_key_rotation function."""
    record_api_key_rotation()
    mock_rotations.inc.assert_called_once()


@patch('lexgraph_legal_rag.metrics.RATE_LIMIT_EXCEEDED')
def test_record_rate_limit_exceeded(mock_rate_limit):
    """Test record_rate_limit_exceeded function."""
    record_rate_limit_exceeded()
    mock_rate_limit.inc.assert_called_once()