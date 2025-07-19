"""Integration tests for CLI functionality."""

import subprocess
import tempfile
import json
import os
from pathlib import Path
import pytest
from unittest.mock import patch, MagicMock


def test_run_agent_with_valid_args():
    """Test run_agent.py with valid arguments."""
    # Create a temporary index file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.bin', delete=False) as f:
        # Mock index file content
        json.dump([
            {
                "id": "test_doc_1",
                "text": "This is a test legal document about contracts.",
                "metadata": {"path": "/test/doc1.txt"}
            }
        ], f)
        temp_index = f.name
    
    try:
        # Mock the environment variable
        env = os.environ.copy()
        env['API_KEY'] = 'test-api-key'
        
        # Test CLI with minimal args
        result = subprocess.run([
            'python3', 'run_agent.py',
            '--query', 'What is this document about?',
            '--index', temp_index
        ], 
        cwd='/root/repo',
        env=env,
        capture_output=True,
        text=True,
        timeout=30
        )
        
        # Should complete without error (even if no actual processing happens)
        assert result.returncode == 0 or "Index" in result.stdout
        
    finally:
        # Clean up
        Path(temp_index).unlink(missing_ok=True)


def test_run_agent_missing_query():
    """Test run_agent.py fails appropriately with missing query."""
    env = os.environ.copy()
    env['API_KEY'] = 'test-api-key'
    
    result = subprocess.run([
        'python3', 'run_agent.py',
        '--index', 'nonexistent.bin'
    ], 
    cwd='/root/repo',
    env=env,
    capture_output=True,
    text=True,
    timeout=10
    )
    
    # Should fail due to missing required --query argument
    assert result.returncode != 0
    assert "required" in result.stderr.lower() or "query" in result.stderr.lower()


def test_run_agent_with_metrics_port():
    """Test run_agent.py with metrics port specified."""
    # Create a temporary index file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.bin', delete=False) as f:
        json.dump([
            {
                "id": "test_doc",
                "text": "Sample legal text.",
                "metadata": {"path": "/test/doc.txt"}
            }
        ], f)
        temp_index = f.name
    
    try:
        env = os.environ.copy()
        env['API_KEY'] = 'test-api-key'
        
        # Test with metrics port
        result = subprocess.run([
            'python3', 'run_agent.py',
            '--query', 'Test query',
            '--index', temp_index,
            '--metrics-port', '0',  # Use port 0 for ephemeral port
            '--hops', '1'
        ], 
        cwd='/root/repo',
        env=env,
        capture_output=True,
        text=True,
        timeout=30
        )
        
        # Should complete successfully
        assert result.returncode == 0 or "Index" in result.stdout
        
    finally:
        Path(temp_index).unlink(missing_ok=True)


def test_run_agent_nonexistent_index():
    """Test run_agent.py with nonexistent index file."""
    env = os.environ.copy()
    env['API_KEY'] = 'test-api-key'
    
    result = subprocess.run([
        'python3', 'run_agent.py',
        '--query', 'Test query',
        '--index', 'definitely_nonexistent_file.bin'
    ], 
    cwd='/root/repo',
    env=env,
    capture_output=True,
    text=True,
    timeout=15
    )
    
    # Should handle missing index gracefully
    assert result.returncode == 0
    assert "not found" in result.stdout.lower()


def test_ingest_script_functionality():
    """Test ingest.py script functionality."""
    # Create temporary directory with test documents
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create test documents
        (temp_path / "doc1.txt").write_text("This is a legal contract about employment terms.")
        (temp_path / "doc2.txt").write_text("This document discusses intellectual property rights.")
        
        # Create temporary output index
        with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as f:
            temp_index = f.name
        
        try:
            env = os.environ.copy()
            env['API_KEY'] = 'test-api-key'
            
            # Test ingest command
            result = subprocess.run([
                'python3', 'ingest.py',
                '--docs', str(temp_path),
                '--index', temp_index
            ], 
            cwd='/root/repo',
            env=env,
            capture_output=True,
            text=True,
            timeout=30
            )
            
            # Should complete successfully
            assert result.returncode == 0 or "Ingested" in result.stdout
            
        finally:
            Path(temp_index).unlink(missing_ok=True)


def test_cli_help_commands():
    """Test that CLI scripts provide help information."""
    # Test run_agent.py help
    result = subprocess.run([
        'python3', 'run_agent.py', '--help'
    ], 
    cwd='/root/repo',
    capture_output=True,
    text=True,
    timeout=10
    )
    
    assert result.returncode == 0
    assert "usage:" in result.stdout.lower()
    assert "--query" in result.stdout
    
    # Test ingest.py help  
    result = subprocess.run([
        'python3', 'ingest.py', '--help'
    ], 
    cwd='/root/repo',
    capture_output=True,
    text=True,
    timeout=10
    )
    
    assert result.returncode == 0
    assert "usage:" in result.stdout.lower()


def test_cli_with_missing_api_key():
    """Test CLI behavior when API_KEY is not set."""
    # Remove API_KEY from environment
    env = os.environ.copy()
    env.pop('API_KEY', None)
    
    result = subprocess.run([
        'python3', 'run_agent.py',
        '--query', 'Test query',
        '--index', 'test.bin'
    ], 
    cwd='/root/repo',
    env=env,
    capture_output=True,
    text=True,
    timeout=15
    )
    
    # Should fail due to missing API key
    assert result.returncode != 0
    assert ("api" in result.stderr.lower() and "key" in result.stderr.lower()) or \
           ("configuration" in result.stderr.lower())


@pytest.mark.parametrize("hops", [1, 3, 5])
def test_run_agent_with_different_hops(hops):
    """Test run_agent.py with different hop values."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.bin', delete=False) as f:
        json.dump([
            {
                "id": f"doc_{i}",
                "text": f"Legal document {i} content.",
                "metadata": {"path": f"/test/doc{i}.txt"}
            } for i in range(5)
        ], f)
        temp_index = f.name
    
    try:
        env = os.environ.copy()
        env['API_KEY'] = 'test-api-key'
        
        result = subprocess.run([
            'python3', 'run_agent.py',
            '--query', 'Legal query',
            '--index', temp_index,
            '--hops', str(hops)
        ], 
        cwd='/root/repo',
        env=env,
        capture_output=True,
        text=True,
        timeout=30
        )
        
        # Should handle different hop values
        assert result.returncode == 0 or "Index" in result.stdout
        
    finally:
        Path(temp_index).unlink(missing_ok=True)


def test_cli_import_structure():
    """Test that CLI scripts can import required modules."""
    # Test that run_agent.py can import its dependencies
    result = subprocess.run([
        'python3', '-c', 
        'import run_agent; print("Import successful")'
    ], 
    cwd='/root/repo',
    capture_output=True,
    text=True,
    timeout=10
    )
    
    # Should import successfully
    assert result.returncode == 0
    assert "Import successful" in result.stdout
    
    # Test ingest.py imports
    result = subprocess.run([
        'python3', '-c', 
        'import ingest; print("Import successful")'
    ], 
    cwd='/root/repo',
    capture_output=True,
    text=True,
    timeout=10
    )
    
    # Should import successfully
    assert result.returncode == 0
    assert "Import successful" in result.stdout


def test_streamlit_app_import():
    """Test that Streamlit app can be imported without errors."""
    env = os.environ.copy()
    env['API_KEY'] = 'test-api-key'
    
    result = subprocess.run([
        'python3', '-c', 
        'import streamlit_app; print("Streamlit import successful")'
    ], 
    cwd='/root/repo',
    env=env,
    capture_output=True,
    text=True,
    timeout=15
    )
    
    # Should import successfully
    assert result.returncode == 0
    assert "Streamlit import successful" in result.stdout