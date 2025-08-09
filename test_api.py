#!/usr/bin/env python3
"""Test the API endpoints."""

import os
import requests
import time
from pathlib import Path

# Set API key for testing
os.environ["API_KEY"] = "test-api-key-12345"

# Start the API server in background for testing
import subprocess
import sys
from threading import Thread
import uvicorn
from src.lexgraph_legal_rag.api import create_api

def start_server():
    """Start the API server."""
    app = create_api(test_mode=True, api_key="test-api-key-12345")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")

def test_api():
    """Test API endpoints."""
    base_url = "http://127.0.0.1:8000"
    headers = {"X-API-Key": "test-api-key-12345"}
    
    print("🧪 Testing LexGraph Legal RAG API")
    print("=" * 40)
    
    # Test health endpoint
    print("1. Testing /health endpoint...")
    try:
        response = requests.get(f"{base_url}/health")
        print(f"✅ Health check: {response.status_code} - {response.json()}")
    except Exception as e:
        print(f"❌ Health check failed: {e}")
    
    # Test readiness endpoint
    print("\n2. Testing /ready endpoint...")
    try:
        response = requests.get(f"{base_url}/ready")
        print(f"✅ Readiness check: {response.status_code} - {response.json()}")
    except Exception as e:
        print(f"❌ Readiness check failed: {e}")
    
    # Test ping endpoint
    print("\n3. Testing /ping endpoint...")
    try:
        response = requests.get(f"{base_url}/ping", headers=headers)
        print(f"✅ Ping: {response.status_code} - {response.json()}")
    except Exception as e:
        print(f"❌ Ping failed: {e}")
    
    # Test add endpoint
    print("\n4. Testing /add endpoint...")
    try:
        response = requests.get(f"{base_url}/add?a=15&b=27", headers=headers)
        print(f"✅ Add: {response.status_code} - {response.json()}")
    except Exception as e:
        print(f"❌ Add failed: {e}")
    
    # Test version endpoint
    print("\n5. Testing /version endpoint...")
    try:
        response = requests.get(f"{base_url}/version", headers=headers)
        print(f"✅ Version: {response.status_code} - {response.json()}")
    except Exception as e:
        print(f"❌ Version failed: {e}")
    
    # Test admin endpoints
    print("\n6. Testing /admin/key-status endpoint...")
    try:
        response = requests.get(f"{base_url}/admin/key-status", headers=headers)
        print(f"✅ Key status: {response.status_code} - {response.json()}")
    except Exception as e:
        print(f"❌ Key status failed: {e}")
    
    # Test metrics endpoint
    print("\n7. Testing /admin/metrics endpoint...")
    try:
        response = requests.get(f"{base_url}/admin/metrics", headers=headers)
        print(f"✅ Metrics: {response.status_code} - Response keys: {list(response.json().keys())}")
    except Exception as e:
        print(f"❌ Metrics failed: {e}")
    
    print("\n🎉 API Testing Complete!")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "server":
        # Start server mode
        start_server()
    else:
        # Test mode - assumes server is already running
        time.sleep(2)  # Wait for server to start
        test_api()