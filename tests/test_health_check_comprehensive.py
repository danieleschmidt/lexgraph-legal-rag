"""Comprehensive tests for health_check.py module.

This module provides extensive test coverage for the health check system
to improve overall test coverage from 0% to 80%+.
"""

import asyncio
import pytest
import time
from unittest.mock import Mock, patch, AsyncMock
from pathlib import Path
import json
import psutil

from lexgraph_legal_rag.health_check import (
    HealthChecker,
    HealthStatus,
    SystemHealth,
    ComponentHealth,
    add_health_endpoints
)


class TestHealthChecker:
    """Test suite for HealthChecker class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.checker = HealthChecker()
    
    def test_init(self):
        """Test HealthChecker initialization."""
        assert hasattr(self.checker, 'checks')
        assert hasattr(self.checker, 'start_time')
        assert hasattr(self.checker, 'version')
        assert isinstance(self.checker.checks, list)
        assert len(self.checker.checks) > 0  # Should have default checks
    
    def test_default_checks_registered(self):
        """Test that default health checks are registered."""
        # The HealthChecker should have registered default checks
        assert len(self.checker.checks) >= 8  # At least 8 default checks
        
        # Check that checks are callable
        for check in self.checker.checks:
            assert callable(check)
    
    @pytest.mark.asyncio
    async def test_get_health_basic(self):
        """Test getting basic health status."""
        # Mock system resources to avoid actual system calls
        with patch('psutil.cpu_percent', return_value=25.0), \
             patch('psutil.virtual_memory', return_value=Mock(percent=50.0, available=4*1024**3)), \
             patch('shutil.disk_usage', return_value=(100*1024**3, 50*1024**3, 50*1024**3)):
            
            health = await self.checker.get_health()
            
            assert isinstance(health, SystemHealth)
            assert health.status in [HealthStatus.HEALTHY, HealthStatus.UNHEALTHY, HealthStatus.DEGRADED]
            assert health.timestamp > 0
            assert health.version is not None
            assert isinstance(health.checks, list)
    
    async def test_run_check_failure(self):
        """Test running a failing health check."""
        def failing_check():
            raise ValueError("Health check failed")
        
        self.checker.register_check("failing_test", failing_check)
        result = await self.checker.run_check("failing_test")
        
        assert result["status"] == "unhealthy"
        assert "Health check failed" in result["message"]
        assert result["error"] == "Health check failed"
        assert "timestamp" in result
    
    async def test_run_check_nonexistent(self):
        """Test running a non-existent health check."""
        result = await self.checker.run_check("nonexistent")
        
        assert result["status"] == "unhealthy"
        assert "not found" in result["message"]
    
    async def test_run_all_checks(self):
        """Test running all registered health checks."""
        def check1():
            return {"status": "healthy", "message": "Check 1 OK"}
        
        def check2():
            return {"status": "healthy", "message": "Check 2 OK"}
        
        self.checker.register_check("check1", check1)
        self.checker.register_check("check2", check2)
        
        results = await self.checker.run_all_checks()
        
        assert len(results) == 2
        assert "check1" in results
        assert "check2" in results
        assert results["check1"]["status"] == "healthy"
        assert results["check2"]["status"] == "healthy"


class TestSystemHealth:
    """Test suite for SystemHealth functionality."""
    
    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    @patch('psutil.disk_usage')
    def test_check_system_resources(self, mock_disk, mock_memory, mock_cpu):
        """Test system resource checking."""
        # Mock system resource data
        mock_cpu.return_value = 25.5
        mock_memory.return_value = Mock(percent=60.2, available=4 * 1024**3)  # 4GB available
        mock_disk.return_value = Mock(percent=45.8)
        
        health_checker = HealthChecker()
        result = health_checker._check_system_resources()
        
        assert result["status"] == "healthy"
        assert result["details"]["cpu_percent"] == 25.5
        assert result["details"]["memory_percent"] == 60.2
        assert result["details"]["disk_percent"] == 45.8
        assert result["details"]["memory_available_gb"] > 0
    
    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    @patch('psutil.disk_usage')
    def test_check_system_resources_high_usage(self, mock_disk, mock_memory, mock_cpu):
        """Test system resource checking with high resource usage."""
        # Mock high resource usage
        mock_cpu.return_value = 95.0
        mock_memory.return_value = Mock(percent=92.0, available=500 * 1024**2)  # 500MB available
        mock_disk.return_value = Mock(percent=88.0)
        
        health_checker = HealthChecker()
        result = health_checker._check_system_resources()
        
        assert result["status"] == "unhealthy"
        assert "high resource usage" in result["message"].lower()
        assert result["details"]["cpu_percent"] == 95.0
        assert result["details"]["memory_percent"] == 92.0
    
    @patch('psutil.cpu_percent')
    def test_check_system_resources_exception(self, mock_cpu):
        """Test system resource checking with exception."""
        # Mock psutil exception
        mock_cpu.side_effect = psutil.Error("Failed to get CPU info")
        
        health_checker = HealthChecker()
        result = health_checker._check_system_resources()
        
        assert result["status"] == "unhealthy"
        assert "failed to check system resources" in result["message"].lower()
    
    @patch('shutil.disk_usage')
    def test_check_disk_space(self, mock_disk_usage):
        """Test disk space checking."""
        # Mock disk usage (total, used, free in bytes)
        mock_disk_usage.return_value = (100 * 1024**3, 60 * 1024**3, 40 * 1024**3)  # 100GB total, 60GB used
        
        health_checker = HealthChecker()
        result = health_checker._check_disk_space()
        
        assert result["status"] == "healthy"
        assert result["details"]["disk_usage_percent"] == 60.0
        assert result["details"]["free_space_gb"] > 0
    
    @patch('shutil.disk_usage')
    def test_check_disk_space_low(self, mock_disk_usage):
        """Test disk space checking with low disk space."""
        # Mock low disk space (95% used)
        mock_disk_usage.return_value = (100 * 1024**3, 95 * 1024**3, 5 * 1024**3)
        
        health_checker = HealthChecker()
        result = health_checker._check_disk_space()
        
        assert result["status"] == "unhealthy"
        assert "disk space low" in result["message"].lower()
        assert result["details"]["disk_usage_percent"] == 95.0
    
    @patch('psutil.virtual_memory')
    def test_check_memory_usage(self, mock_memory):
        """Test memory usage checking."""
        # Mock normal memory usage
        mock_memory.return_value = Mock(
            percent=65.5,
            total=16 * 1024**3,  # 16GB
            available=5.5 * 1024**3,  # 5.5GB available
            used=10.5 * 1024**3  # 10.5GB used
        )
        
        health_checker = HealthChecker()
        result = health_checker._check_memory_usage()
        
        assert result["status"] == "healthy"
        assert result["details"]["memory_percent"] == 65.5
        assert result["details"]["memory_total_gb"] > 0
        assert result["details"]["memory_available_gb"] > 0
    
    @patch('psutil.virtual_memory')
    def test_check_memory_usage_high(self, mock_memory):
        """Test memory usage checking with high memory usage."""
        # Mock high memory usage
        mock_memory.return_value = Mock(
            percent=92.0,
            total=8 * 1024**3,  # 8GB
            available=640 * 1024**2,  # 640MB available
            used=7.36 * 1024**3  # ~7.36GB used
        )
        
        health_checker = HealthChecker()
        result = health_checker._check_memory_usage()
        
        assert result["status"] == "unhealthy"
        assert "memory usage high" in result["message"].lower()
        assert result["details"]["memory_percent"] == 92.0


class TestAPIHealthChecker:
    """Test suite for API health checking functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.api_checker = APIHealthChecker()
    
    @patch('httpx.AsyncClient')
    async def test_check_api_endpoint_success(self, mock_client):
        """Test successful API endpoint health check."""
        # Mock successful HTTP response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "ok"}
        mock_response.raise_for_status.return_value = None
        
        mock_client_instance = AsyncMock()
        mock_client_instance.get.return_value = mock_response
        mock_client.return_value.__aenter__.return_value = mock_client_instance
        
        result = await self.api_checker.check_endpoint("http://test.api/health")
        
        assert result["status"] == "healthy"
        assert result["details"]["status_code"] == 200
        assert result["details"]["response_time_ms"] > 0
    
    @patch('httpx.AsyncClient')
    async def test_check_api_endpoint_failure(self, mock_client):
        """Test API endpoint health check with failure."""
        # Mock HTTP error
        mock_client_instance = AsyncMock()
        mock_client_instance.get.side_effect = Exception("Connection failed")
        mock_client.return_value.__aenter__.return_value = mock_client_instance
        
        result = await self.api_checker.check_endpoint("http://test.api/health")
        
        assert result["status"] == "unhealthy"
        assert "connection failed" in result["message"].lower()
    
    @patch('httpx.AsyncClient')
    async def test_check_api_endpoint_timeout(self, mock_client):
        """Test API endpoint health check with timeout."""
        # Mock timeout
        import httpx
        mock_client_instance = AsyncMock()
        mock_client_instance.get.side_effect = httpx.TimeoutException("Request timeout")
        mock_client.return_value.__aenter__.return_value = mock_client_instance
        
        result = await self.api_checker.check_endpoint("http://test.api/health", timeout=1.0)
        
        assert result["status"] == "unhealthy"
        assert "timeout" in result["message"].lower()
    
    async def test_check_multiple_endpoints(self):
        """Test checking multiple API endpoints."""
        endpoints = [
            "http://api1.test/health",
            "http://api2.test/health",
            "http://api3.test/health"
        ]
        
        with patch('httpx.AsyncClient') as mock_client:
            # Mock successful responses for all endpoints
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"status": "ok"}
            mock_response.raise_for_status.return_value = None
            
            mock_client_instance = AsyncMock()
            mock_client_instance.get.return_value = mock_response
            mock_client.return_value.__aenter__.return_value = mock_client_instance
            
            results = await self.api_checker.check_multiple_endpoints(endpoints)
            
            assert len(results) == 3
            for endpoint, result in results.items():
                assert result["status"] == "healthy"
                assert endpoint in endpoints


class TestHealthIntegration:
    """Integration tests for the health check system."""
    
    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    @patch('psutil.disk_usage')
    async def test_get_health_success(self, mock_disk, mock_memory, mock_cpu):
        """Test the main get_health function with healthy system."""
        # Mock healthy system metrics
        mock_cpu.return_value = 30.0
        mock_memory.return_value = Mock(percent=50.0, available=8 * 1024**3)
        mock_disk.return_value = Mock(percent=40.0)
        
        health_data = await get_health()
        
        assert health_data["status"] == "healthy"
        assert "checks" in health_data
        assert "timestamp" in health_data
        assert "version" in health_data
    
    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    async def test_get_health_unhealthy(self, mock_memory, mock_cpu):
        """Test the main get_health function with unhealthy system."""
        # Mock unhealthy system metrics
        mock_cpu.return_value = 98.0
        mock_memory.return_value = Mock(percent=95.0, available=200 * 1024**2)
        
        health_data = await get_health()
        
        assert health_data["status"] == "unhealthy"
        assert "checks" in health_data
        # Should contain details about what's unhealthy
        assert any(check["status"] == "unhealthy" for check in health_data["checks"].values())
    
    def test_add_health_endpoints(self):
        """Test adding health endpoints to FastAPI app."""
        from fastapi import FastAPI
        
        app = FastAPI()
        initial_routes = len(app.routes)
        
        add_health_endpoints(app)
        
        # Should have added at least one health endpoint
        assert len(app.routes) > initial_routes
        
        # Check that health endpoints were added
        route_paths = [route.path for route in app.routes if hasattr(route, 'path')]
        assert any("/health" in path for path in route_paths)
    
    @patch('lexgraph_legal_rag.health_check.get_health')
    async def test_health_endpoint_response(self, mock_get_health):
        """Test health endpoint HTTP response."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        
        # Mock health response
        mock_get_health.return_value = {
            "status": "healthy",
            "checks": {"system": {"status": "healthy"}},
            "timestamp": time.time(),
            "version": "1.0.0"
        }
        
        app = FastAPI()
        add_health_endpoints(app)
        client = TestClient(app)
        
        response = client.get("/health")
        
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
    
    async def test_comprehensive_health_check_flow(self):
        """Test complete health check workflow."""
        health_checker = HealthChecker()
        
        # Register multiple health checks
        def cpu_check():
            return {"status": "healthy", "message": "CPU OK", "details": {"cpu_percent": 25.0}}
        
        def memory_check():
            return {"status": "healthy", "message": "Memory OK", "details": {"memory_percent": 60.0}}
        
        def api_check():
            return {"status": "healthy", "message": "API OK", "details": {"response_time_ms": 150}}
        
        health_checker.register_check("cpu", cpu_check)
        health_checker.register_check("memory", memory_check)
        health_checker.register_check("api", api_check)
        
        # Run all checks
        results = await health_checker.run_all_checks()
        
        assert len(results) == 3
        assert all(result["status"] == "healthy" for result in results.values())
        
        # Verify all checks have required fields
        for check_name, result in results.items():
            assert "status" in result
            assert "message" in result
            assert "timestamp" in result
            assert "duration_ms" in result
    
    def test_health_status_enum_values(self):
        """Test HealthStatus enum values."""
        assert HealthStatus.HEALTHY.value == "healthy"
        assert HealthStatus.UNHEALTHY.value == "unhealthy"
        assert HealthStatus.DEGRADED.value == "degraded"
        assert HealthStatus.UNKNOWN.value == "unknown"
    
    async def test_concurrent_health_checks(self):
        """Test running health checks concurrently."""
        health_checker = HealthChecker()
        
        # Create slow health checks to test concurrency
        async def slow_check_1():
            await asyncio.sleep(0.1)
            return {"status": "healthy", "message": "Slow check 1"}
        
        async def slow_check_2():
            await asyncio.sleep(0.1)
            return {"status": "healthy", "message": "Slow check 2"}
        
        health_checker.register_check("slow1", slow_check_1)
        health_checker.register_check("slow2", slow_check_2)
        
        start_time = time.time()
        results = await health_checker.run_all_checks()
        end_time = time.time()
        
        # Should complete faster than sequential execution (0.2s)
        assert end_time - start_time < 0.2
        assert len(results) == 2
        assert all(result["status"] == "healthy" for result in results.values())