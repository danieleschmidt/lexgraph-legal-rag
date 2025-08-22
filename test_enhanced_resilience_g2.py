"""
Test Enhanced Resilience System - Generation 2
==============================================

Comprehensive tests for enhanced resilience patterns and security hardening.
"""

import asyncio
import pytest
import time
from unittest.mock import Mock, patch

from src.lexgraph_legal_rag.enhanced_resilience_patterns import (
    EnhancedResilienceSystem,
    ResilienceConfig,
    ResilienceLevel,
    with_resilience,
    ValidationError,
    RateLimitError,
    CircuitBreakerError,
    resilient_operation
)

from src.lexgraph_legal_rag.security_hardening_enhanced import (
    SecurityHardeningSystem,
    SecurityConfig,
    SecurityLevel,
    SecurityError
)


class TestEnhancedResilienceSystem:
    """Test the enhanced resilience system."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = ResilienceConfig(
            max_retries=2,
            retry_delay=0.1,
            rate_limit_per_minute=10,
            circuit_failure_threshold=3
        )
        self.system = EnhancedResilienceSystem(self.config)
    
    @pytest.mark.asyncio
    async def test_successful_operation(self):
        """Test successful operation execution."""
        
        async def mock_operation(value):
            return value * 2
        
        result = await self.system.with_resilience(
            mock_operation, 5,
            operation_name="test_op",
            resilience_level=ResilienceLevel.ENHANCED
        )
        
        assert result == 10
    
    @pytest.mark.asyncio
    async def test_retry_mechanism(self):
        """Test retry mechanism with transient failures."""
        
        call_count = 0
        
        async def failing_operation():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Temporary failure")
            return "success"
        
        result = await self.system.with_resilience(
            failing_operation,
            operation_name="failing_op",
            resilience_level=ResilienceLevel.ENHANCED
        )
        
        assert result == "success"
        assert call_count == 3
    
    @pytest.mark.asyncio
    async def test_input_validation(self):
        """Test input validation."""
        
        async def mock_operation(value):
            return value
        
        # Test with None value (should fail validation)
        with pytest.raises(ValidationError):
            await self.system.with_resilience(
                mock_operation, None,
                operation_name="validation_test"
            )
    
    @pytest.mark.asyncio
    async def test_input_sanitization(self):
        """Test input sanitization."""
        
        async def mock_operation(text):
            return text
        
        # Test with potentially dangerous input
        dangerous_input = "<script>alert('xss')</script>"
        
        result = await self.system.with_resilience(
            mock_operation, dangerous_input,
            operation_name="sanitization_test"
        )
        
        # Should be sanitized
        assert "<script>" not in result
    
    @pytest.mark.asyncio
    async def test_rate_limiting(self):
        """Test rate limiting functionality."""
        
        async def fast_operation():
            return "fast"
        
        # Make requests up to the limit
        for i in range(self.config.rate_limit_per_minute):
            result = await self.system.with_resilience(
                fast_operation,
                operation_name="rate_test"
            )
            assert result == "fast"
        
        # Next request should fail due to rate limit
        with pytest.raises(Exception) as exc_info:
            await self.system.with_resilience(
                fast_operation,
                operation_name="rate_test"
            )
        
        assert "rate limit" in str(exc_info.value).lower()
    
    @pytest.mark.asyncio
    async def test_circuit_breaker(self):
        """Test circuit breaker functionality."""
        
        async def always_failing_operation():
            raise RuntimeError("Always fails")
        
        # Make enough failed requests to open circuit
        for i in range(self.config.circuit_failure_threshold):
            with pytest.raises(RuntimeError):
                await self.system.with_resilience(
                    always_failing_operation,
                    operation_name="circuit_test"
                )
        
        # Next request should fail due to open circuit
        with pytest.raises(Exception) as exc_info:
            await self.system.with_resilience(
                always_failing_operation,
                operation_name="circuit_test"
            )
        
        assert "circuit" in str(exc_info.value).lower()
    
    @pytest.mark.asyncio
    async def test_health_status(self):
        """Test health status reporting."""
        
        async def test_operation():
            return "ok"
        
        # Execute some operations
        await self.system.with_resilience(
            test_operation,
            operation_name="health_test"
        )
        
        health = await self.system.get_health_status()
        
        assert "timestamp" in health
        assert "health_metrics" in health
        assert "circuit_states" in health
        assert health["health_metrics"]["health_test_success_count"] == 1
    
    @pytest.mark.asyncio
    async def test_resilience_decorator(self):
        """Test resilience decorator."""
        
        @with_resilience(
            operation_name="decorated_op",
            resilience_level=ResilienceLevel.ROBUST
        )
        async def decorated_function(value):
            return value * 3
        
        result = await decorated_function(4)
        assert result == 12
    
    @pytest.mark.asyncio
    async def test_resilient_context_manager(self):
        """Test resilient context manager."""
        
        async with resilient_operation("context_test") as resilience:
            # Context provides resilience system
            assert isinstance(resilience, EnhancedResilienceSystem)


class TestSecurityHardeningSystem:
    """Test the security hardening system."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = SecurityConfig(
            rate_limit_requests_per_minute=5,
            max_request_size=1000
        )
        self.system = SecurityHardeningSystem(self.config)
    
    def test_string_sanitization(self):
        """Test string sanitization."""
        
        # Test basic sanitization
        clean_text = self.system.sanitize_input("Hello World")
        assert clean_text == "Hello World"
        
        # Test HTML escaping
        html_text = self.system.sanitize_input("<script>alert('xss')</script>")
        assert "&lt;script&gt;" in html_text
        assert "<script>" not in html_text
    
    def test_sql_injection_detection(self):
        """Test SQL injection detection."""
        
        # Test legitimate input
        safe_input = "SELECT legal documents WHERE category = 'contract'"
        with pytest.raises(SecurityError):
            self.system.sanitize_input(safe_input, SecurityLevel.ENHANCED)
    
    def test_xss_detection(self):
        """Test XSS attack detection."""
        
        malicious_input = "<script>document.cookie</script>"
        with pytest.raises(SecurityError):
            self.system.sanitize_input(malicious_input, SecurityLevel.ENHANCED)
    
    def test_dictionary_sanitization(self):
        """Test dictionary sanitization."""
        
        input_dict = {
            "name": "John Doe",
            "query": "<script>alert('test')</script>",
            "nested": {"key": "value"}
        }
        
        sanitized = self.system.sanitize_input(input_dict, SecurityLevel.STANDARD)
        
        assert sanitized["name"] == "John Doe"
        assert "&lt;script&gt;" in sanitized["query"]
        assert sanitized["nested"]["key"] == "value"
    
    def test_rate_limiting(self):
        """Test rate limiting."""
        
        client_id = "test_client_123"
        
        # Make requests up to limit
        for i in range(self.config.rate_limit_requests_per_minute):
            assert self.system.check_rate_limit(client_id) is True
        
        # Next request should be rate limited
        assert self.system.check_rate_limit(client_id) is False
    
    def test_file_upload_validation(self):
        """Test file upload validation."""
        
        # Test valid file
        valid_content = b"This is a text file content"
        assert self.system.validate_file_upload("document.txt", valid_content) is True
        
        # Test invalid file type
        assert self.system.validate_file_upload("script.exe", valid_content) is False
        
        # Test file too large
        large_content = b"x" * (self.config.max_request_size + 1)
        assert self.system.validate_file_upload("large.txt", large_content) is False
    
    def test_password_hashing(self):
        """Test password hashing and verification."""
        
        password = "MySecurePassword123!"
        
        # Hash password
        password_hash, salt = self.system.hash_password(password)
        
        # Verify correct password
        assert self.system.verify_password(password, password_hash, salt) is True
        
        # Verify incorrect password
        assert self.system.verify_password("wrong", password_hash, salt) is False
    
    def test_session_management(self):
        """Test session management."""
        
        user_id = "user123"
        
        # Create session
        session_id = self.system.create_session(user_id)
        assert len(session_id) > 0
        
        # Validate session
        assert self.system.validate_session(session_id) is True
        
        # Invalidate session
        self.system.invalidate_session(session_id)
        assert self.system.validate_session(session_id) is False
    
    def test_data_encryption(self):
        """Test data encryption and decryption."""
        
        original_data = "Sensitive legal document content"
        
        # Encrypt data
        encrypted = self.system.encrypt_data(original_data)
        assert encrypted != original_data
        
        # Decrypt data
        decrypted = self.system.decrypt_data(encrypted)
        assert decrypted == original_data
    
    def test_security_report(self):
        """Test security report generation."""
        
        # Generate some events
        self.system._log_security_event("test_event", success=True)
        self.system._log_security_event("failed_event", success=False)
        
        report = self.system.get_security_report()
        
        assert "total_security_events" in report
        assert "failed_events" in report
        assert "success_rate" in report
        assert report["total_security_events"] >= 2
        assert report["failed_events"] >= 1


class TestIntegratedResilience:
    """Test integrated resilience and security."""
    
    @pytest.mark.asyncio
    async def test_resilience_with_security(self):
        """Test resilience system with security hardening."""
        
        # Create systems
        resilience_config = ResilienceConfig(max_retries=2)
        security_config = SecurityConfig()
        
        resilience_system = EnhancedResilienceSystem(resilience_config)
        security_system = SecurityHardeningSystem(security_config)
        
        async def secure_operation(user_input):
            # First apply security
            safe_input = security_system.sanitize_input(user_input)
            
            # Then process
            return f"Processed: {safe_input}"
        
        # Test with potentially dangerous input
        result = await resilience_system.with_resilience(
            secure_operation,
            "<script>alert('test')</script>",
            operation_name="secure_op"
        )
        
        assert "Processed:" in result
        assert "&lt;script&gt;" in result


# Performance benchmark
@pytest.mark.asyncio
async def test_resilience_performance():
    """Test resilience system performance."""
    
    system = EnhancedResilienceSystem()
    
    async def fast_operation(value):
        return value * 2
    
    # Measure performance
    start_time = time.time()
    
    tasks = []
    for i in range(100):
        task = system.with_resilience(
            fast_operation, i,
            operation_name=f"perf_test_{i}"
        )
        tasks.append(task)
    
    results = await asyncio.gather(*tasks)
    
    end_time = time.time()
    duration = end_time - start_time
    
    # Should complete 100 operations in reasonable time
    assert len(results) == 100
    assert duration < 5.0  # Less than 5 seconds
    
    print(f"✅ Processed 100 operations in {duration:.3f}s")
    print(f"⚡ Throughput: {len(results)/duration:.1f} ops/sec")


if __name__ == "__main__":
    # Run basic tests
    print("🧪 Testing Enhanced Resilience System (Generation 2)")
    print("=" * 60)
    
    async def run_tests():
        # Test resilience
        resilience_system = EnhancedResilienceSystem()
        
        async def test_op():
            return "success"
        
        result = await resilience_system.with_resilience(
            test_op,
            operation_name="basic_test"
        )
        
        print(f"✅ Basic resilience test: {result}")
        
        # Test security
        security_system = SecurityHardeningSystem()
        
        # Test with basic level (should sanitize but not block)
        safe_input = security_system.sanitize_input("Hello World", SecurityLevel.BASIC)
        print(f"✅ Security sanitization: {safe_input}")
        
        # Test XSS detection
        try:
            security_system.sanitize_input("<script>alert('xss')</script>", SecurityLevel.ENHANCED)
            print("❌ XSS detection failed")
        except SecurityError:
            print("✅ XSS detection working")
        
        # Test health status
        health = await resilience_system.get_health_status()
        print(f"✅ Health metrics: {len(health['health_metrics'])} metrics tracked")
        
        print("\n🎉 Generation 2 Robustness Tests Completed Successfully!")
    
    asyncio.run(run_tests())