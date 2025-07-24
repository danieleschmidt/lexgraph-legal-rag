#!/usr/bin/env python3
"""
Fast test suite runner - runs only critical tests for CI/CD
Focus on core functionality and essential integrations
"""
import subprocess
import sys
import os

def run_fast_tests():
    """Run essential tests quickly for CI/CD gates"""
    
    # Set up environment
    os.environ['PYTHONPATH'] = '/root/repo/src'
    
    # Essential tests that must pass for deployment
    essential_tests = [
        'tests/test_exceptions.py',  # Core error handling
        'tests/test_sample.py',      # Basic functionality  
        'tests/test_models_coverage.py',  # Data models
        'tests/test_alerting.py',    # Monitoring (high business value)
        'tests/test_cors_security.py',  # Security (critical)
    ]
    
    # Run with optimized settings
    cmd = [
        'python', '-m', 'pytest',
        '-n', '4',  # fixed parallelism  
        '--dist', 'worksteal',  # work stealing scheduler
        '--tb=line',  # minimal traceback
        '--no-header',  # reduce output
        '--cov=src/lexgraph_legal_rag',
        '--cov-report=term-missing:skip-covered',
        '--cov-fail-under=0',
        '-q',  # quiet mode
        '--maxfail=5',  # stop after 5 failures
    ] + essential_tests
    
    print("🚀 Running fast test suite for CI/CD...")
    print(f"📝 Running {len(essential_tests)} essential test files...")
    
    try:
        result = subprocess.run(cmd, timeout=120, cwd='/root/repo')
        
        if result.returncode == 0:
            print("✅ Fast test suite PASSED - ready for deployment!")
            return 0
        else:
            print("❌ Fast test suite FAILED - blocking deployment")
            return 1
            
    except subprocess.TimeoutExpired:
        print("⏰ Fast test suite timed out - performance issue detected")
        return 1
    except Exception as e:
        print(f"💥 Fast test suite error: {e}")
        return 1

if __name__ == '__main__':
    exit(run_fast_tests())