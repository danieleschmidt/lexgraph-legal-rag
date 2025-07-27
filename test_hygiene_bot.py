#!/usr/bin/env python3
"""
Test script for GitHub Repository Hygiene Bot
"""

import json
import os
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch, MagicMock

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from github_hygiene_bot import GitHubHygieneBot


class TestGitHubHygieneBot(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures."""
        self.bot = GitHubHygieneBot(token="test_token", owner="test_owner")
        
    def test_initialization(self):
        """Test bot initialization."""
        self.assertEqual(self.bot.token, "test_token")
        self.assertEqual(self.bot.owner, "test_owner")
        self.assertEqual(self.bot.base_url, "https://api.github.com")
        self.assertEqual(self.bot.changes_made, [])
        
    def test_suggest_topics_python(self):
        """Test topic suggestion for Python repositories."""
        repo = {
            'name': 'test-api',
            'language': 'Python',
            'topics': []
        }
        topics = self.bot._suggest_topics(repo)
        
        self.assertIn('python', topics)
        self.assertIn('api', topics)
        self.assertIn('automation', topics)
        self.assertTrue(len(topics) <= 8)
        
    def test_suggest_topics_javascript(self):
        """Test topic suggestion for JavaScript repositories."""
        repo = {
            'name': 'web-app',
            'language': 'JavaScript',
            'topics': ['existing-topic']
        }
        topics = self.bot._suggest_topics(repo)
        
        self.assertIn('existing-topic', topics)
        self.assertIn('javascript', topics)
        self.assertIn('web', topics)
        self.assertTrue(len(topics) <= 8)
        
    def test_suggest_topics_patterns(self):
        """Test topic suggestion based on name patterns."""
        repo = {
            'name': 'security-bot',
            'language': 'Python',
            'topics': []
        }
        topics = self.bot._suggest_topics(repo)
        
        self.assertIn('automation', topics)  # from 'bot'
        self.assertIn('security', topics)    # from 'security'
        
    @patch('requests.Session.request')
    def test_make_request_success(self, mock_request):
        """Test successful API request."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'test': 'data'}
        mock_request.return_value = mock_response
        
        response = self.bot._make_request('GET', '/test')
        
        self.assertEqual(response.status_code, 200)
        mock_request.assert_called_once()
        
    @patch('requests.Session.request')
    @patch('time.sleep')
    def test_make_request_rate_limit(self, mock_sleep, mock_request):
        """Test rate limit handling."""
        # First call returns 429, second call succeeds
        rate_limit_response = Mock()
        rate_limit_response.status_code = 429
        rate_limit_response.headers = {'X-RateLimit-Reset': str(int(time.time()) + 60)}
        
        success_response = Mock()
        success_response.status_code = 200
        
        mock_request.side_effect = [rate_limit_response, success_response]
        
        response = self.bot._make_request('GET', '/test')
        
        self.assertEqual(response.status_code, 200)
        mock_sleep.assert_called_once()
        self.assertEqual(mock_request.call_count, 2)
        
    def test_get_readme_badges(self):
        """Test README badge generation."""
        badges = self.bot._get_readme_badges("test-repo")
        
        self.assertIn("[![License]", badges)
        self.assertIn("[![CI]", badges)
        self.assertIn("[![Security Rating]", badges)
        self.assertIn("[![SBOM]", badges)
        self.assertIn("test_owner/test-repo", badges)
        
    def test_template_methods_return_base64(self):
        """Test that template methods return base64 encoded content."""
        import base64
        
        templates = [
            self.bot._get_apache_license(),
            self.bot._get_code_of_conduct(),
            self.bot._get_contributing_guide(),
            self.bot._get_security_policy(),
            self.bot._get_bug_template(),
            self.bot._get_feature_template(),
            self.bot._get_codeql_workflow(),
            self.bot._get_dependabot_config(),
            self.bot._get_scorecard_workflow(),
            self.bot._get_sbom_workflow(),
            self.bot._get_sbom_diff_workflow()
        ]
        
        for template in templates:
            # Should be valid base64
            try:
                decoded = base64.b64decode(template)
                # Should contain some expected content
                self.assertGreater(len(decoded), 0)
            except Exception as e:
                self.fail(f"Template is not valid base64: {e}")
                
    @patch.object(GitHubHygieneBot, '_make_request')
    def test_list_owned_repositories(self, mock_request):
        """Test repository listing."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {
                'name': 'repo1',
                'fork': False,
                'is_template': False,
                'archived': False
            },
            {
                'name': 'repo2',
                'fork': True,  # Should be filtered out
                'is_template': False,
                'archived': False
            },
            {
                'name': 'repo3',
                'fork': False,
                'is_template': False,
                'archived': False
            }
        ]
        mock_request.return_value = mock_response
        
        repos = self.bot.list_owned_repositories()
        
        self.assertEqual(len(repos), 2)  # repo2 filtered out
        self.assertEqual(repos[0]['name'], 'repo1')
        self.assertEqual(repos[1]['name'], 'repo3')
        
    def test_security_checks(self):
        """Test security scanner checking methods."""
        repo = {'name': 'test-repo'}
        
        # Mock the _make_request method for these tests
        with patch.object(self.bot, '_make_request') as mock_request:
            # Test CodeQL check
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = [
                {'name': 'codeql-analysis.yml'},
                {'name': 'ci.yml'}
            ]
            mock_request.return_value = mock_response
            
            result = self.bot._check_code_scanning(repo)
            self.assertTrue(result)
            
            # Test when no CodeQL workflow exists
            mock_response.json.return_value = [{'name': 'ci.yml'}]
            result = self.bot._check_code_scanning(repo)
            self.assertFalse(result)
            
            # Test Dependabot check
            mock_response.status_code = 200
            result = self.bot._check_dependabot(repo)
            self.assertTrue(result)
            
            mock_response.status_code = 404
            result = self.bot._check_dependabot(repo)
            self.assertFalse(result)


class TestConfiguration(unittest.TestCase):
    """Test configuration file parsing."""
    
    def test_config_file_exists(self):
        """Test that configuration file exists and is valid JSON."""
        config_path = os.path.join(os.path.dirname(__file__), 'hygiene_config.json')
        self.assertTrue(os.path.exists(config_path))
        
        with open(config_path, 'r') as f:
            config = json.load(f)
            
        # Check required sections
        self.assertIn('metadata', config)
        self.assertIn('topics', config)
        self.assertIn('community_files', config)
        self.assertIn('security', config)
        self.assertIn('readme', config)
        
        # Check metadata section
        metadata = config['metadata']
        self.assertIn('default_description_template', metadata)
        self.assertIn('min_topics', metadata)
        self.assertIn('max_topics', metadata)
        
        # Check topics section
        topics = config['topics']
        self.assertIn('language_based', topics)
        self.assertIn('common', topics)
        self.assertIn('patterns', topics)


class TestDocumentation(unittest.TestCase):
    """Test documentation files."""
    
    def test_readme_exists(self):
        """Test that README documentation exists."""
        readme_path = os.path.join(os.path.dirname(__file__), 'HYGIENE_BOT_README.md')
        self.assertTrue(os.path.exists(readme_path))
        
        with open(readme_path, 'r') as f:
            content = f.read()
            
        # Check for required sections
        self.assertIn('## ✨ Features', content)
        self.assertIn('## ⚡ Quick Start', content)
        self.assertIn('## 🔍 What It Does', content)
        self.assertIn('## 🛠 Configuration', content)
        self.assertIn('## 🔒 Security & Permissions', content)
        
    def test_requirements_file_exists(self):
        """Test that requirements file exists."""
        req_path = os.path.join(os.path.dirname(__file__), 'hygiene_requirements.txt')
        self.assertTrue(os.path.exists(req_path))
        
        with open(req_path, 'r') as f:
            content = f.read()
            
        self.assertIn('requests', content)


class TestWorkflowFile(unittest.TestCase):
    """Test GitHub Actions workflow file."""
    
    def test_workflow_exists(self):
        """Test that workflow file exists and has required content."""
        workflow_path = os.path.join(
            os.path.dirname(__file__), 
            '.github/workflows/repository-hygiene.yml'
        )
        self.assertTrue(os.path.exists(workflow_path))
        
        with open(workflow_path, 'r') as f:
            content = f.read()
            
        # Check for required workflow elements
        self.assertIn('name: Repository Hygiene Automation', content)
        self.assertIn('schedule:', content)
        self.assertIn('workflow_dispatch:', content)
        self.assertIn('python3 github_hygiene_bot.py', content)


def run_tests():
    """Run all tests."""
    print("🧪 Running GitHub Hygiene Bot Tests")
    print("=" * 50)
    
    # Discover and run tests
    loader = unittest.TestLoader()
    suite = loader.discover('.', pattern='test_hygiene_bot.py')
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "=" * 50)
    if result.wasSuccessful():
        print("✅ All tests passed!")
        return 0
    else:
        print(f"❌ {len(result.failures)} test(s) failed, {len(result.errors)} error(s)")
        return 1


if __name__ == '__main__':
    # Add time import for rate limit test
    import time
    
    exit_code = run_tests()
    sys.exit(exit_code)