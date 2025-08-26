#!/usr/bin/env python3
"""
🔐 GENERATION 2: ENHANCED SECURITY SYSTEM
Advanced authentication, authorization, and threat detection
"""

import asyncio
import json
import hashlib
import hmac
import secrets
import time
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime, timedelta
import jwt
import bcrypt
from pathlib import Path

import structlog

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.ConsoleRenderer()
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    logger_factory=structlog.stdlib.LoggerFactory(),
    context_class=dict,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)


class SecurityEventType(Enum):
    """Types of security events"""
    LOGIN_SUCCESS = "login_success"
    LOGIN_FAILURE = "login_failure"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    TOKEN_VALIDATION_FAILED = "token_validation_failed"
    PRIVILEGE_ESCALATION = "privilege_escalation"


class ThreatLevel(Enum):
    """Threat severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SecurityEvent:
    """Security event record"""
    id: str
    event_type: SecurityEventType
    threat_level: ThreatLevel
    user_id: Optional[str]
    ip_address: str
    user_agent: str
    resource: str
    details: Dict[str, Any]
    timestamp: datetime
    blocked: bool = False


@dataclass
class User:
    """User account information"""
    id: str
    username: str
    email: str
    password_hash: bytes
    roles: List[str]
    permissions: Set[str]
    created_at: datetime
    last_login: Optional[datetime] = None
    failed_login_attempts: int = 0
    locked_until: Optional[datetime] = None


class EnhancedAuthenticator:
    """Advanced authentication system"""
    
    def __init__(self):
        self.users: Dict[str, User] = {}
        self.sessions: Dict[str, Dict] = {}
        self.jwt_secret = secrets.token_urlsafe(32)
        self.password_salt_rounds = 12
        
        # Create demo users
        self._create_demo_users()
    
    def _create_demo_users(self):
        """Create demo users for testing"""
        demo_users = [
            {
                'id': 'admin_001',
                'username': 'admin',
                'email': 'admin@terragon.dev',
                'password': 'SecurePassword123!',
                'roles': ['admin', 'user'],
                'permissions': {'read', 'write', 'delete', 'admin'}
            },
            {
                'id': 'user_001',
                'username': 'analyst',
                'email': 'analyst@terragon.dev',
                'password': 'AnalystPass456@',
                'roles': ['analyst', 'user'],
                'permissions': {'read', 'analyze'}
            },
            {
                'id': 'guest_001',
                'username': 'guest',
                'email': 'guest@terragon.dev',
                'password': 'GuestPass789#',
                'roles': ['guest'],
                'permissions': {'read'}
            }
        ]
        
        for user_data in demo_users:
            password_hash = bcrypt.hashpw(
                user_data['password'].encode('utf-8'),
                bcrypt.gensalt(rounds=self.password_salt_rounds)
            )
            
            user = User(
                id=user_data['id'],
                username=user_data['username'],
                email=user_data['email'],
                password_hash=password_hash,
                roles=user_data['roles'],
                permissions=user_data['permissions'],
                created_at=datetime.now()
            )
            
            self.users[user.id] = user
            logger.info(f"Created demo user: {user.username}")
    
    async def authenticate(self, username: str, password: str, ip_address: str) -> Optional[str]:
        """Authenticate user and return JWT token"""
        user = self._find_user_by_username(username)
        
        if not user:
            logger.warning(f"Authentication failed: user not found - {username}")
            return None
        
        # Check if account is locked
        if user.locked_until and datetime.now() < user.locked_until:
            logger.warning(f"Authentication failed: account locked - {username}")
            return None
        
        # Verify password
        if not bcrypt.checkpw(password.encode('utf-8'), user.password_hash):
            user.failed_login_attempts += 1
            
            # Lock account after 5 failed attempts
            if user.failed_login_attempts >= 5:
                user.locked_until = datetime.now() + timedelta(minutes=30)
                logger.warning(f"Account locked due to failed attempts: {username}")
            
            logger.warning(f"Authentication failed: invalid password - {username}")
            return None
        
        # Reset failed attempts on successful login
        user.failed_login_attempts = 0
        user.locked_until = None
        user.last_login = datetime.now()
        
        # Generate JWT token
        token = self._generate_jwt_token(user)
        
        # Create session
        session_id = secrets.token_urlsafe(32)
        self.sessions[session_id] = {
            'user_id': user.id,
            'ip_address': ip_address,
            'created_at': datetime.now(),
            'last_activity': datetime.now()
        }
        
        logger.info(f"Authentication successful: {username}")
        return token
    
    def _find_user_by_username(self, username: str) -> Optional[User]:
        """Find user by username"""
        for user in self.users.values():
            if user.username == username:
                return user
        return None
    
    def _generate_jwt_token(self, user: User) -> str:
        """Generate JWT token for user"""
        payload = {
            'user_id': user.id,
            'username': user.username,
            'roles': user.roles,
            'permissions': list(user.permissions),
            'exp': datetime.utcnow() + timedelta(hours=24),
            'iat': datetime.utcnow()
        }
        
        return jwt.encode(payload, self.jwt_secret, algorithm='HS256')
    
    async def validate_token(self, token: str) -> Optional[Dict]:
        """Validate JWT token and return user info"""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=['HS256'])
            
            # Check if user still exists
            user_id = payload.get('user_id')
            if user_id not in self.users:
                logger.warning(f"Token validation failed: user not found - {user_id}")
                return None
            
            return payload
            
        except jwt.ExpiredSignatureError:
            logger.warning("Token validation failed: token expired")
            return None
        except jwt.InvalidTokenError:
            logger.warning("Token validation failed: invalid token")
            return None


class AuthorizationManager:
    """Role-based access control (RBAC) system"""
    
    def __init__(self):
        self.role_permissions: Dict[str, Set[str]] = {
            'admin': {'read', 'write', 'delete', 'admin', 'analyze'},
            'analyst': {'read', 'analyze'},
            'user': {'read'},
            'guest': {'read'}
        }
        
        self.resource_permissions: Dict[str, Set[str]] = {
            '/api/admin': {'admin'},
            '/api/analyze': {'analyze'},
            '/api/documents': {'read'},
            '/api/upload': {'write'},
            '/api/delete': {'delete'}
        }
    
    def check_permission(self, user_roles: List[str], resource: str, action: str) -> bool:
        """Check if user has permission for resource action"""
        # Get required permissions for resource
        required_perms = self.resource_permissions.get(resource, {action})
        
        # Get user permissions from roles
        user_perms = set()
        for role in user_roles:
            user_perms.update(self.role_permissions.get(role, set()))
        
        # Check if user has required permissions
        return bool(required_perms.intersection(user_perms))
    
    def add_role_permission(self, role: str, permission: str):
        """Add permission to role"""
        if role not in self.role_permissions:
            self.role_permissions[role] = set()
        self.role_permissions[role].add(permission)
    
    def add_resource_requirement(self, resource: str, permission: str):
        """Add permission requirement to resource"""
        if resource not in self.resource_permissions:
            self.resource_permissions[resource] = set()
        self.resource_permissions[resource].add(permission)


class ThreatDetector:
    """Intelligent threat detection system"""
    
    def __init__(self):
        self.ip_request_history: Dict[str, List[datetime]] = {}
        self.user_activity_patterns: Dict[str, List[Dict]] = {}
        self.blocked_ips: Set[str] = set()
        self.suspicious_patterns: Dict[str, Any] = {}
        
        # Rate limiting thresholds
        self.rate_limits = {
            'requests_per_minute': 60,
            'login_attempts_per_hour': 10,
            'failed_logins_threshold': 5
        }
    
    async def analyze_request(self, user_id: Optional[str], ip_address: str, 
                             user_agent: str, resource: str) -> Optional[SecurityEvent]:
        """Analyze incoming request for threats"""
        threat_indicators = []
        threat_level = ThreatLevel.LOW
        
        # Check IP rate limiting
        if self._check_rate_limit(ip_address):
            threat_indicators.append("Rate limit exceeded")
            threat_level = ThreatLevel.HIGH
        
        # Check for blocked IPs
        if ip_address in self.blocked_ips:
            threat_indicators.append("Blocked IP")
            threat_level = ThreatLevel.CRITICAL
        
        # Analyze user agent
        if self._is_suspicious_user_agent(user_agent):
            threat_indicators.append("Suspicious user agent")
            threat_level = max(threat_level, ThreatLevel.MEDIUM, key=lambda x: x.value)
        
        # Analyze access patterns
        if user_id and self._detect_unusual_access_pattern(user_id, resource):
            threat_indicators.append("Unusual access pattern")
            threat_level = max(threat_level, ThreatLevel.MEDIUM, key=lambda x: x.value)
        
        if threat_indicators:
            event_id = secrets.token_urlsafe(16)
            event = SecurityEvent(
                id=event_id,
                event_type=SecurityEventType.SUSPICIOUS_ACTIVITY,
                threat_level=threat_level,
                user_id=user_id,
                ip_address=ip_address,
                user_agent=user_agent,
                resource=resource,
                details={'indicators': threat_indicators},
                timestamp=datetime.now(),
                blocked=threat_level == ThreatLevel.CRITICAL
            )
            
            logger.warning(f"Security threat detected: {threat_indicators}")
            return event
        
        return None
    
    def _check_rate_limit(self, ip_address: str) -> bool:
        """Check if IP has exceeded rate limits"""
        now = datetime.now()
        cutoff = now - timedelta(minutes=1)
        
        # Initialize if not exists
        if ip_address not in self.ip_request_history:
            self.ip_request_history[ip_address] = []
        
        # Clean old requests
        self.ip_request_history[ip_address] = [
            req_time for req_time in self.ip_request_history[ip_address]
            if req_time > cutoff
        ]
        
        # Add current request
        self.ip_request_history[ip_address].append(now)
        
        # Check threshold
        return len(self.ip_request_history[ip_address]) > self.rate_limits['requests_per_minute']
    
    def _is_suspicious_user_agent(self, user_agent: str) -> bool:
        """Check if user agent is suspicious"""
        suspicious_patterns = [
            'bot', 'crawler', 'spider', 'scraper',
            'python', 'curl', 'wget', 'postman'
        ]
        
        user_agent_lower = user_agent.lower()
        return any(pattern in user_agent_lower for pattern in suspicious_patterns)
    
    def _detect_unusual_access_pattern(self, user_id: str, resource: str) -> bool:
        """Detect unusual access patterns for user"""
        if user_id not in self.user_activity_patterns:
            self.user_activity_patterns[user_id] = []
        
        # Add current activity
        activity = {
            'resource': resource,
            'timestamp': datetime.now(),
            'hour': datetime.now().hour
        }
        
        self.user_activity_patterns[user_id].append(activity)
        
        # Keep only recent activity (last 24 hours)
        cutoff = datetime.now() - timedelta(hours=24)
        self.user_activity_patterns[user_id] = [
            act for act in self.user_activity_patterns[user_id]
            if act['timestamp'] > cutoff
        ]
        
        # Simple anomaly detection: unusual time of access
        recent_activities = self.user_activity_patterns[user_id]
        if len(recent_activities) > 10:
            typical_hours = {act['hour'] for act in recent_activities[:-1]}
            current_hour = activity['hour']
            
            # If accessing at unusual time (outside typical hours)
            if current_hour not in typical_hours and (current_hour < 6 or current_hour > 22):
                return True
        
        return False
    
    def block_ip(self, ip_address: str):
        """Block an IP address"""
        self.blocked_ips.add(ip_address)
        logger.warning(f"IP address blocked: {ip_address}")


class SecurityAuditor:
    """Security audit and logging system"""
    
    def __init__(self):
        self.security_events: List[SecurityEvent] = []
        self.audit_log_path = Path("security_audit.log")
    
    async def log_security_event(self, event: SecurityEvent):
        """Log security event"""
        self.security_events.append(event)
        
        # Write to audit log
        log_entry = {
            'timestamp': event.timestamp.isoformat(),
            'event_id': event.id,
            'event_type': event.event_type.value,
            'threat_level': event.threat_level.value,
            'user_id': event.user_id,
            'ip_address': event.ip_address,
            'resource': event.resource,
            'blocked': event.blocked,
            'details': event.details
        }
        
        with open(self.audit_log_path, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
        
        logger.info(f"Security event logged: {event.event_type.value}")
    
    def get_security_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get security summary for recent period"""
        cutoff = datetime.now() - timedelta(hours=hours)
        recent_events = [e for e in self.security_events if e.timestamp > cutoff]
        
        event_counts = {}
        threat_levels = {}
        blocked_events = 0
        
        for event in recent_events:
            event_type = event.event_type.value
            threat_level = event.threat_level.value
            
            event_counts[event_type] = event_counts.get(event_type, 0) + 1
            threat_levels[threat_level] = threat_levels.get(threat_level, 0) + 1
            
            if event.blocked:
                blocked_events += 1
        
        return {
            'period_hours': hours,
            'total_events': len(recent_events),
            'blocked_events': blocked_events,
            'event_types': event_counts,
            'threat_levels': threat_levels,
            'top_ips': self._get_top_threat_ips(recent_events),
            'top_resources': self._get_top_targeted_resources(recent_events)
        }
    
    def _get_top_threat_ips(self, events: List[SecurityEvent]) -> List[Dict]:
        """Get top IP addresses by threat activity"""
        ip_counts = {}
        for event in events:
            if event.threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
                ip_counts[event.ip_address] = ip_counts.get(event.ip_address, 0) + 1
        
        return sorted(
            [{'ip': ip, 'count': count} for ip, count in ip_counts.items()],
            key=lambda x: x['count'],
            reverse=True
        )[:5]
    
    def _get_top_targeted_resources(self, events: List[SecurityEvent]) -> List[Dict]:
        """Get most targeted resources"""
        resource_counts = {}
        for event in events:
            resource_counts[event.resource] = resource_counts.get(event.resource, 0) + 1
        
        return sorted(
            [{'resource': res, 'count': count} for res, count in resource_counts.items()],
            key=lambda x: x['count'],
            reverse=True
        )[:5]


class EnhancedSecuritySystem:
    """Complete security system"""
    
    def __init__(self):
        self.authenticator = EnhancedAuthenticator()
        self.authorization_manager = AuthorizationManager()
        self.threat_detector = ThreatDetector()
        self.security_auditor = SecurityAuditor()
    
    async def authenticate_user(self, username: str, password: str, ip_address: str) -> Optional[str]:
        """Authenticate user with security monitoring"""
        # Log authentication attempt
        event = SecurityEvent(
            id=secrets.token_urlsafe(16),
            event_type=SecurityEventType.LOGIN_SUCCESS,
            threat_level=ThreatLevel.LOW,
            user_id=username,
            ip_address=ip_address,
            user_agent="system",
            resource="/auth/login",
            details={'username': username},
            timestamp=datetime.now()
        )
        
        token = await self.authenticator.authenticate(username, password, ip_address)
        
        if not token:
            event.event_type = SecurityEventType.LOGIN_FAILURE
            event.threat_level = ThreatLevel.MEDIUM
        
        await self.security_auditor.log_security_event(event)
        return token
    
    async def authorize_request(self, token: str, resource: str, action: str, 
                               ip_address: str, user_agent: str) -> bool:
        """Authorize request with threat detection"""
        # Validate token
        user_info = await self.authenticator.validate_token(token)
        if not user_info:
            event = SecurityEvent(
                id=secrets.token_urlsafe(16),
                event_type=SecurityEventType.TOKEN_VALIDATION_FAILED,
                threat_level=ThreatLevel.HIGH,
                user_id=None,
                ip_address=ip_address,
                user_agent=user_agent,
                resource=resource,
                details={'action': action},
                timestamp=datetime.now(),
                blocked=True
            )
            await self.security_auditor.log_security_event(event)
            return False
        
        # Check authorization
        authorized = self.authorization_manager.check_permission(
            user_info['roles'], resource, action
        )
        
        if not authorized:
            event = SecurityEvent(
                id=secrets.token_urlsafe(16),
                event_type=SecurityEventType.UNAUTHORIZED_ACCESS,
                threat_level=ThreatLevel.HIGH,
                user_id=user_info['user_id'],
                ip_address=ip_address,
                user_agent=user_agent,
                resource=resource,
                details={'action': action, 'roles': user_info['roles']},
                timestamp=datetime.now(),
                blocked=True
            )
            await self.security_auditor.log_security_event(event)
            return False
        
        # Threat detection
        threat_event = await self.threat_detector.analyze_request(
            user_info['user_id'], ip_address, user_agent, resource
        )
        
        if threat_event:
            await self.security_auditor.log_security_event(threat_event)
            if threat_event.blocked:
                return False
        
        return True
    
    async def get_security_status(self) -> Dict[str, Any]:
        """Get comprehensive security status"""
        summary = self.security_auditor.get_security_summary()
        
        return {
            'timestamp': datetime.now().isoformat(),
            'system_status': 'active',
            'total_users': len(self.authenticator.users),
            'active_sessions': len(self.authenticator.sessions),
            'blocked_ips': len(self.threat_detector.blocked_ips),
            'security_summary': summary,
            'threat_detection_rules': len(self.threat_detector.rate_limits),
            'authorization_roles': len(self.authorization_manager.role_permissions)
        }


async def demo_security_system():
    """Demonstrate enhanced security system"""
    print("🔐 GENERATION 2: ENHANCED SECURITY SYSTEM")
    print("=" * 60)
    print("Advanced authentication, authorization, and threat detection")
    print()
    
    security = EnhancedSecuritySystem()
    
    print("🔑 Testing authentication...")
    
    # Test successful authentication
    token = await security.authenticate_user("admin", "SecurePassword123!", "192.168.1.100")
    if token:
        print("✅ Admin authentication successful")
    else:
        print("❌ Admin authentication failed")
    
    # Test failed authentication
    failed_token = await security.authenticate_user("admin", "wrongpassword", "192.168.1.100")
    if not failed_token:
        print("✅ Failed authentication properly blocked")
    
    print("\n🛡️ Testing authorization...")
    
    # Test authorized request
    if token:
        authorized = await security.authorize_request(
            token, "/api/admin", "admin", "192.168.1.100", "Mozilla/5.0 Browser"
        )
        if authorized:
            print("✅ Admin access authorized")
        else:
            print("❌ Admin access denied")
    
    # Test unauthorized request
    user_token = await security.authenticate_user("guest", "GuestPass789#", "192.168.1.101")
    if user_token:
        unauthorized = await security.authorize_request(
            user_token, "/api/admin", "admin", "192.168.1.101", "Mozilla/5.0 Browser"
        )
        if not unauthorized:
            print("✅ Unauthorized access properly blocked")
    
    print("\n🚨 Testing threat detection...")
    
    # Simulate rate limiting
    for i in range(65):  # Exceed rate limit
        threat_event = await security.threat_detector.analyze_request(
            None, "192.168.1.200", "Python/requests", "/api/analyze"
        )
        if threat_event and threat_event.blocked:
            print(f"✅ Rate limiting activated after {i+1} requests")
            break
    
    # Test suspicious user agent
    threat_event = await security.threat_detector.analyze_request(
        None, "192.168.1.201", "Python-requests/2.28.1", "/api/sensitive"
    )
    if threat_event:
        print("✅ Suspicious user agent detected")
    
    print("\n📊 Security status:")
    status = await security.get_security_status()
    
    print(f"Total Users: {status['total_users']}")
    print(f"Active Sessions: {status['active_sessions']}")
    print(f"Blocked IPs: {status['blocked_ips']}")
    print(f"Security Events (24h): {status['security_summary']['total_events']}")
    print(f"Blocked Events (24h): {status['security_summary']['blocked_events']}")
    
    # Save results
    results_file = Path("enhanced_security_g2_results.json")
    with open(results_file, 'w') as f:
        json.dump(status, f, indent=2, default=str)
    
    print(f"\n📄 Security status saved to {results_file}")
    print("\n🎉 Enhanced security system demonstration complete!")


if __name__ == "__main__":
    asyncio.run(demo_security_system())