 #!/usr/bin/env python3
"""
Configuration Management for Text2SQL MCP Server
Handles multi-database discovery and validation
"""

import os
import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class DatabaseType(Enum):
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"
    SQLITE = "sqlite"

@dataclass
class DatabaseConfig:
    id: str
    type: DatabaseType
    connection_string: str
    alias: str = ""
    domains: List[str] = field(default_factory=list)
    read_only: bool = True
    max_connections: int = 5
    query_timeout: int = 30

@dataclass
class Config:
    """Main configuration class with auto-discovery"""
    
    # Database configurations
    databases: List[DatabaseConfig] = field(default_factory=list)
    
    # LLM Settings
    OPENAI_API_KEY: str = ""
    TEXT2SQL_MODEL: str = "gpt-4o-mini"
    
    # Discovery Settings
    ENABLE_LLM_DISCOVERY: bool = True
    DISCOVERY_ON_STARTUP: bool = True
    DISCOVERY_INTERVAL_HOURS: int = 24
    
    # Performance Settings
    MAX_CONCURRENT_QUERIES: int = 5
    SCHEMA_CACHE_TTL: int = 3600
    MAX_ROWS_RETURNED: int = 1000
    QUERY_TIMEOUT_SECONDS: int = 30
    
    # Safety Settings
    ALLOWED_OPERATIONS: List[str] = field(default_factory=lambda: ["SELECT"])
    MAX_TABLES_PER_QUERY: int = 10
    ENABLE_QUERY_EXPLANATION: bool = True
    
    # Server Settings
    SERVER_HOST: str = "127.0.0.1"
    SERVER_PORT: int = 8000
    LOG_LEVEL: str = "INFO"
    
    def __post_init__(self):
        """Initialize configuration from environment variables"""
        self._load_from_env()
        self._discover_databases()
    
    def _load_from_env(self):
        """Load configuration from environment variables"""
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
        self.TEXT2SQL_MODEL = os.getenv("TEXT2SQL_MODEL", "gpt-4o-mini")
        
        # Discovery settings
        self.ENABLE_LLM_DISCOVERY = os.getenv("ENABLE_LLM_DISCOVERY", "true").lower() == "true"
        self.DISCOVERY_ON_STARTUP = os.getenv("DISCOVERY_ON_STARTUP", "true").lower() == "true"
        self.DISCOVERY_INTERVAL_HOURS = int(os.getenv("DISCOVERY_INTERVAL_HOURS", "24"))
        
        # Performance settings
        self.MAX_CONCURRENT_QUERIES = int(os.getenv("MAX_CONCURRENT_QUERIES", "5"))
        self.SCHEMA_CACHE_TTL = int(os.getenv("SCHEMA_CACHE_TTL", "3600"))
        self.MAX_ROWS_RETURNED = int(os.getenv("MAX_ROWS_RETURNED", "1000"))
        self.QUERY_TIMEOUT_SECONDS = int(os.getenv("QUERY_TIMEOUT_SECONDS", "30"))
        
        # Safety settings
        allowed_ops = os.getenv("ALLOWED_OPERATIONS", "SELECT")
        self.ALLOWED_OPERATIONS = [op.strip().upper() for op in allowed_ops.split(",")]
        self.MAX_TABLES_PER_QUERY = int(os.getenv("MAX_TABLES_PER_QUERY", "10"))
        self.ENABLE_QUERY_EXPLANATION = os.getenv("ENABLE_QUERY_EXPLANATION", "true").lower() == "true"
        
        # Server settings
        self.SERVER_HOST = os.getenv("SERVER_HOST", "127.0.0.1")
        self.SERVER_PORT = int(os.getenv("SERVER_PORT", "8000"))
        self.LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    
    def _discover_databases(self):
        """Auto-discover database configurations from environment"""
        databases = []
        discovered_ids = set()
        
        # Pattern: DB_{ID}_CONNECTION_STRING
        for key in os.environ:
            if key.startswith("DB_") and key.endswith("_CONNECTION_STRING"):
                db_id = key.replace("DB_", "").replace("_CONNECTION_STRING", "").lower()
                discovered_ids.add(db_id)
        
        for db_id in discovered_ids:
            conn_str = os.getenv(f"DB_{db_id.upper()}_CONNECTION_STRING")
            if conn_str:
                try:
                    db_config = DatabaseConfig(
                        id=db_id,
                        type=self._detect_db_type(conn_str),
                        connection_string=conn_str,
                        alias=os.getenv(f"DB_{db_id.upper()}_ALIAS", db_id.replace("_", " ").title()),
                        domains=self._parse_domains(os.getenv(f"DB_{db_id.upper()}_DOMAINS", "")),
                        read_only=os.getenv(f"DB_{db_id.upper()}_READ_ONLY", "true").lower() == "true",
                        max_connections=int(os.getenv(f"DB_{db_id.upper()}_MAX_CONNECTIONS", "5")),
                        query_timeout=int(os.getenv(f"DB_{db_id.upper()}_QUERY_TIMEOUT", "30"))
                    )
                    databases.append(db_config)
                    logger.info(f"Discovered database: {db_id} ({db_config.type.value})")
                except Exception as e:
                    logger.error(f"Failed to configure database {db_id}: {e}")
        
        self.databases = databases
    
    def _detect_db_type(self, connection_string: str) -> DatabaseType:
        """Detect database type from connection string"""
        conn_lower = connection_string.lower()
        if conn_lower.startswith("postgresql://") or conn_lower.startswith("postgres://"):
            return DatabaseType.POSTGRESQL
        elif conn_lower.startswith("mysql://"):
            return DatabaseType.MYSQL
        elif conn_lower.startswith("sqlite://") or conn_lower.endswith(".db"):
            return DatabaseType.SQLITE
        else:
            raise ValueError(f"Unsupported database type in connection string: {connection_string}")
    
    def _parse_domains(self, domains_str: str) -> List[str]:
        """Parse comma-separated domains"""
        if not domains_str:
            return []
        return [domain.strip() for domain in domains_str.split(",") if domain.strip()]
    
    def validate(self) -> Tuple[bool, List[str]]:
        """Validate configuration and return status"""
        errors = []
        
        # Check OpenAI API key
        if not self.OPENAI_API_KEY:
            errors.append("OPENAI_API_KEY is required")
        
        # Check databases
        if not self.databases:
            errors.append("No databases configured. Set DB_{ID}_CONNECTION_STRING environment variables")
        
        # Validate database connections
        for db in self.databases:
            if not db.connection_string:
                errors.append(f"Database {db.id} has empty connection string")
        
        # Validate safety settings
        if self.MAX_ROWS_RETURNED <= 0:
            errors.append("MAX_ROWS_RETURNED must be positive")
        
        if self.QUERY_TIMEOUT_SECONDS <= 0:
            errors.append("QUERY_TIMEOUT_SECONDS must be positive")
        
        valid_operations = {"SELECT", "INSERT", "UPDATE", "DELETE", "CREATE", "ALTER", "DROP"}
        for op in self.ALLOWED_OPERATIONS:
            if op not in valid_operations:
                errors.append(f"Invalid operation in ALLOWED_OPERATIONS: {op}")
        
        return len(errors) == 0, errors
    
    def get_database_by_id(self, database_id: str) -> Optional[DatabaseConfig]:
        """Get database configuration by ID"""
        return next((db for db in self.databases if db.id == database_id), None)
    
    def get_databases_by_domain(self, domain: str) -> List[DatabaseConfig]:
        """Get databases that contain a specific domain"""
        return [db for db in self.databases if domain.lower() in [d.lower() for d in db.domains]]
    
    def get_status_info(self) -> Dict:
        """Get configuration status for health checks"""
        return {
            "databases": {
                "count": len(self.databases),
                "configured": [
                    {
                        "id": db.id,
                        "type": db.type.value,
                        "alias": db.alias,
                        "domains": db.domains,
                        "read_only": db.read_only
                    }
                    for db in self.databases
                ]
            },
            "llm": {
                "model": self.TEXT2SQL_MODEL,
                "configured": bool(self.OPENAI_API_KEY)
            },
            "safety": {
                "allowed_operations": self.ALLOWED_OPERATIONS,
                "max_rows": self.MAX_ROWS_RETURNED,
                "read_only_enforced": all(db.read_only for db in self.databases)
            },
            "performance": {
                "max_concurrent": self.MAX_CONCURRENT_QUERIES,
                "timeout_seconds": self.QUERY_TIMEOUT_SECONDS,
                "cache_ttl": self.SCHEMA_CACHE_TTL
            }
        }

# Global configuration instance
config = Config()

def validate_startup_config() -> bool:
    """Validate configuration at startup"""
    is_valid, errors = config.validate()
    
    if is_valid:
        logger.info("✅ Configuration validated successfully")
        logger.info(f"📊 Configured {len(config.databases)} database(s)")
        for db in config.databases:
            logger.info(f"   - {db.id}: {db.alias} ({db.type.value})")
        return True
    else:
        logger.error("❌ Configuration validation failed:")
        for error in errors:
            logger.error(f"   - {error}")
        return False