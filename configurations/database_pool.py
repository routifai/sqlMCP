#!/usr/bin/env python3
"""
Fixed Database connection pool management with proper event loop handling
"""

import asyncio
import logging
from typing import Dict, Optional
from contextlib import asynccontextmanager
from sqlalchemy.ext.asyncio import create_async_engine, AsyncEngine
from sqlalchemy.pool import StaticPool
from sqlalchemy import text

from .config import config, DatabaseConfig
from .exceptions import DatabaseConnectionError, ResourceExhaustionError

logger = logging.getLogger(__name__)

class DatabasePool:
    """Manages database connection pools for multiple databases with proper event loop handling"""
    
    def __init__(self):
        self.engine_configs: Dict[str, Dict] = {}  # Store engine configs instead of engines
        self.connection_counts: Dict[str, int] = {}
        self._lock = None
    
    async def initialize(self):
        """Initialize connection pool configurations for all configured databases"""
        # Create lock in current event loop
        if self._lock is None:
            self._lock = asyncio.Lock()
            
        async with self._lock:
            for db_config in config.databases:
                try:
                    # Store configuration instead of creating engine immediately
                    self.engine_configs[db_config.id] = self._build_engine_config(db_config)
                    self.connection_counts[db_config.id] = 0
                    
                    # Test connection by creating a temporary engine
                    await self._test_connection_config(db_config)
                    logger.info(f"✅ Database pool configured: {db_config.id}")
                except Exception as e:
                    logger.error(f"❌ Failed to configure pool for {db_config.id}: {e}")
    
    def _build_engine_config(self, db_config: DatabaseConfig) -> Dict:
        """Build engine configuration without creating the actual engine"""
        async_conn_str = self._convert_to_async_url(db_config.connection_string)
        
        engine_config = {
            "url": async_conn_str,
            "pool_size": db_config.max_connections,
            "max_overflow": 2,
            "pool_timeout": 30,
            "pool_recycle": 3600,
            "echo": False,
            "future": True,
        }
        
        # Add database-specific connection args
        if "sqlite" in async_conn_str:
            engine_config["connect_args"] = {
                "check_same_thread": False,
                "timeout": 30
            }
            # Use StaticPool for SQLite to avoid connection issues
            engine_config["poolclass"] = StaticPool
        
        return engine_config
    
    async def _test_connection_config(self, db_config: DatabaseConfig):
        """Test connection configuration by creating a temporary engine"""
        engine_config = self.engine_configs[db_config.id]
        
        # Create temporary engine in current event loop
        temp_engine = create_async_engine(**engine_config)
        
        try:
            async with temp_engine.begin() as conn:
                await conn.execute(text("SELECT 1"))
        except Exception as e:
            raise DatabaseConnectionError(f"Failed to connect to {db_config.id}: {e}")
        finally:
            await temp_engine.dispose()
    
    def _convert_to_async_url(self, connection_string: str) -> str:
        """Convert connection string to async format"""
        if connection_string.startswith("postgresql://"):
            return connection_string.replace("postgresql://", "postgresql+asyncpg://", 1)
        elif connection_string.startswith("mysql://"):
            return connection_string.replace("mysql://", "mysql+aiomysql://", 1)
        elif connection_string.startswith("sqlite://"):
            return connection_string.replace("sqlite://", "sqlite+aiosqlite://", 1)
        else:
            return connection_string
    
    def _get_or_create_engine(self, database_id: str) -> AsyncEngine:
        """Get or create engine in the current event loop"""
        if database_id not in self.engine_configs:
            raise DatabaseConnectionError(f"Database {database_id} not configured")
        
        # Create fresh engine in current event loop
        engine_config = self.engine_configs[database_id]
        return create_async_engine(**engine_config)
    
    @asynccontextmanager
    async def get_connection(self, database_id: str):
        """Get database connection with automatic cleanup - FIXED for event loop compatibility"""
        if database_id not in self.engine_configs:
            raise DatabaseConnectionError(f"Database {database_id} not configured")
        
        # Check connection limits
        db_config = config.get_database_by_id(database_id)
        if db_config and self.connection_counts[database_id] >= db_config.max_connections:
            raise ResourceExhaustionError(f"Connection limit reached for {database_id}")
        
        # Create engine in current event loop
        engine = self._get_or_create_engine(database_id)
        self.connection_counts[database_id] += 1
        
        try:
            # Get connection from engine
            connection = await engine.connect()
            try:
                yield connection
            finally:
                await connection.close()
        except Exception as e:
            logger.error(f"Database connection error for {database_id}: {e}")
            raise
        finally:
            # Clean up engine and decrement counter
            await engine.dispose()
            self.connection_counts[database_id] = max(0, self.connection_counts[database_id] - 1)
    
    async def test_connection(self, database_id: str) -> bool:
        """Test database connection"""
        try:
            async with self.get_connection(database_id) as conn:
                await conn.execute(text("SELECT 1"))
            return True
        except Exception as e:
            logger.error(f"Connection test failed for {database_id}: {e}")
            return False
    
    async def close_all(self):
        """Close all database connections - cleanup configurations"""
        if self._lock:
            async with self._lock:
                # Clear configurations
                self.engine_configs.clear()
                self.connection_counts.clear()
                logger.info("Cleared all database pool configurations")
    
    def get_stats(self) -> Dict:
        """Get connection pool statistics"""
        return {
            "databases": list(self.engine_configs.keys()),
            "connection_counts": self.connection_counts.copy(),
            "total_pools": len(self.engine_configs)
        }

# Global database pool instance
database_pool = DatabasePool()