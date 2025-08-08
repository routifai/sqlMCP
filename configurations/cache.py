 #!/usr/bin/env python3
"""
Caching system for schema and query results
"""

import hashlib
import json
import logging
from typing import Any, Dict, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class CacheEntry:
    value: Any
    created_at: datetime
    access_count: int = 0
    last_accessed: datetime = None
    
    def __post_init__(self):
        if self.last_accessed is None:
            self.last_accessed = self.created_at

class TTLCache:
    """Time-To-Live cache with size limits"""
    
    def __init__(self, maxsize: int = 1000, ttl_seconds: int = 3600):
        self.maxsize = maxsize
        self.ttl = timedelta(seconds=ttl_seconds)
        self._cache: Dict[str, CacheEntry] = {}
        self._access_order = []
    
    def _is_expired(self, entry: CacheEntry) -> bool:
        return datetime.now() - entry.created_at > self.ttl
    
    def _evict_expired(self):
        expired_keys = [
            key for key, entry in self._cache.items() 
            if self._is_expired(entry)
        ]
        for key in expired_keys:
            self._cache.pop(key, None)
            if key in self._access_order:
                self._access_order.remove(key)
    
    def _evict_lru(self):
        while len(self._cache) >= self.maxsize and self._access_order:
            lru_key = self._access_order.pop(0)
            self._cache.pop(lru_key, None)
    
    def get(self, key: str) -> Optional[Any]:
        self._evict_expired()
        
        if key not in self._cache:
            return None
            
        entry = self._cache[key]
        if self._is_expired(entry):
            self._cache.pop(key)
            if key in self._access_order:
                self._access_order.remove(key)
            return None
        
        entry.access_count += 1
        entry.last_accessed = datetime.now()
        
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)
        
        return entry.value
    
    def set(self, key: str, value: Any):
        self._evict_expired()
        self._evict_lru()
        
        now = datetime.now()
        self._cache[key] = CacheEntry(value=value, created_at=now)
        
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)
    
    def clear(self):
        self._cache.clear()
        self._access_order.clear()
    
    def stats(self) -> Dict[str, Any]:
        self._evict_expired()
        return {
            "size": len(self._cache),
            "maxsize": self.maxsize,
            "ttl_seconds": self.ttl.total_seconds()
        }

class SchemaCache:
    """Cache for database schema information"""
    
    def __init__(self, maxsize: int = 500, ttl_seconds: int = 3600):
        self.cache = TTLCache(maxsize, ttl_seconds)
        self.hits = 0
        self.misses = 0
    
    def _schema_key(self, database_id: str, table_name: str = None) -> str:
        if table_name:
            return f"schema_{database_id}_{table_name}"
        return f"schema_{database_id}_all"
    
    def get_table_schema(self, database_id: str, table_name: str) -> Optional[Dict]:
        key = self._schema_key(database_id, table_name)
        result = self.cache.get(key)
        
        if result is not None:
            self.hits += 1
            logger.debug(f"Schema cache hit: {database_id}.{table_name}")
        else:
            self.misses += 1
            logger.debug(f"Schema cache miss: {database_id}.{table_name}")
        
        return result
    
    def set_table_schema(self, database_id: str, table_name: str, schema: Dict):
        key = self._schema_key(database_id, table_name)
        self.cache.set(key, schema)
        logger.debug(f"Cached schema: {database_id}.{table_name}")
    
    def get_database_schema(self, database_id: str) -> Optional[Dict]:
        key = self._schema_key(database_id)
        result = self.cache.get(key)
        
        if result is not None:
            self.hits += 1
        else:
            self.misses += 1
        
        return result
    
    def set_database_schema(self, database_id: str, schema: Dict):
        key = self._schema_key(database_id)
        self.cache.set(key, schema)
    
    def get_hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            "hit_rate": self.get_hit_rate(),
            "hits": self.hits,
            "misses": self.misses,
            "cache_stats": self.cache.stats()
        }

class QueryResultCache:
    """Cache for SQL query results"""
    
    def __init__(self, maxsize: int = 200, ttl_seconds: int = 1800):
        self.cache = TTLCache(maxsize, ttl_seconds)
        self.hits = 0
        self.misses = 0
    
    def _query_key(self, sql: str, database_id: str) -> str:
        query_data = {
            "sql": sql.strip().lower(),
            "database_id": database_id
        }
        return hashlib.md5(
            json.dumps(query_data, sort_keys=True).encode()
        ).hexdigest()
    
    def get_result(self, sql: str, database_id: str) -> Optional[Dict]:
        key = self._query_key(sql, database_id)
        result = self.cache.get(key)
        
        if result is not None:
            self.hits += 1
            logger.debug(f"Query cache hit for {database_id}")
        else:
            self.misses += 1
            logger.debug(f"Query cache miss for {database_id}")
        
        return result
    
    def set_result(self, sql: str, database_id: str, result: Dict):
        key = self._query_key(sql, database_id)
        self.cache.set(key, result)
        logger.debug(f"Cached query result for {database_id}")
    
    def get_hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            "hit_rate": self.get_hit_rate(),
            "hits": self.hits,
            "misses": self.misses,
            "cache_stats": self.cache.stats()
        }