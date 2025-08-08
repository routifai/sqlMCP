#!/usr/bin/env python3
"""
Simple Text2SQL System
One LLM call, real schema, sample data, done.
"""

import asyncio
import logging
import re
from typing import List, Dict, Any, Optional
import openai
from sqlalchemy import text

from configurations.config import config
from configurations.database_pool import database_pool

logger = logging.getLogger(__name__)

class SimpleText2SQL:
    """Simple text-to-SQL with one LLM call approach"""
    
    def __init__(self):
        if not config.OPENAI_API_KEY:
            raise ValueError("OpenAI API key required")
        
        self.client = openai.OpenAI(api_key=config.OPENAI_API_KEY)
    
    async def text_to_sql(self, query: str, database_id: str, execute: bool = True) -> Dict[str, Any]:
        """Main text-to-SQL function - one LLM call approach with complexity analysis"""
        start_time = asyncio.get_event_loop().time()
        
        try:
            logger.info(f"Simple Text2SQL: '{query[:50]}...' for {database_id}")
            
            # Step 1: Analyze query complexity
            complexity = self._analyze_query_complexity(query)
            logger.info(f"Query complexity: {complexity['level']}, likely tables: {complexity['likely_tables']}")
            
            # Step 2: Get fresh schema + sample data (1 second)
            schema = await self._get_relevant_schema(database_id, query)
            
            # Step 3: Generate SQL with LLM (2 seconds)
            sql = await self._generate_sql_with_context(query, schema, complexity)
            
            # Step 4: Execute if requested (1 second)
            results = None
            if execute:
                results = await self._execute_sql(sql, database_id)
            
            processing_time = asyncio.get_event_loop().time() - start_time
            
            return {
                "success": True,
                "query": query,
                "sql": sql,
                "results": results,
                "processing_time": processing_time,
                "tables_used": [table["table"] for table in schema],
                "complexity_analysis": complexity
            }
            
        except Exception as e:
            logger.error(f"Simple Text2SQL failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "query": query,
                "processing_time": asyncio.get_event_loop().time() - start_time
            }
    
    def _analyze_query_complexity(self, query: str) -> Dict[str, Any]:
        """Analyze query complexity to optimize approach"""
        query_lower = query.lower()
        
        # Simple patterns
        simple_patterns = [
            "show", "get", "list", "find", "select", "count", "total"
        ]
        
        # Complex patterns
        complex_patterns = [
            "join", "group by", "order by", "having", "aggregate", "average", "sum", "max", "min",
            "between", "like", "in", "exists", "subquery", "union", "intersect", "except"
        ]
        
        # Relationship patterns
        relationship_patterns = [
            "related", "connected", "linked", "associated", "belongs to", "has many",
            "customer orders", "user posts", "product reviews"
        ]
        
        # Count pattern matches
        simple_count = sum(1 for pattern in simple_patterns if pattern in query_lower)
        complex_count = sum(1 for pattern in complex_patterns if pattern in query_lower)
        relationship_count = sum(1 for pattern in relationship_patterns if pattern in query_lower)
        
        # Determine complexity level
        if complex_count > 2 or relationship_count > 1:
            level = "complex"
            likely_tables = 3
        elif complex_count > 0 or relationship_count > 0:
            level = "medium"
            likely_tables = 2
        else:
            level = "simple"
            likely_tables = 1
        
        # Detect specific query types
        query_types = []
        if any(word in query_lower for word in ["count", "total", "number"]):
            query_types.append("aggregation")
        if any(word in query_lower for word in ["average", "sum", "max", "min"]):
            query_types.append("statistical")
        if any(word in query_lower for word in ["recent", "latest", "newest", "oldest"]):
            query_types.append("time_based")
        if any(word in query_lower for word in ["join", "related", "connected"]):
            query_types.append("relational")
        
        return {
            "level": level,
            "likely_tables": likely_tables,
            "query_types": query_types,
            "simple_patterns": simple_count,
            "complex_patterns": complex_count,
            "relationship_patterns": relationship_count
        }
    
    async def _get_relevant_schema(self, database_id: str, query: str) -> List[Dict[str, Any]]:
        """Get schema + enhanced sample data for relevant tables (no LLM calls)"""
        logger.info(f"Getting enhanced schema for {database_id}")
        
        # Get all table names (fast)
        tables = await self._get_all_table_names(database_id)
        
        # Find relevant tables using simple keyword matching
        relevant_tables = self._find_tables_by_keywords(query, tables)[:5]  # Increased from 3 to 5
        
        logger.info(f"Found relevant tables: {relevant_tables}")
        
        # Get enhanced schema + sample data for each relevant table
        schema_with_samples = []
        for table in relevant_tables:
            enhanced_data = await self._get_enhanced_table_data(database_id, table)
            schema_with_samples.append(enhanced_data)
        
        return schema_with_samples
    
    async def _get_enhanced_table_data(self, database_id: str, table_name: str) -> Dict[str, Any]:
        """Get comprehensive table data including schema, samples, and metadata"""
        columns = await self._get_table_columns(database_id, table_name)
        row_count = await self._get_row_count(database_id, table_name)
        sample_rows = await self._get_diverse_samples(database_id, table_name, limit=5)
        key_columns = await self._identify_key_columns(database_id, table_name)
        foreign_keys = await self._get_foreign_keys(database_id, table_name)
        
        return {
            "table": table_name,
            "columns": columns,
            "row_count": row_count,
            "sample_data": sample_rows,
            "key_columns": key_columns,
            "foreign_keys": foreign_keys,
            "data_quality": await self._get_data_quality_info(database_id, table_name)
        }
    
    async def _get_row_count(self, database_id: str, table_name: str) -> int:
        """Get total row count for a table"""
        query = text(f"SELECT COUNT(*) FROM {table_name}")
        
        try:
            async with database_pool.get_connection(database_id) as conn:
                result = await conn.execute(query)
                return result.scalar()
        except Exception as e:
            logger.warning(f"Could not get row count for {table_name}: {e}")
            return 0
    
    async def _get_diverse_samples(self, database_id: str, table_name: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Get diverse sample rows including recent and varied data"""
        # Try simple SELECT first (most reliable)
        simple_query = text(f"SELECT * FROM {table_name} LIMIT {limit}")
        
        try:
            async with database_pool.get_connection(database_id) as conn:
                # Start with simple query
                result = await conn.execute(simple_query)
                rows = result.fetchall()
                columns = result.keys()
                
                if rows:
                    return [
                        {col: str(val) if val is not None else None for col, val in zip(columns, row)}
                        for row in rows
                    ]
                else:
                    return []
                    
        except Exception as e:
            logger.warning(f"Could not get samples for {table_name}: {e}")
            return []
    
    async def _get_sample_rows(self, database_id: str, table_name: str, limit: int = 3) -> List[Dict[str, Any]]:
        """Get sample rows from a table (fallback method)"""
        query = text(f"SELECT * FROM {table_name} LIMIT {limit}")
        
        try:
            async with database_pool.get_connection(database_id) as conn:
                result = await conn.execute(query)
                rows = result.fetchall()
                columns = result.keys()
                
                return [
                    {col: str(val) if val is not None else None for col, val in zip(columns, row)}
                    for row in rows
                ]
        except Exception as e:
            logger.warning(f"Could not get sample data for {table_name}: {e}")
            return []
    
    async def _identify_key_columns(self, database_id: str, table_name: str) -> List[str]:
        """Identify key columns (primary keys, unique constraints, etc.)"""
        db_config = config.get_database_by_id(database_id)
        
        if db_config.type.value == "postgresql":
            query = text("""
                SELECT column_name 
                FROM information_schema.key_column_usage 
                WHERE table_name = :table_name 
                AND constraint_name LIKE '%_pkey'
            """)
        elif db_config.type.value == "mysql":
            query = text("""
                SELECT COLUMN_NAME 
                FROM information_schema.KEY_COLUMN_USAGE 
                WHERE TABLE_NAME = :table_name 
                AND CONSTRAINT_NAME = 'PRIMARY'
            """)
        else:  # SQLite
            query = text("PRAGMA table_info(:table_name)")
        
        try:
            async with database_pool.get_connection(database_id) as conn:
                result = await conn.execute(query, {"table_name": table_name})
                rows = result.fetchall()
                
                if db_config.type.value == "sqlite":
                    # SQLite returns different format
                    return [row[1] for row in rows if row[5] == 1]  # pk column
                else:
                    return [row[0] for row in rows]
                    
        except Exception as e:
            logger.warning(f"Could not identify key columns for {table_name}: {e}")
            return []
    
    async def _get_foreign_keys(self, database_id: str, table_name: str) -> List[Dict[str, str]]:
        """Get foreign key relationships for a table"""
        db_config = config.get_database_by_id(database_id)
        
        try:
            if db_config.type.value == "postgresql":
                query = text("""
                    SELECT 
                        kcu.column_name as source_column,
                        ccu.table_name as target_table,
                        ccu.column_name as target_column
                    FROM information_schema.table_constraints tc
                    JOIN information_schema.key_column_usage kcu ON tc.constraint_name = kcu.constraint_name
                    JOIN information_schema.constraint_column_usage ccu ON ccu.constraint_name = tc.constraint_name
                    WHERE tc.constraint_type = 'FOREIGN KEY' AND tc.table_name = :table_name
                """)
            elif db_config.type.value == "mysql":
                query = text("""
                    SELECT 
                        COLUMN_NAME as source_column,
                        REFERENCED_TABLE_NAME as target_table,
                        REFERENCED_COLUMN_NAME as target_column
                    FROM information_schema.KEY_COLUMN_USAGE 
                    WHERE TABLE_NAME = :table_name 
                    AND REFERENCED_TABLE_NAME IS NOT NULL
                """)
            else:  # SQLite
                query = text("PRAGMA foreign_key_list(:table_name)")
            
            async with database_pool.get_connection(database_id) as conn:
                result = await conn.execute(query, {"table_name": table_name})
                rows = result.fetchall()
                
                if db_config.type.value == "sqlite":
                    return [
                        {
                            "source_column": row[3],
                            "target_table": row[2], 
                            "target_column": row[4]
                        }
                        for row in rows
                    ]
                else:
                    return [
                        {
                            "source_column": row[0],
                            "target_table": row[1],
                            "target_column": row[2]
                        }
                        for row in rows
                    ]
                    
        except Exception as e:
            logger.warning(f"Could not get foreign keys for {table_name}: {e}")
            return []
    
    async def _get_data_quality_info(self, database_id: str, table_name: str) -> Dict[str, Any]:
        """Get basic data quality information"""
        try:
            # Get total row count first
            total_rows = await self._get_row_count(database_id, table_name)
            
            if total_rows == 0:
                return {
                    "null_statistics": {},
                    "total_rows": 0
                }
            
            # Get columns for analysis
            columns = await self._get_table_columns(database_id, table_name)
            null_stats = {}
            
            # Only analyze first 3 columns to avoid too many queries
            for col in columns[:3]:
                col_name = col['name']
                try:
                    # Use a simple COUNT query
                    null_query = text(f"SELECT COUNT(*) FROM {table_name} WHERE {col_name} IS NULL")
                    async with database_pool.get_connection(database_id) as conn:
                        result = await conn.execute(null_query)
                        null_count = result.scalar()
                        
                        if total_rows > 0:
                            null_stats[col_name] = {
                                "null_count": null_count,
                                "null_percentage": round((null_count / total_rows) * 100, 1)
                            }
                except Exception as e:
                    logger.debug(f"Could not analyze null stats for {col_name}: {e}")
                    continue
            
            return {
                "null_statistics": null_stats,
                "total_rows": total_rows
            }
                
        except Exception as e:
            logger.warning(f"Could not get data quality info for {table_name}: {e}")
            return {"null_statistics": {}, "total_rows": 0}
    
    def _find_tables_by_keywords(self, query: str, all_tables: List[str]) -> List[str]:
        """Smart keyword matching to find relevant tables"""
        query_words = query.lower().split()
        scored_tables = []
        
        for table in all_tables:
            score = 0
            table_lower = table.lower()
            
            # Direct matches
            for word in query_words:
                if word in table_lower:
                    score += 10
                # Plural/singular matching
                if word.rstrip('s') in table_lower or f"{word}s" in table_lower:
                    score += 8
            
            # Common patterns
            if any(word in ['show', 'get', 'find', 'list'] for word in query_words):
                if table_lower.endswith('s'):  # Likely plural table
                    score += 3
            
            # Common table patterns
            if 'user' in query_words and 'user' in table_lower:
                score += 15
            if 'product' in query_words and 'product' in table_lower:
                score += 15
            if 'order' in query_words and 'order' in table_lower:
                score += 15
            
            if score > 0:
                scored_tables.append((table, score))
        
        return [table for table, score in sorted(scored_tables, key=lambda x: x[1], reverse=True)]
    
    async def _get_all_table_names(self, database_id: str) -> List[str]:
        """Get all table names from database"""
        db_config = config.get_database_by_id(database_id)
        
        if db_config.type.value == "postgresql":
            query = text("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_type = 'BASE TABLE'
                ORDER BY table_name
            """)
        elif db_config.type.value == "mysql":
            query = text("""
                SELECT TABLE_NAME 
                FROM information_schema.TABLES 
                WHERE TABLE_SCHEMA = DATABASE()
                AND TABLE_TYPE = 'BASE TABLE'
                ORDER BY TABLE_NAME
            """)
        else:  # SQLite
            query = text("""
                SELECT name 
                FROM sqlite_master 
                WHERE type = 'table' 
                ORDER BY name
            """)
        
        async with database_pool.get_connection(database_id) as conn:
            result = await conn.execute(query)
            return [row[0] for row in result.fetchall()]
    
    async def _get_table_columns(self, database_id: str, table_name: str) -> List[Dict[str, Any]]:
        """Get column information for a table"""
        db_config = config.get_database_by_id(database_id)
        
        if db_config.type.value == "postgresql":
            query = text("""
                SELECT column_name, data_type, is_nullable, column_default
                FROM information_schema.columns 
                WHERE table_name = :table_name 
                AND table_schema = 'public'
                ORDER BY ordinal_position
            """)
        elif db_config.type.value == "mysql":
            query = text("""
                SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE, COLUMN_DEFAULT
                FROM information_schema.COLUMNS 
                WHERE TABLE_NAME = :table_name 
                AND TABLE_SCHEMA = DATABASE()
                ORDER BY ORDINAL_POSITION
            """)
        else:  # SQLite
            query = text("PRAGMA table_info(:table_name)")
        
        async with database_pool.get_connection(database_id) as conn:
            result = await conn.execute(query, {"table_name": table_name})
            rows = result.fetchall()
            
            if db_config.type.value == "sqlite":
                return [
                    {
                        "name": row[1],
                        "data_type": row[2],
                        "nullable": not row[3],
                        "default_value": row[4]
                    }
                    for row in rows
                ]
            else:
                return [
                    {
                        "name": row[0],
                        "data_type": row[1],
                        "nullable": row[2] == "YES",
                        "default_value": row[3]
                    }
                    for row in rows
                ]
    
    async def _generate_sql_with_context(self, query: str, schema_context: List[Dict[str, Any]], complexity_info: Dict[str, Any]) -> str:
        """Generate SQL with one LLM call using enhanced context and complexity analysis"""
        logger.info(f"Generating SQL with enhanced LLM context (complexity: {complexity_info['level']})")
        
        # Format enhanced schema context for LLM
        schema_text = self._format_schema_context(schema_context)
        
        # Build complexity-specific instructions
        complexity_instructions = self._build_complexity_instructions(complexity_info)
        
        prompt = f"""Generate SQL for: "{query}"

Query Analysis:
- Complexity Level: {complexity_info['level']}
- Query Types: {', '.join(complexity_info['query_types'])}
- Likely Tables Needed: {complexity_info['likely_tables']}

Available tables with enhanced metadata:
{schema_text}

Instructions:
{complexity_instructions}

Return only the SQL query, properly formatted."""

        response = await asyncio.to_thread(
            self.client.chat.completions.create,
            model=config.TEXT2SQL_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=800  # Increased for more complex queries
        )
        
        sql = response.choices[0].message.content.strip()
        
        # Clean up the SQL
        sql = self._clean_sql(sql)
        
        logger.info(f"Generated SQL: {sql[:100]}...")
        return sql
    
    def _build_complexity_instructions(self, complexity_info: Dict[str, Any]) -> str:
        """Build complexity-specific instructions for SQL generation"""
        level = complexity_info['level']
        query_types = complexity_info['query_types']
        
        base_instructions = [
            "1. Use the foreign key relationships to create proper JOINs when needed",
            "2. Consider the key columns (primary keys) for efficient queries", 
            "3. Use the data quality information to handle NULL values appropriately",
            "4. Use LIMIT 100 if the query might return many rows",
            "5. Consider the total row counts when deciding on query complexity",
            "6. Return only the SQL query, properly formatted"
        ]
        
        if level == "simple":
            base_instructions.extend([
                "7. For simple queries, focus on basic SELECT with WHERE clauses",
                "8. Use the most relevant table based on the query keywords"
            ])
        elif level == "medium":
            base_instructions.extend([
                "7. Consider using JOINs if the query mentions relationships",
                "8. Use appropriate WHERE clauses for filtering",
                "9. Consider ORDER BY for sorting if mentioned"
            ])
        else:  # complex
            base_instructions.extend([
                "7. Use multiple JOINs based on foreign key relationships",
                "8. Consider GROUP BY for aggregations",
                "9. Use subqueries if needed for complex logic",
                "10. Optimize for performance with proper indexing hints"
            ])
        
        # Add type-specific instructions
        if "aggregation" in query_types:
            base_instructions.append("11. Use COUNT(), SUM(), AVG() as appropriate for aggregations")
        if "statistical" in query_types:
            base_instructions.append("12. Use statistical functions like AVG(), MAX(), MIN()")
        if "time_based" in query_types:
            base_instructions.append("13. Use date/time functions and ORDER BY for time-based queries")
        if "relational" in query_types:
            base_instructions.append("14. Use JOINs to connect related tables")
        
        return "\n".join(base_instructions)
    
    def _format_schema_context(self, schema_context: List[Dict[str, Any]]) -> str:
        """Format enhanced schema context for LLM prompt"""
        context_parts = []
        
        for table_info in schema_context:
            table_name = table_info["table"]
            columns = table_info["columns"]
            samples = table_info["sample_data"]
            row_count = table_info["row_count"]
            key_columns = table_info["key_columns"]
            foreign_keys = table_info["foreign_keys"]
            data_quality = table_info["data_quality"]
            
            context_parts.append(f"=== TABLE: {table_name} ===")
            context_parts.append(f"TOTAL ROWS: {row_count:,}")
            
            # Key columns
            if key_columns:
                context_parts.append(f"KEY COLUMNS: {', '.join(key_columns)}")
            
            # Foreign key relationships
            if foreign_keys:
                context_parts.append("FOREIGN KEY RELATIONSHIPS:")
                for fk in foreign_keys:
                    context_parts.append(f"  - {fk['source_column']} → {fk['target_table']}.{fk['target_column']}")
            
            # Data quality info
            if data_quality.get("null_statistics"):
                context_parts.append("DATA QUALITY:")
                for col, stats in data_quality["null_statistics"].items():
                    context_parts.append(f"  - {col}: {stats['null_percentage']}% null values")
            
            context_parts.append("COLUMNS:")
            for col in columns:
                col_info = f"  - {col['name']} ({col['data_type']}"
                if not col.get('nullable', True):
                    col_info += ", NOT NULL"
                if col.get('default_value'):
                    col_info += f", DEFAULT: {col['default_value']}"
                if col['name'] in key_columns:
                    col_info += ", PRIMARY KEY"
                col_info += ")"
                context_parts.append(col_info)
            
            # Enhanced sample data
            if samples:
                context_parts.append(f"SAMPLE DATA ({len(samples)} rows):")
                for i, row in enumerate(samples):
                    row_display = {}
                    for key, value in list(row.items())[:8]:  # Max 8 columns per row
                        if value is not None:
                            str_val = str(value)
                            if len(str_val) > 40:
                                str_val = str_val[:40] + "..."
                            row_display[key] = str_val
                        else:
                            row_display[key] = "NULL"
                    context_parts.append(f"  Row {i+1}: {row_display}")
            
            context_parts.append("")  # Empty line between tables
        
        return "\n".join(context_parts)
    
    def _clean_sql(self, sql: str) -> str:
        """Clean up generated SQL"""
        # Remove markdown code blocks
        sql = re.sub(r'```sql\s*', '', sql)
        sql = re.sub(r'```\s*', '', sql)
        
        # Remove extra whitespace
        sql = sql.strip()
        
        # Ensure it ends with semicolon
        if not sql.endswith(';'):
            sql += ';'
        
        return sql
    
    async def _execute_sql(self, sql: str, database_id: str) -> Dict[str, Any]:
        """Execute SQL and return results"""
        logger.info(f"Executing SQL: {sql[:50]}...")
        
        try:
            async with database_pool.get_connection(database_id) as conn:
                result = await conn.execute(text(sql))
                rows = result.fetchall()
                columns = result.keys()
                
                # Format results
                formatted_rows = []
                for row in rows:
                    formatted_row = {}
                    for col, val in zip(columns, row):
                        if val is not None:
                            formatted_row[col] = str(val)
                        else:
                            formatted_row[col] = None
                    formatted_rows.append(formatted_row)
                
                return {
                    "success": True,
                    "row_count": len(formatted_rows),
                    "columns": list(columns),
                    "data": formatted_rows,
                    "formatted_output": self._format_results(formatted_rows, list(columns))
                }
                
        except Exception as e:
            logger.error(f"SQL execution failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _format_results(self, rows: List[Dict[str, Any]], columns: List[str]) -> str:
        """Format results for display"""
        if not rows:
            return "No results found."
        
        # Create header
        header = " | ".join(columns)
        separator = "-" * len(header)
        
        # Create rows
        formatted_rows = []
        for row in rows[:10]:  # Limit to 10 rows for display
            row_str = " | ".join(str(row.get(col, "")) for col in columns)
            formatted_rows.append(row_str)
        
        return f"{header}\n{separator}\n" + "\n".join(formatted_rows) 