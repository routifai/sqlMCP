#!/usr/bin/env python3
"""
Simple Query Executor
Execute SQL queries and return results
"""

import logging
from typing import Dict, Any, List
from sqlalchemy import text

from configurations.config import config
from configurations.database_pool import database_pool

logger = logging.getLogger(__name__)

class QueryExecutor:
    """Simple SQL query executor"""
    
    async def execute_sql(self, sql: str, database_id: str) -> Dict[str, Any]:
        """Execute SQL query and return results"""
        try:
            logger.info(f"Executing SQL on {database_id}: {sql[:50]}...")
            
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
                    "database_id": database_id,
                    "row_count": len(formatted_rows),
                    "columns": list(columns),
                    "data": formatted_rows,
                    "formatted_output": self._format_results(formatted_rows, list(columns))
                }
                
        except Exception as e:
            logger.error(f"SQL execution failed: {e}")
            return {
                "success": False,
                "database_id": database_id,
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
        for row in rows[:20]:  # Limit to 20 rows for display
            row_str = " | ".join(str(row.get(col, "")) for col in columns)
            formatted_rows.append(row_str)
        
        result = f"{header}\n{separator}\n" + "\n".join(formatted_rows)
        
        if len(rows) > 20:
            result += f"\n... and {len(rows) - 20} more rows"
        
        return result 