#!/usr/bin/env python3
"""
Simple Text2SQL MCP Server
One LLM call, real schema, sample data, done.
"""

import asyncio
import logging
import signal
import sys
from datetime import datetime
from fastmcp import FastMCP
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from configurations.config import config, validate_startup_config
from configurations.database_pool import database_pool
from configurations.exceptions import handle_error, ConfigurationError
from simple_text2sql import SimpleText2SQL
from query_executor import QueryExecutor

# Setup logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL.upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SimpleText2SQLMCPServer:
    """Simple Text2SQL MCP Server - One LLM call approach"""
    
    def __init__(self):
        # Validate configuration
        if not validate_startup_config():
            raise ConfigurationError("Invalid configuration. Please check your environment variables.")
        
        self.text2sql = SimpleText2SQL()
        self.query_executor = QueryExecutor()
        self.app = FastMCP("simple-text2sql-server")
        self._setup_tools()
        self._setup_routes()
    
    def _setup_tools(self):
        """Setup MCP tools with simple text2SQL"""
        
        @self.app.tool()
        async def text_to_sql(query: str, database_id: str = None, execute: bool = True) -> str:
            """
            Convert natural language to SQL and execute it.
            
            This tool takes your natural language question and converts it to SQL using
            real database schema and sample data, then executes the query to return results.
            
            Args:
                query: Natural language question about your data (e.g., "show me all users")
                database_id: Database to query (optional, uses first available if not provided)
                execute: Whether to execute the SQL and return results (default: True)
                
            Returns:
                Generated SQL with explanations and execution results
            """
            try:
                logger.info(f"Text2SQL request: '{query[:100]}...' (execute: {execute})")
                
                # Use first database if none specified
                if not database_id and config.databases:
                    database_id = config.databases[0].id
                elif not database_id:
                    return "❌ No databases configured"
                
                # Use simple text2SQL - one LLM call approach
                result = await self.text2sql.text_to_sql(query, database_id, execute)
                
                if not result.get("success", False):
                    return f"❌ Failed to process query: {result.get('error', 'Unknown error')}"
                
                return self._format_text2sql_response(result)
                
            except Exception as e:
                error_msg = handle_error(e, "text_to_sql")
                logger.error(f"Text2SQL failed: {error_msg}")
                return error_msg
        
        @self.app.tool()
        async def execute_sql(sql: str, database_id: str) -> str:
            """
            Execute raw SQL query.
            
            Args:
                sql: SQL query to execute
                database_id: Database to execute the query against
                
            Returns:
                Query execution results
            """
            try:
                logger.info(f"Direct SQL execution: {database_id}")
                
                result = await self.query_executor.execute_sql(sql, database_id)
                return self._format_execution_response(result)
                
            except Exception as e:
                error_msg = handle_error(e, "execute_sql")
                logger.error(f"SQL execution failed: {error_msg}")
                return error_msg
        
        @self.app.tool()
        async def list_databases() -> str:
            """
            List available databases.
            
            Returns:
                List of configured databases
            """
            try:
                response_parts = ["# Available Databases\n"]
                
                for db in config.databases:
                    response_parts.extend([
                        f"## {db.id}",
                        f"**Type:** {db.type.value}",
                        f"**Alias:** {db.alias}",
                        f"**Read-only:** {db.read_only}",
                        ""
                    ])
                
                return "\n".join(response_parts)
                
            except Exception as e:
                error_msg = handle_error(e, "list_databases")
                logger.error(f"List databases failed: {error_msg}")
                return error_msg
    
    def _format_text2sql_response(self, result: dict) -> str:
        """Format text2sql response for user display with enhanced information"""
        
        response_parts = [
            f"# 🧠 Enhanced Text2SQL Results",
            f"**Query:** {result['query']}",
            f"**Processing Time:** {result['processing_time']:.2f}s",
            f"**Tables Used:** {', '.join(result['tables_used'])}",
            f""
        ]
        
        # Add complexity analysis if available
        if result.get('complexity_analysis'):
            complexity = result['complexity_analysis']
            response_parts.extend([
                f"## 📊 Query Analysis",
                f"**Complexity Level:** {complexity['level'].title()}",
                f"**Query Types:** {', '.join(complexity['query_types']) if complexity['query_types'] else 'None'}",
                f"**Likely Tables:** {complexity['likely_tables']}",
                f""
            ])
        
        response_parts.extend([
            f"## Generated SQL",
            f"```sql",
            f"{result['sql']}",
            f"```",
            f""
        ])
        
        # Add execution results if available
        if result.get('results'):
            results = result['results']
            if results.get('success'):
                response_parts.extend([
                    f"## Execution Results",
                    f"**Rows Returned:** {results.get('row_count', 0)}",
                    f"**Columns:** {', '.join(results.get('columns', []))}",
                    f"",
                    f"### Data",
                    f"```",
                    f"{results.get('formatted_output', 'No results')}",
                    f"```"
                ])
            else:
                response_parts.append(f"**Execution Error:** {results.get('error', 'Unknown error')}")
        
        return "\n".join(response_parts)
    
    def _format_execution_response(self, result: dict) -> str:
        """Format SQL execution response"""
        
        if not result.get('success'):
            return f"❌ Execution failed: {result.get('error', 'Unknown error')}"
        
        response_parts = [
            f"# Query Execution Results",
            f"**Database:** {result.get('database_id', 'unknown')}",
            f"**Rows Returned:** {result.get('row_count', 0)}",
            f"**Columns:** {', '.join(result.get('columns', []))}\n"
        ]
        
        if result.get('formatted_output'):
            response_parts.extend([
                "## Results",
                f"```\n{result['formatted_output']}\n```"
            ])
        
        return "\n".join(response_parts)
    
    def _setup_routes(self):
        """Setup HTTP routes for health checks"""
        
        @self.app.custom_route("/health", methods=["GET"])
        async def health_check(request):
            """Simple health check"""
            from starlette.responses import JSONResponse
            
            try:
                return JSONResponse({
                    "status": "healthy",
                    "timestamp": datetime.now().isoformat(),
                    "version": "2.0.0-simple",
                    "databases": len(config.databases),
                    "model": config.TEXT2SQL_MODEL
                }, status_code=200)
                
            except Exception as e:
                logger.error(f"Health check failed: {e}")
                return JSONResponse({
                    "status": "unhealthy",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }, status_code=503)
    
    async def startup(self):
        """Server startup"""
        logger.info("🚀 Starting Simple Text2SQL MCP Server v2.0.0")
        
        # Display configuration
        logger.info(f"📊 Configured {len(config.databases)} database(s):")
        for db in config.databases:
            logger.info(f"   - {db.id}: {db.alias} ({db.type.value})")
        
        # Initialize database pools
        await database_pool.initialize()
        logger.info("✅ Database connections initialized")
        
        logger.info(f"🌐 Server starting on: http://{config.SERVER_HOST}:{config.SERVER_PORT}")
        logger.info("🎯 MCP Tools: text_to_sql, execute_sql, list_databases")
        logger.info("🔄 Ready for simple text2sql requests")
    
    async def shutdown(self):
        """Server shutdown cleanup"""
        logger.info("🛑 Shutting down Simple Text2SQL MCP Server...")
        
        # Close database connections
        await database_pool.close_all()
        logger.info("✅ Database connections closed")
        
        logger.info("👋 Server stopped")
    
    def run(self):
        """Run the MCP server"""
        
        # Setup signal handlers
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, starting graceful shutdown...")
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        try:
            # Initialize
            asyncio.run(self.startup())
            
            # Run server
            self.app.run(transport="streamable-http", host=config.SERVER_HOST, port=config.SERVER_PORT)
            
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt")
        except Exception as e:
            logger.error(f"Server error: {e}")
            raise
        finally:
            # Cleanup
            asyncio.run(self.shutdown())

def main():
    """Main entry point"""
    try:
        server = SimpleText2SQLMCPServer()
        server.run()
    except ConfigurationError as e:
        logger.error(f"❌ Configuration error: {e}")
        print("\n💡 Setup Guide:")
        print("   1. Copy .env.example to .env")
        print("   2. Configure database connections: DB_{ID}_CONNECTION_STRING")
        print("   3. Set OpenAI API key: OPENAI_API_KEY")
        print("   4. Start server: python server.py")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Failed to start server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()