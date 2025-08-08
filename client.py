#!/usr/bin/env python3
"""
SQL MCP Client with Streamable HTTP Transport
Interactive client using OpenAI to intelligently interact with SQL MCP tools.
"""

import asyncio
import json
import logging
import os
from typing import Optional, Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from mcp.shared.exceptions import McpError
import openai

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SQLMCPClient:
    """Interactive MCP client using OpenAI for intelligent SQL tool usage"""
    
    def __init__(self, 
                 server_url: str = "http://localhost:8000/mcp",
                 model: str = "gpt-4o-mini"):
        self.server_url = server_url
        self.model = model
        
        # Initialize OpenAI client
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment")
        
        self.openai_client = openai.OpenAI(api_key=api_key)
        
        logger.info(f"SQL MCP Client initialized with model: {self.model}")
    
    async def chat_with_sql_tools(self, user_message: str) -> str:
        """Chat using OpenAI with SQL MCP tools"""
        try:
            # Connect to MCP server for this chat session
            async with streamablehttp_client(self.server_url) as (read_stream, write_stream, _):
                async with ClientSession(read_stream, write_stream) as session:
                    # Initialize connection
                    await session.initialize()
                    
                    # Get available tools
                    tools_response = await session.list_tools()
                    
                    # Convert MCP tools to OpenAI format
                    openai_tools = []
                    for tool in tools_response.tools:
                        # FastMCP tools have better schemas by default
                        openai_function = {
                            "type": "function",
                            "function": {
                                "name": tool.name,
                                "description": tool.description,
                                "parameters": tool.inputSchema or {
                                    "type": "object",
                                    "properties": {},
                                    "required": []
                                }
                            }
                        }
                        openai_tools.append(openai_function)
                    
                    logger.info(f"📋 Loaded {len(openai_tools)} SQL tools for OpenAI")
                    
                    # Create messages for OpenAI
                    messages = [
                        {
                            "role": "system",
                            "content": """You are a helpful AI assistant with access to SQL database tools. 
                            
                            CRITICAL RULE: NEVER modify, rewrite, or change user queries when calling tools. Use the exact query the user provided.
                            
                            Available SQL tools:
                            - text_to_sql: Convert natural language to SQL and optionally execute it
                            - explore_database: Explore learned database knowledge and structure
                            - discover_database_knowledge: Trigger intelligent discovery of database knowledge
                            - find_relevant_tables: Find relevant tables for a query using learned intelligence
                            - get_business_domains: Get discovered business domains and their tables
                            - execute_sql: Execute raw SQL query with safety checks
                            
                            Use these tools when users:
                            - Ask questions about data in natural language
                            - Want to explore database structure
                            - Need to find specific tables or data
                            - Want to execute SQL queries
                            - Ask for business domain information
                            
                            IMPORTANT: When calling text_to_sql:
                            - Use the EXACT query the user provided
                            - Do NOT add years, dates, or modify the query
                            - Do NOT rewrite or "improve" the user's query
                            - Pass the user's original words as-is
                            
                            Be helpful and use the appropriate tools when needed, but preserve user intent exactly."""
                        },
                        {
                            "role": "user", 
                            "content": user_message
                        }
                    ]
                    
                    # First call to OpenAI
                    response = self.openai_client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        tools=openai_tools,
                        tool_choice="auto",
                        max_tokens=1000
                    )
                    
                    response_message = response.choices[0].message
                    messages.append(response_message)
                    
                    # Check if tools were called
                    if response_message.tool_calls:
                        print(f"\n🔧 LLM decided to use SQL tools: {[tc.function.name for tc in response_message.tool_calls]}")
                        
                        for tool_call in response_message.tool_calls:
                            tool_name = tool_call.function.name
                            tool_args = json.loads(tool_call.function.arguments)
                            
                            print(f"📞 Calling SQL MCP tool '{tool_name}' with args: {tool_args}")
                            
                            # Call the MCP tool
                            try:
                                result = await session.call_tool(tool_name, tool_args)
                                
                                if result.content and len(result.content) > 0:
                                    content = result.content[0]
                                    tool_result = content.text if hasattr(content, 'text') else str(content)
                                else:
                                    tool_result = "No response from MCP tool"
                                
                                print(f"✅ Tool result: {tool_result[:200]}...")
                                
                                # Add tool result to conversation
                                messages.append({
                                    "role": "tool",
                                    "tool_call_id": tool_call.id,
                                    "content": tool_result
                                })
                                
                            except McpError as e:
                                logger.error(f"MCP Error: {e}")
                                tool_result = f"MCP Error: {e}"
                                messages.append({
                                    "role": "tool",
                                    "tool_call_id": tool_call.id,
                                    "content": tool_result
                                })
                        
                        # Get final response from OpenAI
                        final_response = self.openai_client.chat.completions.create(
                            model=self.model,
                            messages=messages,
                            max_tokens=1000
                        )
                        
                        final_message = final_response.choices[0].message.content
                        print(f"🤖 Final LLM response: {final_message}")
                        
                        return final_message
                    else:
                        print(f"\n💬 LLM responded directly (no tools used): {response_message.content}")
                        return response_message.content
                        
        except Exception as e:
            logger.error(f"Error in chat: {e}")
            logger.exception("Chat error details:")
            return f"Error: {e}"
    
    async def test_connection(self) -> bool:
        """Test connection to both LLM and MCP server"""
        try:
            # Test LLM connection
            test_response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=5
            )
            logger.info("LLM connection successful")
            
            # Test MCP server connection
            async with streamablehttp_client(self.server_url) as (read_stream, write_stream, _):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    tools = await session.list_tools()
                    logger.info(f"MCP server connection successful, {len(tools.tools)} tools available")
            
            return True
            
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False

async def main():
    """Main interactive chat function"""
    print("🚀 SQL MCP Client v1.0")
    print("=" * 50)
    
    # Create MCP client
    try:
        client = SQLMCPClient()
        
        # Test connections
        if not await client.test_connection():
            print("❌ Failed to connect to LLM or MCP server. Check your configuration.")
            return
        
        print("✅ All connections successful!")
        
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        return
    
    print("\n🤖 Starting interactive SQL chat...")
    print("💡 The LLM will automatically use SQL MCP tools when appropriate.")
    print("🛑 Type 'quit', 'exit', or 'bye' to exit.")
    print("\n💬 Try these examples:")
    print("   - 'Show me all users' → Should trigger text_to_sql")
    print("   - 'Find products with price greater than $100' → Should trigger text_to_sql")
    print("   - 'Get the top 10 customers by total spent' → Should trigger text_to_sql")
    print("   - 'Explore the database structure' → Should trigger explore_database")
    print("   - 'Find relevant tables for customer data' → Should trigger find_relevant_tables")
    print("   - 'Get business domains' → Should trigger get_business_domains")
    print("-" * 50)
    
    while True:
        try:
            # Get user input
            user_input = input("\n💬 You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("👋 Goodbye!")
                break
            
            # Get LLM response
            print("\n🔄 Processing...")
            response = await client.chat_with_sql_tools(user_input)
            
            print(f"\n🤖 Assistant: {response}")
            print("-" * 50)
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            logger.exception("Main loop error:")
            print("-" * 50)

if __name__ == "__main__":
    asyncio.run(main()) 