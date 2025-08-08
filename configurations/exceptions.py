 #!/usr/bin/env python3
"""
Exception classes for Text2SQL MCP Server
"""

class Text2SQLError(Exception):
    """Base exception for all Text2SQL server errors"""
    def __init__(self, message: str, details: str = None):
        self.message = message
        self.details = details or ""
        super().__init__(self.message)

class ConfigurationError(Text2SQLError):
    """Configuration validation errors"""
    pass

class DatabaseError(Text2SQLError):
    """Database connection and operation errors"""
    pass

class DatabaseConnectionError(DatabaseError):
    """Database connection failed"""
    pass

class DatabaseTimeoutError(DatabaseError):
    """Database operation timed out"""
    pass

class SchemaError(Text2SQLError):
    """Schema introspection errors"""
    pass

class SQLGenerationError(Text2SQLError):
    """SQL generation failed"""
    pass

class SQLExecutionError(Text2SQLError):
    """SQL execution failed"""
    pass

class QueryAnalysisError(Text2SQLError):
    """Query analysis failed"""
    pass

class SecurityError(Text2SQLError):
    """Security violation detected"""
    pass

class ResourceExhaustionError(Text2SQLError):
    """System resources exhausted"""
    pass

def handle_error(error: Exception, context: str = "") -> str:
    """Convert exceptions to user-friendly error messages"""
    context_prefix = f"[{context}] " if context else ""
    
    if isinstance(error, DatabaseConnectionError):
        return f"🔌 {context_prefix}Cannot connect to database. Please check connection settings."
    
    elif isinstance(error, DatabaseTimeoutError):
        return f"⏱️ {context_prefix}Database query timed out. Try a simpler query."
    
    elif isinstance(error, SQLGenerationError):
        return f"🧠 {context_prefix}Could not generate SQL from your question. Please rephrase."
    
    elif isinstance(error, SQLExecutionError):
        return f"⚠️ {context_prefix}SQL execution failed. Please check your query."
    
    elif isinstance(error, SecurityError):
        return f"🔒 {context_prefix}Query blocked for security reasons."
    
    elif isinstance(error, SchemaError):
        return f"📋 {context_prefix}Cannot access database schema information."
    
    elif isinstance(error, ConfigurationError):
        return f"⚙️ {context_prefix}Server configuration error. Contact administrator."
    
    elif isinstance(error, ResourceExhaustionError):
        return f"🔄 {context_prefix}Server overloaded. Please try again later."
    
    else:
        return f"❌ {context_prefix}Unexpected error: {str(error)}"