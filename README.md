# Simple Text2SQL MCP Server

**One LLM call, real schema, sample data, done.**

A simple and fast text-to-SQL MCP server that converts natural language queries to SQL using real database schema and sample data.

## 🚀 Features

- **Simple & Fast**: One LLM call per query (~3 seconds total)
- **Real Schema**: Uses actual database structure and sample data
- **Smart Matching**: Intelligent keyword matching to find relevant tables
- **Multiple Databases**: Supports PostgreSQL, MySQL, and SQLite
- **MCP Protocol**: Works with any MCP client

## 📁 File Structure

```
├── server.py                    # MCP server
├── client.py                    # MCP client for testing
├── simple_text2sql.py          # Core text2SQL logic
├── query_executor.py           # SQL execution
├── configurations/             # Configuration management
└── requirements.txt            # Dependencies
```

## 🎯 How It Works

1. **User Query**: "show me all users"
2. **Schema Discovery**: Get relevant tables using keyword matching (1 second)
3. **LLM Generation**: Generate SQL with real schema + sample data (2 seconds)
4. **Execution**: Execute SQL and return results (1 second)

## ⚡ Quick Start

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Configure environment:**
```bash
cp env.example .env
# Edit .env with your database and OpenAI settings
```

3. **Start server:**
```bash
python server.py
```

4. **Test with client:**
```bash
python client.py
```

## 🔧 Configuration

The server can be configured using environment variables:

### Database Configuration
- `DB_{ID}_CONNECTION_STRING`: Connection string for database
- `DB_{ID}_ALIAS`: Human-readable name for database

### LLM Settings
- `OPENAI_API_KEY`: Your OpenAI API key
- `TEXT2SQL_MODEL`: LLM model to use (default: gpt-4o-mini)

### Performance Settings
- `MAX_CONCURRENT_QUERIES`: Maximum concurrent queries (default: 5)
- `MAX_ROWS_RETURNED`: Maximum rows returned per query (default: 1000)
- `QUERY_TIMEOUT_SECONDS`: Query timeout in seconds (default: 30)

## 🎯 MCP Tools

- **`text_to_sql`**: Convert natural language to SQL and execute
- **`execute_sql`**: Execute raw SQL queries
- **`list_databases`**: List available databases

## 📊 Performance

| Metric | Simple Approach |
|--------|----------------|
| Response Time | ~3 seconds |
| LLM Calls | 1 per query |
| Code Complexity | 3 core files |
| Maintenance | Low |

## 🧠 Smart Schema Discovery

The system uses intelligent keyword matching to find relevant tables:

- **Direct matches**: "users" → `users` table
- **Plural/singular**: "user" → `users` table  
- **Common patterns**: "show all" → tables ending in 's'
- **Sample data**: Real data gives LLM perfect context

## 🚀 Example Usage

```python
# Natural language query
"show me all users with email addresses"

# Generated SQL
SELECT id, username, email, first_name, last_name 
FROM users 
WHERE email IS NOT NULL 
LIMIT 100;

# Results
id | username | email           | first_name | last_name
---|----------|-----------------|------------|----------
1  | johndoe  | john@doe.com   | John       | Doe
2  | janedoe  | jane@doe.com   | Jane       | Doe
```

## 🔍 Why This Works Better

1. **LLM is smart enough** - With real schema + sample data, GPT-4 generates excellent SQL
2. **Fresh data** - No stale cache issues
3. **Fast schema lookup** - Simple SQL queries are very fast
4. **Keyword matching works** - Simple but effective for finding relevant tables
5. **Sample data is key** - Shows LLM actual data patterns and formats

The beauty of this approach: **The LLM does the intelligence, not your code.**