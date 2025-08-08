# SQL MCP Server Setup Guide

## 🚀 Quick Setup

### 1. Install PostgreSQL
```bash
# macOS (using Homebrew)
brew install postgresql
brew services start postgresql

# Ubuntu/Debian
sudo apt-get install postgresql postgresql-contrib
sudo systemctl start postgresql

# Windows
# Download from https://www.postgresql.org/download/windows/
```

### 2. Create Database and User
```bash
# Connect to PostgreSQL
psql postgres

# Create user and database
CREATE USER postgres WITH PASSWORD 'postgres';
CREATE DATABASE test_sqlmcp;
GRANT ALL PRIVILEGES ON DATABASE test_sqlmcp TO postgres;
\q
```

### 3. Install Python Dependencies
```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 4. Set Up Test Database
```bash
# Run the database setup script
python setup_test_db.py
```

This will create:
- **10 tables** with realistic relationships
- **1,000 users** with profiles
- **500 products** across 10 categories
- **2,000 orders** with order items
- **3,000 reviews** with ratings
- **5,000 sales records** with regional data
- **50 suppliers** with ratings
- **Inventory and customer data**

### 5. Configure Environment
```bash
# Copy the example environment file
cp env.example .env

# Edit .env and add your OpenAI API key
# OPENAI_API_KEY=your_actual_api_key_here
```

### 6. Start the MCP Server
```bash
python server.py
```

## 📊 Database Schema

### Core Tables:
- **users** - Customer accounts and profiles
- **products** - Product catalog with categories
- **orders** - Order headers with status
- **order_items** - Individual items in orders
- **reviews** - Product reviews and ratings
- **customers** - Extended customer data
- **sales** - Sales transactions by region
- **inventory** - Stock levels by warehouse
- **suppliers** - Supplier information
- **categories** - Product categories

### Sample Data Volume:
- **1,000 users** with subscription types
- **500 products** across 10 categories
- **2,000 orders** with 1-5 items each
- **3,000 reviews** with ratings 1-5
- **5,000 sales records** with regional data
- **50 suppliers** with contact info
- **Inventory tracking** for all products

## 🧪 Testing the Server

### 1. Health Check
```bash
curl http://127.0.0.1:8000/health
```

### 2. Test Queries
The server supports these MCP tools:

- `text_to_sql` - Convert natural language to SQL
- `explore_database` - Explore database structure
- `discover_database_knowledge` - Trigger intelligent discovery
- `find_relevant_tables` - Find tables for a query
- `get_business_domains` - Get business domains
- `execute_sql` - Execute raw SQL

### 3. Sample Natural Language Queries

```python
# User queries
"Show me all users"
"Find users with premium subscription"
"Get users who haven't logged in recently"

# Product queries
"Find products with price greater than $100"
"Show me products in the Electronics category"
"Get products with low stock"

# Order queries
"Show me orders from the last 30 days"
"Find orders with total amount over $500"
"Get orders by status"

# Analytics queries
"Get total revenue by month"
"Show me sales by region"
"Find top 10 customers by total spent"
"Get average rating for each product category"
```

## 🔧 Configuration Options

### Database Connection
```env
DB_TEST_CONNECTION_STRING=postgresql://postgres:postgres@localhost:5432/test_sqlmcp
```

### Server Settings
```env
SERVER_HOST=127.0.0.1
SERVER_PORT=8000
LOG_LEVEL=INFO
```

### Performance Settings
```env
MAX_CONCURRENT_QUERIES=5
SCHEMA_CACHE_TTL=3600
MAX_ROWS_RETURNED=1000
QUERY_TIMEOUT_SECONDS=30
```

### Safety Settings
```env
ALLOWED_OPERATIONS=SELECT
MAX_TABLES_PER_QUERY=10
ENABLE_QUERY_EXPLANATION=true
```

## 🐛 Troubleshooting

### Common Issues:

1. **PostgreSQL Connection Error**
   ```bash
   # Check if PostgreSQL is running
   brew services list | grep postgresql
   # or
   sudo systemctl status postgresql
   ```

2. **Missing Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **OpenAI API Key**
   - Get your API key from https://platform.openai.com/api-keys
   - Add it to the .env file

4. **Port Already in Use**
   ```bash
   # Change port in .env file
   SERVER_PORT=8001
   ```

## 📈 Performance Tips

- The server includes intelligent caching for schema and query results
- Database discovery happens automatically on startup
- Large queries are limited to 1000 rows by default
- Connection pooling is enabled for better performance

## 🔒 Security Notes

- Only SELECT operations are allowed by default
- Query execution is limited to prevent abuse
- All queries are logged for monitoring
- Input validation is performed on all queries 