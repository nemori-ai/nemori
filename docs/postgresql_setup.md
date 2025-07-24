# PostgreSQL Support for Nemori

This guide explains how to set up and use PostgreSQL as a storage backend for the Nemori episodic memory system.

## Prerequisites

1. **PostgreSQL Server**: Make sure you have PostgreSQL 12+ installed and running
2. **Database**: Create a database for Nemori (e.g., `nemori`)
3. **User Permissions**: Ensure your PostgreSQL user has CREATE, INSERT, UPDATE, DELETE, and SELECT permissions

## Installation

The PostgreSQL dependencies are already included when you install Nemori:

```bash
uv sync
```

The required packages are:
- `psycopg2-binary>=2.9.10` - PostgreSQL adapter for Python
- `asyncpg>=0.30.0` - Async PostgreSQL driver
- `sqlmodel>=0.0.24` - SQLModel for type-safe database operations

## Configuration

### Option 1: Using the Helper Function

```python
from nemori.storage import create_postgresql_config

config = create_postgresql_config(
    host="localhost",
    port=5432,
    database="nemori",
    username="postgres",
    password="your_password",  # Optional
    batch_size=500,
    cache_size=5000,
)
```

### Option 2: Using StorageConfig Directly

```python
from nemori.storage import StorageConfig

config = StorageConfig(
    backend_type="postgresql",
    connection_string="postgresql+asyncpg://user:password@localhost:5432/nemori",
    batch_size=1000,
    cache_size=10000,
)
```

### Option 3: Using Environment Variables

```python
import os
from nemori.storage import StorageConfig

config = StorageConfig(
    backend_type="postgresql",
    connection_string=os.getenv("POSTGRESQL_URL"),
)
```

Set the environment variable:
```bash
export POSTGRESQL_URL="postgresql+asyncpg://user:password@localhost:5432/nemori"
```

## Basic Usage

```python
import asyncio
from nemori.storage import create_repositories, create_postgresql_config
from nemori.core.data_types import DataType, RawEventData, TemporalInfo
from datetime import datetime

async def main():
    # Create configuration
    config = create_postgresql_config(
        host="localhost",
        database="nemori",
        username="postgres",
        password="your_password"
    )
    
    # Create repositories
    raw_repo, episode_repo = create_repositories(config)
    
    try:
        # Initialize repositories (creates tables if they don't exist)
        await raw_repo.initialize()
        await episode_repo.initialize()
        
        # Check health
        assert await raw_repo.health_check()
        assert await episode_repo.health_check()
        
        # Store some data
        raw_data = RawEventData(
            data_id="example_1",
            data_type=DataType.CONVERSATION,
            content={"message": "Hello, world!"},
            source="example",
            temporal_info=TemporalInfo(timestamp=datetime.now()),
            metadata={},
        )
        
        stored_id = await raw_repo.store_raw_data(raw_data)
        print(f"Stored data with ID: {stored_id}")
        
        # Retrieve data
        retrieved = await raw_repo.get_raw_data(stored_id)
        print(f"Retrieved: {retrieved.content}")
        
    finally:
        # Clean up
        await raw_repo.close()
        await episode_repo.close()

if __name__ == "__main__":
    asyncio.run(main())
```

## Database Schema

Nemori automatically creates the following tables in your PostgreSQL database:

### `raw_data` Table
- `data_id` (VARCHAR, PRIMARY KEY) - Unique identifier for raw event data
- `data_type` (VARCHAR) - Type of data (e.g., CONVERSATION, EMAIL)
- `content` (TEXT) - JSON content of the event
- `source` (VARCHAR) - Source system or application
- `timestamp` (TIMESTAMP) - When the event occurred
- `duration` (FLOAT) - Duration of the event in seconds
- `timezone` (VARCHAR) - Timezone information
- `precision` (VARCHAR) - Temporal precision (second, minute, etc.)
- `event_metadata` (TEXT) - JSON metadata
- `processed` (BOOLEAN) - Whether the data has been processed
- `processing_version` (VARCHAR) - Version of processing pipeline
- `created_at` (TIMESTAMP) - When the record was created

### `episodes` Table
- `episode_id` (VARCHAR, PRIMARY KEY) - Unique identifier for episode
- `owner_id` (VARCHAR) - ID of the owner/user
- `episode_type` (VARCHAR) - Type of episode
- `level` (INTEGER) - Episode level (atomic, composite, etc.)
- `title` (VARCHAR) - Episode title
- `content` (TEXT) - Episode content/description
- `summary` (TEXT) - Episode summary
- `timestamp` (TIMESTAMP) - When the episode occurred
- `duration` (FLOAT) - Duration in seconds
- `timezone` (VARCHAR) - Timezone information
- `precision` (VARCHAR) - Temporal precision
- `event_metadata` (TEXT) - JSON metadata
- `structured_data` (TEXT) - JSON structured data
- `search_keywords` (TEXT) - JSON array of search keywords
- `embedding_vector` (TEXT) - JSON array of embedding vector
- `recall_count` (INTEGER) - Number of times accessed
- `importance_score` (FLOAT) - Importance score (0.0-1.0)
- `last_accessed` (TIMESTAMP) - Last access time
- `created_at` (TIMESTAMP) - When the record was created

### `episode_raw_data` Table
- `episode_id` (VARCHAR) - References episodes table
- `raw_data_id` (VARCHAR) - References raw_data table
- Composite primary key on both columns

## Performance Considerations

### Indexing
Nemori automatically creates indexes on commonly queried columns:
- `data_type` on raw_data table
- `source` on raw_data table
- `processed` on raw_data table
- `timestamp` on both tables
- `owner_id` on episodes table
- `episode_type` on episodes table
- `level` on episodes table
- `importance_score` on episodes table

### Connection Pooling
The PostgreSQL implementation uses SQLAlchemy's async engine with connection pooling:
- Default pool size: 10 connections
- Max overflow: 0 (no additional connections beyond pool size)
- You can adjust these settings by modifying the engine creation in the repository

### Batch Operations
Use batch operations when inserting multiple records:

```python
# Store multiple raw data items at once
batch_data = [raw_data1, raw_data2, raw_data3]
batch_ids = await raw_repo.store_raw_data_batch(batch_data)

# Store multiple episodes at once  
batch_episodes = [episode1, episode2, episode3]
episode_ids = await episode_repo.store_episode_batch(batch_episodes)
```

## Backup and Restore

### Backup
```python
# Create a backup
success = await raw_repo.backup("/path/to/backup.sql")
```

### Restore
```python
# Restore from backup
success = await raw_repo.restore("/path/to/backup.sql")
```

Note: Backup and restore operations use PostgreSQL's `pg_dump` and `psql` commands, which must be available in your system PATH.

## Testing

To run PostgreSQL-specific tests:

1. Set up a test database:
```bash
createdb nemori_test
```

2. Set the test environment variable:
```bash
export POSTGRESQL_TEST_URL="postgresql+asyncpg://postgres@localhost/nemori_test"
```

3. Run the tests:
```bash
uv run pytest tests/test_postgresql_storage.py -v
```

## Troubleshooting

### Connection Issues
- Ensure PostgreSQL is running: `pg_ctl status`
- Check connection string format: `postgresql+asyncpg://user:password@host:port/database`
- Verify user permissions: The user must be able to create tables and indexes
- Test connection manually: `psql -h localhost -U postgres -d nemori`

### Performance Issues
- Monitor query performance using `EXPLAIN ANALYZE`
- Consider adding custom indexes for your query patterns
- Adjust batch sizes in the configuration
- Monitor connection pool usage

### Memory Issues
- Reduce `cache_size` in configuration
- Use pagination with `limit` and `offset` for large result sets
- Consider using streaming for very large data exports

## Migration from Other Backends

### From DuckDB
The table schemas are compatible between DuckDB and PostgreSQL. You can export data from DuckDB and import it into PostgreSQL using standard SQL tools.

### From Memory Storage
Memory storage is non-persistent, so there's no direct migration path. You'll need to re-process your raw data to populate the PostgreSQL backend.

## Security Considerations

1. **Connection Security**: Always use SSL in production:
   ```python
   connection_string = "postgresql+asyncpg://user:password@host:port/database?ssl=require"
   ```

2. **User Permissions**: Use a dedicated database user with minimal required permissions

3. **Password Management**: Store passwords in environment variables or secure vaults, not in code

4. **SQL Injection**: The implementation uses parameterized queries and input validation to prevent SQL injection

5. **Data Encryption**: Consider using PostgreSQL's built-in encryption features for sensitive data

## Example Applications

See `examples/postgresql_example.py` for a complete example demonstrating all PostgreSQL features including:
- Configuration setup
- Raw data storage and retrieval
- Episode creation and search
- Batch operations
- Statistics and monitoring
- Error handling