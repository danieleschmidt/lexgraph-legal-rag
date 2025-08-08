
                CREATE INDEX IF NOT EXISTS idx_documents_content_fts 
                ON documents USING GIN(to_tsvector('english', content));
                
                CREATE TABLE IF NOT EXISTS search_queries (
                    id SERIAL PRIMARY KEY,
                    query_text VARCHAR(1000) NOT NULL,
                    result_count INTEGER,
                    execution_time_ms INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                