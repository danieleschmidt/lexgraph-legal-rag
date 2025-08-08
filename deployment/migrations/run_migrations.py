#!/usr/bin/env python3
import os
import psycopg2
from pathlib import Path

def run_migrations():
    """Run database migrations."""
    db_url = os.environ.get('DATABASE_URL')
    if not db_url:
        print("DATABASE_URL environment variable not set")
        return False
    
    migrations_dir = Path(__file__).parent
    migration_files = sorted(migrations_dir.glob("*.sql"))
    
    try:
        conn = psycopg2.connect(db_url)
        cursor = conn.cursor()
        
        # Create migrations table if it doesn't exist
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version VARCHAR(10) PRIMARY KEY,
                applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        for migration_file in migration_files:
            version = migration_file.name.split('_')[0]
            
            # Check if migration already applied
            cursor.execute("SELECT version FROM schema_migrations WHERE version = %s", (version,))
            if cursor.fetchone():
                print(f"Migration {version} already applied, skipping")
                continue
            
            print(f"Applying migration {version}...")
            with open(migration_file) as f:
                cursor.execute(f.read())
            
            # Record migration as applied
            cursor.execute("INSERT INTO schema_migrations (version) VALUES (%s)", (version,))
            conn.commit()
            print(f"Migration {version} applied successfully")
        
        cursor.close()
        conn.close()
        print("All migrations applied successfully")
        return True
        
    except Exception as e:
        print(f"Migration failed: {e}")
        return False

if __name__ == "__main__":
    run_migrations()
