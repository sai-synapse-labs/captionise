import os
import psycopg2
from psycopg2 import pool
from dotenv import load_dotenv
from captionise.utils.logger import logger

load_dotenv()

class DatabaseManager:
    """Manages PostgreSQL database connections for the Caption Subnet."""
    
    def __init__(self):
        """Initialize the database connection pool."""
        # Get DB configuration from environment variables
        self.db_params = {
            'database': os.getenv('POSTGRES_DB', 'caption_subnet'),
            'user': os.getenv('POSTGRES_USER', 'postgres'),
            'password': os.getenv('POSTGRES_PASSWORD', 'postgres'),
            'host': os.getenv('POSTGRES_HOST', 'localhost'),
            'port': os.getenv('POSTGRES_PORT', '5432')
        }
        
        # Create a connection pool
        self.connection_pool = psycopg2.pool.SimpleConnectionPool(
            1, 10, **self.db_params
        )
        logger.info("Database connection pool initialized")

    def initialize_schema(self):
        """Create the necessary tables if they don't exist."""
        conn = self.get_connection()
        try:
            with conn.cursor() as cursor:
                # Create jobs table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS jobs (
                        job_id SERIAL PRIMARY KEY,
                        audio_segment BYTEA,
                        audio_hash VARCHAR(64) UNIQUE,
                        transcript_source TEXT,
                        transcript_submitted TEXT,
                        language VARCHAR(10),
                        processed BOOLEAN DEFAULT FALSE,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create job_results table for tracking validator scores
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS job_results (
                        result_id SERIAL PRIMARY KEY,
                        job_id INTEGER REFERENCES jobs(job_id),
                        miner_hotkey VARCHAR(64),
                        score FLOAT,
                        wer FLOAT,
                        processing_time FLOAT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                conn.commit()
                logger.success("Database schema initialized successfully")
        except Exception as e:
            conn.rollback()
            logger.error(f"Error initializing database schema: {e}")
        finally:
            self.return_connection(conn)

    def get_connection(self):
        """Get a connection from the pool."""
        return self.connection_pool.getconn()

    def return_connection(self, conn):
        """Return a connection to the pool."""
        self.connection_pool.putconn(conn)

    def close_all(self):
        """Close all database connections."""
        self.connection_pool.closeall()
        logger.info("Closed all database connections")

# Global instance
db_manager = DatabaseManager() 