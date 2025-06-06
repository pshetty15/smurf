-- Database initialization script for SMURF
-- This runs automatically when the PostgreSQL container starts

-- Enable the pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Log the initialization
DO $$
BEGIN
    RAISE NOTICE 'Initializing SMURF database with pgvector extension...';
END
$$;