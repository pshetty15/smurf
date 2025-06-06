-- Migration: Add source_type index for better filtering
-- Run this after core.sql and github.sql

-- Add source_type index if it doesn't exist
CREATE INDEX IF NOT EXISTS idx_crawled_pages_source_type ON crawled_pages(source_id);

-- Add processor metadata table for future use
CREATE TABLE IF NOT EXISTS processors (
    name TEXT PRIMARY KEY,
    enabled BOOLEAN DEFAULT true,
    config JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Insert default processors
INSERT INTO processors (name, enabled, config) VALUES 
    ('web', true, '{"max_depth": 3, "timeout": 30}'),
    ('github', true, '{"cache_dir": "/app/data/repos", "max_file_size": 1048576}')
ON CONFLICT (name) DO NOTHING;

-- Add processor tracking to sources table if not exists
ALTER TABLE sources ADD COLUMN IF NOT EXISTS processor_name TEXT DEFAULT 'web';
CREATE INDEX IF NOT EXISTS idx_sources_processor ON sources(processor_name);