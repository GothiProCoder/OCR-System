-- ============================================
-- PostgreSQL Initialization Script
-- ============================================
-- This script runs automatically when the PostgreSQL
-- container starts for the first time.

-- Create extensions (if needed)
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";  -- For text search

-- Grant permissions (if using custom user)
-- GRANT ALL PRIVILEGES ON DATABASE ocr_system TO ocr_admin;

-- Log successful initialization
DO $$
BEGIN
    RAISE NOTICE 'OCR System database initialized successfully!';
END $$;
