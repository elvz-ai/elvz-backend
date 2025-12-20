-- Elvz.ai Database Initialization
-- Run on first database setup

-- Enable extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Users table
CREATE TABLE IF NOT EXISTS users (
    id VARCHAR(36) PRIMARY KEY DEFAULT uuid_generate_v4()::text,
    email VARCHAR(255) UNIQUE NOT NULL,
    name VARCHAR(255) NOT NULL,
    hashed_password VARCHAR(255) NOT NULL,
    subscription_tier VARCHAR(50) DEFAULT 'free' NOT NULL,
    is_active BOOLEAN DEFAULT true NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL
);

-- User profiles table
CREATE TABLE IF NOT EXISTS user_profiles (
    id VARCHAR(36) PRIMARY KEY DEFAULT uuid_generate_v4()::text,
    user_id VARCHAR(36) UNIQUE NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    brand_name VARCHAR(255),
    industry VARCHAR(100),
    company_size VARCHAR(50),
    website_url VARCHAR(500),
    brand_voice TEXT,
    tone_preferences JSONB,
    target_audience JSONB,
    buyer_personas JSONB,
    business_goals JSONB,
    content_preferences JSONB,
    social_platforms JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL
);

-- Brand voice profiles table
CREATE TABLE IF NOT EXISTS brand_voice_profiles (
    id VARCHAR(36) PRIMARY KEY DEFAULT uuid_generate_v4()::text,
    user_id VARCHAR(36) UNIQUE NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    tone_characteristics JSONB DEFAULT '{}',
    vocabulary_patterns JSONB DEFAULT '{}',
    sentence_structure JSONB DEFAULT '{}',
    personality_traits JSONB DEFAULT '[]',
    content_patterns JSONB DEFAULT '{}',
    sample_phrases JSONB DEFAULT '[]',
    samples_analyzed INTEGER DEFAULT 0,
    confidence_score FLOAT DEFAULT 0.0,
    analyzed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL
);

-- Content history table
CREATE TABLE IF NOT EXISTS content_history (
    id VARCHAR(36) PRIMARY KEY DEFAULT uuid_generate_v4()::text,
    user_id VARCHAR(36) NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    elf_type VARCHAR(50) NOT NULL,
    content_type VARCHAR(50) NOT NULL,
    platform VARCHAR(50),
    content_text TEXT NOT NULL,
    content_metadata JSONB,
    input_request JSONB NOT NULL,
    performance_metrics JSONB,
    performance_score FLOAT,
    user_rating INTEGER,
    user_feedback TEXT,
    was_published BOOLEAN DEFAULT false,
    was_edited BOOLEAN DEFAULT false,
    model_used VARCHAR(100),
    tokens_used INTEGER,
    generation_cost FLOAT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
    published_at TIMESTAMP WITH TIME ZONE
);

-- Tasks table
CREATE TABLE IF NOT EXISTS tasks (
    id VARCHAR(36) PRIMARY KEY DEFAULT uuid_generate_v4()::text,
    user_id VARCHAR(36) NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    session_id VARCHAR(36),
    elf_type VARCHAR(50) NOT NULL,
    task_type VARCHAR(100) NOT NULL,
    request_data JSONB NOT NULL,
    result_data JSONB,
    status VARCHAR(50) DEFAULT 'pending' NOT NULL,
    error_message TEXT,
    retry_count INTEGER DEFAULT 0,
    execution_time_ms INTEGER,
    tokens_used INTEGER,
    estimated_cost FLOAT,
    execution_trace JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE
);

-- Analytics table
CREATE TABLE IF NOT EXISTS analytics (
    id VARCHAR(36) PRIMARY KEY DEFAULT uuid_generate_v4()::text,
    user_id VARCHAR(36) NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    metric_type VARCHAR(50) NOT NULL,
    platform VARCHAR(50),
    metric_value FLOAT NOT NULL,
    content_id VARCHAR(36),
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    hour_of_day INTEGER,
    day_of_week INTEGER
);

-- API usage table
CREATE TABLE IF NOT EXISTS api_usage (
    id VARCHAR(36) PRIMARY KEY DEFAULT uuid_generate_v4()::text,
    user_id VARCHAR(36) NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    elf_type VARCHAR(50) NOT NULL,
    endpoint VARCHAR(200) NOT NULL,
    prompt_tokens INTEGER DEFAULT 0,
    completion_tokens INTEGER DEFAULT 0,
    total_tokens INTEGER DEFAULT 0,
    estimated_cost FLOAT DEFAULT 0.0,
    model_used VARCHAR(100),
    request_duration_ms INTEGER,
    success BOOLEAN DEFAULT true,
    error_code VARCHAR(50),
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL
);

-- Usage summaries table
CREATE TABLE IF NOT EXISTS usage_summaries (
    id VARCHAR(36) PRIMARY KEY DEFAULT uuid_generate_v4()::text,
    user_id VARCHAR(36) NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    period_type VARCHAR(20) NOT NULL,
    period_start TIMESTAMP WITH TIME ZONE NOT NULL,
    period_end TIMESTAMP WITH TIME ZONE NOT NULL,
    total_requests INTEGER DEFAULT 0,
    total_tokens INTEGER DEFAULT 0,
    total_cost FLOAT DEFAULT 0.0,
    requests_by_elf JSONB DEFAULT '{}',
    tokens_by_elf JSONB DEFAULT '{}',
    cost_by_elf JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_content_history_user_id ON content_history(user_id);
CREATE INDEX IF NOT EXISTS idx_content_history_elf_type ON content_history(elf_type);
CREATE INDEX IF NOT EXISTS idx_content_history_created_at ON content_history(created_at);
CREATE INDEX IF NOT EXISTS idx_tasks_user_id ON tasks(user_id);
CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);
CREATE INDEX IF NOT EXISTS idx_tasks_created_at ON tasks(created_at);
CREATE INDEX IF NOT EXISTS idx_analytics_user_id ON analytics(user_id);
CREATE INDEX IF NOT EXISTS idx_analytics_timestamp ON analytics(timestamp);
CREATE INDEX IF NOT EXISTS idx_api_usage_user_id ON api_usage(user_id);
CREATE INDEX IF NOT EXISTS idx_api_usage_timestamp ON api_usage(timestamp);

-- Create updated_at trigger function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply updated_at triggers
CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_user_profiles_updated_at BEFORE UPDATE ON user_profiles
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_brand_voice_profiles_updated_at BEFORE UPDATE ON brand_voice_profiles
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_usage_summaries_updated_at BEFORE UPDATE ON usage_summaries
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

