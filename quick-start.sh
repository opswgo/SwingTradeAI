#!/bin/bash
# SwingTradeAI - Quick Start Script
# User Acceptance: User can run system locally in <5 minutes with single command

set -e

echo "🚀 SwingTradeAI Quick Start"
echo "=========================="

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker Desktop and try again."
    exit 1
fi

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating .env file from template..."
    cp .env.example .env
    echo "✅ .env file created. You can modify it later for API keys."
fi

# Create necessary directories
echo "📁 Creating data directories..."
mkdir -p data/{raw,processed,models,backtests}
mkdir -p logs
mkdir -p sql

# Create basic SQL initialization
cat > sql/init.sql << 'EOF'
-- SwingTradeAI Database Initialization
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create basic tables
CREATE TABLE IF NOT EXISTS system_health (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status VARCHAR(20) NOT NULL,
    message TEXT
);

-- Insert initial health check
INSERT INTO system_health (status, message) VALUES ('healthy', 'Database initialized successfully');
EOF

echo "🐳 Starting Docker containers..."
docker-compose up -d

echo "⏳ Waiting for services to be ready..."
sleep 10

# Health check
echo "🔍 Checking service health..."
if curl -f http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ API service is healthy"
else
    echo "⚠️  API service not ready yet (this is normal on first run)"
fi

if docker-compose ps postgres | grep -q "healthy"; then
    echo "✅ Database is healthy"
else
    echo "⚠️  Database still starting up..."
fi

echo ""
echo "🎉 SwingTradeAI is starting up!"
echo "📊 Dashboard: http://localhost:8501"
echo "🔧 API Docs: http://localhost:8000/docs"
echo "🗄️  PgAdmin: http://localhost:8080 (dev profile)"
echo ""
echo "📝 Next steps:"
echo "   1. Wait 1-2 minutes for all services to fully start"
echo "   2. Visit the dashboard to test data pipeline"
echo "   3. Check API docs for endpoint testing"
echo ""
echo "🛑 To stop: docker-compose down"
echo "🔄 To restart: docker-compose restart"
