#!/bin/bash
set -e

echo "🚀 Redis OpenAI Agents - Quick Setup"
echo ""

# Check if running from project root
if [ ! -f "pyproject.toml" ]; then
    echo "❌ Error: Please run this script from the redis-openai-agents project root"
    exit 1
fi

# Step 1: Check for .env file
echo "1️⃣  Checking environment configuration..."
if [ ! -f "examples/.env" ]; then
    if [ -f ".env" ]; then
        echo "   Copying .env to examples/"
        cp .env examples/.env
    elif [ -f "examples/.env.example" ]; then
        echo "   Creating examples/.env from template"
        cp examples/.env.example examples/.env
        echo "   ⚠️  Please edit examples/.env and add your OPENAI_API_KEY"
        echo ""
    else
        echo "   ❌ Error: No .env.example found"
        exit 1
    fi
else
    echo "   ✅ examples/.env exists"
fi

# Step 2: Check if OPENAI_API_KEY is set
if grep -q "OPENAI_API_KEY=sk-..." examples/.env 2>/dev/null; then
    echo ""
    echo "⚠️  OPENAI_API_KEY is not set in examples/.env"
    echo ""
    echo "Please edit examples/.env and add your OpenAI API key:"
    echo "   nano examples/.env"
    echo ""
    echo "Then run: make examples"
    exit 0
fi

echo "   ✅ OPENAI_API_KEY is configured"
echo ""

# Step 3: Check Docker
echo "2️⃣  Checking Docker..."
if ! command -v docker &> /dev/null; then
    echo "   ❌ Docker not found. Please install Docker Desktop."
    exit 1
fi

if ! docker info &> /dev/null; then
    echo "   ❌ Docker daemon not running. Please start Docker Desktop."
    exit 1
fi

echo "   ✅ Docker is running"
echo ""

# Step 4: Check Docker Compose
echo "3️⃣  Checking Docker Compose..."
if docker compose version &> /dev/null; then
    echo "   ✅ Docker Compose available"
else
    echo "   ❌ Docker Compose not found"
    exit 1
fi
echo ""

# Step 5: Offer to start services
echo "4️⃣  Ready to start!"
echo ""
echo "Start the development environment with:"
echo "   cd examples && docker compose up --build"
echo ""
echo "Or use the Makefile:"
echo "   make examples"
echo ""
echo "Then open:"
echo "   - Jupyter:      http://localhost:8888"
echo "   - Redis Insight: http://localhost:8001"
echo ""
