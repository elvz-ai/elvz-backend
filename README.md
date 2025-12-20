# Elvz.ai Backend

Multi-Agent AI Platform providing specialized AI assistants ("Elves") for social media management, SEO optimization, content creation, and business automation.

## 🏗️ Architecture

### Three-Layer Hierarchy

1. **Platform Orchestrator** - Routes requests to appropriate Elf agents
2. **Elf Agents** - Domain-specific macro agents (4 types)
3. **Mini-Agents** - Task-specific micro agents within each Elf

### Elf Agents

| Elf | Purpose | Mini-Agents |
|-----|---------|-------------|
| **Social Media Manager** | Create, optimize, and schedule social content | Strategy, Content Generator, Hashtag Research, Timing Optimizer, Visual Advisor |
| **SEO Optimizer** | Audit websites and provide SEO insights | Technical Auditor, Keyword Analyzer, Competitor Analyzer, Content Optimizer, Meta/Schema Generator |
| **Copy Writer** | Generate high-quality content | Content Strategist, Blog Writer, Ad Copy Writer, Product Description Writer, Tone Adapter |
| **AI Assistant** | Manage tasks, communication, research | Task Manager, Email Manager, Researcher, Meeting Assistant, Document Generator |

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Docker & Docker Compose
- OpenAI API Key (required)
- Anthropic API Key (optional)

### Installation

1. **Clone and setup environment**
```bash
git clone <repository-url>
cd elvz-backend

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

2. **Configure environment variables**
```bash
cp .env.example .env
# Edit .env with your API keys
```

3. **Start with Docker Compose**
```bash
docker-compose up -d
```

4. **Or run locally**
```bash
# Start Redis and PostgreSQL separately, then:
uvicorn app.api.main:app --reload
```

### API Documentation

Once running, access:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 📡 API Endpoints

### Chat (Conversational Interface)
```bash
POST /api/v1/chat
{
  "message": "Create a LinkedIn post about AI in healthcare",
  "session_id": "optional-session-id"
}
```

### Social Media Manager
```bash
POST /api/v1/elves/social-media/create-post
{
  "platform": "linkedin",
  "topic": "The future of AI",
  "content_type": "thought_leadership",
  "goals": ["engagement"]
}
```

### SEO Optimizer
```bash
POST /api/v1/elves/seo/audit-site
{
  "website_url": "https://example.com",
  "include_competitors": true,
  "competitor_urls": ["https://competitor.com"]
}
```

### Copywriter
```bash
POST /api/v1/elves/copywriter/write-blog
{
  "topic": "10 Tips for Remote Work",
  "target_keywords": ["remote work", "productivity"],
  "word_count": 1500,
  "tone": "professional"
}
```

### AI Assistant
```bash
POST /api/v1/elves/assistant/manage-tasks
{
  "message": "I need to prepare for the board meeting and review Q4 budget"
}
```

### WebSocket (Real-time Updates)
```javascript
const ws = new WebSocket('ws://localhost:8000/ws/stream/client-123');
ws.send(JSON.stringify({
  type: 'chat',
  message: 'Create a post about AI',
  user_id: 'user-123'
}));
```

## 🛠️ Technology Stack

- **Language**: Python 3.11+
- **Agent Framework**: LangGraph
- **API Framework**: FastAPI
- **Database**: PostgreSQL
- **Cache/Session**: Redis
- **Queue**: Celery
- **LLM Providers**: OpenAI (GPT-4o), Anthropic (Claude 3.5 Sonnet)

## 📁 Project Structure

```
elvz-backend/
├── app/
│   ├── api/
│   │   ├── routes/
│   │   │   ├── chat.py
│   │   │   ├── social_media.py
│   │   │   ├── seo.py
│   │   │   ├── copywriter.py
│   │   │   └── assistant.py
│   │   ├── deps.py
│   │   ├── websocket.py
│   │   └── main.py
│   ├── agents/
│   │   ├── platform_orchestrator/
│   │   │   ├── intent_classifier.py
│   │   │   ├── router.py
│   │   │   ├── response_aggregator.py
│   │   │   └── orchestrator.py
│   │   └── elves/
│   │       ├── social_media_manager/
│   │       ├── seo_optimizer/
│   │       ├── copywriter/
│   │       └── ai_assistant/
│   ├── core/
│   │   ├── config.py
│   │   ├── database.py
│   │   ├── cache.py
│   │   ├── llm_clients.py
│   │   └── vector_store.py
│   ├── tools/
│   │   ├── registry.py
│   │   ├── social_media_tools.py
│   │   ├── seo_tools.py
│   │   └── research_tools.py
│   ├── models/
│   │   ├── user.py
│   │   ├── content.py
│   │   └── analytics.py
│   └── utils/
│       ├── prompts.py
│       ├── validators.py
│       └── formatters.py
├── tests/
├── migrations/
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── README.md
```

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov-report=html

# Run specific test file
pytest tests/test_api.py -v
```

## 🔧 Development

### Running Locally (Step by Step)

#### Option 1: Full Docker Setup (Recommended)

```bash
# 1. Copy environment file
cp .env.example .env

# 2. Edit .env and add your API keys (REQUIRED)
# At minimum, set: OPENAI_API_KEY=sk-your-key-here

# 3. Start everything with Docker
docker-compose up -d

# 4. Check it's running
curl http://localhost:8000/health
```

#### Option 2: Local Development (without Docker for app)

```bash
# 1. Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Copy and configure environment
cp .env.example .env
# Edit .env with your API keys

# 4. Start PostgreSQL and Redis (using Docker)
docker-compose up -d db redis

# 5. Run the API with hot reload
uvicorn app.api.main:app --reload --port 8000
```

#### Option 3: Minimal Setup (No Docker)

If you don't have Docker, you can run with just Python:

```bash
# 1. Setup
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Edit .env - add your OPENAI_API_KEY

# 3. Run without external services (limited functionality)
# The app will work but without Redis/PostgreSQL persistence
uvicorn app.api.main:app --reload --port 8000
```

### Making API Requests

#### 1. Health Check
```bash
curl http://localhost:8000/health
```

#### 2. Chat API (Main Interface)
```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Create a LinkedIn post about AI trends",
    "user_id": "test-user-123"
  }'
```

#### 3. Social Media Manager - Create Post
```bash
curl -X POST http://localhost:8000/api/v1/elves/social-media/create-post \
  -H "Content-Type: application/json" \
  -d '{
    "platform": "linkedin",
    "topic": "The future of artificial intelligence in healthcare",
    "content_type": "thought_leadership",
    "goals": ["engagement", "awareness"]
  }'
```

#### 4. Social Media Manager - Full Example with Response
```bash
curl -X POST http://localhost:8000/api/v1/elves/social-media/create-post \
  -H "Content-Type: application/json" \
  -d '{
    "platform": "twitter",
    "topic": "5 productivity tips for developers",
    "content_type": "educational",
    "goals": ["engagement"],
    "additional_context": "Target audience is junior developers"
  }' | jq .
```

#### 5. SEO Optimizer - Audit Website
```bash
curl -X POST http://localhost:8000/api/v1/elves/seo/audit-site \
  -H "Content-Type: application/json" \
  -d '{
    "website_url": "https://example.com",
    "include_competitors": true,
    "competitor_urls": ["https://competitor.com"]
  }'
```

#### 6. Copywriter - Write Blog Post
```bash
curl -X POST http://localhost:8000/api/v1/elves/copywriter/write-blog \
  -H "Content-Type: application/json" \
  -d '{
    "topic": "10 Tips for Remote Work Productivity",
    "target_keywords": ["remote work", "productivity", "work from home"],
    "word_count": 1500,
    "tone": "professional"
  }'
```

#### 7. AI Assistant - Manage Tasks
```bash
curl -X POST http://localhost:8000/api/v1/elves/assistant/manage-tasks \
  -H "Content-Type: application/json" \
  -d '{
    "message": "I need to prepare for the board meeting next week and review Q4 budget"
  }'
```

### Using Python requests

```python
import requests

# Chat API
response = requests.post(
    "http://localhost:8000/api/v1/chat",
    json={
        "message": "Create a LinkedIn post about AI trends",
        "user_id": "test-user-123"
    }
)
print(response.json())

# Social Media Manager
response = requests.post(
    "http://localhost:8000/api/v1/elves/social-media/create-post",
    json={
        "platform": "linkedin",
        "topic": "The future of AI",
        "content_type": "thought_leadership",
        "goals": ["engagement"]
    }
)
print(response.json())
```

### Using HTTPie (alternative to curl)

```bash
# Install: pip install httpie

# Chat
http POST localhost:8000/api/v1/chat message="Create a post about AI" user_id="test"

# Social Media
http POST localhost:8000/api/v1/elves/social-media/create-post \
  platform=linkedin \
  topic="AI trends" \
  content_type=thought_leadership \
  goals:='["engagement"]'
```

### Celery Workers (for background tasks)

```bash
# Run Celery worker
celery -A app.worker worker --loglevel=info

# Run Celery beat (scheduled tasks)
celery -A app.worker beat --loglevel=info
```

### Code Quality

```bash
# Format code
black app/
isort app/

# Lint
ruff app/

# Type check
mypy app/
```

## 📊 Content Generation Methods

| Method | Use Case | Description |
|--------|----------|-------------|
| **RAG + Dynamic Context** | Knowledge-grounded tasks | Retrieves knowledge from vector DB, injects user context |
| **Few-Shot + RAG** | Creative content | Uses high-performing examples as templates |
| **Tool-Augmented Generation** | Data-driven tasks | LLM orchestrates tool calls for real-time data |
| **RAFT** | Brand consistency | Uses analyzed voice profile for consistent tone |

## 🔐 Authentication

The API uses JWT Bearer tokens. In development mode, requests without tokens use a default user.

```bash
# Header format
Authorization: Bearer <your-jwt-token>
```

## 📈 Performance Targets

- Simple queries: < 3 seconds
- Complex workflows: < 10 seconds
- Multi-Elf coordination: < 20 seconds
- Success rate: > 95%
- Cache hit rate: > 70%

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details.

## 🆘 Support

For issues or questions, please open a GitHub issue.
