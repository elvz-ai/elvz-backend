# Mini-Agents Architecture

This document explains how each mini-agent works and where they get their data.

---

## 🏗️ Mini-Agent Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    User Request                                          │
│  "Create a LinkedIn post about AI trends in 2025"                       │
└─────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STRATEGY AGENT (runs first)                                             │
│  ───────────────────────────────                                         │
│  Data Sources:                                                           │
│  1. User request (topic, platform, goals)                               │
│  2. Vector DB → Best practices for LinkedIn thought leadership          │
│  3. Context → Brand voice, industry info                                │
│                                                                          │
│  Output: Strategy brief (tone, key messages, target audience)           │
└─────────────────────────────────────────────────────────────────────────┘
                                   │
                    strategy flows to ↓ all parallel agents
                                   │
     ┌──────────────┬──────────────┼──────────────┬──────────────┐
     ▼              ▼              ▼              ▼              ▼
┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐   (Already done)
│ CONTENT │   │ HASHTAG │   │ TIMING  │   │ VISUAL  │
│  AGENT  │   │  AGENT  │   │  AGENT  │   │  AGENT  │
└─────────┘   └─────────┘   └─────────┘   └─────────┘
```

---

## 1️⃣ Strategy Agent

| Aspect | Details |
|--------|---------|
| **Method** | RAG + Dynamic Context |
| **File** | `mini_agents/strategy.py` |

### Data Sources:

```python
# 1. User Request (from state)
request = state.get("user_request", {})
platform = request.get("platform")      # "linkedin"
topic = request.get("topic")            # "AI trends in 2025"
goals = request.get("goals")            # ["engagement"]

# 2. Vector Database (RAG) - Best Practices
knowledge_context = await vector_store.search_knowledge(
    query=f"Best practices for {content_type} on {platform} about {topic}",
    platform=platform,
    top_k=3,
)
# Returns: ["Use line breaks on LinkedIn", "Hook in first line", ...]

# 3. Context (brand info from user profile)
brand_context = context.get("brand_voice", {})
# Returns: {"tone": "professional", "industry": "tech", ...}
```

### Output → Strategy Brief:

```json
{
  "tone": "Professional yet conversational",
  "key_messages": ["AI is transforming work", "2025 is pivotal"],
  "cta_approach": "Ask a thought-provoking question",
  "target_audience": "Tech leaders, CTOs"
}
```

---

## 2️⃣ Content Generator Agent

| Aspect | Details |
|--------|---------|
| **Method** | Few-Shot + RAFT (brand voice) |
| **File** | `mini_agents/content.py` |

### Data Sources:

```python
# 1. Strategy (from previous agent)
strategy = state.get("strategy")
tone = strategy.get("tone")              # "Professional yet conversational"
key_messages = strategy.get("key_messages")  # ["AI is transforming work", ...]

# 2. Vector DB - High-performing examples (Few-Shot)
examples = await vector_store.search_examples(
    platform=platform,
    content_type=content_type,
    topic=topic,
    top_k=3,
)
# Returns: Past successful posts as templates

# 3. Redis Cache - Brand Voice Profile (RAFT)
brand_voice = await cache.get_brand_voice(user_id)
# Returns: {"vocabulary": [...], "phrases_to_avoid": [...], ...}

# 4. User Request
topic = request.get("topic")  # "AI trends in 2025"
```

### Output → 3 Content Variations:

```json
{
  "variations": [
    {"version": "hook_focused", "post_text": "2025 isn't coming. It's here..."},
    {"version": "story_focused", "post_text": "Last week, I saw an AI..."},
    {"version": "value_focused", "post_text": "3 AI trends for 2025:..."}
  ]
}
```

---

## 3️⃣ Hashtag Research Agent

| Aspect | Details |
|--------|---------|
| **Method** | Tool-Augmented Generation (TAG) |
| **File** | `mini_agents/hashtag.py` |

### Data Sources:

```python
# 1. Tools (External APIs simulated)
hashtag_tool = tool_registry.get_tool("hashtag_research")
tool_results = await hashtag_tool.execute({
    "topic": topic,
    "platform": platform,
})
# Returns: {"hashtags": [{"tag": "#AI", "volume": 1.2M, "trending": true}, ...]}

# 2. Strategy (key messages for relevance)
key_messages = state.get("strategy", {}).get("key_messages", [])

# 3. Platform limits
HASHTAG_LIMITS = {
    "twitter": 2,
    "linkedin": 5,
    "instagram": 30,
}
```

### Output → Optimized Hashtags:

```json
{
  "hashtags": [
    {"tag": "#AI", "volume": "high", "relevance": 0.95},
    {"tag": "#AITrends2025", "volume": "medium", "relevance": 0.92},
    {"tag": "#FutureOfWork", "volume": "high", "relevance": 0.88}
  ]
}
```

---

## 4️⃣ Timing Optimizer Agent

| Aspect | Details |
|--------|---------|
| **Method** | Tool-Augmented Generation (Analytics) |
| **File** | `mini_agents/timing.py` |

### Data Sources:

```python
# 1. Tools (Analytics APIs)
timing_tool = tool_registry.get_tool("optimal_timing")
tool_results = await timing_tool.execute({
    "platform": platform,
    "audience_timezone": context.get("timezone", "UTC"),
})
# Returns: {"best_times": [{"day": "Tuesday", "hour": 10, "score": 0.95}, ...]}

# 2. Strategy (target audience)
target_audience = state.get("strategy", {}).get("target_audience")
# "Tech leaders, CTOs" → suggests morning weekday posts

# 3. User context
timezone = context.get("timezone", "UTC")
```

### Output → Posting Schedule:

```json
{
  "timing": {
    "recommended_datetime": "2025-12-19 09:30",
    "timezone": "UTC",
    "day_of_week": "Tuesday",
    "confidence": 0.95,
    "reasoning": "Highest engagement for tech audience"
  }
}
```

---

## 5️⃣ Visual Advisor Agent

| Aspect | Details |
|--------|---------|
| **Method** | RAG + Few-Shot |
| **File** | `mini_agents/visual.py` |

### Data Sources:

```python
# 1. Vector DB - Visual best practices
visual_practices = await vector_store.search_knowledge(
    query=f"Visual content best practices for {platform}",
    category="visual_design",
)
# Returns: ["Carousels get 3x engagement", "Use blue tones on LinkedIn", ...]

# 2. Strategy (tone, key messages)
tone = state.get("strategy", {}).get("tone")
key_messages = state.get("strategy", {}).get("key_messages")

# 3. Platform specs (hardcoded)
PLATFORM_SPECS = {
    "linkedin": {"image": "1200x628", "carousel": "1080x1080"},
    "twitter": {"image": "1600x900"},
}
```

### Output → Visual Recommendations:

```json
{
  "visual_advice": {
    "primary_recommendation": {
      "type": "carousel",
      "description": "5-slide carousel with AI trend per slide"
    },
    "design_specs": {
      "dimensions": "1080x1080",
      "colors": ["#0077B5", "#FFFFFF"]
    }
  }
}
```

---

## 📊 Complete Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DATA SOURCES                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐ │
│  │ User Request │   │ Vector DB    │   │ Redis Cache  │   │ External     │ │
│  │              │   │ (Pinecone/   │   │              │   │ Tools/APIs   │ │
│  │ • topic      │   │  Weaviate)   │   │ • brand voice│   │              │ │
│  │ • platform   │   │              │   │ • sessions   │   │ • hashtag    │ │
│  │ • goals      │   │ • best       │   │ • tool cache │   │   research   │ │
│  │ • message    │   │   practices  │   │              │   │ • analytics  │ │
│  │              │   │ • examples   │   │              │   │ • trending   │ │
│  └──────┬───────┘   └──────┬───────┘   └──────┬───────┘   └──────┬───────┘ │
│         │                  │                  │                  │          │
└─────────┼──────────────────┼──────────────────┼──────────────────┼──────────┘
          │                  │                  │                  │
          ▼                  ▼                  ▼                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           MINI-AGENTS                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────┐    ┌────────────┐    ┌────────────┐    ┌────────────┐       │
│  │  Strategy  │───▶│  Content   │    │  Hashtag   │    │  Timing    │       │
│  │   Agent    │    │   Agent    │    │   Agent    │    │   Agent    │       │
│  │            │    │            │    │            │    │            │       │
│  │ Uses:      │    │ Uses:      │    │ Uses:      │    │ Uses:      │       │
│  │ • Request  │    │ • Strategy │    │ • Tools    │    │ • Tools    │       │
│  │ • VectorDB │    │ • VectorDB │    │ • Strategy │    │ • Strategy │       │
│  │ • Context  │    │ • Cache    │    │            │    │            │       │
│  └────────────┘    └────────────┘    └────────────┘    └────────────┘       │
│                                                                              │
│                    ┌────────────┐                                            │
│                    │  Visual    │                                            │
│                    │   Agent    │                                            │
│                    │ Uses:      │                                            │
│                    │ • VectorDB │                                            │
│                    │ • Strategy │                                            │
│                    └────────────┘                                            │
└─────────────────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  LLM (Gemini) - Generates content based on all gathered data                │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🔑 Summary

| Agent | Method | Primary Data Source | Secondary Sources |
|-------|--------|---------------------|-------------------|
| **Strategy** | RAG + Context | Vector DB (best practices) | User request, brand context |
| **Content** | Few-Shot + RAFT | Strategy output | Vector DB (examples), brand voice cache |
| **Hashtag** | TAG (Tools) | External hashtag tool | Strategy (relevance) |
| **Timing** | TAG (Analytics) | Analytics tool | User timezone, audience |
| **Visual** | RAG + Few-Shot | Vector DB (visual practices) | Strategy (tone) |

---

## ⚠️ Current Limitations

If Vector Database is not configured with proper embeddings, the following errors may appear:

```
Knowledge retrieval failed: OpenAI API key error
Visual practices retrieval failed: OpenAI API key error
Example retrieval failed: OpenAI API key error
```

This happens because the Vector Database (Pinecone/Weaviate) uses OpenAI for embeddings by default. When using only Gemini, the vector search falls back to **hardcoded best practices** instead.

### Solutions:

1. **Configure Vector DB to use Gemini embeddings** - Update the embedding service
2. **Expand the hardcoded fallback data** - Add more comprehensive best practices
3. **Use a local embedding model** - No external API required

---

## 🔄 LLM Calls per Agent

| Agent | LLM Calls | Purpose |
|-------|-----------|---------|
| Strategy | 1 | Generate strategy brief |
| Content | 1 | Generate 3 content variations |
| Hashtag | 1 | Synthesize hashtag recommendations |
| Timing | 1 | Generate posting schedule |
| Visual | 1 | Generate visual recommendations |

**Total LLM calls per request: 5** (1 sequential + 4 parallel)

---

## 📁 File Structure

```
app/agents/elves/social_media_manager/
├── orchestrator.py          # Main workflow controller
├── state.py                 # State definitions
└── mini_agents/
    ├── __init__.py
    ├── strategy.py          # Strategy Agent
    ├── content.py           # Content Generator Agent
    ├── hashtag.py           # Hashtag Research Agent
    ├── timing.py            # Timing Optimizer Agent
    └── visual.py            # Visual Advisor Agent
```

