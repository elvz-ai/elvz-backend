# Elvz.ai — Database Contract

This document defines all the tables, columns, and API operations required by the **Elvz.ai backend**. If you manage the PostgreSQL database, please expose these operations as REST endpoints so that the Elvz backend can interact with your database without a direct connection.

---

## Connection Requirements

- **Database**: PostgreSQL 14+
- **Extensions needed**: `uuid-ossp` (for UUIDs), `pg_trgm` (optional, for text search)
- **Column types used**: `VARCHAR`, `TEXT`, `BOOLEAN`, `INTEGER`, `FLOAT`, `TIMESTAMP WITH TIME ZONE`, `JSONB`, `VARCHAR[]` (array)

---

## Tables & Schema

### 1. `users`
Core account table. Every user in the system must have a row here.

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | `VARCHAR(36)` | PRIMARY KEY | UUID string |
| `email` | `VARCHAR(255)` | UNIQUE, NOT NULL, INDEX | User email address |
| `name` | `VARCHAR(255)` | NOT NULL | Display name |
| `hashed_password` | `VARCHAR(255)` | NOT NULL | Bcrypt hashed password |
| `subscription_tier` | `VARCHAR(50)` | NOT NULL, DEFAULT `'free'` | `free` / `starter` / `professional` / `enterprise` |
| `is_active` | `BOOLEAN` | NOT NULL, DEFAULT `true` | Whether account is active |
| `created_at` | `TIMESTAMPTZ` | NOT NULL, DEFAULT `now()` | |
| `updated_at` | `TIMESTAMPTZ` | NOT NULL, DEFAULT `now()` | Auto-update on change |

---

### 2. `user_profiles`
Brand and business information per user. Used to personalize AI-generated content.

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | `VARCHAR(36)` | PRIMARY KEY | UUID string |
| `user_id` | `VARCHAR(36)` | FK → `users.id` CASCADE, UNIQUE | One profile per user |
| `brand_name` | `VARCHAR(255)` | NULLABLE | Company/brand name |
| `industry` | `VARCHAR(100)` | NULLABLE | e.g. `SaaS`, `Healthcare` |
| `company_size` | `VARCHAR(50)` | NULLABLE | e.g. `1-10`, `50-200` |
| `website_url` | `VARCHAR(500)` | NULLABLE | |
| `brand_voice` | `TEXT` | NULLABLE | Text description of brand tone |
| `tone_preferences` | `JSONB` | NULLABLE | `{"formal": 0.7, "friendly": 0.8}` |
| `target_audience` | `JSONB` | NULLABLE | Demographics object |
| `buyer_personas` | `JSONB` | NULLABLE | Array of persona objects |
| `business_goals` | `JSONB` | NULLABLE | `["brand_awareness", "leads"]` |
| `content_preferences` | `JSONB` | NULLABLE | Content style preferences |
| `social_platforms` | `JSONB` | NULLABLE | `["linkedin", "instagram"]` |
| `created_at` | `TIMESTAMPTZ` | NOT NULL, DEFAULT `now()` | |
| `updated_at` | `TIMESTAMPTZ` | NOT NULL, DEFAULT `now()` | |

---

### 3. `brand_voice_profiles`
AI-analyzed brand voice patterns. Used to make generated content match the user's writing style.

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | `VARCHAR(36)` | PRIMARY KEY | UUID string |
| `user_id` | `VARCHAR(36)` | FK → `users.id` CASCADE, UNIQUE | One per user |
| `tone_characteristics` | `JSONB` | DEFAULT `{}` | `{"formal":0.6,"friendly":0.8}` |
| `vocabulary_patterns` | `JSONB` | DEFAULT `{}` | `{"common_words":["innovative"],"jargon_level":"moderate"}` |
| `sentence_structure` | `JSONB` | DEFAULT `{}` | `{"avg_sentence_length":15,"uses_questions":true}` |
| `personality_traits` | `JSONB` | DEFAULT `[]` | `["innovative","customer-centric"]` |
| `content_patterns` | `JSONB` | DEFAULT `{}` | `{"typical_cta_style":"action-oriented"}` |
| `sample_phrases` | `JSONB` | DEFAULT `[]` | Example phrases array |
| `samples_analyzed` | `INTEGER` | DEFAULT `0` | How many content samples were analyzed |
| `confidence_score` | `FLOAT` | DEFAULT `0.0` | 0.0 to 1.0 |
| `analyzed_at` | `TIMESTAMPTZ` | NOT NULL, DEFAULT `now()` | |
| `updated_at` | `TIMESTAMPTZ` | NOT NULL, DEFAULT `now()` | |

---

### 4. `conversations`
Each chat session between a user and the AI.

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | `VARCHAR(36)` | PRIMARY KEY | UUID string |
| `user_id` | `VARCHAR(36)` | FK → `users.id` CASCADE, NOT NULL, INDEX | |
| `thread_id` | `VARCHAR(36)` | UNIQUE, NOT NULL, INDEX | LangGraph checkpoint thread ID |
| `title` | `VARCHAR(255)` | NULLABLE | Auto-generated conversation title |
| `status` | `VARCHAR(50)` | NOT NULL, DEFAULT `'active'` | `active` / `archived` / `deleted` |
| `metadata` | `JSONB` | DEFAULT `{}` | `{"platforms":["linkedin"],"total_tokens":15000}` |
| `created_at` | `TIMESTAMPTZ` | NOT NULL, DEFAULT `now()` | |
| `updated_at` | `TIMESTAMPTZ` | NOT NULL, DEFAULT `now()` | |
| `last_message_at` | `TIMESTAMPTZ` | NULLABLE | Timestamp of last message |

**Indexes needed:**
- `(user_id, status)`
- `(last_message_at)`

---

### 5. `messages`
Individual messages within a conversation.

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | `VARCHAR(36)` | PRIMARY KEY | UUID string |
| `conversation_id` | `VARCHAR(36)` | FK → `conversations.id` CASCADE, NOT NULL, INDEX | |
| `role` | `VARCHAR(20)` | NOT NULL | `user` / `assistant` / `system` |
| `content` | `TEXT` | NOT NULL | Message text |
| `metadata` | `JSONB` | DEFAULT `{}` | `{"intent":"artifact","tokens_used":500,"model":"gpt-4o"}` |
| `created_at` | `TIMESTAMPTZ` | NOT NULL, DEFAULT `now()` | |

**Indexes needed:**
- `(conversation_id, created_at)`

---

### 6. `query_decompositions`
Tracks how complex multi-platform queries are split up.

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | `VARCHAR(36)` | PRIMARY KEY | UUID string |
| `conversation_id` | `VARCHAR(36)` | FK → `conversations.id` CASCADE, NOT NULL, INDEX | |
| `message_id` | `VARCHAR(36)` | FK → `messages.id` SET NULL, NULLABLE | |
| `original_query` | `TEXT` | NOT NULL | The full user query |
| `is_multi_platform` | `BOOLEAN` | DEFAULT `false` | Was this decomposed? |
| `decomposed_queries` | `JSONB` | DEFAULT `[]` | `[{"platform":"linkedin","query":"...","priority":1}]` |
| `execution_strategy` | `VARCHAR(50)` | DEFAULT `'sequential'` | `sequential` / `parallel` |
| `created_at` | `TIMESTAMPTZ` | NOT NULL, DEFAULT `now()` | |

---

### 7. `artifact_batches`
Groups of content generated from a single request (e.g. one request → LinkedIn + Instagram + Facebook posts).

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | `VARCHAR(36)` | PRIMARY KEY | UUID string |
| `conversation_id` | `VARCHAR(36)` | FK → `conversations.id` CASCADE, NOT NULL, INDEX | |
| `query_decomposition_id` | `VARCHAR(36)` | FK → `query_decompositions.id` SET NULL, NULLABLE | |
| `platforms` | `VARCHAR[]` | NOT NULL | `["linkedin","instagram"]` |
| `topic` | `VARCHAR(500)` | NULLABLE | What the content is about |
| `status` | `VARCHAR(50)` | NOT NULL, DEFAULT `'pending'` | `pending` / `in_progress` / `complete` / `partial` / `failed` |
| `execution_strategy` | `VARCHAR(50)` | DEFAULT `'sequential'` | |
| `total_tokens_used` | `INTEGER` | DEFAULT `0` | |
| `total_cost` | `FLOAT` | DEFAULT `0.0` | USD cost |
| `execution_time_ms` | `INTEGER` | DEFAULT `0` | |
| `metadata` | `JSONB` | DEFAULT `{}` | |
| `created_at` | `TIMESTAMPTZ` | NOT NULL, DEFAULT `now()` | |
| `completed_at` | `TIMESTAMPTZ` | NULLABLE | |

---

### 8. `artifacts`
Individual generated content pieces (social post, image, blog post, etc.).

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | `VARCHAR(36)` | PRIMARY KEY | UUID string |
| `conversation_id` | `VARCHAR(36)` | FK → `conversations.id` CASCADE, NOT NULL, INDEX | |
| `message_id` | `VARCHAR(36)` | FK → `messages.id` SET NULL, NULLABLE | |
| `batch_id` | `VARCHAR(36)` | FK → `artifact_batches.id` SET NULL, NULLABLE, INDEX | |
| `artifact_type` | `VARCHAR(50)` | NOT NULL | `social_post` / `image` / `video_script` / `hashtags` / `blog_post` / `ad_copy` |
| `platform` | `VARCHAR(50)` | NULLABLE, INDEX | `linkedin` / `instagram` / `facebook` / `twitter` / `tiktok` |
| `content` | `JSONB` | NOT NULL | See content structure below |
| `status` | `VARCHAR(50)` | NOT NULL, DEFAULT `'draft'` | `draft` / `approved` / `published` / `rejected` / `archived` |
| `user_rating` | `INTEGER` | NULLABLE | 1–5 stars |
| `user_feedback` | `TEXT` | NULLABLE | Free text feedback |
| `was_edited` | `BOOLEAN` | DEFAULT `false` | Did user edit this content? |
| `was_published` | `BOOLEAN` | DEFAULT `false` | Was it published? |
| `generation_metadata` | `JSONB` | DEFAULT `{}` | `{"model":"gpt-4o","tokens_used":1500,"cost":0.05}` |
| `created_at` | `TIMESTAMPTZ` | NOT NULL, DEFAULT `now()` | |
| `updated_at` | `TIMESTAMPTZ` | NOT NULL, DEFAULT `now()` | |
| `published_at` | `TIMESTAMPTZ` | NULLABLE | |

**`content` JSONB structure (for `social_post` type):**
```json
{
  "text": "Post content here...",
  "hook": "Attention-grabbing opener",
  "cta": "Call to action text",
  "hashtags": ["#AI", "#Innovation"],
  "image_url": "https://firebase.../image.jpg",
  "image_description": "Description of generated image",
  "schedule": {"datetime": "2024-02-15T10:00:00Z", "timezone": "UTC"},
  "engagement_estimate": {"reach": 1000, "engagement_rate": 0.05}
}
```

**Indexes needed:**
- `(artifact_type, platform)`
- `(status)`
- `(created_at)`

---

### 9. `hitl_requests`
Human-in-the-Loop: when the AI needs to ask the user a question or get approval before proceeding.

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | `VARCHAR(36)` | PRIMARY KEY | UUID string |
| `conversation_id` | `VARCHAR(36)` | FK → `conversations.id` CASCADE, NOT NULL, INDEX | |
| `artifact_id` | `VARCHAR(36)` | FK → `artifacts.id` SET NULL, NULLABLE | Related artifact if any |
| `request_type` | `VARCHAR(50)` | NOT NULL | `clarification` / `approval` / `edit` / `platform_selection` / `data_missing` / `confirmation` |
| `status` | `VARCHAR(50)` | NOT NULL, DEFAULT `'pending'` | `pending` / `approved` / `rejected` / `modified` / `expired` / `cancelled` |
| `prompt` | `TEXT` | NOT NULL | The question shown to the user |
| `options` | `JSONB` | NULLABLE | `[{"id":"opt1","label":"LinkedIn","description":"..."}]` |
| `response` | `TEXT` | NULLABLE | User's text response |
| `selected_options` | `JSONB` | NULLABLE | Array of selected option IDs |
| `context` | `JSONB` | DEFAULT `{}` | Internal state snapshot for graph resumption |
| `expires_at` | `TIMESTAMPTZ` | NULLABLE | When this request expires (default 5 min) |
| `requested_at` | `TIMESTAMPTZ` | NOT NULL, DEFAULT `now()` | |
| `responded_at` | `TIMESTAMPTZ` | NULLABLE | When user responded |
| `requester_notes` | `TEXT` | NULLABLE | Why this was triggered |
| `responder_notes` | `TEXT` | NULLABLE | User's additional notes |

**Indexes needed:**
- `(status)`
- `(conversation_id, status)`
- `(expires_at)`

---

## Required API Endpoints

Please expose these endpoints. All requests/responses use **JSON**. All IDs are UUID strings.

---

### Users

| Method | Path | Description | Request Body | Response |
|--------|------|-------------|-------------|----------|
| `GET` | `/users/{id}` | Get user by ID | — | User object |
| `POST` | `/users` | Create user | `{id, email, name, hashed_password, subscription_tier?}` | User object |
| `GET` | `/users/{id}/profile` | Get user + profile + brand voice | — | `{user, profile, brand_voice}` |
| `PUT` | `/users/{id}/profile` | Create/update user profile | Profile fields | Profile object |

---

### Conversations

| Method | Path | Description | Request Body | Response |
|--------|------|-------------|-------------|----------|
| `POST` | `/conversations` | Create conversation | `{id, user_id, thread_id, title?, metadata?}` | Conversation object |
| `GET` | `/conversations/{id}` | Get by ID | — | Conversation object |
| `GET` | `/conversations?user_id=&status=&limit=&offset=` | List user conversations | — | `{items: [], total: N}` |
| `PATCH` | `/conversations/{id}` | Update (title, status, metadata, last_message_at) | Partial fields | Conversation object |
| `DELETE` | `/conversations/{id}` | Hard delete | — | `{success: true}` |
| `GET` | `/conversations/count?user_id=&status=` | Count conversations | — | `{count: N}` |

---

### Messages

| Method | Path | Description | Request Body | Response |
|--------|------|-------------|-------------|----------|
| `POST` | `/messages` | Create message | `{id, conversation_id, role, content, metadata?}` | Message object |
| `GET` | `/messages?conversation_id=&limit=&order=asc\|desc` | List messages | — | `{items: []}` |

---

### Artifacts

| Method | Path | Description | Request Body | Response |
|--------|------|-------------|-------------|----------|
| `POST` | `/artifact-batches` | Create batch | `{id, conversation_id, platforms, topic?, status?}` | Batch object |
| `GET` | `/artifact-batches/{id}` | Get batch | — | Batch object |
| `PATCH` | `/artifact-batches/{id}` | Update batch status | `{status, completed_at?, total_tokens_used?, total_cost?}` | Batch object |
| `POST` | `/artifacts` | Create artifact | `{id, conversation_id, artifact_type, platform?, content, batch_id?, message_id?}` | Artifact object |
| `GET` | `/artifacts/{id}` | Get artifact | — | Artifact object |
| `GET` | `/artifacts?conversation_id=&platform=&artifact_type=&limit=` | List artifacts | — | `{items: []}` |
| `GET` | `/artifacts?batch_id=` | Get all artifacts in a batch | — | `{items: []}` |
| `PATCH` | `/artifacts/{id}` | Update (status, user_rating, user_feedback, was_edited, was_published) | Partial fields | Artifact object |
| `DELETE` | `/artifacts/{id}` | Delete artifact | — | `{success: true}` |

---

### HITL Requests

| Method | Path | Description | Request Body | Response |
|--------|------|-------------|-------------|----------|
| `POST` | `/hitl-requests` | Create HITL request | `{id, conversation_id, request_type, prompt, options?, context?, expires_at?}` | HITL object |
| `GET` | `/hitl-requests/{id}` | Get by ID | — | HITL object |
| `GET` | `/hitl-requests?conversation_id=&status=` | List for a conversation | — | `{items: []}` |
| `PATCH` | `/hitl-requests/{id}` | Respond to HITL | `{status, response?, selected_options?, responded_at}` | HITL object |

---

## Standard Response Format

All endpoints should return:

**Success:**
```json
{
  "data": { ...object or array... },
  "success": true
}
```

**Error:**
```json
{
  "success": false,
  "error": "Human-readable message",
  "code": "ERROR_CODE"
}
```

**HTTP Status Codes:**
- `200` — Success
- `201` — Created
- `404` — Not found
- `409` — Conflict (duplicate ID)
- `422` — Validation error
- `500` — Server error

---

## Notes

1. All `id` fields are **UUID v4 strings** (36 chars) — the Elvz backend generates them, your DB just stores them.
2. `JSONB` columns accept any valid JSON object or array.
3. `metadata` and `content` columns must support arbitrary nested JSON.
4. `TIMESTAMPTZ` (timestamp with timezone) — always store in UTC.
5. Foreign key `ON DELETE CASCADE` means: when a `conversation` is deleted, all its messages/artifacts/hitl_requests are automatically deleted too.
6. The Elvz backend manages its own UUIDs — your API should accept client-supplied IDs on `POST` (no auto-generation needed on the DB side).
