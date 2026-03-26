# Plan B: Artifact Versioning with `parent_id` Rows

## Context

Currently, artifact edits are tracked in a JSONB `edit_history` array on a single row. This plan evaluates replacing it with **multiple rows** using a `parent_id` FK, where each modification creates a new row pointing back to the original.

**Current (Approach A):** 1 row per artifact, `content` + `original_content` + `edit_history[]`
**Proposed (Approach B):** N rows per artifact, `parent_id` FK, no `edit_history`/`original_content` columns

---

## Data Model

```
Original artifact:  id=art-1, parent_id=NULL, content={v0}
After edit 1:       id=art-2, parent_id=art-1, content={v1}
After edit 2:       id=art-3, parent_id=art-1, content={v2}  ← current
```

- `parent_id IS NULL` → original (no `is_edited` column needed)
- `parent_id IS NOT NULL` → modified version
- All children point to original (flat, not linked list)
- Latest version = `WHERE parent_id = :id ORDER BY created_at DESC LIMIT 1`
- All versions = `WHERE (id = :id OR parent_id = :id) ORDER BY created_at ASC`

### Columns to add/remove on `artifacts` table

```sql
-- Add
ALTER TABLE artifacts ADD COLUMN parent_id VARCHAR(36) REFERENCES artifacts(id) ON DELETE CASCADE;
CREATE INDEX idx_artifacts_parent_id ON artifacts(parent_id);

-- Remove (after migration)
ALTER TABLE artifacts DROP COLUMN original_content;
ALTER TABLE artifacts DROP COLUMN edit_history;
```

- `original_content` → replaced by the original row (parent_id IS NULL)
- `edit_history` → replaced by version rows
- `was_edited` → can be derived: `EXISTS(SELECT 1 FROM artifacts WHERE parent_id = :id)` or kept for convenience

---

## What breaks and how to fix it

### Problem 1: Artifact ID stability

**Impact**: Every API endpoint, SSE event, and state reference uses artifact_id as a stable identifier. With Approach B, "the artifact" is a chain of rows, not one row.

**Solution**: The frontend always references the **original artifact ID** (parent). Backend resolves "current content" by finding the latest child.

Add a helper to `artifact_service`:
```python
async def get_current_artifact(artifact_id: str) -> Artifact:
    """Get the latest version of an artifact (latest child, or self if no children)."""
    latest_child = await db.execute(
        select(Artifact)
        .where(Artifact.parent_id == artifact_id)
        .order_by(Artifact.created_at.desc())
        .limit(1)
    )
    child = latest_child.scalar_one_or_none()
    return child or await get_artifact(artifact_id)
```

### Problem 2: API endpoints (6 endpoints affected)

Every endpoint that does `get_artifact(artifact_id)` must change to `get_current_artifact(artifact_id)` for display, but use the original ID for identity.

| Endpoint | Change |
|----------|--------|
| `GET /artifacts/{id}` | Use `get_current_artifact()` + include version history |
| `PUT /artifacts/{id}/content` | Create new child row instead of updating in place |
| `POST /artifacts/{id}/optimize` | Read from current, create new child row |
| `POST /artifacts/{id}/feedback` | Apply to current version row |
| `DELETE /artifacts/{id}` | CASCADE deletes all children |
| `GET /batch/{batch_id}` | Return only originals (parent_id IS NULL), each with current content |
| `POST /posts/{id}/generate-image` | Read current, create new child with image |
| `POST /posts/{id}/regenerate-image` | Read current, create new child with new image |

### Problem 3: `update_artifact_content()` becomes `create_artifact_version()`

Currently merges updates into the row. Must change to:
1. Get current version (latest child or original)
2. Merge updates into content
3. Compute diff (for display, store in `generation_metadata` or a `diff` column)
4. INSERT new row with `parent_id = original_id`
5. Return new row

### Problem 4: Artifact modifier node

`artifact_modifier.py` — tools call `artifact_service.update_artifact_content()`. Each tool would now create a new version row. For compound edits (2+ tools), each tool creates a separate version.

After tools complete, `state["artifacts"]` and `state["last_artifact"]` must reference the **original ID** (for cross-turn modification), not the version ID.

### Problem 5: Cross-turn modification (state/working memory)

`state["last_artifact"]` stores artifact dict with `id` field. The router uses this to resolve "modify the post". If `id` changes with each edit, the next turn can't find the artifact.

**Fix**: Always store the original (parent) ID in `last_artifact` and `artifact_history`.

### Problem 6: SSE events and chat response

`chat_v2.py` returns `artifacts[].id` to frontend. Must always return the **original ID**, not the version ID.

### Problem 7: Wizard flow (posts.py)

`_persist_all()` creates artifacts with pre-generated UUIDs. No change needed for creation. But `generate-image` and `regenerate-image` endpoints would now create version rows instead of updating.

### Problem 8: Graph persistence (_persist_artifacts_to_db)

`graph.py` creates `Artifact()` rows. These are always originals (parent_id=NULL). No change needed for initial creation.

---

## Frontend rendering

### Show current content
```
GET /artifacts/{original_id}
→ Returns latest version's content (resolved by backend)
```

### Version history list
```
GET /artifacts/{original_id}/versions   ← NEW endpoint
→ Returns:
{
  "artifact_id": "art-1",
  "versions": [
    { "version": 0, "content": {...}, "created_at": "...", "source": "generated" },
    { "version": 1, "content": {...}, "created_at": "...", "source": "regeneration", "prompt": "make shorter" },
    { "version": 2, "content": {...}, "created_at": "...", "source": "user_edit" }
  ],
  "current_version": 2
}
```

### Navigate to previous version (modal)
```ts
// Frontend builds version list from /versions endpoint
const versions = response.versions
// Click version i → show versions[i].content in modal
```

### Restore previous version
```
PUT /artifacts/{original_id}/content
body: { text: versions[i].content.text, hashtags: [...] }
→ Creates a new version row (version 3) with the restored content
```

---

## Files to modify

| File | Change |
|------|--------|
| `app/models/artifact.py` | Add `parent_id` FK column, remove `original_content` + `edit_history`, update `to_dict()` |
| `app/services/artifact_service.py` | Add `get_current_artifact()`, `get_versions()`. Change `update_artifact_content()` → `create_version()`. Update all methods that fetch artifacts |
| `app/api/routes/artifacts.py` | All 6 endpoints: use `get_current_artifact()`. Add `GET /{id}/versions`. Change PUT to create version |
| `app/api/routes/posts.py` | `generate-image` and `regenerate-image`: create version rows |
| `app/agents/conversational_graph/nodes/artifact_modifier.py` | Tools create versions. State stores original ID |
| `app/agents/conversational_graph/graph.py` | No change for creation. `_persist_artifacts_to_db` stays same |
| `app/agents/conversational_graph/nodes/saver.py` | Ensure original ID in artifact_history/last_artifact |
| `app/api/routes/chat_v2.py` | Return original ID in response |

---

## SQL migration

```sql
-- Add parent_id
ALTER TABLE artifacts ADD COLUMN parent_id VARCHAR(36) REFERENCES artifacts(id) ON DELETE CASCADE;
CREATE INDEX idx_artifacts_parent_id ON artifacts(parent_id);

-- Migrate existing edit_history to version rows (one-time script)
-- For each artifact with non-empty edit_history:
--   INSERT a row per edit_history entry with parent_id = original artifact id

-- After migration verified:
ALTER TABLE artifacts DROP COLUMN original_content;
ALTER TABLE artifacts DROP COLUMN edit_history;
```

---

## Pros vs Cons summary

| | Approach A (current) | Approach B (parent_id) |
|---|---|---|
| **Query for current content** | Read `content` column | JOIN/subquery to find latest child |
| **Version navigation** | Walk `edit_history` array | `SELECT WHERE parent_id = X ORDER BY created_at` |
| **ID stability** | Stable | Must always resolve to original |
| **API complexity** | Simple (1 row = 1 artifact) | Every endpoint needs version resolution |
| **Storage** | JSONB grows per edit | Fixed-size rows |
| **SQL queryability** | Need JSONB operators | Standard SQL |
| **Files to change** | 0 (already working) | 8 files, ~15 methods |
| **Migration risk** | None | Existing edit_history must be migrated to rows |
| **New endpoint needed** | No | Yes (`GET /versions`) |

---

## Verification
1. Generate artifact → single row, parent_id=NULL
2. Edit via PUT → new row with parent_id, GET returns latest content
3. Edit via optimize → new row with parent_id
4. GET /versions → returns all versions in order
5. Cross-turn modification → "make it shorter" still finds the artifact
6. Delete original → CASCADE deletes all versions
7. Batch listing → shows originals with current content
8. SSE events → return original artifact ID
