"""Knowledge store — markdown files + FTS5 SQLite index + semantic search.

Markdown files in knowledge/ are the source of truth.
SQLite is a derived index, rebuildable via reindex.
FTS5 keyword search is always available.
Semantic search (sentence-transformers) is used when available, with
graceful fallback to FTS5-only when not installed.
"""

import hashlib
import re
import sqlite3
import time
import uuid
from pathlib import Path

import yaml

from corc import embeddings

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    file_path TEXT UNIQUE NOT NULL,
    type TEXT NOT NULL,
    project TEXT,
    title TEXT NOT NULL,
    status TEXT DEFAULT 'active',
    source TEXT DEFAULT 'human',
    created TEXT NOT NULL,
    updated TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    file_mtime REAL,
    supersedes TEXT
);

CREATE TABLE IF NOT EXISTS document_tags (
    document_id TEXT,
    tag TEXT NOT NULL,
    PRIMARY KEY (document_id, tag)
);

CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
    title, content,
    tokenize='porter unicode61'
);

CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    heading TEXT,
    content TEXT NOT NULL,
    token_estimate INTEGER NOT NULL,
    embedding BLOB,
    FOREIGN KEY (document_id) REFERENCES documents(id),
    UNIQUE(document_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS idx_documents_type ON documents(type);
CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status);
CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id);
"""


def _parse_frontmatter(content: str) -> tuple[dict, str]:
    """Parse YAML frontmatter from markdown content."""
    if not content.startswith("---"):
        return {}, content
    end = content.find("---", 3)
    if end == -1:
        return {}, content
    fm_str = content[3:end].strip()
    body = content[end + 3:].strip()
    try:
        fm = yaml.safe_load(fm_str) or {}
    except yaml.YAMLError:
        fm = {}
    return fm, body


def _extract_title(body: str) -> str:
    for line in body.split("\n"):
        line = line.strip()
        if line.startswith("# "):
            return line[2:].strip()
    return "Untitled"


def _content_hash(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()


# --- Content Chunking ---

TARGET_TOKENS = 500
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)


def estimate_tokens(text: str) -> int:
    """Estimate token count from text. ~1.3 tokens per whitespace-delimited word."""
    words = text.split()
    return int(len(words) * 1.3) if words else 0


def _split_by_headings(body: str) -> list[dict]:
    """Split markdown body into sections delimited by headings.

    Returns a list of dicts: {"heading": str | None, "level": int, "content": str}
    Content before the first heading gets heading=None, level=0.
    """
    matches = list(_HEADING_RE.finditer(body))

    if not matches:
        # No headings — entire body is one section
        return [{"heading": None, "level": 0, "content": body.strip()}]

    sections: list[dict] = []

    # Content before the first heading
    pre = body[: matches[0].start()].strip()
    if pre:
        sections.append({"heading": None, "level": 0, "content": pre})

    for i, m in enumerate(matches):
        level = len(m.group(1))
        heading = m.group(2).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        content = body[start:end].strip()
        sections.append({"heading": heading, "level": level, "content": content})

    return sections


def _split_long_text(text: str, target: int = TARGET_TOKENS) -> list[str]:
    """Split text that exceeds the target token count into smaller pieces.

    Splits by paragraphs first; if a single paragraph is still too long,
    splits by sentences.
    """
    paragraphs = re.split(r"\n\n+", text)
    pieces: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for para in paragraphs:
        para_tokens = estimate_tokens(para)

        if para_tokens > target and para_tokens > 0:
            # Flush current buffer
            if current:
                pieces.append("\n\n".join(current))
                current = []
                current_tokens = 0
            # Split this paragraph by sentences
            sentences = re.split(r"(?<=[.!?])\s+", para)
            if len(sentences) > 1:
                sent_buf: list[str] = []
                sent_tokens = 0
                for sent in sentences:
                    st = estimate_tokens(sent)
                    if sent_buf and sent_tokens + st > target:
                        pieces.append(" ".join(sent_buf))
                        sent_buf = []
                        sent_tokens = 0
                    sent_buf.append(sent)
                    sent_tokens += st
                if sent_buf:
                    pieces.append(" ".join(sent_buf))
            else:
                # No sentence boundaries — split by word count
                words = para.split()
                # ~500 tokens / 1.3 ≈ 385 words per chunk
                words_per_chunk = max(1, int(target / 1.3))
                for wi in range(0, len(words), words_per_chunk):
                    pieces.append(" ".join(words[wi : wi + words_per_chunk]))
        elif current_tokens + para_tokens > target and current:
            pieces.append("\n\n".join(current))
            current = [para]
            current_tokens = para_tokens
        else:
            current.append(para)
            current_tokens += para_tokens

    if current:
        pieces.append("\n\n".join(current))

    return pieces


def chunk_markdown(body: str, target_tokens: int = TARGET_TOKENS) -> list[dict]:
    """Chunk markdown body into pieces of ~target_tokens tokens.

    Returns list of dicts: {"heading": str | None, "content": str, "token_estimate": int}
    """
    sections = _split_by_headings(body)
    chunks: list[dict] = []

    for section in sections:
        heading = section["heading"]
        content = section["content"]

        # Include heading text in content for context
        if heading:
            full_text = f"{'#' * section['level']} {heading}\n\n{content}" if content else f"{'#' * section['level']} {heading}"
        else:
            full_text = content

        if not full_text.strip():
            continue

        tokens = estimate_tokens(full_text)

        if tokens <= target_tokens:
            chunks.append({
                "heading": heading,
                "content": full_text,
                "token_estimate": tokens,
            })
        else:
            # Section too long — split it
            pieces = _split_long_text(content, target_tokens)
            for j, piece in enumerate(pieces):
                # First sub-chunk gets the heading prefix
                if heading and j == 0:
                    piece_text = f"{'#' * section['level']} {heading}\n\n{piece}"
                elif heading:
                    piece_text = f"{'#' * section['level']} {heading} (continued)\n\n{piece}"
                else:
                    piece_text = piece
                chunks.append({
                    "heading": heading,
                    "content": piece_text,
                    "token_estimate": estimate_tokens(piece_text),
                })

    # Handle edge case: empty body produces no chunks
    if not chunks and body.strip():
        chunks.append({
            "heading": None,
            "content": body.strip(),
            "token_estimate": estimate_tokens(body.strip()),
        })

    return chunks


class KnowledgeStore:
    def __init__(self, knowledge_dir: Path, db_path: Path):
        self.knowledge_dir = Path(knowledge_dir)
        self.knowledge_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self._migrate_embedding_column()
        self._migrate_mtime_column()
        self._embeddings_available = embeddings.is_available()

    def _migrate_embedding_column(self) -> None:
        """Add embedding column to chunks table if it doesn't exist (migration)."""
        cursor = self.conn.execute("PRAGMA table_info(chunks)")
        columns = {row[1] for row in cursor.fetchall()}
        if "embedding" not in columns:
            self.conn.execute("ALTER TABLE chunks ADD COLUMN embedding BLOB")
            self.conn.commit()

    def _migrate_mtime_column(self) -> None:
        """Add file_mtime column to documents table if it doesn't exist (migration)."""
        cursor = self.conn.execute("PRAGMA table_info(documents)")
        columns = {row[1] for row in cursor.fetchall()}
        if "file_mtime" not in columns:
            self.conn.execute("ALTER TABLE documents ADD COLUMN file_mtime REAL")
            self.conn.commit()

    def add(self, file_path: Path | None = None, content: str | None = None,
            doc_type: str = "note", project: str | None = None, tags: list[str] | None = None) -> str:
        """Add a document. Either from file_path or raw content."""
        mtime: float | None = None
        if file_path:
            file_path = Path(file_path).resolve()
            content = file_path.read_text()
            mtime = file_path.stat().st_mtime
            try:
                rel_path = str(file_path.relative_to(self.knowledge_dir.resolve()))
            except ValueError:
                # File is outside knowledge dir — copy it in
                dest = self.knowledge_dir / (doc_type + "s") / file_path.name
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(content)
                rel_path = str(dest.relative_to(self.knowledge_dir.resolve()))
                mtime = dest.stat().st_mtime
        elif content:
            doc_id = str(uuid.uuid4())[:8]
            filename = f"{time.strftime('%Y-%m-%d')}-{doc_id}.md"
            rel_path = f"{doc_type}s/{filename}" if doc_type != "note" else filename
            file_path = self.knowledge_dir / rel_path
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content)
            mtime = file_path.stat().st_mtime
        else:
            raise ValueError("Either file_path or content must be provided")

        fm, body = _parse_frontmatter(content)
        doc_id = fm.get("id", str(uuid.uuid4())[:8])
        title = _extract_title(body) or fm.get("title", "Untitled")
        ch = _content_hash(content)
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        self.conn.execute(
            """INSERT OR REPLACE INTO documents(id, file_path, type, project, title, status,
               source, created, updated, content_hash, file_mtime, supersedes)
               VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                doc_id,
                rel_path,
                fm.get("type", doc_type),
                fm.get("project", project),
                title,
                fm.get("status", "active"),
                fm.get("source", "human"),
                fm.get("created", now),
                fm.get("updated", now),
                ch,
                mtime,
                fm.get("supersedes"),
            ),
        )

        # FTS
        self.conn.execute(
            "INSERT OR REPLACE INTO documents_fts(rowid, title, content) VALUES((SELECT rowid FROM documents WHERE id=?), ?, ?)",
            (doc_id, title, body),
        )

        # Tags
        all_tags = list(set((tags or []) + fm.get("tags", [])))
        self.conn.execute("DELETE FROM document_tags WHERE document_id=?", (doc_id,))
        for tag in all_tags:
            self.conn.execute(
                "INSERT OR IGNORE INTO document_tags(document_id, tag) VALUES(?, ?)",
                (doc_id, tag),
            )

        # Chunks
        self.conn.execute("DELETE FROM chunks WHERE document_id=?", (doc_id,))
        chunks = chunk_markdown(body)

        # Generate embeddings if sentence-transformers is available
        chunk_embeddings: list[bytes | None] = [None] * len(chunks)
        if self._embeddings_available:
            try:
                texts = [c["content"] for c in chunks]
                vecs = embeddings.encode(texts)
                chunk_embeddings = [embeddings.embedding_to_blob(v) for v in vecs]
            except Exception:
                # If embedding fails for any reason, continue without embeddings
                chunk_embeddings = [None] * len(chunks)

        for idx, chunk in enumerate(chunks):
            self.conn.execute(
                "INSERT INTO chunks(document_id, chunk_index, heading, content, token_estimate, embedding) VALUES(?, ?, ?, ?, ?, ?)",
                (doc_id, idx, chunk["heading"], chunk["content"], chunk["token_estimate"], chunk_embeddings[idx]),
            )

        self.conn.commit()
        return doc_id

    def _refresh_stale_docs(self) -> int:
        """Check for stale or new docs and reindex them. Returns count of refreshed docs.

        Compares file mtimes against stored file_mtime values. Files whose mtime
        is newer than the stored value are candidates for reindexing. A content_hash
        comparison avoids unnecessary reindexing when mtime changed but content didn't
        (e.g., after `touch`). New files not yet indexed are also added. Indexed files
        that no longer exist on disk are removed.
        """
        refreshed = 0

        # Build a map of indexed file_path -> (content_hash, file_mtime, id)
        rows = self.conn.execute(
            "SELECT id, file_path, content_hash, file_mtime FROM documents"
        ).fetchall()
        indexed = {}
        for row in rows:
            indexed[row[1]] = {
                "id": row[0],
                "content_hash": row[2],
                "file_mtime": row[3],
            }

        # Scan all markdown files in knowledge_dir
        on_disk = set()
        for md_file in self.knowledge_dir.rglob("*.md"):
            if md_file.name.startswith("_"):
                continue
            try:
                rel_path = str(md_file.relative_to(self.knowledge_dir.resolve()))
            except ValueError:
                continue
            on_disk.add(rel_path)

            current_mtime = md_file.stat().st_mtime
            entry = indexed.get(rel_path)

            if entry is None:
                # New file — not yet indexed
                try:
                    self.add(file_path=md_file)
                    refreshed += 1
                except Exception:
                    pass
            elif entry["file_mtime"] is None or current_mtime > entry["file_mtime"]:
                # Mtime is newer — check if content actually changed
                content = md_file.read_text()
                ch = _content_hash(content)
                if ch != entry["content_hash"]:
                    # Content changed — reindex
                    try:
                        self.add(file_path=md_file)
                        refreshed += 1
                    except Exception:
                        pass
                else:
                    # Content unchanged — just update stored mtime
                    self.conn.execute(
                        "UPDATE documents SET file_mtime = ? WHERE id = ?",
                        (current_mtime, entry["id"]),
                    )
                    self.conn.commit()

        # Remove indexed docs whose files no longer exist
        for rel_path, entry in indexed.items():
            if rel_path not in on_disk:
                doc_id = entry["id"]
                self.conn.execute("DELETE FROM chunks WHERE document_id=?", (doc_id,))
                self.conn.execute("DELETE FROM document_tags WHERE document_id=?", (doc_id,))
                self.conn.execute(
                    "DELETE FROM documents_fts WHERE rowid = (SELECT rowid FROM documents WHERE id=?)",
                    (doc_id,),
                )
                self.conn.execute("DELETE FROM documents WHERE id=?", (doc_id,))
                self.conn.commit()
                refreshed += 1

        return refreshed

    def search(self, query: str, limit: int = 10, doc_type: str | None = None,
               project: str | None = None) -> list[dict]:
        """FTS5 keyword search with BM25 ranking."""
        self._refresh_stale_docs()
        sql = """
            SELECT d.*, bm25(documents_fts) as score
            FROM documents_fts fts
            JOIN documents d ON fts.rowid = (SELECT rowid FROM documents WHERE id = d.id)
            WHERE documents_fts MATCH ?
        """
        params: list = [query]

        if doc_type:
            sql += " AND d.type = ?"
            params.append(doc_type)
        if project:
            sql += " AND d.project = ?"
            params.append(project)

        sql += " ORDER BY score LIMIT ?"
        params.append(limit)

        rows = self.conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def semantic_search(self, query: str, limit: int = 10, doc_type: str | None = None,
                        project: str | None = None) -> list[dict]:
        """Semantic search using sentence-transformers embeddings and cosine similarity.

        Falls back to FTS5 keyword search if sentence-transformers is unavailable
        or if no chunks have embeddings.
        """
        self._refresh_stale_docs()
        if not self._embeddings_available:
            return self.search(query, limit=limit, doc_type=doc_type, project=project)

        try:
            query_vec = embeddings.encode_single(query)
        except Exception:
            # Encoding failed — fall back to FTS5
            return self.search(query, limit=limit, doc_type=doc_type, project=project)

        # Fetch all chunks that have embeddings, with document metadata
        sql = """
            SELECT c.id as chunk_id, c.document_id, c.chunk_index, c.heading,
                   c.content as chunk_content, c.token_estimate, c.embedding,
                   d.type, d.project, d.title, d.status, d.file_path, d.source,
                   d.created, d.updated
            FROM chunks c
            JOIN documents d ON c.document_id = d.id
            WHERE c.embedding IS NOT NULL AND d.status = 'active'
        """
        params: list = []
        if doc_type:
            sql += " AND d.type = ?"
            params.append(doc_type)
        if project:
            sql += " AND d.project = ?"
            params.append(project)

        rows = self.conn.execute(sql, params).fetchall()

        if not rows:
            # No embeddings stored — fall back to FTS5
            return self.search(query, limit=limit, doc_type=doc_type, project=project)

        # Compute cosine similarity for each chunk
        scored: list[tuple[float, dict]] = []
        for row in rows:
            row_dict = dict(row)
            chunk_vec = embeddings.blob_to_embedding(row_dict["embedding"])
            sim = embeddings.cosine_similarity(query_vec, chunk_vec)
            row_dict["score"] = sim
            # Remove the raw embedding blob from results
            del row_dict["embedding"]
            scored.append((sim, row_dict))

        # Sort by similarity descending, take top-N
        scored.sort(key=lambda x: x[0], reverse=True)

        return [item[1] for item in scored[:limit]]

    def hybrid_search(self, query: str, limit: int = 10, doc_type: str | None = None,
                      project: str | None = None, semantic_weight: float = 0.6,
                      keyword_weight: float | None = None) -> list[dict]:
        """Hybrid search combining FTS5 keyword and semantic similarity.

        Runs both FTS5 keyword search and semantic search, normalizes each
        score set to [0, 1] via min-max normalization, then combines them
        with configurable weights (default: 0.4 keyword + 0.6 semantic).

        Args:
            query: Search query string.
            limit: Maximum number of results to return.
            doc_type: Optional document type filter.
            project: Optional project filter.
            semantic_weight: Weight for semantic scores (default 0.6).
            keyword_weight: Weight for keyword scores.  If None, computed as
                ``1.0 - semantic_weight``.

        Returns:
            List of result dicts, deduplicated by document, sorted by combined
            score descending.  Each result includes ``score``, ``fts_score``,
            and ``semantic_score`` keys (all in [0, 1]).

        Falls back to FTS5 if semantic search is unavailable.
        """
        kw_weight = keyword_weight if keyword_weight is not None else (1.0 - semantic_weight)

        # Note: freshness check happens inside search() and semantic_search()
        # Run both FTS5 keyword and semantic searches with expanded limits
        fts_results = self.search(query, limit=limit * 2, doc_type=doc_type, project=project)

        if not self._embeddings_available:
            return fts_results[:limit]

        sem_results = self.semantic_search(query, limit=limit * 2, doc_type=doc_type, project=project)

        # If semantic search fell back to FTS5, just return FTS results
        # (detected by checking if results have 'chunk_id' key)
        if sem_results and "chunk_id" not in sem_results[0]:
            return fts_results[:limit]

        # --- Normalize FTS scores to [0, 1] ---
        # BM25 scores are negative; more negative = better match.
        fts_by_doc: dict[str, float] = {}
        fts_data: dict[str, dict] = {}
        if fts_results:
            raw_fts = [r["score"] for r in fts_results]
            min_fts = min(raw_fts)
            max_fts = max(raw_fts)
            fts_range = max_fts - min_fts
            for r in fts_results:
                # Invert and normalize: most negative BM25 → 1.0, least negative → 0.0
                if fts_range != 0:
                    normalized = (max_fts - r["score"]) / fts_range
                else:
                    # Single result or all identical scores → give full score
                    normalized = 1.0
                fts_by_doc[r["id"]] = normalized
                fts_data[r["id"]] = r

        # --- Normalize semantic scores to [0, 1] ---
        # First, deduplicate by document keeping best chunk score.
        sem_by_doc: dict[str, float] = {}
        sem_data: dict[str, dict] = {}
        for r in sem_results:
            doc_id = r["document_id"]
            # Keep best chunk score per document
            if doc_id not in sem_by_doc or r["score"] > sem_by_doc[doc_id]:
                sem_by_doc[doc_id] = r["score"]
                sem_data[doc_id] = r

        # Min-max normalize per-document semantic scores to [0, 1].
        if sem_by_doc:
            raw_sem = list(sem_by_doc.values())
            min_sem = min(raw_sem)
            max_sem = max(raw_sem)
            sem_range = max_sem - min_sem
            if sem_range != 0:
                sem_by_doc = {
                    doc_id: (score - min_sem) / sem_range
                    for doc_id, score in sem_by_doc.items()
                }
            else:
                # All identical scores → give full score
                sem_by_doc = {doc_id: 1.0 for doc_id in sem_by_doc}

        # --- Merge scores ---
        all_doc_ids = set(fts_by_doc.keys()) | set(sem_by_doc.keys())
        merged: list[tuple[float, dict]] = []

        for doc_id in all_doc_ids:
            fts_score = fts_by_doc.get(doc_id, 0.0)
            sem_score = sem_by_doc.get(doc_id, 0.0)
            combined = kw_weight * fts_score + semantic_weight * sem_score

            # Use FTS result data if available, else reconstruct from semantic data
            if doc_id in fts_data:
                result = dict(fts_data[doc_id])
                result["score"] = combined
                result["fts_score"] = fts_score
                result["semantic_score"] = sem_score
            else:
                sr = sem_data[doc_id]
                result = {
                    "id": doc_id,
                    "file_path": sr["file_path"],
                    "type": sr["type"],
                    "project": sr["project"],
                    "title": sr["title"],
                    "status": sr["status"],
                    "source": sr["source"],
                    "created": sr["created"],
                    "updated": sr["updated"],
                    "score": combined,
                    "fts_score": fts_score,
                    "semantic_score": sem_score,
                }
            merged.append((combined, result))

        merged.sort(key=lambda x: x[0], reverse=True)
        return [item[1] for item in merged[:limit]]

    def get(self, doc_id: str) -> dict | None:
        row = self.conn.execute("SELECT * FROM documents WHERE id=?", (doc_id,)).fetchone()
        if not row:
            return None
        d = dict(row)
        file_path = self.knowledge_dir / d["file_path"]
        if file_path.exists():
            d["content"] = file_path.read_text()
        return d

    def list_docs(self, doc_type: str | None = None, status: str = "active") -> list[dict]:
        sql = "SELECT * FROM documents WHERE status=?"
        params: list = [status]
        if doc_type:
            sql += " AND type=?"
            params.append(doc_type)
        sql += " ORDER BY created DESC"
        return [dict(r) for r in self.conn.execute(sql, params).fetchall()]

    def get_chunks(self, doc_id: str) -> list[dict]:
        """Get all chunks for a document, ordered by chunk_index."""
        rows = self.conn.execute(
            "SELECT * FROM chunks WHERE document_id=? ORDER BY chunk_index",
            (doc_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def reindex(self):
        """Full reindex from markdown files."""
        self.conn.execute("DELETE FROM documents")
        self.conn.execute("DELETE FROM documents_fts")
        self.conn.execute("DELETE FROM document_tags")
        self.conn.execute("DELETE FROM chunks")
        self.conn.commit()

        for md_file in self.knowledge_dir.rglob("*.md"):
            if md_file.name.startswith("_"):
                continue
            try:
                self.add(file_path=md_file)
            except Exception as e:
                print(f"Warning: failed to index {md_file}: {e}")

    def stats(self) -> dict:
        total = self.conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        by_type = {}
        for row in self.conn.execute("SELECT type, COUNT(*) as cnt FROM documents GROUP BY type").fetchall():
            by_type[row[0]] = row[1]
        by_status = {}
        for row in self.conn.execute("SELECT status, COUNT(*) as cnt FROM documents GROUP BY status").fetchall():
            by_status[row[0]] = row[1]
        return {"total": total, "by_type": by_type, "by_status": by_status}
