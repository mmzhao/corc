"""Knowledge store — markdown files + FTS5 SQLite index.

Markdown files in knowledge/ are the source of truth.
SQLite is a derived index, rebuildable via reindex.
Phase 0: FTS5 keyword search only. Semantic search added in Phase 1A.
"""

import hashlib
import re
import sqlite3
import time
import uuid
from pathlib import Path

import yaml

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

    def add(self, file_path: Path | None = None, content: str | None = None,
            doc_type: str = "note", project: str | None = None, tags: list[str] | None = None) -> str:
        """Add a document. Either from file_path or raw content."""
        if file_path:
            file_path = Path(file_path).resolve()
            content = file_path.read_text()
            try:
                rel_path = str(file_path.relative_to(self.knowledge_dir.resolve()))
            except ValueError:
                # File is outside knowledge dir — copy it in
                dest = self.knowledge_dir / (doc_type + "s") / file_path.name
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(content)
                rel_path = str(dest.relative_to(self.knowledge_dir.resolve()))
        elif content:
            doc_id = str(uuid.uuid4())[:8]
            filename = f"{time.strftime('%Y-%m-%d')}-{doc_id}.md"
            rel_path = f"{doc_type}s/{filename}" if doc_type != "note" else filename
            file_path = self.knowledge_dir / rel_path
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content)
        else:
            raise ValueError("Either file_path or content must be provided")

        fm, body = _parse_frontmatter(content)
        doc_id = fm.get("id", str(uuid.uuid4())[:8])
        title = _extract_title(body) or fm.get("title", "Untitled")
        ch = _content_hash(content)
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        self.conn.execute(
            """INSERT OR REPLACE INTO documents(id, file_path, type, project, title, status,
               source, created, updated, content_hash, supersedes)
               VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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
        for idx, chunk in enumerate(chunks):
            self.conn.execute(
                "INSERT INTO chunks(document_id, chunk_index, heading, content, token_estimate) VALUES(?, ?, ?, ?, ?)",
                (doc_id, idx, chunk["heading"], chunk["content"], chunk["token_estimate"]),
            )

        self.conn.commit()
        return doc_id

    def search(self, query: str, limit: int = 10, doc_type: str | None = None,
               project: str | None = None) -> list[dict]:
        """FTS5 keyword search with BM25 ranking."""
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
