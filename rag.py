"""Context building for transcript Q&A.

Short transcripts (the common case) are stuffed into the prompt whole.
Long transcripts are split into overlapping chunks and the most relevant ones
are picked with BM25, a classic keyword-ranking algorithm (no extra deps,
no embeddings API needed).
"""
import math
import os
import re
from collections import Counter

# ~4 chars per token. 24k chars ~= 6k tokens, which leaves room for the answer
# inside Groq's free-tier per-minute token limits.
STUFF_LIMIT_CHARS = int(os.getenv("STUFF_LIMIT_CHARS", "24000"))
CHUNK_CHARS = 1500
CHUNK_OVERLAP = 300
TOP_K = 8

STOPWORDS = set("""
a an the and or but if of to in on at by for with from as is are was were be been being it its this that
these those i you he she we they me him her us them my your his our their what which who whom when where why
how do does did doing have has had having not no so than too very can will just should would could about
into over after before then there here all any some such only own same s t don now also yeah okay ok um uh
like got get going go
""".split())


def tokenize(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9']+", text.lower()) if w not in STOPWORDS and len(w) > 1]


def chunk_text(text: str, size: int = CHUNK_CHARS, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split on line boundaries where possible so speaker turns stay intact."""
    print('rag pipeline')
    lines = text.splitlines(keepends=True)
    chunks, current = [], ""
    for line in lines:
        # A single giant line (e.g. no line breaks at all) gets hard-split.
        while len(line) > size:
            if current:
                chunks.append(current)
                current = ""
            chunks.append(line[:size])
            line = line[size - overlap:]
        if len(current) + len(line) > size and current:
            chunks.append(current)
            current = current[-overlap:] if overlap else ""
        current += line
    if current.strip():
        chunks.append(current)
    return chunks


def bm25_rank(chunks: list[str], query: str, k1: float = 1.5, b: float = 0.75) -> list[tuple[int, float]]:
    docs = [tokenize(c) for c in chunks]
    q_terms = tokenize(query)
    if not docs or not q_terms:
        return [(i, 0.0) for i in range(len(chunks))]
    n = len(docs)
    avgdl = sum(len(d) for d in docs) / n or 1
    df = Counter(term for d in docs for term in set(d))
    scores = []
    for i, d in enumerate(docs):
        tf = Counter(d)
        score = 0.0
        for term in q_terms:
            if term not in tf:
                continue
            idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
            freq = tf[term]
            score += idf * freq * (k1 + 1) / (freq + k1 * (1 - b + b * len(d) / avgdl))
        scores.append((i, score))
    return sorted(scores, key=lambda x: x[1], reverse=True)


def build_context(transcript: str, question: str, history_text: str = "") -> tuple[str, str]:
    """Return (context, mode) where mode is 'full' or 'retrieved'."""
    if len(transcript) <= STUFF_LIMIT_CHARS:
        return transcript, "full"
    chunks = chunk_text(transcript)
    # Include recent conversation so follow-ups like "and who owns that?" still retrieve well.
    ranked = bm25_rank(chunks, f"{question} {history_text}")
    picked, budget = [], STUFF_LIMIT_CHARS
    for idx, _score in ranked:
        if len(chunks[idx]) > budget:
            continue
        picked.append(idx)
        budget -= len(chunks[idx])
        if len(picked) >= TOP_K:
            break
    picked.sort()  # keep transcript order so the conversation reads naturally
    parts = [f"[Excerpt {n + 1}]\n{chunks[i].strip()}" for n, i in enumerate(picked)]
    return "\n\n".join(parts), "retrieved"
