"""All calls to Groq live here: transcript analysis, follow-up Q&A, audio transcription.

Groq offers a free tier (no credit card) for open-source models such as Llama 3.3
and Whisper. Get a key at https://console.groq.com/keys.
"""
import json
import os
import re

from groq import Groq

import rag

CHAT_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b  ")
WHISPER_MODEL = os.getenv("GROQ_WHISPER_MODEL", "whisper-large-v3-turbo")
# Transcripts longer than this are analysed in parts and merged (map-reduce),
# so very long meetings don't blow past the model's free-tier token limits.
ANALYSIS_CHUNK_CHARS = int(os.getenv("ANALYSIS_CHUNK_CHARS", "20000"))

SENTIMENT_LABELS = ["Positive", "Neutral", "Negative", "Mixed", "Tense", "Productive"]


class LLMError(Exception):
    """Raised with a message that is safe to show to the user."""


_client = None


def get_client() -> Groq:
    global _client
    key = os.getenv("GROQ_API_KEY", "").strip()
    if not key:
        raise LLMError("GROQ_API_KEY is not set. Add it to your .env file (free key: https://console.groq.com/keys).")
    if _client is None or _client.api_key != key:
        _client = Groq(api_key=key, max_retries=2, timeout=90)
    return _client


def _friendly_error(exc: Exception) -> LLMError:
    name = type(exc).__name__
    msg = str(exc)
    if "AuthenticationError" in name or "401" in msg:
        return LLMError("Groq rejected the API key. Check GROQ_API_KEY in your .env file.")
    if "RateLimit" in name or "429" in msg:
        return LLMError("Groq's free-tier rate limit was hit. Wait about a minute and try again.")
    if "APIConnectionError" in name or "Timeout" in name:
        return LLMError("Couldn't reach Groq. Check your internet connection and try again.")
    if "413" in msg or "too large" in msg.lower():
        return LLMError("The request was too large for the model. Try a shorter transcript or audio file.")
    return LLMError(f"Groq request failed: {msg[:300]}")


def chat_completion(messages, json_mode=False, temperature=0.2, max_tokens=2048) -> str:
    client = get_client()
    kwargs = dict(model=CHAT_MODEL, messages=messages, temperature=temperature, max_tokens=max_tokens)
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    try:
        resp = client.chat.completions.create(**kwargs)
    except Exception as exc:  # groq raises several subclasses; normalise them
        raise _friendly_error(exc) from exc
    return (resp.choices[0].message.content or "").strip()


# --------------------------------------------------------------------------- analysis

ANALYSIS_SYSTEM = f"""You are an expert meeting analyst. You read a meeting transcript and return ONLY a JSON object with this exact shape:

{{
  "title": "short descriptive meeting title, max 8 words",
  "summary": ["3 to 5 concise bullet points covering the main discussion"],
  "action_items": [
    {{"task": "what needs to be done, starting with a verb", "owner": "person responsible", "due_date": "deadline exactly as stated, or null"}}
  ],
  "decisions": ["each concrete decision that was agreed on"],
  "sentiment": {{"label": "one of {', '.join(SENTIMENT_LABELS)}", "description": "one sentence describing the tone of the meeting"}}
}}

Rules:
- Use only information in the transcript. Never invent people, tasks, dates or decisions.
- owner: use the name of the person who took or was given the task. If nobody was named, use "Unassigned".
- due_date: copy the deadline as spoken (e.g. "Friday", "end of Q3", "March 14"). If none was mentioned, use null.
- decisions: only things the group actually agreed or settled, not ideas that were merely suggested. Use [] if there were none.
- summary: 3 to 5 bullets, each one sentence, most important first.
- Return valid JSON only. No markdown, no commentary."""

PARTIAL_NOTE = ("This is PART {part} of {total} of a longer meeting transcript. Analyse only this part; "
                "the parts will be merged later. The summary may have 2 to 5 bullets.")

MERGE_SYSTEM = f"""You merge partial analyses of ONE meeting (produced from consecutive parts of its transcript) into a single final analysis.
Return ONLY a JSON object with the same shape as the inputs:
{{"title": str, "summary": [3-5 str], "action_items": [{{"task": str, "owner": str, "due_date": str|null}}],
"decisions": [str], "sentiment": {{"label": one of {', '.join(SENTIMENT_LABELS)}, "description": str}}}}
Rules: deduplicate action items and decisions that refer to the same thing (keep the most specific owner/due date);
if a later part reverses an earlier decision, keep only the final one; write a summary of the whole meeting in 3-5 bullets;
judge the overall sentiment. Use only information present in the partial analyses."""


def parse_json(text: str) -> dict:
    """Parse model output as JSON, tolerating stray code fences or prose around it."""
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
    raise LLMError("The model returned output that wasn't valid JSON. Try running the analysis again.")


def _clean_str(value) -> str:
    return re.sub(r"\s+", " ", str(value)).strip() if value is not None else ""


def normalize_analysis(data: dict) -> dict:
    """Coerce whatever the model returned into a predictable structure for the UI."""
    if not isinstance(data, dict):
        data = {}

    summary = data.get("summary") or []
    if isinstance(summary, str):
        summary = [s.strip("-• ").strip() for s in summary.splitlines() if s.strip()]
    summary = [_clean_str(s) for s in summary if _clean_str(s)][:5]

    actions = []
    for item in data.get("action_items") or []:
        if isinstance(item, str):
            item = {"task": item}
        if not isinstance(item, dict):
            continue
        task = _clean_str(item.get("task") or item.get("action") or item.get("description"))
        if not task:
            continue
        owner = _clean_str(item.get("owner") or item.get("assignee")) or "Unassigned"
        due = _clean_str(item.get("due_date") or item.get("due") or item.get("deadline"))
        if due.lower() in {"null", "none", "n/a", "not mentioned", "unspecified", "tbd", ""}:
            due = None
        actions.append({"task": task, "owner": owner, "due_date": due, "done": bool(item.get("done", False))})

    decisions = data.get("decisions") or []
    if isinstance(decisions, str):
        decisions = [decisions]
    decisions = [_clean_str(d if not isinstance(d, dict) else d.get("decision", "")) for d in decisions]
    decisions = [d for d in decisions if d]

    sentiment = data.get("sentiment") or {}
    if isinstance(sentiment, str):
        sentiment = {"label": "Neutral", "description": sentiment}
    label = _clean_str(sentiment.get("label")).capitalize() or "Neutral"
    if label not in SENTIMENT_LABELS:
        label = "Mixed" if label else "Neutral"
    description = _clean_str(sentiment.get("description")) or "Tone could not be determined."

    return {
        "title": _clean_str(data.get("title"))[:120] or "Untitled meeting",
        "summary": summary,
        "action_items": actions,
        "decisions": decisions,
        "sentiment": {"label": label, "description": description},
    }


def _analyze_once(transcript: str, note: str = "") -> dict:
    user = (note + "\n\n" if note else "") + f"TRANSCRIPT:\n\"\"\"\n{transcript}\n\"\"\""
    raw = chat_completion(
        [{"role": "system", "content": ANALYSIS_SYSTEM}, {"role": "user", "content": user}],
        json_mode=True, temperature=0.1,
    )
    return parse_json(raw)


def analyze_transcript(transcript: str) -> dict:
    if len(transcript) <= ANALYSIS_CHUNK_CHARS:
        return normalize_analysis(_analyze_once(transcript))

    parts = rag.chunk_text(transcript, size=ANALYSIS_CHUNK_CHARS, overlap=500)
    partials = [
        normalize_analysis(_analyze_once(p, PARTIAL_NOTE.format(part=i + 1, total=len(parts))))
        for i, p in enumerate(parts)
    ]
    raw = chat_completion(
        [{"role": "system", "content": MERGE_SYSTEM},
         {"role": "user", "content": "PARTIAL ANALYSES (in meeting order):\n" + json.dumps(partials, indent=1)}],
        json_mode=True, temperature=0.1,
    )
    return normalize_analysis(parse_json(raw))


# --------------------------------------------------------------------------- Q&A

QA_SYSTEM = """You are a helpful assistant answering questions about one specific meeting.
Ground every answer in the transcript provided. When useful, mention who said something.
If the transcript doesn't contain the answer, say so plainly instead of guessing.
Keep answers short and direct. You may use simple Markdown: **bold**, bullet lists with "- ", and line breaks."""


def answer_question(transcript: str, analysis: dict, history: list[dict], question: str) -> dict:
    recent = history[-10:]
    history_text = " ".join(m["content"] for m in recent if m["role"] == "user")[-500:]
    context, mode = rag.build_context(transcript, question, history_text)

    scope = ("the full transcript" if mode == "full"
             else "the transcript excerpts most relevant to the question (the full meeting is longer)")
    grounding = (
        f"Here is {scope}:\n\"\"\"\n{context}\n\"\"\"\n\n"
        f"Structured notes already extracted from the meeting (for reference):\n{json.dumps(analysis, indent=1)}"
    )
    messages = [{"role": "system", "content": QA_SYSTEM + "\n\n" + grounding}]
    messages += [{"role": m["role"], "content": m["content"]} for m in recent]
    messages.append({"role": "user", "content": question})
    answer = chat_completion(messages, temperature=0.3, max_tokens=1024)
    return {"answer": answer or "I couldn't produce an answer. Try rephrasing the question.", "context_mode": mode}


# --------------------------------------------------------------------------- audio

def transcribe_audio(data: bytes, filename: str) -> str:
    client = get_client()
    try:
        result = client.audio.transcriptions.create(
            file=(filename, data), model=WHISPER_MODEL, response_format="json", temperature=0.0,
        )
    except Exception as exc:
        raise _friendly_error(exc) from exc
    text = getattr(result, "text", None) or (result.get("text") if isinstance(result, dict) else str(result))
    return (text or "").strip()
