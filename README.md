# Smart Meeting Notes Assistant

A Flask web app that turns a meeting transcript (pasted text, a `.txt` file, or an audio recording) into:

- a 3–5 bullet **summary**
- **action items** with owner and due date (when mentioned), which you can tick off
- **key decisions**
- a one-line **sentiment / tone**

…and lets you **chat with the transcript**. Every meeting and its chat history is saved in SQLite, so you can come back to it later.

It runs entirely on **Groq's free tier** with open-source models — no paid API needed:

| Job | Model |
| --- | --- |
| Analysis + chat | `openai/gpt-oss-120b`  |
| Audio → text | `whisper-large-v3-turbo` (OpenAI Whisper, open weights) |

---

## 1. Setup (5 minutes)

**Requirements:** Python 3.10+

```bash
cd smart-meeting-notes

# create a virtual environment
python -m venv venv
# Windows:      venv\Scripts\activate
# macOS/Linux:  source venv/bin/activate

pip install -r requirements.txt
```

**Get a free Groq API key** (no credit card): sign in at <https://console.groq.com/keys> → *Create API Key*.

```bash
# copy the example env file and paste your key in it
cp .env          # Windows: copy .env.example .env
```

Edit `.env`:

```
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxx
```

## 2. Run

```bash
python app.py
```

Open **http://127.0.0.1:5000**. Click **Use a sample transcript** → **Generate notes** to try it.

The status line at the bottom of the sidebar shows a green dot when your key is loaded.

## 3. Run the tests

```bash
pytest -q
```

The 22 tests mock Groq, so they run offline in under a second. They cover: text/file/subtitle/audio input, validation, Groq error handling, persistence across restarts, renaming, action-item toggling, export, chat history, retrieval for long transcripts, map-reduce analysis, and messy-LLM-output normalisation.

---

## Features

- **Three input modes** — paste, upload `.txt` (also `.md`, `.vtt`, `.srt` — subtitle timestamps are stripped), or upload audio (`mp3, m4a, wav, webm, ogg, flac, mp4`, up to 25 MB) which is transcribed with Whisper on Groq.
- **Structured analysis** — the model is forced into JSON mode with a strict schema; the backend then normalises whatever comes back (stray code fences, missing owners, "N/A" dates, too many bullets…) so the UI never breaks.
- **Long meetings** — transcripts over ~20k characters are analysed in parts and merged (map-reduce), so they stay within free-tier token limits.
- **Chat with the transcript** — short transcripts are stuffed into the prompt whole; long ones use **BM25 retrieval** (implemented in `rag.py`, no extra dependencies) to pick the most relevant excerpts. The last 10 chat turns are sent as history so follow-ups like "and when is that due?" work.
- **Persistence** — SQLite (`data/meetings.db`). Past meetings are listed in the sidebar with search; each has its own URL (`/#/meeting/3`), so refresh/bookmark works.
- **Manage notes** — tick off action items, rename a meeting by clicking its title, re-run analysis, export to Markdown, delete.
- Responsive layout (desktop three-column, tablet, mobile), keyboard accessible, clear error messages (bad key, rate limit, no connection, file too large…).

## Project structure

```
smart-meeting-notes/
├── app.py              # Flask app: routes, upload handling, validation, errors
├── llm.py              # Groq calls: analysis (JSON + map-reduce), Q&A, Whisper
├── rag.py              # chunking + BM25 retrieval for long transcripts
├── db.py               # SQLite schema and queries
├── templates/
│   └── index.html      # single-page UI
├── static/
│   ├── style.css
│   ├── app.js          # vanilla JS frontend (no build step)
│   └── sample_transcript.txt
├── tests/test_app.py
├── data/               # meetings.db is created here
├── requirements.txt
└── .env.example
```

## API

| Method | Endpoint | Purpose |
| --- | --- | --- |
| GET | `/api/health` | Server status, whether the key is set, model names |
| GET | `/api/meetings` | List meetings (newest first) |
| POST | `/api/meetings` | Create. `multipart/form-data` with `title`, and `transcript` **or** `file`; or JSON `{title, transcript}` |
| GET | `/api/meetings/<id>` | Meeting + analysis + transcript + chat messages |
| PATCH | `/api/meetings/<id>` | Rename: `{"title": "..."}` |
| DELETE | `/api/meetings/<id>` | Delete meeting and its chat |
| POST | `/api/meetings/<id>/reanalyze` | Run the analysis again |
| PATCH | `/api/meetings/<id>/action-items/<index>` | `{"done": true}` |
| GET | `/api/meetings/<id>/export` | Download Markdown |
| POST | `/api/meetings/<id>/chat` | Ask: `{"question": "..."}` |
| DELETE | `/api/meetings/<id>/chat` | Clear chat history |

Example with curl:

```bash
curl -X POST http://127.0.0.1:5000/api/meetings -F "file=@static/sample_transcript.txt"
curl -X POST http://127.0.0.1:5000/api/meetings/1/chat \
     -H "Content-Type: application/json" -d '{"question":"Who owns the crash fix?"}'
```

## Configuration (`.env`)

| Variable | Default | Notes |
| --- | --- | --- |
| `GROQ_API_KEY` | — | Required |
| `GROQ_MODEL` | `openai/gpt-oss-120b `| is faster with higher free limits |
| `GROQ_WHISPER_MODEL` | `whisper-large-v3-turbo` | or `whisper-large-v3` (slower, slightly more accurate) |
| `DATABASE_PATH` | `data/meetings.db` | |
| `PORT` | `5000` | |
| `FLASK_DEBUG` | off | set `1` for auto-reload while developing |

## Troubleshooting

- **"GROQ_API_KEY is not set"** — make sure the file is named exactly `.env` (not `.env.txt`) and sits next to `app.py`; restart the server.
- **"rate limit was hit"** — the free tier limits tokens per minute. Wait a minute, or switch `GROQ_MODEL` to `llama-3.1-8b-instant`.
- **Audio over 25 MB** — compress it first, e.g. `ffmpeg -i meeting.wav -ac 1 -b:a 32k meeting.mp3`.
- **Want to start fresh?** Stop the server and delete `data/meetings.db`.
