"""Smart Meeting Notes Assistant — Flask backend.

Run:  python app.py   then open http://127.0.0.1:5000
"""
import os
import re

from dotenv import load_dotenv

load_dotenv()  # must run before importing llm so GROQ_* settings are picked up

from flask import Flask, Response, jsonify, render_template, request  # noqa: E402
from werkzeug.exceptions import HTTPException  # noqa: E402

import db  # noqa: E402
import llm  # noqa: E402

TEXT_EXTENSIONS = {".txt", ".md", ".vtt", ".srt"}
AUDIO_EXTENSIONS = {".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".wav", ".webm", ".ogg", ".flac"}
MIN_TRANSCRIPT_CHARS = 40
MAX_TRANSCRIPT_CHARS = 400_000
MAX_UPLOAD_MB = 25  # Groq's free-tier limit for audio files


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024
    app.json.sort_keys = False
    db.init_db()

    # ------------------------------------------------------------------ helpers
    def error(message: str, status: int = 400):
        return jsonify({"error": message}), status

    def load_meeting_or_404(meeting_id: int):
        meeting = db.get_meeting(meeting_id)
        if meeting is None:
            return None, error("Meeting not found.", 404)
        return meeting, None

    def decode_text_file(raw: bytes) -> str:
        for enc in ("utf-8-sig", "utf-16", "latin-1"):
            try:
                text = raw.decode(enc)
                if enc == "utf-16" and "\x00" in text:
                    continue
                return text
            except UnicodeDecodeError:
                continue
        return raw.decode("utf-8", errors="replace")

    def clean_subtitles(text: str) -> str:
        """Strip WEBVTT/SRT numbering and timestamps so only the dialogue remains."""
        lines = []
        for line in text.splitlines():
            s = line.strip()
            if s == "WEBVTT" or re.fullmatch(r"\d+", s) or "-->" in s:
                continue
            lines.append(line)
        return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()

    def tidy_transcript(text: str) -> str:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"[ \t]+\n", "\n", text)
        return re.sub(r"\n{4,}", "\n\n\n", text).strip()

    # ------------------------------------------------------------------ pages
    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/api/health")
    def health():
        return jsonify({
            "ok": True,
            "groq_key_configured": bool(os.getenv("GROQ_API_KEY", "").strip()),
            "model": llm.CHAT_MODEL,
            "whisper_model": llm.WHISPER_MODEL,
        })

    # ------------------------------------------------------------------ meetings
    @app.get("/api/meetings")
    def list_meetings():
        return jsonify(db.list_meetings())

    @app.post("/api/meetings")
    def create_meeting():
        """Accepts multipart/form-data (title, transcript, file) or JSON (title, transcript)."""
        if request.is_json:
            payload = request.get_json(silent=True) or {}
            title = (payload.get("title") or "").strip()
            transcript = payload.get("transcript") or ""
            upload = None
        else:
            title = (request.form.get("title") or "").strip()
            transcript = request.form.get("transcript") or ""
            upload = request.files.get("file")

        source_type, source_name = "text", None
        if upload and upload.filename:
            ext = os.path.splitext(upload.filename)[1].lower()
            raw = upload.read()
            if not raw:
                return error("The uploaded file is empty.")
            source_name = upload.filename
            if ext in TEXT_EXTENSIONS:
                transcript = decode_text_file(raw)
                if ext in {".vtt", ".srt"}:
                    transcript = clean_subtitles(transcript)
                source_type = "file"
            elif ext in AUDIO_EXTENSIONS:
                try:
                    spoken = llm.transcribe_audio(raw, upload.filename)
                except llm.LLMError as exc:
                    return error(str(exc), 502)
                # Whisper returns one long paragraph; one sentence per line reads far better.
                transcript = re.sub(r"(?<=[.!?])\s+", "\n", spoken)
                source_type = "audio"
            else:
                allowed = ", ".join(sorted(TEXT_EXTENSIONS | AUDIO_EXTENSIONS))
                return error(f"Unsupported file type '{ext or 'none'}'. Use one of: {allowed}.")

        transcript = tidy_transcript(transcript)
        if len(transcript) < MIN_TRANSCRIPT_CHARS:
            if source_type == "audio":
                return error("No speech could be detected in that audio file.")
            return error(f"The transcript is too short. Provide at least {MIN_TRANSCRIPT_CHARS} characters.")
        if len(transcript) > MAX_TRANSCRIPT_CHARS:
            return error(f"The transcript is too long (max {MAX_TRANSCRIPT_CHARS:,} characters).")

        try:
            analysis = llm.analyze_transcript(transcript)
        except llm.LLMError as exc:
            return error(str(exc), 502)

        meeting_id = db.create_meeting(
            title=title or analysis["title"],
            transcript=transcript,
            analysis=analysis,
            source_type=source_type,
            source_name=source_name,
        )
        meeting = db.get_meeting(meeting_id)
        meeting["messages"] = []
        return jsonify(meeting), 201

    @app.get("/api/meetings/<int:meeting_id>")
    def get_meeting(meeting_id):
        meeting, err = load_meeting_or_404(meeting_id)
        if err:
            return err
        meeting["messages"] = db.get_messages(meeting_id)
        return jsonify(meeting)

    @app.patch("/api/meetings/<int:meeting_id>")
    def rename_meeting(meeting_id):
        meeting, err = load_meeting_or_404(meeting_id)
        if err:
            return err
        title = ((request.get_json(silent=True) or {}).get("title") or "").strip()
        if not title:
            return error("Title can't be empty.")
        db.update_meeting(meeting_id, title=title[:200])
        return jsonify(db.get_meeting(meeting_id))

    @app.delete("/api/meetings/<int:meeting_id>")
    def delete_meeting(meeting_id):
        if not db.delete_meeting(meeting_id):
            return error("Meeting not found.", 404)
        return jsonify({"deleted": meeting_id})

    @app.post("/api/meetings/<int:meeting_id>/reanalyze")
    def reanalyze(meeting_id):
        meeting, err = load_meeting_or_404(meeting_id)
        if err:
            return err
        try:
            analysis = llm.analyze_transcript(meeting["transcript"])
        except llm.LLMError as exc:
            return error(str(exc), 502)
        db.update_meeting(meeting_id, analysis=analysis)
        updated = db.get_meeting(meeting_id)
        updated["messages"] = db.get_messages(meeting_id)
        return jsonify(updated)

    @app.patch("/api/meetings/<int:meeting_id>/action-items/<int:index>")
    def toggle_action_item(meeting_id, index):
        meeting, err = load_meeting_or_404(meeting_id)
        if err:
            return err
        items = meeting["analysis"]["action_items"]
        if not 0 <= index < len(items):
            return error("Action item not found.", 404)
        done = (request.get_json(silent=True) or {}).get("done")
        items[index]["done"] = bool(done) if done is not None else not items[index].get("done", False)
        db.update_meeting(meeting_id, analysis=meeting["analysis"])
        return jsonify(items[index])

    @app.get("/api/meetings/<int:meeting_id>/export")
    def export_meeting(meeting_id):
        meeting, err = load_meeting_or_404(meeting_id)
        if err:
            return err
        a = meeting["analysis"]
        lines = [f"# {meeting['title']}", "", f"_Created {meeting['created_at']}_", "", "## Summary"]
        lines += [f"- {s}" for s in a["summary"]] or ["- (none)"]
        lines += ["", "## Action items"]
        for item in a["action_items"]:
            due = f" — due {item['due_date']}" if item.get("due_date") else ""
            lines.append(f"- [{'x' if item.get('done') else ' '}] {item['task']} (**{item['owner']}**{due})")
        if not a["action_items"]:
            lines.append("- (none)")
        lines += ["", "## Decisions"] + ([f"- {d}" for d in a["decisions"]] or ["- (none)"])
        lines += ["", "## Sentiment", f"**{a['sentiment']['label']}** — {a['sentiment']['description']}"]
        lines += ["", "## Transcript", "", meeting["transcript"], ""]
        slug = re.sub(r"[^a-z0-9]+", "-", meeting["title"].lower()).strip("-")[:60] or "meeting"
        return Response(
            "\n".join(lines), mimetype="text/markdown",
            headers={"Content-Disposition": f'attachment; filename="{slug}.md"'},
        )

    # ------------------------------------------------------------------ chat
    @app.post("/api/meetings/<int:meeting_id>/chat")
    def chat(meeting_id):
        meeting, err = load_meeting_or_404(meeting_id)
        if err:
            return err
        question = ((request.get_json(silent=True) or {}).get("question") or "").strip()
        if not question:
            return error("Type a question first.")
        if len(question) > 2000:
            return error("Questions are limited to 2,000 characters.")

        history = db.get_messages(meeting_id)
        try:
            result = llm.answer_question(meeting["transcript"], meeting["analysis"], history, question)
        except llm.LLMError as exc:
            return error(str(exc), 502)

        user_msg = db.add_message(meeting_id, "user", question)
        bot_msg = db.add_message(meeting_id, "assistant", result["answer"])
        return jsonify({"question": user_msg, "answer": bot_msg, "context_mode": result["context_mode"]})

    @app.delete("/api/meetings/<int:meeting_id>/chat")
    def clear_chat(meeting_id):
        meeting, err = load_meeting_or_404(meeting_id)
        if err:
            return err
        db.clear_messages(meeting_id)
        return jsonify({"cleared": meeting_id})

    # ------------------------------------------------------------------ errors
    @app.errorhandler(413)
    def too_large(_e):
        return error(f"File is too large. The limit is {MAX_UPLOAD_MB} MB.", 413)

    @app.errorhandler(HTTPException)
    def http_error(e):
        if request.path.startswith("/api/"):
            return error(e.description or e.name, e.code)
        return e

    @app.errorhandler(Exception)
    def unhandled(e):
        app.logger.exception("Unhandled error")
        return error("Something went wrong on the server. Check the terminal for details.", 500)

    return app


app = create_app()

if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    if not os.getenv("GROQ_API_KEY"):
        print("\n  ! GROQ_API_KEY is not set. Copy .env.example to .env and add your free key.\n")
    app.run(host="127.0.0.1", port=port, debug=os.getenv("FLASK_DEBUG") == "1")
