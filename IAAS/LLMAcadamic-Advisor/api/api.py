
import sys
import os

# Optimize performance and suppress noisy telemetry & cache warnings
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("CHROMA_TELEMETRY", "False")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import io
import logging
import hashlib
import json
import sqlite3
import threading

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add the parent directory to Python path BEFORE importing LLM modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now import LLM modules
from LLM.main_copy import setup_environment
from LLM.main_copy import aggregate_and_generate_response
from LLM.main_copy import generate_conversational_response
from LLM.main_copy import get_execution_trace_for_user
from LLM.main_copy import clear_search_context_for_user
from LLM.main_copy import reset_uploaded_transcript_state
from LLM.majorsGraph import MajorGraph

# Import Flask modules
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
app = Flask(__name__)

def _configured_cors_origins():
    configured = os.getenv("LLM_CORS_ORIGINS", "")
    if configured.strip():
        return [origin.strip() for origin in configured.split(",") if origin.strip()]
    return [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://13.58.145.194",
        "https://13.58.145.194",
    ]

CORS(app, origins=_configured_cors_origins())

class SessionStateStore:
    """Conversation/transcript metadata store.

    Defaults to in-memory state for local development. Set MERIDIAN_SESSION_DB_PATH
    to use SQLite-backed state that survives process restarts and can be moved
    behind a managed database adapter later without changing route logic.
    """

    def __init__(self, sqlite_path: str = ""):
        self._sqlite_path = (sqlite_path or "").strip()
        self._lock = threading.RLock()
        self._memory_conversations = {}
        self._memory_transcript_hashes = {}
        self._conn = None
        if self._sqlite_path:
            db_dir = os.path.dirname(os.path.abspath(self._sqlite_path))
            if db_dir:
                os.makedirs(db_dir, exist_ok=True)
            self._conn = sqlite3.connect(self._sqlite_path, check_same_thread=False)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS session_state (
                    user_id TEXT PRIMARY KEY,
                    conversation_json TEXT NOT NULL DEFAULT '[]',
                    transcript_hash TEXT
                )
                """
            )
            self._conn.commit()

    def _ensure_row(self, user_id: str):
        self._conn.execute(
            "INSERT OR IGNORE INTO session_state (user_id, conversation_json) VALUES (?, '[]')",
            (user_id,),
        )

    @staticmethod
    def _safe_history(raw_json: str):
        try:
            parsed = json.loads(raw_json or "[]")
            return parsed if isinstance(parsed, list) else []
        except (TypeError, json.JSONDecodeError):
            return []

    def get_conversation(self, user_id: str):
        if not self._conn:
            return list(self._memory_conversations.get(user_id, []))
        with self._lock:
            row = self._conn.execute(
                "SELECT conversation_json FROM session_state WHERE user_id = ?",
                (user_id,),
            ).fetchone()
            return self._safe_history(row[0]) if row else []

    def set_conversation(self, user_id: str, history):
        safe_history = list(history or [])[-50:]
        if not self._conn:
            self._memory_conversations[user_id] = safe_history
            return safe_history
        payload = json.dumps(safe_history)
        with self._lock:
            self._ensure_row(user_id)
            self._conn.execute(
                "UPDATE session_state SET conversation_json = ? WHERE user_id = ?",
                (payload, user_id),
            )
            self._conn.commit()
        return safe_history

    def append_messages(self, user_id: str, messages):
        history = self.get_conversation(user_id)
        history.extend(messages)
        return self.set_conversation(user_id, history)

    def clear_conversation(self, user_id: str):
        return self.set_conversation(user_id, [])

    def get_transcript_hash(self, user_id: str) -> str:
        if not self._conn:
            return self._memory_transcript_hashes.get(user_id, "no_transcript")
        with self._lock:
            row = self._conn.execute(
                "SELECT transcript_hash FROM session_state WHERE user_id = ?",
                (user_id,),
            ).fetchone()
            return row[0] if row and row[0] else "no_transcript"

    def set_transcript_hash(self, user_id: str, doc_hash: str):
        if not self._conn:
            self._memory_transcript_hashes[user_id] = doc_hash
            return
        with self._lock:
            self._ensure_row(user_id)
            self._conn.execute(
                "UPDATE session_state SET transcript_hash = ? WHERE user_id = ?",
                (doc_hash, user_id),
            )
            self._conn.commit()

    def delete_transcript_hash(self, user_id: str):
        if not self._conn:
            self._memory_transcript_hashes.pop(user_id, None)
            return
        with self._lock:
            self._ensure_row(user_id)
            self._conn.execute(
                "UPDATE session_state SET transcript_hash = NULL WHERE user_id = ?",
                (user_id,),
            )
            self._conn.commit()

    def close(self):
        if self._conn:
            with self._lock:
                self._conn.close()
                self._conn = None


SESSION_STORE = SessionStateStore(os.getenv("MERIDIAN_SESSION_DB_PATH", ""))


def get_cache_key(user_id: str, doc_hash: str, user_question: str,
                  model_version: str, prompt_version: str,
                  conversation_tail=None) -> str:
    """Generate cache key with all required scoping."""
    import hashlib
    import json
    if conversation_tail is None:
        conversation_tail = []
    key_data = {
        "user_id": user_id,
        "doc_hash": doc_hash,
        "question": user_question,
        "conversation_tail": conversation_tail,
        "model": model_version,
        "prompt": prompt_version
    }
    key_str = json.dumps(key_data, sort_keys=True)
    return hashlib.sha256(key_str.encode()).hexdigest()

def get_user_transcript_hash(user_id: str) -> str:
    """Get transcript hash for user/session."""
    return SESSION_STORE.get_transcript_hash(user_id)

def compute_file_hash(file_path: str) -> str:
    """Compute SHA256 hash of a file."""
    hasher = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            hasher.update(chunk)
    return hasher.hexdigest()

RESPONSE_CACHE = {}  # In production: Redis
CACHE_TTL_HOURS = 24


def _is_debug_trace_enabled(debug_trace_flag) -> bool:
    if isinstance(debug_trace_flag, str):
        debug_trace_flag = debug_trace_flag.strip().lower() in {'1', 'true', 'yes', 'on'}
    env_flag = str(os.getenv("AGENT_DEBUG_TRACE", "0")).strip().lower() in {'1', 'true', 'yes', 'on'}
    return bool(debug_trace_flag) or env_flag


def _is_dynamic_deadline_query(text: str) -> bool:
    if not text:
        return False
    t = text.lower()
    return (
        ("deadline" in t or "date" in t)
        and any(k in t for k in ["late drop", "drop", "withdraw", "registration", "current semester", "this semester", "next semester", "upcoming semester"])
    )

@app.route('/llm/generate-response', methods=['POST'])
def generate_graduation_plan():
    try:
        data = request.get_json()
        user_question = data.get('user_question', '')
        conversation_history = data.get('conversation_history', [])
        user_id = data.get('user_id', 'default')  # Use session_id or user_id from frontend
        debug_trace = _is_debug_trace_enabled(data.get('debug_trace', False))
        
        if not user_question:
            return jsonify({"error": "User question is required"}), 400

        # Get doc_hash if transcript exists
        doc_hash = get_user_transcript_hash(user_id)
        model_version = os.getenv("ADVISOR_MODEL_NAME", "gpt-5.4-mini")
        prompt_version = "v1"  # Or get from config
        stored_history = SESSION_STORE.get_conversation(user_id)
        
        # Response cache is intentionally disabled so every query executes full
        # reasoning/planning/tool-use/validation with fresh context.
        history_tail_for_cache = conversation_history[-6:] if conversation_history else stored_history[-6:]
        cache_key = get_cache_key(
            user_id,
            doc_hash,
            user_question,
            model_version,
            prompt_version,
            conversation_tail=history_tail_for_cache
        )
        skip_cache = True
        if (not skip_cache) and doc_hash != "no_transcript" and cache_key in RESPONSE_CACHE:
            cached = RESPONSE_CACHE[cache_key]
            from datetime import datetime, timedelta
            if datetime.now() - cached['timestamp'] < timedelta(hours=CACHE_TTL_HOURS):
                if conversation_history and len(conversation_history) > 0:
                    stored_history = SESSION_STORE.set_conversation(user_id, conversation_history)
                
                return jsonify({
                    "graduation_plan": cached['response'],
                    "conversation_history": stored_history,
                    "cached": True,
                    "debug_trace": get_execution_trace_for_user(user_id)
                }), 200

        # Update with frontend history if provided (sync frontend and backend state)
        if conversation_history and len(conversation_history) > 0:
            stored_history = SESSION_STORE.set_conversation(user_id, conversation_history)
        
        # Generate response with existing conversation history (before adding current message)
        # Pass user's transcript status so we don't use another user's transcript
        user_has_transcript = (doc_hash != "no_transcript")
        result = generate_conversational_response(
            user_question,
            stored_history,
            user_id=user_id,
            user_has_transcript=user_has_transcript,
            debug_mode=debug_trace
        )
        
        # Response cache disabled by design (see skip_cache=True above).
        if (not skip_cache) and doc_hash != "no_transcript":
            from datetime import datetime
            RESPONSE_CACHE[cache_key] = {
                "response": result,
                "timestamp": datetime.now()
            }
        
        # Add user message and AI response to history after generation
        stored_history = SESSION_STORE.append_messages(user_id, [
            {
                'role': 'user',
                'content': user_question
            },
            {
                'role': 'assistant',
                'content': result
            },
        ])

        return jsonify({
            "graduation_plan": result,
            "conversation_history": stored_history,
            "cached": False,
            "debug_trace": get_execution_trace_for_user(user_id)
        }), 200
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route('/llm/clear-conversation', methods=['POST'])
def clear_conversation():
    """
    Clear everything tied to the current chat session:
      - conversation history
      - active web-search context
      - per-user transcript hash (so the agent no longer scopes caches to an old doc)
      - module-level transcript state (pdf retriever, stored docs, extracted student data)
      - any cached responses scoped to this user

    The transcript vectorstore collection on disk is left in place (it is TTL-managed);
    wiping the in-memory state is enough to stop the agent from consulting the document.
    """
    try:
        data = request.get_json() or {}
        user_id = data.get('user_id', 'default')

        SESSION_STORE.clear_conversation(user_id)
        clear_search_context_for_user(user_id)

        SESSION_STORE.delete_transcript_hash(user_id)
        reset_uploaded_transcript_state()

        stale_cache_keys = [
            k for k, v in RESPONSE_CACHE.items()
            if isinstance(v, dict) and v.get('user_id') == user_id
        ]
        for k in stale_cache_keys:
            RESPONSE_CACHE.pop(k, None)

        return jsonify({
            "message": "Conversation cleared successfully",
            "cleared": {
                "conversation_history": True,
                "search_context": True,
                "transcript_state": True,
                "response_cache_entries": len(stale_cache_keys),
            }
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/llm/setup-environment', methods=['POST'])
def handle_setup_environment():
    if 'transcript' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    transcript = request.files['transcript']
    user_id = request.form.get('user_id', 'default')  # Get user_id from form data
    
    # Validate file
    if transcript.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not transcript.filename.lower().endswith('.pdf'):
        return jsonify({'error': 'Only PDF files are allowed'}), 400
    
    try:
        # Save file temporarily to a more accessible location
        import tempfile
        import os
        
        # Create a temporary file with proper extension
        temp_dir = tempfile.gettempdir()
        temp_filename = f"transcript_{os.urandom(8).hex()}.pdf"
        temp_path = os.path.join(temp_dir, temp_filename)
        
        # Save the uploaded file
        transcript.save(temp_path)

        # Track transcript hash per user for cache scoping
        doc_hash = compute_file_hash(temp_path)
        SESSION_STORE.set_transcript_hash(user_id, doc_hash)
        
        # Verify the file was saved and is readable
        if not os.path.exists(temp_path) or os.path.getsize(temp_path) == 0:
            return jsonify({'error': 'Failed to save uploaded file'}), 500
        
        # Create a mock request object with the saved file
        class MockFile:
            def __init__(self, path, original_filename):
                self.path = path
                self.filename = original_filename
            
            def save(self, destination):
                import shutil
                shutil.copy2(self.path, destination)
        
        # Create mock request object
        class MockRequest:
            def __init__(self, file_path, filename):
                self.files = {'transcript': MockFile(file_path, filename)}
        
        mock_request = MockRequest(temp_path, transcript.filename)
        
        # Call setup_environment with user_id
        setup_environment(mock_request, user_id=user_id)
        
        # Clean up temporary file
        try:
            os.unlink(temp_path)
        except:
            pass
            
        return jsonify({
            "message": "Environment setup successfully",
            "doc_hash": doc_hash
        }), 200
        
    except Exception as e:
        # Clean up temporary file on error
        try:
            if 'temp_path' in locals() and os.path.exists(temp_path):
                os.unlink(temp_path)
        except:
            pass
            
        import traceback
        error_details = str(e)
        traceback_str = traceback.format_exc()
        logger.error(f"Setup environment error: {error_details}")
        logger.error(f"Traceback: {traceback_str}")
        # Also print to stderr so systemd can capture it
        import sys
        print(f"Setup environment error: {error_details}", file=sys.stderr)
        print(f"Traceback: {traceback_str}", file=sys.stderr)
        return jsonify({"error": f"Setup failed: {error_details}"}), 500


@app.route('/llm/generate-major-graph', methods=['POST'])
def generate_major_graph():
    try:
        data = request.get_json()
        major_name = data.get('major_name', '')

        if not major_name:
            return jsonify({"error": "Major name is required"}), 400

        # Initialize MajorGraph with the CSV file path
        csv_path = os.path.join(os.path.dirname(__file__), '..', 'LLM', 'nith_majors.csv')
        if not os.path.exists(csv_path):
            csv_path = os.path.join(os.path.dirname(__file__), '..', 'LLM', 'nith_majors.csv')
        if not os.path.exists(csv_path):
            csv_path = os.path.join(os.path.dirname(__file__), '..', 'LLM', 'all_abington_majors_combined(5).csv')
        
        # Check if CSV file exists
        if not os.path.exists(csv_path):
            logger.error(f"CSV file not found at: {csv_path}")
            return jsonify({"error": f"CSV file not found at: {csv_path}"}), 500
        
        logger.info(f"Generating graph for major: {major_name}")
        mg = MajorGraph(csv_path)

        # Generate the graph and get the image buffer
        image_buffer = mg.draw_major_graph(major_name)

        # Return the image as a response
        return send_file(
            image_buffer,
            mimetype='image/png',
            as_attachment=False
        )

    except ValueError as e:
        # This is raised when major is not found
        logger.error(f"Major not found: {str(e)}")
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        logger.error(f"Error generating major graph: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({"error": f"Failed to generate graph: {str(e)}"}), 500


@app.route('/llm/clear-major-graph-cache', methods=['POST'])
def clear_major_graph_cache():
    try:
        csv_path = os.path.join(os.path.dirname(__file__), '..', 'LLM', 'nith_majors.csv')
        if not os.path.exists(csv_path):
            csv_path = os.path.join(os.path.dirname(__file__), '..', 'LLM', 'all_abington_majors_combined(5).csv')
        if not os.path.exists(csv_path):
            return jsonify({"error": f"CSV file not found at: {csv_path}"}), 500
        mg = MajorGraph(csv_path)
        removed = mg.clear_cache()
        return jsonify({"removed": removed}), 200
    except Exception as e:
        logger.error(f"Error clearing cache: {str(e)}")
        return jsonify({"error": f"Failed to clear cache: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5002)
