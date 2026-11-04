import sys
import tempfile
import unittest
from pathlib import Path


API_DIR = Path(__file__).resolve().parent / "LLMAcadamic-Advisor" / "api"
if str(API_DIR) not in sys.path:
    sys.path.insert(0, str(API_DIR))

from api import SessionStateStore  # noqa: E402


class SessionStateStoreTests(unittest.TestCase):
    def test_sqlite_store_persists_session_metadata_and_trims_history(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "session-state.sqlite3"
            store = SessionStateStore(str(db_path))
            store.set_conversation("session-1", [{"role": "user", "content": "hello"}])
            store.set_transcript_hash("session-1", "abc123")
            store.close()

            reopened = SessionStateStore(str(db_path))
            self.assertEqual(
                reopened.get_conversation("session-1"),
                [{"role": "user", "content": "hello"}],
            )
            self.assertEqual(reopened.get_transcript_hash("session-1"), "abc123")

            messages = [{"role": "user", "content": str(i)} for i in range(60)]
            trimmed = reopened.append_messages("session-1", messages)
            self.assertEqual(len(trimmed), 50)
            self.assertEqual(trimmed[0]["content"], "10")

            reopened.clear_conversation("session-1")
            reopened.delete_transcript_hash("session-1")
            self.assertEqual(reopened.get_conversation("session-1"), [])
            self.assertEqual(reopened.get_transcript_hash("session-1"), "no_transcript")
            reopened.close()


if __name__ == "__main__":
    unittest.main()
