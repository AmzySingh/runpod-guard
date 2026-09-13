from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import tempfile
import unittest

from runpod_guard.state import LeaseStore


class StateTests(unittest.TestCase):
    def test_corrupt_records_do_not_hide_valid_expiry(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)
            (path / "scalar.json").write_text("[]")
            (path / "naive.json").write_text(json.dumps({"expires_at": "2026-01-01T00:00:00"}))
            store = LeaseStore(path)
            store.put("valid", {
                "pod_id": "valid",
                "expires_at": (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat(),
            })
            expired = store.expired()
            self.assertEqual([lease["pod_id"] for lease in expired], ["valid"])
            self.assertEqual(len(store.errors), 2)


if __name__ == "__main__":
    unittest.main()
