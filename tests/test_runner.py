from __future__ import annotations

import tempfile
import threading
import unittest
import subprocess
import signal
from pathlib import Path
from unittest.mock import patch

from runpod_guard.models import JobSpec
from runpod_guard.runner import RunpodRunner
from runpod_guard.state import LeaseStore


class FakeAPI:
    def __init__(self):
        self.deleted = []
        self.body = None
        self.identity = "fake-api"

    def create_pod(self, body):
        self.body = body
        return {"id": "pod-1", "costPerHr": "0.40"}

    def get_pod(self, _pod_id):
        return {"desiredStatus": "RUNNING", "publicIp": "127.0.0.1", "portMappings": {"22": 22}}

    def delete_and_confirm(self, pod_id):
        self.deleted.append(pod_id)
        return True

    def list_pods(self):
        return []


class RunnerTests(unittest.TestCase):
    def runner(self, root, api=None):
        key = Path(root) / "key"
        subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(key)],
                       check=True)
        return RunpodRunner(api=api or FakeAPI(), ssh_key=key,
                            leases=LeaseStore(Path(root) / "leases"), log=lambda _: None)

    def test_execute_arms_watchdog_and_deletes(self):
        with tempfile.TemporaryDirectory() as root:
            api = FakeAPI()
            runner = self.runner(root, api)
            scripts = []

            def ssh(_ip, _port, script, _timeout):
                scripts.append(script)
                return 0

            with patch.object(runner, "_wait_for_address", return_value=("127.0.0.1", 22)), \
                 patch.object(runner, "_wait_for_ssh"), patch.object(runner, "_ssh", side_effect=ssh):
                result = runner.execute(JobSpec(repo="https://example/repo", ref="abc",
                                                command="python test.py", max_minutes=10,
                                                max_cost_per_hour=0.5))
            self.assertTrue(result.ok)
            self.assertEqual(api.deleted, ["pod-1"])
            self.assertEqual(api.body["volumeInGb"], 0)
            self.assertNotIn("RUNPOD_API_KEY", api.body["env"])
            self.assertIn("PUBLIC_KEY", api.body["env"])
            self.assertIn("runpodctl pod delete", scripts[0])
            self.assertIn("while true", scripts[0])
            self.assertIn("nvidia-smi -L", scripts[1])
            self.assertIn("git checkout --detach FETCH_HEAD", scripts[2])

    def test_over_budget_still_deletes(self):
        with tempfile.TemporaryDirectory() as root:
            api = FakeAPI()
            runner = self.runner(root, api)
            with self.assertRaises(RuntimeError):
                runner.execute(JobSpec(repo="https://example/repo", ref="y", command="z",
                                       max_cost_per_hour=0.10))
            self.assertEqual(api.deleted, ["pod-1"])

    def test_nonfinite_provider_cost_fails_closed(self):
        with tempfile.TemporaryDirectory() as root:
            api = FakeAPI()
            api.create_pod = lambda body: {"id": "pod-1", "costPerHr": "NaN"}
            runner = self.runner(root, api)
            with self.assertRaises(RuntimeError):
                runner.execute(JobSpec(repo="https://example/repo", ref="y", command="z",
                                       max_cost_per_hour=1.0))
            self.assertEqual(api.deleted, ["pod-1"])

    def test_logger_failure_still_deletes(self):
        with tempfile.TemporaryDirectory() as root:
            api = FakeAPI()
            runner = self.runner(root, api)
            runner.log = lambda _message: (_ for _ in ()).throw(BrokenPipeError())
            with patch.object(runner, "_wait_for_address", return_value=("127.0.0.1", 22)), \
                 patch.object(runner, "_wait_for_ssh"), patch.object(runner, "_ssh", return_value=0):
                result = runner.execute(JobSpec(repo="https://example/repo", ref="y", command="z"))
            self.assertTrue(result.ok)
            self.assertEqual(api.deleted, ["pod-1"])

    def test_library_can_execute_in_worker_thread(self):
        with tempfile.TemporaryDirectory() as root:
            api = FakeAPI()
            runner = self.runner(root, api)
            results = []

            def target():
                with patch.object(runner, "_wait_for_address", return_value=("127.0.0.1", 22)), \
                     patch.object(runner, "_wait_for_ssh"), patch.object(runner, "_ssh", return_value=0):
                    results.append(runner.execute(JobSpec(repo="https://example/repo", ref="y", command="z")))

            thread = threading.Thread(target=target)
            thread.start()
            thread.join()
            self.assertTrue(results[0].ok)
            self.assertEqual(api.deleted, ["pod-1"])

    def test_post_create_lease_failure_deletes_and_restores_signals(self):
        with tempfile.TemporaryDirectory() as root:
            api = FakeAPI()
            runner = self.runner(root, api)
            original_int = signal.getsignal(signal.SIGINT)
            original_term = signal.getsignal(signal.SIGTERM)
            original_put = runner.leases.put
            calls = 0

            def fail_second_put(key, value):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError("disk full")
                return original_put(key, value)

            runner.leases.put = fail_second_put
            with self.assertRaises(OSError):
                runner.execute(JobSpec(repo="https://example/repo", ref="y", command="z"))
            self.assertEqual(api.deleted, ["pod-1"])
            self.assertIs(signal.getsignal(signal.SIGINT), original_int)
            self.assertIs(signal.getsignal(signal.SIGTERM), original_term)

    def test_reaper_continues_after_lease_bookkeeping_failure(self):
        class BrokenLeases:
            errors = []

            def expired(self):
                return [
                    {"pod_id": "a-gone", "name": "rpg-gone", "api_identity": "fake-api"},
                    {"pod_id": "z-live", "name": "rpg-live", "api_identity": "fake-api"},
                ]

            def remove(self, pod_id):
                if pod_id == "a-gone":
                    raise PermissionError("read only")

        with tempfile.TemporaryDirectory() as root:
            api = FakeAPI()
            api.list_pods = lambda: [{"id": "z-live", "name": "rpg-live"}]
            runner = self.runner(root, api)
            runner.leases = BrokenLeases()
            with patch("runpod_guard.runner.time.sleep"):
                removed = runner.reap()
            self.assertEqual(removed, ["z-live"])
            self.assertEqual(api.deleted, ["z-live"])
            self.assertTrue(runner.reap_failures)


if __name__ == "__main__":
    unittest.main()
