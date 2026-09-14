from __future__ import annotations

import tempfile
import threading
import unittest
import subprocess
import signal
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from runpod_guard.api import RunpodAPIError
from runpod_guard.models import JobSpec
from runpod_guard.runner import RunpodRunner
from runpod_guard.state import LeaseStore


class FakeAPI:
    def __init__(self):
        self.deleted = []
        self.started = []
        self.stopped = []
        self.body = None
        self.identity = "fake-api"
        self.pods = {}

    def create_pod(self, body):
        self.body = body
        pod = {"id": "pod-1", "name": body["name"], "costPerHr": "0.40",
               "desiredStatus": "RUNNING", "env": dict(body["env"])}
        self.pods[pod["id"]] = pod
        return pod

    def get_pod(self, pod_id):
        return {**self.pods.get(pod_id, {}), "publicIp": "127.0.0.1",
                "portMappings": {"22": 22}}

    def start_pod(self, pod_id):
        self.started.append(pod_id)
        self.pods[pod_id]["desiredStatus"] = "RUNNING"

    def stop_and_confirm(self, pod_id):
        self.stopped.append(pod_id)
        self.pods[pod_id]["desiredStatus"] = "EXITED"
        return True

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
            self.assertIn("git checkout --detach --force FETCH_HEAD", scripts[2])
            self.assertIn("/root/runpod-guard-job", scripts[2])
            self.assertIn("RUNPOD_GUARD_CACHE=/root/runpod-guard-cache", scripts[2])
            self.assertNotIn("RUNPOD_GUARD_CACHE=/workspace/runpod-guard-cache", scripts[2])
            self.assertNotIn("--filter", scripts[2])

    def test_completed_job_can_pause_for_bounded_retest_and_be_reused(self):
        with tempfile.TemporaryDirectory() as root:
            api = FakeAPI()
            runner = self.runner(root, api)
            scripts = []
            with patch.object(runner, "_wait_for_address", return_value=("127.0.0.1", 22)), \
                 patch.object(runner, "_wait_for_ssh"), \
                 patch.object(runner, "_ssh", side_effect=lambda _i, _p, s, _t: scripts.append(s) or 0):
                first = runner.execute(JobSpec(
                    repo="https://example/repo", ref="abc", command="pytest",
                    max_minutes=10, retest_window_minutes=15,
                ))
            self.assertTrue(first.ok)
            self.assertTrue(first.paused)
            self.assertFalse(first.terminated)
            self.assertIsNotNone(first.retest_expires_at)
            self.assertEqual(api.stopped, ["pod-1"])
            self.assertEqual(api.body["volumeInGb"], 20)
            self.assertEqual(api.body["volumeMountPath"], "/workspace")
            self.assertIn("/workspace/runpod-guard-job", scripts[2])
            self.assertIn("RUNPOD_GUARD_CACHE=/workspace/runpod-guard-cache", scripts[2])

            with patch.object(runner, "_wait_for_address", return_value=("127.0.0.1", 22)), \
                 patch.object(runner, "_wait_for_ssh"), patch.object(runner, "_ssh", return_value=0):
                second = runner.execute(JobSpec(
                    repo="https://example/repo", ref="def", command="pytest",
                    max_minutes=10, reuse_pod_id="pod-1",
                ))
            self.assertTrue(second.ok)
            self.assertEqual(api.started, ["pod-1"])
            self.assertEqual(api.deleted, ["pod-1"])

    def test_stopped_retest_lease_can_be_extended_without_starting_gpu(self):
        with tempfile.TemporaryDirectory() as root:
            api = FakeAPI()
            runner = self.runner(root, api)
            with patch.object(runner, "_wait_for_address", return_value=("127.0.0.1", 22)), \
                 patch.object(runner, "_wait_for_ssh"), patch.object(runner, "_ssh", return_value=0):
                runner.execute(JobSpec(
                    repo="https://example/repo", ref="abc", command="pytest",
                    max_minutes=10, retest_window_minutes=15,
                ))
            before = datetime.now(timezone.utc)
            expiry = datetime.fromisoformat(runner.extend_retest("pod-1", 1440))
            self.assertGreaterEqual((expiry - before).total_seconds(), 1439 * 60)
            self.assertEqual(api.started, [])
            self.assertEqual(api.pods["pod-1"]["desiredStatus"], "EXITED")
            self.assertEqual(runner.leases.all()[0]["expires_at"], expiry.isoformat())

    def test_extend_refuses_running_or_unowned_pod_and_invalid_window(self):
        with tempfile.TemporaryDirectory() as root:
            api = FakeAPI()
            runner = self.runner(root, api)
            with patch.object(runner, "_wait_for_address", return_value=("127.0.0.1", 22)), \
                 patch.object(runner, "_wait_for_ssh"), patch.object(runner, "_ssh", return_value=0):
                runner.execute(JobSpec(
                    repo="https://example/repo", ref="abc", command="pytest",
                    max_minutes=10, retest_window_minutes=15,
                ))
            api.pods["pod-1"]["desiredStatus"] = "RUNNING"
            with self.assertRaisesRegex(RuntimeError, "is not stopped"):
                runner.extend_retest("pod-1", 60)
            api.pods["pod-1"]["desiredStatus"] = "EXITED"
            lease = runner.leases.all()[0]
            lease["api_identity"] = "someone-else"
            runner.leases.put("pod-1", {key: value for key, value in lease.items() if key != "_path"})
            with self.assertRaisesRegex(RuntimeError, "this API identity"):
                runner.extend_retest("pod-1", 60)
            for minutes in (0, 1441):
                with self.assertRaisesRegex(ValueError, "between 1 and 1440"):
                    runner.extend_retest("pod-1", minutes)

    def test_reuse_rejects_changed_immutable_pod_configuration(self):
        with tempfile.TemporaryDirectory() as root:
            api = FakeAPI()
            runner = self.runner(root, api)
            with patch.object(runner, "_wait_for_address", return_value=("127.0.0.1", 22)), \
                 patch.object(runner, "_wait_for_ssh"), patch.object(runner, "_ssh", return_value=0):
                runner.execute(JobSpec(
                    repo="https://example/repo", ref="abc", command="pytest",
                    max_minutes=10, retest_window_minutes=15,
                ))
            with self.assertRaisesRegex(RuntimeError, "configuration differs"):
                runner.execute(JobSpec(
                    repo="https://example/repo", ref="def", command="pytest",
                    image="different/image", max_minutes=10, reuse_pod_id="pod-1",
                ))
            self.assertEqual(api.started, [])
            self.assertEqual(api.deleted, [])
            self.assertEqual(runner.leases.all()[0]["state"], "paused-for-retest")

    def test_reuse_rejects_changed_ssh_key_before_starting_pod(self):
        with tempfile.TemporaryDirectory() as root:
            api = FakeAPI()
            runner = self.runner(root, api)
            with patch.object(runner, "_wait_for_address", return_value=("127.0.0.1", 22)), \
                 patch.object(runner, "_wait_for_ssh"), patch.object(runner, "_ssh", return_value=0):
                runner.execute(JobSpec(
                    repo="https://example/repo", ref="abc", command="pytest",
                    max_minutes=10, retest_window_minutes=15,
                ))

            replacement_key = Path(root) / "replacement-key"
            subprocess.run(
                ["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(replacement_key)],
                check=True,
            )
            runner.ssh_key = replacement_key
            with self.assertRaisesRegex(RuntimeError, "configuration differs"):
                runner.execute(JobSpec(
                    repo="https://example/repo", ref="def", command="pytest",
                    max_minutes=10, reuse_pod_id="pod-1",
                ))
            self.assertEqual(api.started, [])
            self.assertEqual(api.deleted, [])

    def test_reuse_upgrades_legacy_lease_after_verifying_pod_ssh_key(self):
        with tempfile.TemporaryDirectory() as root:
            api = FakeAPI()
            runner = self.runner(root, api)
            with patch.object(runner, "_wait_for_address", return_value=("127.0.0.1", 22)), \
                 patch.object(runner, "_wait_for_ssh"), patch.object(runner, "_ssh", return_value=0):
                runner.execute(JobSpec(
                    repo="https://example/repo", ref="abc", command="pytest",
                    max_minutes=10, retest_window_minutes=15,
                ))

            lease = runner.leases.all()[0]
            lease["pod_configuration"].pop("ssh_public_key_sha256")
            runner.leases.put("pod-1", {key: value for key, value in lease.items() if key != "_path"})
            api.start_pod = lambda _pod_id: (_ for _ in ()).throw(
                RunpodAPIError("GPU is no longer available")
            )
            with self.assertRaises(RunpodAPIError):
                runner.execute(JobSpec(
                    repo="https://example/repo", ref="def", command="pytest",
                    max_minutes=10, reuse_pod_id="pod-1", reuse_start_attempts=1,
                ))
            upgraded = runner.leases.all()[0]["pod_configuration"]
            self.assertIn("ssh_public_key_sha256", upgraded)
            self.assertEqual(api.started, [])

    def test_reuse_rejects_legacy_lease_when_pod_ssh_key_does_not_match(self):
        with tempfile.TemporaryDirectory() as root:
            api = FakeAPI()
            runner = self.runner(root, api)
            with patch.object(runner, "_wait_for_address", return_value=("127.0.0.1", 22)), \
                 patch.object(runner, "_wait_for_ssh"), patch.object(runner, "_ssh", return_value=0):
                runner.execute(JobSpec(
                    repo="https://example/repo", ref="abc", command="pytest",
                    max_minutes=10, retest_window_minutes=15,
                ))

            lease = runner.leases.all()[0]
            lease["pod_configuration"].pop("ssh_public_key_sha256")
            runner.leases.put("pod-1", {key: value for key, value in lease.items() if key != "_path"})
            api.pods["pod-1"]["env"]["SSH_PUBLIC_KEY"] = "different"
            with self.assertRaisesRegex(RuntimeError, "does not match.*SSH public key"):
                runner.execute(JobSpec(
                    repo="https://example/repo", ref="def", command="pytest",
                    max_minutes=10, reuse_pod_id="pod-1",
                ))
            self.assertEqual(api.started, [])
            self.assertEqual(api.deleted, [])
            self.assertNotIn(
                "ssh_public_key_sha256",
                runner.leases.all()[0]["pod_configuration"],
            )

    def test_reuse_checks_refreshed_running_cost(self):
        class ChangingCostAPI(FakeAPI):
            def get_pod(self, pod_id):
                pod = super().get_pod(pod_id)
                pod["costPerHr"] = "0.80" if pod.get("desiredStatus") == "RUNNING" else "0.01"
                return pod

        with tempfile.TemporaryDirectory() as root:
            api = ChangingCostAPI()
            runner = self.runner(root, api)
            with patch.object(runner, "_wait_for_address", return_value=("127.0.0.1", 22)), \
                 patch.object(runner, "_wait_for_ssh"), patch.object(runner, "_ssh", return_value=0):
                runner.execute(JobSpec(
                    repo="https://example/repo", ref="abc", command="pytest",
                    max_minutes=10, max_cost_per_hour=1.0, retest_window_minutes=15,
                ))
            api.stopped.clear()
            with patch.object(runner, "_wait_for_address", return_value=("127.0.0.1", 22)), \
                 self.assertRaisesRegex(RuntimeError, "above.*limit"):
                runner.execute(JobSpec(
                    repo="https://example/repo", ref="def", command="pytest",
                    max_minutes=10, max_cost_per_hour=0.5, reuse_pod_id="pod-1",
                ))
            self.assertEqual(api.started, ["pod-1"])
            self.assertEqual(api.stopped, ["pod-1"])
            self.assertEqual(api.deleted, [])

    def test_failed_pause_falls_back_to_verified_delete(self):
        with tempfile.TemporaryDirectory() as root:
            api = FakeAPI()
            api.stop_and_confirm = lambda _pod_id: False
            runner = self.runner(root, api)
            with patch.object(runner, "_wait_for_address", return_value=("127.0.0.1", 22)), \
                 patch.object(runner, "_wait_for_ssh"), patch.object(runner, "_ssh", return_value=0):
                result = runner.execute(JobSpec(
                    repo="https://example/repo", ref="abc", command="pytest",
                    max_minutes=10, retest_window_minutes=15,
                ))
            self.assertTrue(result.ok)
            self.assertFalse(result.paused)
            self.assertTrue(result.terminated)
            self.assertEqual(api.deleted, ["pod-1"])

    def test_failed_test_is_paused_when_retest_was_requested(self):
        with tempfile.TemporaryDirectory() as root:
            api = FakeAPI()
            runner = self.runner(root, api)
            calls = 0

            def ssh(_ip, _port, _script, _timeout):
                nonlocal calls
                calls += 1
                return 1 if calls == 3 else 0

            with patch.object(runner, "_wait_for_address", return_value=("127.0.0.1", 22)), \
                 patch.object(runner, "_wait_for_ssh"), patch.object(runner, "_ssh", side_effect=ssh):
                result = runner.execute(JobSpec(
                    repo="https://example/repo", ref="abc", command="pytest",
                    max_minutes=10, retest_window_minutes=15,
                ))
            self.assertFalse(result.ok)
            self.assertEqual(result.returncode, 1)
            self.assertTrue(result.paused)
            self.assertEqual(api.stopped, ["pod-1"])
            self.assertEqual(api.deleted, [])

    def test_timeout_after_execution_starts_is_paused_for_retest(self):
        with tempfile.TemporaryDirectory() as root:
            api = FakeAPI()
            runner = self.runner(root, api)
            calls = 0

            def ssh(_ip, _port, _script, timeout):
                nonlocal calls
                calls += 1
                if calls == 3:
                    raise subprocess.TimeoutExpired("ssh", timeout)
                return 0

            with patch.object(runner, "_wait_for_address", return_value=("127.0.0.1", 22)), \
                 patch.object(runner, "_wait_for_ssh"), patch.object(runner, "_ssh", side_effect=ssh):
                result = runner.execute(JobSpec(
                    repo="https://example/repo", ref="abc", command="pytest",
                    max_minutes=10, retest_window_minutes=15,
                ))
            self.assertTrue(result.timed_out)
            self.assertTrue(result.paused)
            self.assertEqual(api.deleted, [])

    def test_unconfirmed_stop_and_delete_keep_short_running_lease(self):
        with tempfile.TemporaryDirectory() as root:
            api = FakeAPI()
            api.stop_and_confirm = lambda _pod_id: False
            api.delete_and_confirm = lambda _pod_id: False
            runner = self.runner(root, api)
            with patch.object(runner, "_wait_for_address", return_value=("127.0.0.1", 22)), \
                 patch.object(runner, "_wait_for_ssh"), patch.object(runner, "_ssh", return_value=0):
                result = runner.execute(JobSpec(
                    repo="https://example/repo", ref="abc", command="pytest",
                    max_minutes=10, retest_window_minutes=60,
                ))
            self.assertFalse(result.ok)
            lease = runner.leases.all()[0]
            self.assertEqual(lease["state"], "running")
            self.assertEqual(lease["max_minutes"], 10)

    def test_concurrent_reuse_claim_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            api = FakeAPI()
            runner = self.runner(root, api)
            with patch.object(runner, "_wait_for_address", return_value=("127.0.0.1", 22)), \
                 patch.object(runner, "_wait_for_ssh"), patch.object(runner, "_ssh", return_value=0):
                runner.execute(JobSpec(
                    repo="https://example/repo", ref="abc", command="pytest",
                    max_minutes=10, retest_window_minutes=15,
                ))
            with runner.leases.claim("pod-1"), self.assertRaisesRegex(
                    RuntimeError, "already claimed"):
                runner.execute(JobSpec(
                    repo="https://example/repo", ref="def", command="pytest",
                    max_minutes=10, reuse_pod_id="pod-1",
                ))
            self.assertEqual(api.started, [])
            self.assertEqual(api.deleted, [])

    def test_restart_failure_restores_original_paused_lease(self):
        with tempfile.TemporaryDirectory() as root:
            api = FakeAPI()
            runner = self.runner(root, api)
            with patch.object(runner, "_wait_for_address", return_value=("127.0.0.1", 22)), \
                 patch.object(runner, "_wait_for_ssh"), patch.object(runner, "_ssh", return_value=0):
                first = runner.execute(JobSpec(
                    repo="https://example/repo", ref="abc", command="pytest",
                    max_minutes=10, retest_window_minutes=15,
                ))
            original_expiry = first.retest_expires_at
            api.stopped.clear()
            api.start_pod = lambda _pod_id: (_ for _ in ()).throw(
                RunpodAPIError("GPU is no longer available")
            )
            with patch("runpod_guard.runner.time.sleep") as sleep, self.assertRaises(RunpodAPIError):
                runner.execute(JobSpec(
                    repo="https://example/repo", ref="def", command="pytest",
                    max_minutes=10, reuse_pod_id="pod-1",
                ))
            self.assertEqual(3, sleep.call_count)
            lease = runner.leases.all()[0]
            self.assertEqual(lease["state"], "paused-for-retest")
            self.assertEqual(lease["expires_at"], original_expiry)
            self.assertEqual(api.stopped, ["pod-1"])
            self.assertEqual(api.deleted, [])

    def test_reuse_retries_temporary_start_failure_then_runs(self):
        with tempfile.TemporaryDirectory() as root:
            api = FakeAPI()
            runner = self.runner(root, api)
            with patch.object(runner, "_wait_for_address", return_value=("127.0.0.1", 22)), \
                 patch.object(runner, "_wait_for_ssh"), patch.object(runner, "_ssh", return_value=0):
                runner.execute(JobSpec(
                    repo="https://example/repo", ref="abc", command="pytest",
                    max_minutes=10, retest_window_minutes=15,
                ))
            calls = 0
            original_start = api.start_pod

            def start(pod_id):
                nonlocal calls
                calls += 1
                if calls < 3:
                    raise RunpodAPIError("no free GPU", 500)
                original_start(pod_id)

            api.start_pod = start
            with patch("runpod_guard.runner.time.sleep") as sleep, \
                 patch.object(runner, "_wait_for_address", return_value=("127.0.0.1", 22)), \
                 patch.object(runner, "_wait_for_ssh"), patch.object(runner, "_ssh", return_value=0):
                result = runner.execute(JobSpec(
                    repo="https://example/repo", ref="def", command="pytest",
                    max_minutes=10, reuse_pod_id="pod-1",
                ))
            self.assertTrue(result.ok)
            self.assertEqual(3, calls)
            self.assertEqual([20, 20], [call.args[0] for call in sleep.call_args_list])

    def test_reuse_accepts_start_confirmed_after_lost_response(self):
        with tempfile.TemporaryDirectory() as root:
            api = FakeAPI()
            runner = self.runner(root, api)
            with patch.object(runner, "_wait_for_address", return_value=("127.0.0.1", 22)), \
                 patch.object(runner, "_wait_for_ssh"), patch.object(runner, "_ssh", return_value=0):
                runner.execute(JobSpec(
                    repo="https://example/repo", ref="abc", command="pytest",
                    max_minutes=10, retest_window_minutes=15,
                ))

            calls = 0

            def start(pod_id):
                nonlocal calls
                calls += 1
                api.pods[pod_id]["desiredStatus"] = "RUNNING"
                raise RunpodAPIError("start response was lost")

            api.start_pod = start
            with patch("runpod_guard.runner.time.sleep") as sleep, \
                 patch.object(runner, "_wait_for_address", return_value=("127.0.0.1", 22)), \
                 patch.object(runner, "_wait_for_ssh"), patch.object(runner, "_ssh", return_value=0):
                result = runner.execute(JobSpec(
                    repo="https://example/repo", ref="def", command="pytest",
                    max_minutes=10, reuse_pod_id="pod-1",
                ))
            self.assertTrue(result.ok)
            self.assertEqual(1, calls)
            sleep.assert_not_called()

    def test_signal_during_reuse_retry_restores_stopped_lease(self):
        with tempfile.TemporaryDirectory() as root:
            api = FakeAPI()
            runner = self.runner(root, api)
            with patch.object(runner, "_wait_for_address", return_value=("127.0.0.1", 22)), \
                 patch.object(runner, "_wait_for_ssh"), patch.object(runner, "_ssh", return_value=0):
                first = runner.execute(JobSpec(
                    repo="https://example/repo", ref="abc", command="pytest",
                    max_minutes=10, retest_window_minutes=15,
                ))

            original_handler = signal.getsignal(signal.SIGTERM)
            api.stopped.clear()
            api.start_pod = lambda _pod_id: (_ for _ in ()).throw(
                RunpodAPIError("temporary server error", 500)
            )
            with patch("runpod_guard.runner.time.sleep", side_effect=lambda _: signal.raise_signal(
                    signal.SIGTERM)), self.assertRaises(KeyboardInterrupt):
                runner.execute(JobSpec(
                    repo="https://example/repo", ref="def", command="pytest",
                    max_minutes=10, reuse_pod_id="pod-1",
                ))
            lease = runner.leases.all()[0]
            self.assertEqual("paused-for-retest", lease["state"])
            self.assertEqual(first.retest_expires_at, lease["expires_at"])
            self.assertEqual(["pod-1"], api.stopped)
            self.assertIs(signal.getsignal(signal.SIGTERM), original_handler)

    def test_reuse_does_not_retry_permanent_start_failure(self):
        with tempfile.TemporaryDirectory() as root:
            api = FakeAPI()
            runner = self.runner(root, api)
            with patch.object(runner, "_wait_for_address", return_value=("127.0.0.1", 22)), \
                 patch.object(runner, "_wait_for_ssh"), patch.object(runner, "_ssh", return_value=0):
                runner.execute(JobSpec(
                    repo="https://example/repo", ref="abc", command="pytest",
                    max_minutes=10, retest_window_minutes=15,
                ))
            calls = 0

            def start(_pod_id):
                nonlocal calls
                calls += 1
                raise RunpodAPIError("forbidden", 403)

            api.start_pod = start
            with patch("runpod_guard.runner.time.sleep") as sleep, self.assertRaises(RunpodAPIError), \
                 patch.object(runner, "_wait_for_address", return_value=("127.0.0.1", 22)), \
                 patch.object(runner, "_wait_for_ssh"), patch.object(runner, "_ssh", return_value=0):
                runner.execute(JobSpec(
                    repo="https://example/repo", ref="def", command="pytest",
                    max_minutes=10, reuse_pod_id="pod-1",
                ))
            self.assertEqual(1, calls)
            sleep.assert_not_called()

    def test_reuse_does_not_start_again_after_retry_deadline(self):
        with tempfile.TemporaryDirectory() as root:
            api = FakeAPI()
            runner = self.runner(root, api)
            calls = 0
            budget_checks = 0

            def start(_pod_id):
                nonlocal calls
                calls += 1
                raise RunpodAPIError("temporary server error", 500)

            def remaining():
                nonlocal budget_checks
                budget_checks += 1
                if budget_checks > 2:
                    raise TimeoutError("deadline reached")
                return 1

            api.start_pod = start
            spec = JobSpec(
                repo="https://example/repo", ref="abc", command="pytest",
                reuse_pod_id="pod-1",
            )
            with patch("runpod_guard.runner.time.sleep") as sleep, \
                 self.assertRaisesRegex(TimeoutError, "deadline reached"):
                runner._start_retained_pod("pod-1", spec, remaining)
            self.assertEqual(1, calls)
            sleep.assert_called_once_with(1)

    def test_local_source_is_archived_uploaded_and_extracted(self):
        with tempfile.TemporaryDirectory() as root:
            source = Path(root) / "source"
            source.mkdir()
            subprocess.run(["git", "init", "-q", str(source)], check=True)
            subprocess.run(["git", "-C", str(source), "config", "user.email", "test@example.com"], check=True)
            subprocess.run(["git", "-C", str(source), "config", "user.name", "Test"], check=True)
            (source / "tracked.txt").write_text("tracked\n")
            (source / ".env").write_text("SECRET=not-uploaded\n")
            subprocess.run(["git", "-C", str(source), "add", "tracked.txt"], check=True)
            subprocess.run(["git", "-C", str(source), "commit", "-qm", "fixture"], check=True)
            api = FakeAPI()
            runner = self.runner(root, api)
            scripts = []
            archives = []

            def ssh(_ip, _port, script, _timeout):
                scripts.append(script)
                return 0

            def upload(_ip, _port, archive, _timeout):
                listing = subprocess.run(["tar", "-tf", str(archive)], check=True,
                                         text=True, stdout=subprocess.PIPE).stdout.splitlines()
                archives.extend(listing)
                return True

            with patch.object(runner, "_wait_for_address", return_value=("127.0.0.1", 22)), \
                 patch.object(runner, "_wait_for_ssh"), \
                 patch.object(runner, "_ssh", side_effect=ssh), \
                 patch.object(runner, "_upload_source", side_effect=upload):
                result = runner.execute(JobSpec(
                    repo=None, source_dir=source, ref="HEAD", command="true",
                    max_minutes=10, max_cost_per_hour=0.5,
                ))
            self.assertTrue(result.ok)
            self.assertIn("tracked.txt", archives)
            self.assertNotIn(".env", archives)
            self.assertIn("tar -xf /tmp/runpod-guard-source.tar", scripts[2])

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

    def test_reaper_skips_a_pod_claimed_by_active_retest(self):
        with tempfile.TemporaryDirectory() as root:
            api = FakeAPI()
            api.pods["pod-1"] = {"id": "pod-1", "name": "rpg-tests-1234"}
            api.list_pods = lambda: list(api.pods.values())
            runner = self.runner(root, api)
            runner.leases.put("pod-1", {
                "pod_id": "pod-1", "name": "rpg-tests-1234",
                "expires_at": "2020-01-01T00:00:00+00:00",
                "api_identity": api.identity, "state": "running",
            })
            with runner.leases.claim("pod-1"), patch("runpod_guard.runner.time.sleep"):
                removed = runner.reap()
            self.assertEqual(removed, [])
            self.assertEqual(api.deleted, [])
            self.assertEqual(runner.reap_failures, [])


if __name__ == "__main__":
    unittest.main()
