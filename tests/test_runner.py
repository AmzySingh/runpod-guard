from __future__ import annotations

import tempfile
import json
import threading
import unittest
import subprocess
import signal
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from runpod_guard.api import RunpodAPIError, RunpodPodNotFound
from runpod_guard.models import Artifact, JobResult, JobSpec
from runpod_guard.runner import RetainedPodLeaseMissing, RetainedPodUnavailable, RunpodRunner
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
            self.assertTrue(result.job_started)
            self.assertIsNone(result.failure_stage)
            self.assertEqual(api.deleted, ["pod-1"])
            self.assertEqual(api.body["volumeInGb"], 0)
            self.assertEqual(api.body["gpuTypePriority"], "custom")
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

    def test_availability_gpu_priority_is_sent_to_runpod(self):
        with tempfile.TemporaryDirectory() as root:
            api = FakeAPI()
            runner = self.runner(root, api)
            with patch.object(runner, "_wait_for_address", return_value=("127.0.0.1", 22)), \
                 patch.object(runner, "_wait_for_ssh"), patch.object(runner, "_ssh", return_value=0):
                result = runner.execute(JobSpec(
                    repo="https://example/repo", ref="abc", command="true",
                    gpu_priority="availability",
                ))
            self.assertTrue(result.ok)
            self.assertEqual("availability", api.body["gpuTypePriority"])
            self.assertEqual("availability", result.requested["gpu_priority"])

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

    def test_reuse_can_fall_back_to_fresh_only_after_retryable_start_attempts(self):
        with tempfile.TemporaryDirectory() as root:
            api = FakeAPI()
            runner = self.runner(root, api)
            with patch.object(runner, "_wait_for_address", return_value=("127.0.0.1", 22)), \
                 patch.object(runner, "_wait_for_ssh"), patch.object(runner, "_ssh", return_value=0):
                first = runner.execute(JobSpec(
                    repo="https://example/repo", ref="abc", command="pytest",
                    max_minutes=10, retest_window_minutes=15,
                ))

            def unavailable(_pod_id):
                raise RunpodAPIError("no free GPU", 500)

            def fresh(body):
                api.body = body
                pod = {"id": "pod-2", "name": body["name"], "costPerHr": "0.40",
                       "desiredStatus": "RUNNING", "env": dict(body["env"])}
                api.pods[pod["id"]] = pod
                return pod

            api.start_pod = unavailable
            api.create_pod = fresh
            with patch("runpod_guard.runner.time.sleep") as sleep, \
                 patch.object(runner, "_wait_for_address", return_value=("127.0.0.1", 22)), \
                 patch.object(runner, "_wait_for_ssh"), patch.object(runner, "_ssh", return_value=0):
                result = runner.execute(JobSpec(
                    repo="https://example/repo", ref="def", command="pytest",
                    max_minutes=10, reuse_pod_ids=("pod-1",), reuse_start_attempts=2,
                    gpu_priority="availability",
                    fallback_fresh_on_reuse_unavailable=True,
                ))

            self.assertTrue(result.ok)
            self.assertTrue(result.requested["reuse_requested"])
            self.assertEqual(10, result.requested["max_minutes"])
            self.assertTrue(result.fresh_fallback_used)
            self.assertEqual(9, result.fresh_fallback_max_minutes)
            self.assertTrue(result.retained_pod_preserved_at_fallback)
            self.assertEqual(("unavailable-preserved",), result.reuse_candidate_dispositions)
            self.assertGreater(result.elapsed_seconds, 0)
            encoded = result.to_dict()
            self.assertTrue(encoded["fresh_fallback_used"])
            self.assertEqual(9, encoded["fresh_fallback_max_minutes"])
            self.assertTrue(encoded["retained_pod_preserved_at_fallback"])
            self.assertEqual(["unavailable-preserved"], encoded["reuse_candidate_dispositions"])
            self.assertEqual([20], [call.args[0] for call in sleep.call_args_list])
            self.assertEqual("availability", api.body["gpuTypePriority"])
            self.assertEqual("availability", result.requested["gpu_priority"])
            self.assertEqual(["pod-2"], api.deleted)
            retained = runner.leases.all()[0]
            self.assertEqual("pod-1", retained["pod_id"])
            self.assertEqual(first.retest_expires_at, retained["expires_at"])

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
                    signal.SIGTERM)), self.assertRaises(KeyboardInterrupt) as caught:
                runner.execute(JobSpec(
                    repo="https://example/repo", ref="def", command="pytest",
                    max_minutes=10, reuse_pod_id="pod-1",
                ))
            lease = runner.leases.all()[0]
            self.assertEqual("paused-for-retest", lease["state"])
            self.assertEqual(first.retest_expires_at, lease["expires_at"])
            self.assertEqual(["pod-1"], api.stopped)
            self.assertIs(signal.getsignal(signal.SIGTERM), original_handler)
            self.assertTrue(caught.exception.job_result.paused)
            self.assertFalse(caught.exception.job_result.job_started)
            self.assertEqual(caught.exception.job_result.failure_stage, "retained_start")

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
                    fallback_fresh_on_reuse_unavailable=True,
                ))
            self.assertEqual(1, calls)
            sleep.assert_not_called()

    def test_fresh_fallback_refuses_when_retained_cleanup_is_unconfirmed(self):
        with tempfile.TemporaryDirectory() as root:
            api = FakeAPI()
            runner = self.runner(root, api)
            with patch.object(runner, "_wait_for_address", return_value=("127.0.0.1", 22)), \
                 patch.object(runner, "_wait_for_ssh"), patch.object(runner, "_ssh", return_value=0):
                runner.execute(JobSpec(
                    repo="https://example/repo", ref="abc", command="pytest",
                    max_minutes=10, retest_window_minutes=15,
                ))
            creates = 0
            original_create = api.create_pod

            def create(body):
                nonlocal creates
                creates += 1
                return original_create(body)

            api.create_pod = create
            api.start_pod = lambda _pod_id: (_ for _ in ()).throw(
                RunpodAPIError("no free GPU", 500)
            )
            api.stop_and_confirm = lambda _pod_id: False
            api.delete_and_confirm = lambda _pod_id: False
            with self.assertRaisesRegex(RuntimeError, "not confirmed stopped"):
                runner.execute(JobSpec(
                    repo="https://example/repo", ref="def", command="pytest",
                    max_minutes=10, reuse_pod_id="pod-1", reuse_start_attempts=1,
                    fallback_fresh_on_reuse_unavailable=True,
                ))
            self.assertEqual(0, creates)

    def test_fresh_fallback_refuses_when_retry_uses_whole_deadline(self):
        with tempfile.TemporaryDirectory() as root:
            runner = self.runner(root)
            runner.leases.put("pod-1", {"pod_id": "pod-1", "state": "paused-for-retest"})
            unavailable = RetainedPodUnavailable("busy", 500)
            unavailable.job_result = JobResult(
                pod_id="pod-1", returncode=None, timed_out=False,
                artifacts_ok=True, terminated=False, paused=True, elapsed_seconds=1,
                failure_stage="retained_start",
            )
            spec = JobSpec(
                repo="https://example/repo", ref="abc", command="pytest",
                max_minutes=10, reuse_pod_id="pod-1",
                fallback_fresh_on_reuse_unavailable=True,
            )
            with patch.object(runner, "_execute", side_effect=unavailable) as execute, \
                 patch("runpod_guard.runner.time.monotonic", side_effect=[0, 0, 600, 600]), \
                 self.assertRaisesRegex(TimeoutError, "no job budget remains") as caught:
                runner.execute(spec)
            execute.assert_called_once_with(spec, monotonic_deadline=600)
            self.assertTrue(caught.exception.job_result.timed_out)
            self.assertEqual("fresh_fallback_budget",
                             caught.exception.job_result.failure_stage)
            self.assertEqual(
                ("unavailable-preserved",),
                caught.exception.job_result.reuse_candidate_dispositions,
            )

    def test_ordered_reuse_keeps_receipt_when_later_candidate_fails_validation(self):
        with tempfile.TemporaryDirectory() as root:
            runner = self.runner(root)
            for pod_id in ("pod-1", "pod-2"):
                runner.leases.put(pod_id, {
                    "pod_id": pod_id, "state": "paused-for-retest",
                })
            unavailable = RetainedPodUnavailable("busy", 500)
            unavailable.job_result = JobResult(
                pod_id="pod-1", returncode=None, timed_out=False,
                artifacts_ok=True, terminated=False, paused=True, elapsed_seconds=1,
                failure_stage="retained_start",
            )
            with patch.object(runner, "_execute", side_effect=[
                    unavailable, RuntimeError("second candidate is invalid")]), \
                 patch("runpod_guard.runner.time.monotonic", return_value=100), \
                 self.assertRaisesRegex(RuntimeError, "second candidate") as caught:
                runner.execute(JobSpec(
                    repo="https://example/repo", ref="abc", command="pytest",
                    reuse_pod_ids=("pod-1", "pod-2"),
                ))
            receipt = caught.exception.job_result
            self.assertEqual("pod-1", receipt.pod_id)
            self.assertEqual("retained_candidate", receipt.failure_stage)
            self.assertEqual(
                ("unavailable-preserved", "failed"),
                receipt.reuse_candidate_dispositions,
            )

    def test_fresh_preallocation_failure_keeps_retained_receipt(self):
        with tempfile.TemporaryDirectory() as root:
            runner = self.runner(root)
            runner.leases.put("pod-1", {"pod_id": "pod-1", "state": "paused-for-retest"})
            unavailable = RetainedPodUnavailable("busy", 500)
            unavailable.job_result = JobResult(
                pod_id="pod-1", returncode=None, timed_out=False,
                artifacts_ok=True, terminated=False, paused=True, elapsed_seconds=1,
                failure_stage="retained_start",
            )
            with patch.object(runner, "_execute", side_effect=[
                    unavailable, RuntimeError("fresh preallocation failed")]), \
                 patch("runpod_guard.runner.time.monotonic", return_value=100), \
                 self.assertRaisesRegex(RuntimeError, "fresh preallocation") as caught:
                runner.execute(JobSpec(
                    repo="https://example/repo", ref="abc", command="pytest",
                    max_minutes=10, reuse_pod_id="pod-1",
                    fallback_fresh_on_reuse_unavailable=True,
                ))
            receipt = caught.exception.job_result
            self.assertEqual("fresh_fallback", receipt.failure_stage)
            self.assertFalse(receipt.fresh_fallback_used)
            self.assertTrue(receipt.retained_pod_preserved_at_fallback)
            self.assertEqual(10, receipt.fresh_fallback_max_minutes)

    def test_ordered_reuse_advances_after_unavailable_candidate(self):
        with tempfile.TemporaryDirectory() as root:
            api = FakeAPI()
            runner = self.runner(root, api)
            with patch.object(runner, "_wait_for_address", return_value=("127.0.0.1", 22)), \
                 patch.object(runner, "_wait_for_ssh"), patch.object(runner, "_ssh", return_value=0):
                runner.execute(JobSpec(
                    repo="https://example/repo", ref="abc", command="pytest",
                    max_minutes=10, retest_window_minutes=15,
                ))
            first_lease = runner.leases.all()[0]
            second_name = "retained-second"
            api.pods["pod-2"] = {**api.pods["pod-1"], "id": "pod-2", "name": second_name}
            runner.leases.put("pod-2", {
                **{key: value for key, value in first_lease.items() if key != "_path"},
                "pod_id": "pod-2", "name": second_name,
            })
            original_start = api.start_pod

            def start(pod_id):
                if pod_id == "pod-1":
                    raise RunpodAPIError("no free GPU", 500)
                original_start(pod_id)

            api.start_pod = start
            with patch("runpod_guard.runner.time.sleep"), \
                 patch.object(runner, "_wait_for_address", return_value=("127.0.0.1", 22)), \
                 patch.object(runner, "_wait_for_ssh"), patch.object(runner, "_ssh", return_value=0):
                result = runner.execute(JobSpec(
                    repo="https://example/repo", ref="def", command="pytest",
                    max_minutes=10, reuse_pod_ids=("pod-1", "pod-2"),
                    reuse_start_attempts=1,
                ))

            self.assertTrue(result.ok)
            self.assertEqual(("unavailable-preserved", "used"),
                             result.reuse_candidate_dispositions)
            self.assertEqual(2, result.requested["reuse_candidate_count"])
            self.assertNotIn("pod-1", str(result.to_dict()))
            self.assertEqual(["pod-2"], api.started)
            self.assertEqual(["pod-2"], api.deleted)
            self.assertEqual("paused-for-retest", runner.leases.all()[0]["state"])

    def test_ordered_reuse_retires_absent_candidate_then_uses_next(self):
        with tempfile.TemporaryDirectory() as root:
            api = FakeAPI()
            runner = self.runner(root, api)
            with patch.object(runner, "_wait_for_address", return_value=("127.0.0.1", 22)), \
                 patch.object(runner, "_wait_for_ssh"), patch.object(runner, "_ssh", return_value=0):
                runner.execute(JobSpec(
                    repo="https://example/repo", ref="abc", command="pytest",
                    max_minutes=10, retest_window_minutes=15,
                ))
            first_lease = runner.leases.all()[0]
            second_name = "retained-second"
            api.pods["pod-2"] = {**api.pods["pod-1"], "id": "pod-2", "name": second_name}
            runner.leases.put("pod-2", {
                **{key: value for key, value in first_lease.items() if key != "_path"},
                "pod_id": "pod-2", "name": second_name,
            })
            original_get = api.get_pod

            def get_pod(pod_id):
                if pod_id == "pod-1":
                    raise RunpodPodNotFound("Runpod Pod was not found", 404)
                return original_get(pod_id)

            api.get_pod = get_pod
            with patch.object(runner, "_wait_for_address", return_value=("127.0.0.1", 22)), \
                 patch.object(runner, "_wait_for_ssh"), patch.object(runner, "_ssh", return_value=0):
                result = runner.execute(JobSpec(
                    repo="https://example/repo", ref="def", command="pytest",
                    max_minutes=10, reuse_pod_ids=("pod-1", "pod-2"),
                ))

            self.assertTrue(result.ok)
            self.assertEqual(("absent-retired", "used"),
                             result.reuse_candidate_dispositions)
            self.assertNotIn("pod-1", str(result.requested))
            self.assertEqual(["pod-2"], api.started)
            self.assertEqual([], runner.leases.all())

    def test_absent_only_without_fresh_opt_in_returns_sanitized_receipt(self):
        with tempfile.TemporaryDirectory() as root:
            runner = self.runner(root)
            runner.leases.put("pod-1", {"pod_id": "pod-1", "state": "paused-for-retest"})
            spec = JobSpec(
                repo="https://example/repo", ref="abc", command="pytest",
                reuse_pod_ids=("pod-1",),
            )
            with patch.object(runner, "_execute",
                              side_effect=RunpodPodNotFound("not found", 404)) as execute, \
                 self.assertRaises(RunpodPodNotFound) as caught:
                runner.execute(spec)

            execute.assert_called_once()
            receipt = caught.exception.job_result
            self.assertEqual("", receipt.pod_id)
            self.assertTrue(receipt.terminated)
            self.assertFalse(receipt.ok)
            self.assertEqual("retained_candidate_absent", receipt.failure_stage)
            self.assertEqual(("absent-retired",), receipt.reuse_candidate_dispositions)
            self.assertEqual(spec.receipt, receipt.requested)
            self.assertEqual([], runner.leases.all())

    def test_absent_only_can_use_opt_in_fresh_fallback(self):
        with tempfile.TemporaryDirectory() as root:
            runner = self.runner(root)
            runner.leases.put("pod-1", {"pod_id": "pod-1", "state": "paused-for-retest"})
            fresh = JobResult(
                pod_id="fresh", returncode=0, timed_out=False, artifacts_ok=True,
                terminated=True, elapsed_seconds=1,
            )
            with patch.object(runner, "_execute", side_effect=[
                    RunpodPodNotFound("not found", 404), fresh]) as execute, \
                 patch("runpod_guard.runner.time.monotonic", return_value=100):
                result = runner.execute(JobSpec(
                    repo="https://example/repo", ref="abc", command="pytest",
                    max_minutes=10, reuse_pod_ids=("pod-1",),
                    fallback_fresh_on_reuse_unavailable=True,
                ))

            self.assertEqual(2, execute.call_count)
            self.assertTrue(result.ok)
            self.assertTrue(result.fresh_fallback_used)
            self.assertFalse(result.retained_pod_preserved_at_fallback)
            self.assertEqual(("absent-retired",), result.reuse_candidate_dispositions)

    def test_fresh_flag_without_candidates_is_an_ordinary_fresh_run(self):
        with tempfile.TemporaryDirectory() as root:
            runner = self.runner(root)
            with patch.object(runner, "_wait_for_address", return_value=("127.0.0.1", 22)), \
                 patch.object(runner, "_wait_for_ssh"), patch.object(runner, "_ssh", return_value=0):
                result = runner.execute(JobSpec(
                    repo="https://example/repo", ref="abc", command="pytest",
                    fallback_fresh_on_reuse_unavailable=True,
                ))
            self.assertTrue(result.ok)
            self.assertFalse(result.fresh_fallback_used)
            self.assertEqual((), result.reuse_candidate_dispositions)

    def test_reaped_local_lease_can_fall_back_after_fresh_absence_lookup(self):
        with tempfile.TemporaryDirectory() as root:
            api = FakeAPI()
            original_get = api.get_pod

            def get_pod(pod_id):
                if pod_id == "stale-candidate":
                    raise RunpodPodNotFound("Runpod Pod was not found", 404)
                return original_get(pod_id)

            api.get_pod = get_pod
            runner = self.runner(root, api)
            with patch.object(runner, "_wait_for_address", return_value=("127.0.0.1", 22)), \
                 patch.object(runner, "_wait_for_ssh"), patch.object(runner, "_ssh", return_value=0):
                result = runner.execute(JobSpec(
                    repo="https://example/repo", ref="abc", command="pytest",
                    max_minutes=10, reuse_pod_ids=("stale-candidate",),
                    fallback_fresh_on_reuse_unavailable=True,
                ))

            self.assertTrue(result.ok)
            self.assertTrue(result.fresh_fallback_used)
            self.assertFalse(result.retained_pod_preserved_at_fallback)
            self.assertEqual(("absent-retired",), result.reuse_candidate_dispositions)
            self.assertIsNotNone(api.body)

    def test_missing_local_lease_does_not_continue_when_provider_has_candidate(self):
        with tempfile.TemporaryDirectory() as root:
            api = FakeAPI()
            runner = self.runner(root, api)
            with self.assertRaises(RetainedPodLeaseMissing):
                runner.execute(JobSpec(
                    repo="https://example/repo", ref="abc", command="pytest",
                    reuse_pod_ids=("still-present", "next-candidate"),
                    fallback_fresh_on_reuse_unavailable=True,
                ))
            self.assertIsNone(api.body)
            self.assertEqual([], api.started)

    def test_ambiguous_404_or_transient_error_does_not_advance_candidate(self):
        for error in (RunpodAPIError("start endpoint returned 404", 404),
                      RunpodAPIError("temporary lookup failure", 500)):
            with self.subTest(status=error.status_code), tempfile.TemporaryDirectory() as root:
                runner = self.runner(root)
                with patch.object(runner, "_execute", side_effect=error) as execute, \
                     self.assertRaises(RunpodAPIError):
                    runner.execute(JobSpec(
                        repo="https://example/repo", ref="abc", command="pytest",
                        reuse_pod_ids=("pod-1", "pod-2"),
                        fallback_fresh_on_reuse_unavailable=True,
                    ))
                self.assertEqual(1, execute.call_count)

    def test_absent_candidate_lease_retirement_failure_aborts(self):
        with tempfile.TemporaryDirectory() as root:
            runner = self.runner(root)
            with patch.object(runner, "_execute",
                              side_effect=RunpodPodNotFound("not found", 404)) as execute, \
                 patch.object(runner.leases, "remove", side_effect=OSError("disk failure")), \
                 self.assertRaisesRegex(OSError, "disk failure"):
                runner.execute(JobSpec(
                    repo="https://example/repo", ref="abc", command="pytest",
                    reuse_pod_ids=("pod-1", "pod-2"),
                    fallback_fresh_on_reuse_unavailable=True,
                ))
            self.assertEqual(1, execute.call_count)

    def test_ordered_reuse_all_unavailable_needs_fresh_opt_in(self):
        with tempfile.TemporaryDirectory() as root:
            runner = self.runner(root)
            for pod_id in ("pod-1", "pod-2"):
                runner.leases.put(pod_id, {
                    "pod_id": pod_id, "state": "paused-for-retest",
                })
            spec = JobSpec(
                repo="https://example/repo", ref="abc", command="pytest",
                reuse_pod_ids=("pod-1", "pod-2"),
            )

            def unavailable(candidate, monotonic_deadline=None):
                self.assertIsNotNone(monotonic_deadline)
                raise RetainedPodUnavailable("busy", 500)

            with patch.object(runner, "_execute", side_effect=unavailable) as execute, \
                 self.assertRaises(RetainedPodUnavailable):
                runner.execute(spec)
            self.assertEqual(2, execute.call_count)

    def test_ordered_reuse_falls_back_fresh_only_after_every_candidate(self):
        with tempfile.TemporaryDirectory() as root:
            runner = self.runner(root)
            for pod_id in ("pod-1", "pod-2"):
                runner.leases.put(pod_id, {
                    "pod_id": pod_id, "state": "paused-for-retest",
                })
            calls = []

            def execute(candidate, monotonic_deadline=None):
                calls.append((candidate.reuse_pod_id, monotonic_deadline))
                if candidate.reuse_pod_id:
                    raise RetainedPodUnavailable("busy", 500)
                return JobResult(
                    pod_id="fresh-result", returncode=0, timed_out=False,
                    artifacts_ok=True, terminated=True, elapsed_seconds=1,
                    requested=candidate.receipt,
                )

            with patch("runpod_guard.runner.time.monotonic", return_value=100), \
                 patch.object(runner, "_execute", side_effect=execute):
                result = runner.execute(JobSpec(
                    repo="https://example/repo", ref="abc", command="pytest",
                    max_minutes=10, reuse_pod_ids=("pod-1", "pod-2"),
                    fallback_fresh_on_reuse_unavailable=True,
                ))

            self.assertEqual([("pod-1", 700), ("pod-2", 700), (None, 700)], calls)
            self.assertTrue(result.fresh_fallback_used)
            self.assertEqual(10, result.fresh_fallback_max_minutes)
            self.assertEqual(
                ("unavailable-preserved", "unavailable-preserved"),
                result.reuse_candidate_dispositions,
            )
            self.assertEqual(2, result.requested["reuse_candidate_count"])

    def test_ordered_reuse_aborts_on_permanent_error_without_trying_next(self):
        with tempfile.TemporaryDirectory() as root:
            runner = self.runner(root)
            spec = JobSpec(
                repo="https://example/repo", ref="abc", command="pytest",
                reuse_pod_ids=("pod-1", "pod-2"),
                fallback_fresh_on_reuse_unavailable=True,
            )
            with patch.object(runner, "_execute", side_effect=RunpodAPIError("forbidden", 403)) as execute, \
                 self.assertRaises(RunpodAPIError):
                runner.execute(spec)
            self.assertEqual(1, execute.call_count)

    def test_ordered_reuse_shares_one_deadline_and_aborts_timeout(self):
        with tempfile.TemporaryDirectory() as root:
            runner = self.runner(root)
            for pod_id in ("pod-1", "pod-2"):
                runner.leases.put(pod_id, {
                    "pod_id": pod_id, "state": "paused-for-retest",
                })
            deadlines = []

            def execute(_candidate, monotonic_deadline=None):
                deadlines.append(monotonic_deadline)
                if len(deadlines) == 1:
                    raise RetainedPodUnavailable("busy", 500)
                raise TimeoutError("shared deadline reached")

            with patch("runpod_guard.runner.time.monotonic", return_value=100), \
                 patch.object(runner, "_execute", side_effect=execute), \
                 self.assertRaisesRegex(TimeoutError, "shared deadline"):
                runner.execute(JobSpec(
                    repo="https://example/repo", ref="abc", command="pytest",
                    max_minutes=10, reuse_pod_ids=("pod-1", "pod-2"),
                    fallback_fresh_on_reuse_unavailable=True,
                ))
            self.assertEqual([700, 700], deadlines)

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

    def test_upload_failure_receipt_follows_teardown_without_starting_job(self):
        for upload_error in (None, RuntimeError("private transport detail"),
                             subprocess.TimeoutExpired("secret command", 120)):
            for deletion in (True, False, RunpodAPIError("private API detail")):
                with self.subTest(upload_error=upload_error, deletion=deletion), \
                     tempfile.TemporaryDirectory() as root:
                    runner = self.runner(root)
                    archive = Path(root) / "source.tar"
                    archive.touch()
                    (Path(root) / ".git").mkdir()
                    spec = JobSpec(
                        repo=None, source_dir=Path(root), ref="abc", command="secret command",
                        env={"TOKEN": "secret value"}, retest_window_minutes=15,
                    )
                    original_handler = signal.getsignal(signal.SIGTERM)
                    with patch.object(runner, "_wait_for_address", return_value=("127.0.0.1", 22)), \
                         patch.object(runner, "_wait_for_ssh"), \
                         patch.object(runner, "_ssh", return_value=0) as ssh, \
                         patch.object(runner, "_source_archive", return_value=archive), \
                         patch.object(runner, "_upload_source", return_value=False,
                                      side_effect=upload_error) as upload, \
                         patch.object(runner, "_fetch") as fetch, \
                         patch.object(runner.api, "delete_and_confirm", return_value=deletion,
                                      side_effect=deletion if isinstance(deletion, Exception) else None) as delete, \
                         self.assertRaises(type(upload_error) if upload_error else RuntimeError) as caught:
                        runner.execute(spec)
                    result = caught.exception.job_result
                    self.assertEqual(result.pod_id, "pod-1")
                    self.assertEqual(result.failure_stage, "source_upload")
                    self.assertFalse(result.job_started)
                    self.assertIsNone(result.returncode)
                    self.assertFalse(result.ok)
                    self.assertFalse(result.paused)
                    self.assertEqual(result.terminated, deletion is True)
                    self.assertEqual(result.timed_out, isinstance(upload_error, subprocess.TimeoutExpired))
                    self.assertEqual(result.cost_per_hour, 0.4)
                    self.assertGreaterEqual(result.elapsed_seconds, 0)
                    self.assertEqual(result.requested, spec.receipt)
                    self.assertNotIn("secret", json.dumps(result.to_dict()))
                    self.assertNotIn("private", json.dumps(result.to_dict()))
                    self.assertEqual(ssh.call_count, 2)  # Watchdog and bootstrap only.
                    upload.assert_called_once()
                    fetch.assert_not_called()
                    delete.assert_called_once_with("pod-1")
                    self.assertEqual(bool(runner.leases.all()), deletion is not True)
                    self.assertFalse(archive.exists())
                    self.assertIs(signal.getsignal(signal.SIGTERM), original_handler)
                    if upload_error is not None:
                        self.assertIs(caught.exception, upload_error)

    def test_job_exception_receipt_keeps_stage_after_artifact_fetch(self):
        with tempfile.TemporaryDirectory() as root:
            runner = self.runner(root)
            with patch.object(runner, "_wait_for_address", return_value=("127.0.0.1", 22)), \
                 patch.object(runner, "_wait_for_ssh"), \
                 patch.object(runner, "_ssh", side_effect=[0, 0, RuntimeError("transport")]), \
                 patch.object(runner, "_fetch", return_value=True), \
                 self.assertRaises(RuntimeError) as caught:
                runner.execute(JobSpec(
                    repo="https://example/repo", ref="abc", command="true",
                    artifacts=(Artifact("output"),),
                ))
            result = caught.exception.job_result
            self.assertTrue(result.job_started)
            self.assertEqual(result.failure_stage, "job")
            self.assertTrue(result.terminated)

    def test_failed_fresh_fallback_receipt_keeps_request_and_dispositions(self):
        with tempfile.TemporaryDirectory() as root:
            runner = self.runner(root)
            runner.leases.put("retained", {"pod_id": "retained", "state": "paused-for-retest"})
            spec = JobSpec(
                repo="https://example/repo", ref="abc", command="true", max_minutes=10,
                reuse_pod_id="retained", fallback_fresh_on_reuse_unavailable=True,
            )
            error = RuntimeError("upload failed")
            error.job_result = JobResult(
                pod_id="fresh", returncode=None, timed_out=False, artifacts_ok=True,
                terminated=True, elapsed_seconds=1, failure_stage="source_upload",
            )
            with patch.object(runner, "_execute", side_effect=[RetainedPodUnavailable("busy"), error]), \
                 patch("runpod_guard.runner.time.monotonic", return_value=100), \
                 self.assertRaises(RuntimeError) as caught:
                runner.execute(spec)
            result = caught.exception.job_result
            self.assertEqual(result.requested, spec.receipt)
            self.assertTrue(result.fresh_fallback_used)
            self.assertEqual(result.fresh_fallback_max_minutes, 10)
            self.assertTrue(result.retained_pod_preserved_at_fallback)
            self.assertEqual(result.reuse_candidate_dispositions, ("unavailable-preserved",))

    def test_artifact_exception_cannot_report_success_after_zero_job_exit(self):
        with tempfile.TemporaryDirectory() as root:
            runner = self.runner(root)
            with patch.object(runner, "_wait_for_address", return_value=("127.0.0.1", 22)), \
                 patch.object(runner, "_wait_for_ssh"), patch.object(runner, "_ssh", return_value=0), \
                 patch.object(runner, "_fetch", side_effect=OSError("disk full")), \
                 self.assertRaises(OSError) as caught:
                runner.execute(JobSpec(
                    repo="https://example/repo", ref="abc", command="true",
                    artifacts=(Artifact("output"),),
                ))
            result = caught.exception.job_result
            self.assertEqual(result.returncode, 0)
            self.assertTrue(result.terminated)
            self.assertEqual(result.failure_stage, "artifact_fetch")
            self.assertFalse(result.ok)

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
            with self.assertRaises(OSError) as caught:
                runner.execute(JobSpec(repo="https://example/repo", ref="y", command="z"))
            self.assertEqual(api.deleted, ["pod-1"])
            self.assertEqual(caught.exception.job_result.failure_stage, "lease")
            self.assertTrue(caught.exception.job_result.terminated)
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
