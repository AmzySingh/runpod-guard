from pathlib import Path
from contextlib import redirect_stdout
import io
import json
import unittest
from unittest.mock import patch

from runpod_guard.cli import main, parser
from runpod_guard.models import JobResult


class CLITests(unittest.TestCase):
    def test_failure_receipt_is_json_with_nonzero_exit_and_no_exception_details(self):
        for error, code in ((RuntimeError("secret transport detail"), 1),
                            (KeyboardInterrupt("secret interruption detail"), 130)):
            with self.subTest(error=type(error).__name__):
                error.job_result = JobResult(
                    pod_id="pod-1", returncode=None, timed_out=False, artifacts_ok=True,
                    terminated=True, elapsed_seconds=12.5, cost_per_hour=0.4,
                    failure_stage="source_upload",
                )
                output = io.StringIO()
                with patch("runpod_guard.cli.load_dotenv"), \
                     patch("runpod_guard.cli.RunpodRunner") as runner, redirect_stdout(output):
                    runner.return_value.execute.side_effect = error
                    status = main([
                        "run", "--repo", "https://example/repo", "--ref", "abc",
                        "--command", "secret command",
                    ])
                self.assertEqual(status, code)
                receipt = json.loads(output.getvalue())
                self.assertEqual(receipt, error.job_result.to_dict())
                self.assertFalse(receipt["ok"])
                self.assertNotIn("secret", output.getvalue())

    def test_exception_before_allocation_is_not_given_a_receipt(self):
        output = io.StringIO()
        with patch("runpod_guard.cli.load_dotenv"), \
             patch("runpod_guard.cli.RunpodRunner") as runner, redirect_stdout(output):
            runner.return_value.execute.side_effect = RuntimeError("preflight failed")
            with self.assertRaisesRegex(RuntimeError, "preflight failed"):
                main(["run", "--repo", "https://example/repo", "--ref", "abc", "--command", "true"])
        self.assertEqual(output.getvalue(), "")

    def test_default_env_file_is_user_wide(self):
        with patch.object(Path, "home", return_value=Path("/users/test")):
            args = parser().parse_args(["list"])
        self.assertEqual(args.env_file, Path("/users/test/.config/runpod-guard/env"))

    def test_retest_options(self):
        args = parser().parse_args([
            "run", "--repo", "https://example/repo", "--ref", "abc",
            "--command", "pytest", "--retest-window-minutes", "15",
            "--reuse-pod", "pod-1",
            "--reuse-pod", "pod-2",
            "--fallback-fresh-on-reuse-unavailable",
        ])
        self.assertEqual(args.retest_window_minutes, 15)
        self.assertEqual(args.reuse_pod, ["pod-1", "pod-2"])
        self.assertEqual(args.reuse_start_attempts, 4)
        self.assertEqual(args.reuse_start_delay_seconds, 20)
        self.assertEqual(args.gpu_priority, "custom")
        self.assertTrue(args.fallback_fresh_on_reuse_unavailable)

    def test_gpu_priority_option(self):
        args = parser().parse_args([
            "run", "--repo", "https://example/repo", "--ref", "abc",
            "--command", "true", "--gpu-priority", "availability",
        ])
        self.assertEqual(args.gpu_priority, "availability")

    def test_extend_options(self):
        args = parser().parse_args(["extend", "pod-1", "--minutes", "1440"])
        self.assertEqual(args.action, "extend")
        self.assertEqual(args.pod_id, "pod-1")
        self.assertEqual(args.minutes, 1440)


if __name__ == "__main__":
    unittest.main()
