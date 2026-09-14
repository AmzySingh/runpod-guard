from pathlib import Path
import unittest
from unittest.mock import patch

from runpod_guard.cli import parser


class CLITests(unittest.TestCase):
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
