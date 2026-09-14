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
        ])
        self.assertEqual(args.retest_window_minutes, 15)
        self.assertEqual(args.reuse_pod, "pod-1")
        self.assertEqual(args.reuse_start_attempts, 4)
        self.assertEqual(args.reuse_start_delay_seconds, 20)


if __name__ == "__main__":
    unittest.main()
