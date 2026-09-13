from pathlib import Path
import unittest
from unittest.mock import patch

from runpod_guard.cli import parser


class CLITests(unittest.TestCase):
    def test_default_env_file_is_user_wide(self):
        with patch.object(Path, "home", return_value=Path("/users/test")):
            args = parser().parse_args(["list"])
        self.assertEqual(args.env_file, Path("/users/test/.config/runpod-guard/env"))


if __name__ == "__main__":
    unittest.main()
