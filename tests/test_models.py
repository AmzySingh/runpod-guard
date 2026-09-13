from pathlib import Path
import unittest

from runpod_guard.models import Artifact, JobSpec


class ModelTests(unittest.TestCase):
    def test_profile_and_artifact(self):
        spec = JobSpec(repo="https://example/repo", ref="abc", command="true")
        self.assertGreater(len(spec.selected_gpus), 1)
        self.assertEqual(Artifact("out/result", Path("here")).remote, "out/result")

    def test_rejects_unsafe_values(self):
        with self.assertRaises(ValueError):
            Artifact("../outside")
        with self.assertRaises(ValueError):
            Artifact("out/result;touch-pwned")
        with self.assertRaises(ValueError):
            JobSpec(repo="https://example/repo", ref="y", command="z", max_minutes=0)
        with self.assertRaises(ValueError):
            JobSpec(repo="https://example/repo", ref="y", command="z",
                    retest_window_minutes=1441)
        with self.assertRaises(ValueError):
            JobSpec(repo="https://example/repo", ref="y", command="z", reuse_pod_id="bad/id")
        with self.assertRaises(ValueError):
            JobSpec(repo="https://example/repo", ref="y", command="z", max_cost_per_hour=float("nan"))
        with self.assertRaises(ValueError):
            JobSpec(repo="https://example/repo", ref="y", command="z", env={"RUNPOD_API_KEY": "secret"})
        with self.assertRaises(ValueError):
            JobSpec(repo="file:///tmp/repo", ref="y", command="z")
        with self.assertRaises(ValueError):
            JobSpec(repo="https://example/repo", ref="--upload-pack=bad", command="z")
        with self.assertRaises(ValueError):
            JobSpec(repo="https://example/repo", ref="y", command="z",
                    env={"RUNPOD_API_URL": "https://attacker.example"})


if __name__ == "__main__":
    unittest.main()
