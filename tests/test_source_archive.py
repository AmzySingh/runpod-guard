from pathlib import Path
import subprocess
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from runpod_guard.models import JobSpec
from runpod_guard.runner import RunpodRunner


class SourceArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.source = Path(self.temporary.name)
        self.git("init", "-q")
        self.git("config", "user.email", "test@example.com")
        self.git("config", "user.name", "Test")
        (self.source / "runtime").mkdir()
        (self.source / "runtime/main.py").write_text("print('committed')\n")
        (self.source / "large-data.txt").write_text("unneeded\n")
        (self.source / "file with spaces.txt").write_text("spaces\n")
        (self.source / ".source-revision").write_text("$Format:%H$\n")
        (self.source / ".gitattributes").write_text("/.source-revision export-subst\n")
        self.git("add", ".")
        self.git("commit", "-qm", "fixture")
        self.commit = self.git("rev-parse", "HEAD").strip()
        (self.source / "untracked.txt").write_text("private\n")
        (self.source / "runtime/main.py").write_text("local dirty edit\n")

    def git(self, *args):
        return subprocess.run(
            ["git", "-C", str(self.source), *args], check=True,
            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        ).stdout

    def spec(self, *paths, ref="HEAD"):
        return JobSpec(repo=None, source_dir=self.source, ref=ref,
                       command="true", source_paths=paths)

    def archive_contents(self, spec):
        archive = RunpodRunner._source_archive(spec)
        try:
            self.assertEqual(archive.stat().st_mode & 0o777, 0o600)
            with tarfile.open(archive) as tar:
                return {member.name: tar.extractfile(member).read()
                        for member in tar.getmembers() if member.isfile()}
        finally:
            archive.unlink()

    def test_allow_list_uses_pinned_commit_and_substitutes_marker(self):
        spec = RunpodRunner._pin_source_selection(
            self.spec("runtime/", ".source-revision", "file with spaces.txt")
        )
        self.assertEqual(spec.ref, self.commit)
        # Move the branch after selection: upload still uses the original tree.
        self.git("add", "runtime/main.py")
        self.git("commit", "-qm", "later commit")
        contents = self.archive_contents(spec)
        self.assertEqual(set(contents), {
            "runtime/main.py", ".source-revision", "file with spaces.txt",
        })
        self.assertEqual(contents["runtime/main.py"], b"print('committed')\n")
        self.assertEqual(contents[".source-revision"], (self.commit + "\n").encode())

    def test_no_allow_list_preserves_full_archive(self):
        contents = self.archive_contents(self.spec())
        self.assertIn("large-data.txt", contents)
        self.assertNotIn("untracked.txt", contents)
        self.assertEqual(contents["runtime/main.py"], b"print('committed')\n")

    def test_rejects_unsafe_or_nonliteral_selections(self):
        for path in ("", "/tmp/file", "../outside", "runtime/../large-data.txt",
                     "--output=elsewhere", "-runtime", ".", "./runtime", "runtime/*",
                     ":(exclude)runtime", "C:/file", "runtime\\file", "runtime\nfile"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                self.spec(path)

    def test_selection_requires_local_source(self):
        with self.assertRaisesRegex(ValueError, "requires source_dir"):
            JobSpec(repo="https://example/repo", ref="HEAD", command="true",
                    source_paths=("runtime",))

    def test_missing_and_untracked_paths_fail_before_allocation(self):
        runner = object.__new__(RunpodRunner)
        for path in ("missing", "untracked.txt", "runtime/missing"):
            with self.subTest(path=path), \
                 patch.object(runner, "_execute_candidates") as allocate, \
                 self.assertRaisesRegex(ValueError, "no tracked files"):
                runner.execute(self.spec("runtime", path))
            allocate.assert_not_called()

    def test_file_added_after_requested_ref_is_refused(self):
        self.git("add", "untracked.txt")
        self.git("commit", "-qm", "track later")
        with self.assertRaisesRegex(ValueError, "no tracked files"):
            RunpodRunner._pin_source_selection(self.spec("untracked.txt", ref=self.commit))


if __name__ == "__main__":
    unittest.main()
