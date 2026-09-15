"""
Regression tests for utils.py's RunDir.

Run:  python src/test_utils.py

The bug this suite exists for: run directory names are derived from
data/head/seed, not the full config, so two runs that differ only in
something like `unlearn.epochs` land in the SAME directory. RunDir.create
used to silently reuse it -- overwriting config.json/env.json (so they'd
describe the wrong run) while appending to results.jsonl/trajectory.jsonl,
mixing rows from incompatible configs into one file. That is how CE's
`finetune` row in results.jsonl ended up describing a 30-epoch run under a
label everything else assumes is 3 epochs (see notes/decisions.md,
2026-09-14). RunDir.create must now refuse to start rather than reuse.
"""

import subprocess
import sys, os
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import RunDir, git_dirty_files


def check(name, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond:
        raise AssertionError(name)


def test_rundir_create_succeeds_on_a_fresh_directory():
    print("RunDir.create: fresh (nonexistent) target directory succeeds")
    with tempfile.TemporaryDirectory() as base:
        target = Path(base) / "a_run"
        check("target does not exist yet", not target.exists())
        rd = RunDir.create(base, "a_run", {"seed": 0})
        check("run directory was created", target.is_dir())
        check("config.json was written", (target / "config.json").exists())
        check("env.json was written", (target / "env.json").exists())


def test_rundir_create_succeeds_on_an_empty_existing_directory():
    print("RunDir.create: pre-existing but empty target directory succeeds")
    with tempfile.TemporaryDirectory() as base:
        target = Path(base) / "a_run"
        target.mkdir()
        check("target exists and is empty", target.is_dir() and not any(target.iterdir()))
        RunDir.create(base, "a_run", {"seed": 0})
        check("config.json was written", (target / "config.json").exists())


def test_rundir_create_refuses_a_nonempty_directory():
    print("RunDir.create: refuses instead of silently reusing a nonempty directory")
    with tempfile.TemporaryDirectory() as base:
        RunDir.create(base, "a_run", {"seed": 0, "unlearn": {"epochs": 3}})
        try:
            RunDir.create(base, "a_run", {"seed": 0, "unlearn": {"epochs": 30}})
            check("raised FileExistsError on reuse", False)
        except FileExistsError:
            check("raised FileExistsError on reuse", True)


def test_rundir_refusal_leaves_existing_files_untouched():
    print("RunDir.create: refusing a reuse does not overwrite what's already there")
    with tempfile.TemporaryDirectory() as base:
        rd = RunDir.create(base, "a_run", {"seed": 0, "unlearn": {"epochs": 3}})
        rd.append_jsonl("results.jsonl", {"method": "finetune", "output_forget": 0.859})
        original_config = (rd.root / "config.json").read_text()
        original_results = (rd.root / "results.jsonl").read_text()

        try:
            RunDir.create(base, "a_run", {"seed": 0, "unlearn": {"epochs": 30}})
        except FileExistsError:
            pass

        check("config.json unchanged", (rd.root / "config.json").read_text() == original_config)
        check("results.jsonl unchanged",
              (rd.root / "results.jsonl").read_text() == original_results)


def test_rundir_create_refuses_even_a_single_stray_file():
    print("RunDir.create: any file at all counts as nonempty, not just its own outputs")
    with tempfile.TemporaryDirectory() as base:
        target = Path(base) / "a_run"
        target.mkdir()
        (target / "unrelated.txt").write_text("leftover from something else")
        try:
            RunDir.create(base, "a_run", {"seed": 0})
            check("raised FileExistsError", False)
        except FileExistsError:
            check("raised FileExistsError", True)


def test_git_dirty_files_distinguishes_clean_dirty_and_unverifiable():
    """Scripts gate scientific execution on this, so the three cases must stay
    distinct. The dangerous one is the third: a tree whose state git cannot
    report must come back as None (UNKNOWN), never as an empty list, or an
    unverifiable run would pass a cleanliness check."""
    print("git_dirty_files: clean vs dirty vs unverifiable")

    def git(*args, cwd):
        subprocess.run(["git", *args], cwd=cwd, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    with tempfile.TemporaryDirectory() as base:
        repo = Path(base) / "repo"
        repo.mkdir()
        git("init", "-q", cwd=repo)
        git("config", "user.email", "t@example.com", cwd=repo)
        git("config", "user.name", "t", cwd=repo)
        (repo / "a.txt").write_text("one\n")
        git("add", "a.txt", cwd=repo)
        git("commit", "-qm", "init", cwd=repo)

        check("a committed tree is clean (empty list, not None)",
              git_dirty_files(cwd=repo) == [])

        (repo / "a.txt").write_text("two\n")
        modified = git_dirty_files(cwd=repo)
        check("a modified tracked file makes it dirty",
              modified and any("a.txt" in line for line in modified))

        git("checkout", "--", "a.txt", cwd=repo)
        check("reverting makes it clean again", git_dirty_files(cwd=repo) == [])

        (repo / "untracked.txt").write_text("x\n")
        untracked = git_dirty_files(cwd=repo)
        check("an untracked file also counts as dirty",
              untracked and any("untracked.txt" in line for line in untracked))

        outside = Path(base) / "not_a_repo"
        outside.mkdir()
        result = git_dirty_files(cwd=outside)
        check("outside a repository the answer is None (unverifiable), not []",
              result is None)
        check("None is distinguishable from clean", result is not [] and result is None)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print()
    print(f"{len(tests)} test groups passed.")
