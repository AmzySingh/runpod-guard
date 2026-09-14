from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .models import Artifact, GPU_PROFILES, JobSpec
from .runner import RunpodRunner


def load_dotenv(path: Path) -> None:
    """Load simple KEY=VALUE secrets without overriding the real environment."""
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if value[:1] == value[-1:] and value[:1] in {"'", '"'}:
            value = value[1:-1]
        if key.isidentifier():
            os.environ.setdefault(key, value)


def artifact(value: str) -> Artifact:
    remote, separator, local = value.partition("=")
    try:
        return Artifact(remote, Path(local) if separator else Path("runpod-output"))
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(prog="runpod-guard")
    command.add_argument(
        "--env-file", type=Path,
        default=Path.home() / ".config" / "runpod-guard" / "env",
        help="API-key env file (default: ~/.config/runpod-guard/env)",
    )
    command.add_argument("--ssh-key", type=Path,
                         default=Path.home() / ".ssh" / "runpod_engram")
    subcommands = command.add_subparsers(dest="action", required=True)

    subcommands.add_parser("preflight", help="read-only API and local SSH-key check")
    subcommands.add_parser("list", help="list current Pods")
    reap = subcommands.add_parser("reap", help="delete expired leased Pods")
    reap.add_argument("--all-managed", action="store_true",
                      help="delete every Pod whose name starts rpg-, even before its deadline")

    extend = subcommands.add_parser(
        "extend", help="extend the lease of a guard-owned stopped Pod"
    )
    extend.add_argument("pod_id", help="ID of the retained Pod")
    extend.add_argument("--minutes", type=int, required=True,
                        help="new retention window from now (1-1440 minutes)")

    run = subcommands.add_parser("run", help="run one bounded job")
    source = run.add_mutually_exclusive_group(required=True)
    source.add_argument("--repo", help="public HTTPS Git repository")
    source.add_argument("--source", type=Path,
                        help="local Git tree; uploads only files tracked at --ref")
    run.add_argument("--ref", required=True, help="commit SHA, tag, or branch")
    run.add_argument("--command", required=True, help="shell command inside the checkout")
    run.add_argument("--setup", default="", help="shell setup command before the job")
    run.add_argument("--profile", choices=GPU_PROFILES, default="small")
    run.add_argument("--gpu", action="append", default=[], help="exact GPU id; repeat for fallbacks")
    run.add_argument("--cloud", choices=["COMMUNITY", "SECURE"], default="SECURE")
    run.add_argument("--max-minutes", type=int, default=60)
    run.add_argument("--max-cost-per-hour", type=float, default=1.0,
                     help="post-allocation hourly-rate ceiling (default: 1.00)")
    run.add_argument("--image", default="runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404")
    run.add_argument("--disk-gb", type=int, default=40)
    run.add_argument("--workspace-gb", type=int, default=20,
                     help="persistent workspace size when retaining for a retest (default: 20)")
    run.add_argument("--retest-window-minutes", type=int, default=0,
                     help="stop instead of delete after a completed job; reaper deletes at expiry")
    run.add_argument("--reuse-pod", help="restart a Pod retained by an earlier guarded run")
    run.add_argument("--reuse-start-attempts", type=int, default=4,
                     help="start attempts for a retained Pod (default: 4)")
    run.add_argument("--reuse-start-delay-seconds", type=int, default=20,
                     help="wait between retained-Pod start attempts (default: 20)")
    run.add_argument("--fallback-fresh-on-reuse-unavailable", action="store_true",
                     help="allocate a fresh Pod after all retryable retained-Pod start attempts fail")
    run.add_argument("--fetch", action="append", type=artifact, default=[])
    run.add_argument("--name", default="job")
    return command


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    load_dotenv(args.env_file)
    runner = RunpodRunner(ssh_key=args.ssh_key)
    if args.action == "preflight":
        count = runner.preflight()
        print(f"preflight passed; {count} Pod(s) currently listed")
        return 0
    if args.action == "list":
        pods = runner.api.list_pods()
        print(json.dumps([{key: pod.get(key) for key in
                           ("id", "name", "desiredStatus", "costPerHr", "machineId")}
                          for pod in pods], indent=2))
        return 0
    if args.action == "reap":
        removed = runner.reap(args.all_managed)
        print(json.dumps({"removed": removed, "failures": runner.reap_failures}))
        return 1 if runner.reap_failures else 0
    if args.action == "extend":
        expires_at = runner.extend_retest(args.pod_id, args.minutes)
        print(json.dumps({"pod_id": args.pod_id, "retest_expires_at": expires_at}, indent=2))
        return 0
    spec = JobSpec(
        repo=args.repo, source_dir=args.source, ref=args.ref,
        command=args.command, setup=args.setup,
        profile=args.profile, gpu_types=tuple(args.gpu), cloud=args.cloud,
        max_minutes=args.max_minutes, max_cost_per_hour=args.max_cost_per_hour,
        image=args.image, container_disk_gb=args.disk_gb,
        workspace_gb=args.workspace_gb,
        retest_window_minutes=args.retest_window_minutes,
        reuse_pod_id=args.reuse_pod,
        reuse_start_attempts=args.reuse_start_attempts,
        reuse_start_delay_seconds=args.reuse_start_delay_seconds,
        fallback_fresh_on_reuse_unavailable=args.fallback_fresh_on_reuse_unavailable,
        artifacts=tuple(args.fetch), name=args.name,
    )
    result = runner.execute(spec)
    print(json.dumps(result.to_dict(), indent=2))
    return 0 if result.ok else 1


if __name__ == "__main__":
    sys.exit(main())
