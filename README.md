# runpod-guard

Run one bounded job on a disposable Runpod GPU, retrieve its outputs, and prove the
Pod was deleted. It is deliberately project-independent: a job is a Git repository,
an exact revision, setup and run commands, a hardware profile, and optional outputs.

It is intended for CI jobs, model tests, short training runs and ad-hoc GPU work—not
for an always-on inference service.

## Safety model

No single cleanup mechanism is trusted:

1. The caller deletes the Pod in `finally` on success, failure, timeout, Ctrl-C and
   SIGTERM, then lists Pods to confirm the ID is absent.
2. A detached watchdog on the Pod deletes its own Pod five minutes after the requested
   deadline and keeps retrying until deletion succeeds.
   It survives a dead SSH connection or a dead caller host.
3. A pending lease is written before creation. The systemd timer
   reaps expired leases every five minutes after a caller crash or reboot.
4. `max_minutes` is the caller-side target for provisioning, installation, execution
   and artifact retrieval. Local deletion starts by that deadline. In the host-loss
   case, the Pod watchdog deliberately has a five-minute grace before it takes over.
5. `max_cost_per_hour` rejects a machine whose reported rate is over budget and
   immediately enters verified teardown.
6. Jobs use no Pod volume or network volume. There is no storage left charging after
   deletion, and stopped Pods are never used.
7. The account API key stays on the caller. It is never placed in Pod environment
   variables. The on-Pod watchdog uses Runpod's injected Pod-scoped identity.

`kill -9`, sudden power loss and network partitions mean no client can make cleanup
literally infallible. Hardened operation requires the scheduled reaper; the Pod
watchdog is an additional fallback once SSH has been reached.
Keep auto-pay disabled during initial use and enable Runpod's low-balance/stale-Pod
notifications as the final account-level guard.

## Install

Python 3.11+, `git`, OpenSSH and SCP are required on the caller:

```bash
python3 -m venv .venv
.venv/bin/pip install -e .
mkdir -p ~/.local/bin
ln -sf "$PWD/.venv/bin/runpod-guard" ~/.local/bin/runpod-guard
```

On minimal hosts without `python3-venv`, a wrapper that sets `PYTHONPATH` is also
enough because the package has no runtime dependencies.

Set the API key in the process environment or an ignored `.env` in the directory from
which the CLI runs:

```bash
RUNPOD_API_KEY='...'
```

The default SSH identity is `~/.ssh/runpod_engram`. Register its `.pub` file in
Runpod account settings, then run the read-only check:

```bash
runpod-guard preflight
```

## CLI

An infrastructure-only smoke test against this public repository:

```bash
runpod-guard run \
  --repo https://github.com/AmzySingh/runpod-guard.git \
  --ref COMMIT_SHA \
  --profile small \
  --cloud COMMUNITY \
  --max-minutes 15 \
  --max-cost-per-hour 0.60 \
  --command 'python3 -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name())"' \
  --name smoke
```

A project job with an artifact:

```bash
runpod-guard run \
  --repo https://github.com/ORG/PROJECT.git \
  --ref 0123456789abcdef0123456789abcdef01234567 \
  --profile fast \
  --max-minutes 45 \
  --max-cost-per-hour 0.90 \
  --setup 'curl -LsSf https://astral.sh/uv/install.sh | sh && export PATH="$HOME/.local/bin:$PATH" && uv sync --extra gpu' \
  --command 'uv run --no-sync python -m project.model_test --out artifacts/result.json' \
  --fetch artifacts/result.json=runpod-output \
  --name model-test
```

For a private repository, upload a Git archive from an existing local checkout. This
does not forward GitHub credentials and includes only files tracked at the requested
revision, so ignored files such as `.env` are not copied:

```bash
runpod-guard run \
  --source /path/to/private-project \
  --ref COMMIT_SHA \
  --profile small \
  --max-minutes 45 \
  --command 'python -m project.model_test'
```

Profiles are ordered availability fallbacks:

| Profile | Intended size | Default GPU choices |
|---|---|---|
| `small` | 24 GB, economical tests | L4, A5000, 3090 |
| `fast` | 24–32 GB, high throughput | 4090, 5090 |
| `large` | 48 GB | A6000, A40, L40S |
| `xlarge` | 80 GB | A100 PCIe, A100 SXM |

Use repeated `--gpu` flags to replace a profile with exact Runpod GPU IDs.

Operational commands:

```bash
runpod-guard list
runpod-guard reap                 # expired local leases only
runpod-guard reap --all-managed   # emergency: every rpg-* Pod
```

The emergency command deliberately cannot touch Pods without the `rpg-` name prefix.
The normal reaper scopes leases to the API-key identity that created them and reports
malformed state or unconfirmed deletion with a nonzero exit.

## Python API

```python
from pathlib import Path
from runpod_guard import Artifact, JobSpec, RunpodRunner

result = RunpodRunner().execute(JobSpec(
    repo="https://github.com/ORG/PROJECT.git",
    ref="0123456789abcdef0123456789abcdef01234567",
    setup="pip install -e .",
    command="python -m project.gpu_test --out artifacts/result.json",
    profile="small",
    max_minutes=30,
    max_cost_per_hour=0.60,
    artifacts=(Artifact("artifacts/result.json", Path("runpod-output")),),
    name="project-test",
))
if not result.ok:
    raise RuntimeError(result.to_dict())
```

## Scheduled reaper

Install after the package itself. Store only the Runpod API key in
`~/.config/runpod-guard/env`, mode `0600`.

```bash
mkdir -p ~/.config/systemd/user ~/.config/runpod-guard
printf 'RUNPOD_API_KEY=%s\n' "$RUNPOD_API_KEY" > ~/.config/runpod-guard/env
cp systemd/runpod-guard-reaper.* ~/.config/systemd/user/
chmod 600 ~/.config/runpod-guard/env
systemctl --user daemon-reload
systemctl --user enable --now runpod-guard-reaper.timer
loginctl enable-linger "$USER"
systemctl --user list-timers runpod-guard-reaper.timer
```

The timer is defense in depth. The normal caller and the on-Pod watchdog remain the
primary teardown paths.

## Current limitations

- Remote Git cloning assumes a public repository. Private repositories can use
  `--source`; credentials and untracked or ignored files are never uploaded.
- Repository setup and job commands run as root and are intentionally arbitrary.
  Treat the selected repository and exact revision as trusted code.
- Artifacts are copied with SCP before deletion and each transfer has a two-minute
  timeout. For large outputs, use object storage in the job itself.
- The default official Runpod PyTorch image is configurable with `--image`.
- A hostile root workload can kill an in-container watchdog. The independently
  scheduled reaper is required for that threat model.
- Publishing this directory to GitHub is a separate external action; create the
  destination repository first, then add it as `origin` and push.
