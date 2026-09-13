from pathlib import Path

from runpod_guard import Artifact, JobSpec, RunpodRunner


runner = RunpodRunner()
result = runner.execute(JobSpec(
    repo="https://github.com/your-org/your-project.git",
    ref="0123456789abcdef0123456789abcdef01234567",
    setup="pip install -e .",
    command="python -m your_project.gpu_test --output artifacts/result.json",
    profile="small",
    max_minutes=30,
    max_cost_per_hour=0.60,
    artifacts=(Artifact("artifacts/result.json", Path("runpod-output")),),
    name="gpu-test",
))
if not result.ok:
    raise SystemExit(result.to_dict())
