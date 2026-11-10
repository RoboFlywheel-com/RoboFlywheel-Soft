# SPDX-License-Identifier: Apache-2.0
"""Scheduler adapters: local execution and Slurm array command rendering."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SlurmArraySpec:
    run_dir: str
    shard_count: int
    partition: str = "gpu"
    time: str = "02:00:00"
    gpus: int = 1
    cpus_per_task: int = 4
    memory: str = "16G"
    episode_executor: str | None = None
    target_provider: str | None = None
    process_isolation: bool = True

    def command(self) -> str:
        if self.shard_count <= 0:
            raise ValueError("shard_count must be positive")
        last = self.shard_count - 1
        command = (
            f"sbatch --array=0-{last} --partition={self.partition} --time={self.time} "
            f"--gpus={self.gpus} --cpus-per-task={self.cpus_per_task} --mem={self.memory} "
            f"--wrap='softsim-exp run --run {self.run_dir} --shard-index $SLURM_ARRAY_TASK_ID"
        )
        if self.episode_executor:
            command += f" --episode-executor {self.episode_executor}"
        if self.target_provider:
            command += f" --target-provider {self.target_provider}"
        if self.process_isolation:
            command += " --process-isolation"
        return command + "'"


def local_command(run_dir: str, shard_index: int = 0, workers: int = 1, *, episode_executor: str | None = None, target_provider: str | None = None, process_isolation: bool = True) -> list[str]:
    command = ["softsim-exp", "run", "--run", str(run_dir), "--shard-index", str(int(shard_index)), "--workers", str(int(workers))]
    if episode_executor:
        command.extend(["--episode-executor", episode_executor])
    if target_provider:
        command.extend(["--target-provider", target_provider])
    if process_isolation:
        command.append("--process-isolation")
    return command
