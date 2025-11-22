import contextlib
import io
import json
import logging
import os
import re
import sys
import time
from typing import Dict, List, Tuple

from _path_helper import ensure_project_root_on_path

ensure_project_root_on_path()

from benchmark.reward_model import RewardModel
from sampling_methods.nested_sampler import NestedSampler

TOTAL_WORK = 192
MAX_GPU_BATCH = 64
MAX_COMBINATIONS = 4
RUNS_PER_COMBINATION = 3
CUSTOM_COMBINATIONS = [
    (64, 8),
    (16, 32),
    (64, 4),
    (32, 8),
    (16, 16),
    (16, 8),
    (64, 1),
    (32, 4),
    (16, 4),
    (32, 1),
    (16, 1),
]
PROGRESS_BAR_WIDTH = 20
AVG_PATTERN = re.compile(
    r"Average reward:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)(?:\s*\|\s*Max reward:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?))?"
)
ESS_PATTERN = re.compile(r"outer ess:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")


class TeeStdout(io.StringIO):
    """Capture stdout while still streaming to the original destination."""

    def __init__(self, original):
        super().__init__()
        self.original = original

    def write(self, s):
        self.original.write(s)
        return super().write(s)

    def flush(self):
        self.original.flush()
        return super().flush()


def factor_pairs(product: int) -> List[Tuple[int, int]]:
    pairs: List[Tuple[int, int]] = []
    for divisor in range(1, product + 1):
        if product % divisor == 0:
            reward_particles = divisor
            n = product // divisor
            pairs.append((reward_particles, n))
    return pairs


def extract_metrics(buffer_value: str) -> List[Dict[str, float]]:
    metrics: List[Dict[str, float]] = []
    pending_avg = None
    pending_max = None
    for line in buffer_value.splitlines():
        avg_match = AVG_PATTERN.search(line)
        if avg_match:
            pending_avg = float(avg_match.group(1))
            if avg_match.group(2) is not None:
                pending_max = float(avg_match.group(2))
            else:
                pending_max = None
            continue

        ess_match = ESS_PATTERN.search(line)
        if ess_match and pending_avg is not None:
            ess_value = float(ess_match.group(1))
            metrics.append(
                {
                    "step": len(metrics),
                    "average_reward": pending_avg,
                    "max_reward": pending_max,
                    "outer_ess": ess_value,
                }
            )
            pending_avg = None
            pending_max = None
    return metrics


def progress_bar(completed: int, total: int, width: int = PROGRESS_BAR_WIDTH) -> str:
    if total <= 0:
        return "[unknown]"
    completed = max(0, min(completed, total))
    filled = int(width * (completed / total))
    return f"[{'#' * filled}{'.' * (width - filled)}]"


def run_configuration(
    reward_model: RewardModel,
    reward_particles: int,
    n: int,
    prompt: str,
) -> Dict:
    sampler = NestedSampler(
        resample_every_n=2,
        reward_fn=reward_model.reward_fn,
        reward_particles=reward_particles,
    )
    tee_stream = TeeStdout(sys.stdout)
    start_time = time.time()
    with contextlib.redirect_stdout(tee_stream):
        sampler.generate(prompt, n=n, max_gpu_n=MAX_GPU_BATCH)
    duration = time.time() - start_time
    metrics = extract_metrics(tee_stream.getvalue())
    return {
        "generation_seconds": duration,
        "steps": metrics,
    }


def main():
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "grid_test_logs.out")
    logger = logging.getLogger("grid_test_logger")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.FileHandler(log_path)
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    prompt = "You are a professional restaurant critic. Give me a review about for the Peruvian restaurant, el casa, in Amsterdam."
    reward_model = RewardModel()
    if CUSTOM_COMBINATIONS:
        sorted_custom = sorted(
            CUSTOM_COMBINATIONS,
            key=lambda combo: (combo[0] * combo[1], combo[0], combo[1]),
            reverse=True,
        )
        combinations = [(m, n) for n, m in sorted_custom]
    else:
        combinations = factor_pairs(TOTAL_WORK)[:MAX_COMBINATIONS]
    total_combos = len(combinations)
    results = []

    for combo_idx, (reward_particles, n) in enumerate(combinations, start=1):
        combo_runs = []
        logger.info(
            "Starting combo %d/%d (reward_particles=%d, n=%d)",
            combo_idx,
            total_combos,
            reward_particles,
            n,
        )
        for run_id in range(1, RUNS_PER_COMBINATION + 1):
            logger.info(
                "Combo %d/%d progress %s (%d/%d runs completed)",
                combo_idx,
                total_combos,
                progress_bar(run_id - 1, RUNS_PER_COMBINATION),
                run_id - 1,
                RUNS_PER_COMBINATION,
            )
            run_result = run_configuration(
                reward_model=reward_model,
                reward_particles=reward_particles,
                n=n,
                prompt=prompt,
            )
            run_result["run_id"] = run_id
            combo_runs.append(run_result)
            logger.info(
                "Finished run %d/%d for combo %d/%d in %.2fs; progress %s",
                run_id,
                RUNS_PER_COMBINATION,
                combo_idx,
                total_combos,
                run_result["generation_seconds"],
                progress_bar(run_id, RUNS_PER_COMBINATION),
            )

        results.append(
            {
                "reward_particles": reward_particles,
                "n": n,
                "max_gpu_n": MAX_GPU_BATCH,
                "runs": combo_runs,
            }
        )

    results_path = os.path.join(log_dir, "grid_test_results.json")
    with open(results_path, "w", encoding="utf-8") as fp:
        json.dump(results, fp, indent=2)
    logger.info("Stored grid search metrics at %s", results_path)


if __name__ == "__main__":
    main()
