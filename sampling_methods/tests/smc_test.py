from _path_helper import ensure_project_root_on_path

ensure_project_root_on_path()

from sampling_methods.smc_sampler import SMC
import logging
import os
import time
from benchmark.haiku_reward_model import HaikuRewardModel



def main():
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "smc_test_logs.out")
    logger = logging.getLogger("smc_test_logger")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.FileHandler(log_path)
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    reward_model = HaikuRewardModel()
    
    # TODO: don't resample after a fixed steps. Look at effective sample size.
    sampler = SMC(reward_fn=reward_model.reward_fn, detail_steps=128)
    prompt = "You are a poet. Write me a haiku."

    start_generate = time.time()
    samples = sampler.generate(prompt, n=128, max_gpu_n=128)
    end_generate = time.time()
    logger.info(f"Time taken for sampler.generate: {end_generate - start_generate:.2f} seconds")
    for sample in samples:
        logger.info("Sample: %s", sample)

    rewards = reward_model.reward_fn(samples)

    average_reward = sum(rewards) / len(rewards) if rewards else 0
    logger.info(f"Average FINAL reward: {average_reward:.4f}")

if __name__ == "__main__":
    main()
