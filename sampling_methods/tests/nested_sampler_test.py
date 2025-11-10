from sampling_methods.nested_sampler import NestedSampler
import logging
import os
import time
from benchmark.reward_model import RewardModel



def main():
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "nested_test_logs.out")
    logger = logging.getLogger("nested_test_logger")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.FileHandler(log_path)
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    reward_model = RewardModel()
    
    # TODO: don't resample after a fixed steps. Look at effective sample size.
    sampler = NestedSampler(resample_every_n=2, reward_fn=reward_model.reward_fn, reward_particles=6)
    prompt = "You are a professional restaurant ciritc. Give me a review about for the Peruvian restaurant, el casa, in Amsterdam."

    start_generate = time.time()
    samples = sampler.generate(prompt, n=32, max_gpu_n=192)
    end_generate = time.time()
    logger.info(f"Time taken for sampler.generate: {end_generate - start_generate:.2f} seconds")
    for sample in samples:
        logger.info("Sample: %s", sample)

    # now sample the best out of the n generated samples
    # reward_model = RewardModel()
    # start_sample = time.time()
    # best_sample = sampler.sample(samples, reward_model.reward_fn)
    # end_sample = time.time()
    # logger.info(f"Time taken for sampler.sample: {end_sample - start_sample:.2f} seconds")
    # logger.info(f"Best sample: {best_sample}")

if __name__ == "__main__":
    main()
