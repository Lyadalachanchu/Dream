# implements the best of n sampling procedure.
# given a prompt, generate n samples; given a reward function, pick the best sample

from sampling_methods.base_sampler import BaseSampler
from typing import Callable
import torch
from transformers import AutoModel, AutoTokenizer
import logging
import os

class BestOfNSampler(BaseSampler):
    def __init__(self):
        model_path = "Dream-org/Dream-v0-Instruct-7B"
        model = AutoModel.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
        )
        # set left padding
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, padding_side='left') 
        self.model = model.to("cuda").eval()

        log_dir = "logs"
        log_file = os.path.join(log_dir, "best_of_n_logs.out")
        self.logger = logging.getLogger("best_of_n_logger")
        self.logger.setLevel(logging.INFO)
        # ensure logs directory exists before configuring handler
        os.makedirs(log_dir, exist_ok=True)
        if not self.logger.handlers:
            handler = logging.FileHandler(log_file)
            formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    # Given the prompt, generate n e2e trajectories.
    def generate(self, prompt: str, n=128, max_gpu_n=64) -> list[str]:
        if max_gpu_n <= 0:
            raise ValueError("max_gpu_n must be a positive integer")

        messages = [[{"role": "user", "content": prompt}] for _ in range(n)]
        generations: list[str] = []

        for batch_idx, start in enumerate(range(0, n, max_gpu_n), start=1):
            end = min(start + max_gpu_n, n)
            batch_messages = messages[start:end]
            inputs = self.tokenizer.apply_chat_template(
                batch_messages,
                return_tensors="pt",
                return_dict=True,
                add_generation_prompt=True,
                padding=True,
            )
            input_ids = inputs.input_ids.to(device="cuda")
            attention_mask = inputs.attention_mask.to(device="cuda")

            self.logger.info(
                "Batch %d: input_ids shape %s",
                batch_idx,
                tuple(input_ids.shape),
            )

            output = self.model.diffusion_generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=256,
                output_history=True,
                return_dict_in_generate=True,
                steps=64,
                temperature=0.2,
                top_p=0.95,
            )
            batch_generations = [
                self.tokenizer.decode(g[len(p) :].tolist()).split(self.tokenizer.eos_token)[0]
                for p, g in zip(input_ids, output.sequences)
            ]
            generations.extend(batch_generations)

        self.logger.info("Completed generation; produced %d trajectories", len(generations))
        return generations



    # given the samples, return the best sample according to the reward_fn
    def sample(self, samples: list[str], reward_fn: 'Callable[[str], [float]]'):
        rewards = reward_fn(samples)
        self.logger.info(f"Completed reward calculation; Rewards: {rewards}")
        best_sample = samples[rewards.index(max(rewards))]
        return best_sample
