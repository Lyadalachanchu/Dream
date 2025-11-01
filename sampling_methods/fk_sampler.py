# implements fk steering approach from https://arxiv.org/pdf/2501.06848
from sampling_methods.base_sampler import BaseSampler
from typing import Callable
import torch
from transformers import AutoModel, AutoTokenizer
import logging
import os
import torch.nn as nn

class FKSampler(BaseSampler):
    def __init__(self, potential:str, resample_every_n:int, reward_fn: 'Callable[[str], [float]]', detail_steps=64):
        self.potential = potential
        self.resample_every_n = resample_every_n

        model_path = "Dream-org/Dream-v0-Instruct-7B"
        model = AutoModel.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
        )
        # set left padding
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, padding_side='left') 
        self.model = model.to("cuda").eval()
        self.reward_fn = reward_fn
        self.detail_steps = detail_steps

        # reward 2d array to keep track of reward history for each particle at latest timestep
        # TODO: This takes an uncessary amount of memory (to be general). Implement a different reward history for each potential implemented
        self.reward_history = []

        log_dir = "logs"
        log_file = os.path.join(log_dir, "fk_sampler.out")
        self.logger = logging.getLogger("fk_sampler_logger")
        self.logger.setLevel(logging.INFO)
        # ensure logs directory exists before configuring handler
        os.makedirs(log_dir, exist_ok=True)
        if not self.logger.handlers:
            handler = logging.FileHandler(log_file)
            formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    # create hook to calculate score and resample intermediate x's
    def generation_tokens_hook_func(self, step, x, logits):
        if step != None and logits != None and x != None and step % self.resample_every_n == 0:
            print(f"potential: {self.potential}")
            print(f"x shape: {x.shape}")
            # convert ids to text and calculate the rewards
            texts = self.tokenizer.batch_decode(x.tolist(), skip_special_tokens=True)
            # use the intermediate text itself to calculate the reward
            # TO_EXPLORE: Jump straight to t=0 and calculate reward on that?
            rewards = self.reward_fn(texts)
            print(f"rewards: {rewards}")

            # first add the rewards to their respective particle; there are n rewards and n rows in self.reward_history
            [self.reward_history[i].append(rewards[i]) for i in range(len(rewards))]

            # calculate the difference potential
            reward_diff = [rewards[i]-self.reward_history[i][-2] for i in range(len(rewards))]
            reward_diff = torch.tensor(reward_diff, dtype=torch.float32)

            # sample with particles with replacement based on the normalized rewards
            # normalize the rewards
            alpha = 2
            # rewards = rewards.pow(alpha)
            reward_diff = reward_diff*alpha
            reward_diff = reward_diff.exp()
            normalized_rewards = reward_diff / (reward_diff.sum() + 1e-8)

            print(f"normalized_rewards: {normalized_rewards}")

            # resample
            idx = torch.multinomial(normalized_rewards, num_samples=len(x), replacement=True)

            # resample reward histories
            self.reward_history = [self.reward_history[i].copy() for i in idx.tolist()]

            print(f"reward history: {self.reward_history}")

            X_sampled = x[idx]
            print(f"idx: {idx}")
            # Assume the proposal function is just the diffusion model 
            # (for now; later we'll change it to the optimal proposal function with nested sampling)
            return X_sampled
            
        return x


    def generate(self, prompt: str, n=128, max_gpu_n=64) -> list[str]:
        if max_gpu_n <= 0:
            raise ValueError("max_gpu_n must be a positive integer")

        messages = [[{"role": "user", "content": prompt}] for _ in range(n)]
        generations: list[str] = []

        # we have to keep track of reward histories for n particles at each time step
        # we use the difference potential (use a different initialization, other than 0 for the other potentials)
        self.reward_history.extend([[0] for _ in range(n)])

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
                steps=self.detail_steps,
                temperature=0.2,
                top_p=0.95,
                generation_tokens_hook_func=self.generation_tokens_hook_func
            )
            batch_generations = [
                self.tokenizer.decode(g[len(p) :].tolist()).split(self.tokenizer.eos_token)[0]
                for p, g in zip(input_ids, output.sequences)
            ]
            generations.extend(batch_generations)

        self.logger.info("Completed generation; produced %d trajectories", len(generations))
        return generations

    def sample(reward_fn: 'Callable[[str], float]'):
        pass
