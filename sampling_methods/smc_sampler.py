from sampling_methods.base_sampler import BaseSampler
from typing import Callable
import torch
from transformers import AutoModel, AutoTokenizer
import logging
import os
from tqdm import tqdm
import torch.nn as nn

class SMCSampler(BaseSampler):
    def __init__(self, reward_fn: 'Callable[[str], [float]]', detail_steps=64):
        self.detail_steps = detail_steps
        model_path = "Dream-org/Dream-v0-Instruct-7B"
        model = AutoModel.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
        )
        # set left padding
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, padding_side='left') 
        self.model = model.to("cuda").eval()
        self.reward_fn = reward_fn

        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, "smc_sampler.out")
        self.logger = logging.getLogger("smc_sampler_logger")
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.FileHandler(log_file)
            formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    # create hook to calculate score and resample intermediate x's
    def generation_tokens_hook_func(self, step, x, logits):
        if step != None and step > 40 and logits != None and x != None:
            # 1. Calculate reward
            # convert ids to text and calculate the rewards
            texts = self.tokenizer.batch_decode(x.tolist(), skip_special_tokens=True)
            response_texts = [self._strip_prompt_for_reward(text) for text in texts]
            reward_values = self.reward_fn(response_texts)
            for idx, (text, reward) in enumerate(zip(response_texts, reward_values)):
                self.logger.info(
                    "Step %s | Particle %d text: %s | reward: %.4f",
                    step,
                    idx,
                    text,
                    reward,
                )
            # use the intermediate text itself to calculate the reward
            rewards = torch.tensor(reward_values, dtype=torch.float32)

            # 2. Normalize rewards
            rewards = rewards.exp()
            normalized_rewards = rewards / (rewards.sum() + 1e-8)

            # 3. Resample based on normalized rewards
            idx = torch.multinomial(normalized_rewards, num_samples=len(x), replacement=True)
            X_sampled = x[idx]
            return X_sampled
            
        return x


    def generate(self, prompt: str, n=128, max_gpu_n=64) -> list[str]:
        if max_gpu_n <= 0:
            raise ValueError("max_gpu_n must be a positive integer")

        messages = [[{"role": "user", "content": prompt}] for _ in range(n)]
        generations: list[str] = []


        for batch_idx, start in enumerate(tqdm(range(0, n, max_gpu_n), desc="SMC sampling batches"), start=1):
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

            output = self.model.diffusion_generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=128,
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

        return generations

    def sample(reward_fn: 'Callable[[str], float]'):
        pass

    def _strip_prompt_for_reward(self, text: str) -> str:
        marker = "<|im_start|>assistant"
        if marker not in text:
            return text
        response = text.rsplit(marker, 1)[1]
        # Drop any trailing assistant end token if it is already present
        end_token = "<|im_end|>"
        if end_token in response:
            response = response.split(end_token, 1)[0]
        return response.lstrip()

# Backwards compatibility for older imports
SMC = SMCSampler
