# implements fk steering approach from https://arxiv.org/pdf/2501.06848
from sampling_methods.base_sampler import BaseSampler
from typing import Callable, Literal, Optional
import torch
from transformers import AutoModel, AutoTokenizer
import logging
import os
import torch.nn as nn
import torch.nn.functional as F
from src.diffllm.gen_utils import sample_tokens


class NestedSampler(BaseSampler):
    def __init__(
        self,
        resample_every_n: int,
        reward_fn: "Callable[[list[str]], list[float]]",
        reward_particles: int = 2,
        ess_ratio: float = 0.7,
    ):
        self.resample_every_n = resample_every_n

        model_path = "Dream-org/Dream-v0-Instruct-7B"
        model = AutoModel.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
        )
        # set left padding
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, padding_side='left') 
        self.model = model.to("cuda").eval()
        self.reward_fn = reward_fn
        self.reward_particles = max(1, int(reward_particles))
        if not (0.0 < ess_ratio <= 1.0):
            raise ValueError("ess_ratio must be within (0, 1].")
        self.ess_ratio = ess_ratio
        self._hook_context = None

        log_dir = "logs"
        log_file = os.path.join(log_dir, "nested_sampler.out")
        self.logger = logging.getLogger("nested_sampler_logger")
        self.logger.setLevel(logging.INFO)
        # ensure logs directory exists before configuring handler
        os.makedirs(log_dir, exist_ok=True)
        if not self.logger.handlers:
            handler = logging.FileHandler(log_file)
            formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def _normalize_weights(self, weights: torch.Tensor) -> torch.Tensor:
        weights = weights.clamp_min(1e-8)
        weight_sum = weights.sum()
        if weight_sum <= 0:
            return torch.full_like(weights, 1.0 / weights.shape[0])
        return weights / weight_sum

    def _prepare_hook_attention(self, seq_len: int):
        if self._hook_context is None:
            return "full", None

        cached_attention = self._hook_context.get("prepared_attention_mask")
        cached_tok_idx = self._hook_context.get("tok_idx")
        if cached_attention is not None:
            return cached_attention, cached_tok_idx

        attention_mask = self._hook_context.get("attention_mask")
        if attention_mask is None or not torch.any(attention_mask == 0):
            self._hook_context["prepared_attention_mask"] = "full"
            self._hook_context["tok_idx"] = None
            return "full", None

        pad = seq_len - attention_mask.shape[1]
        if pad > 0:
            attention_mask = F.pad(attention_mask, (0, pad), value=1.0)

        tok_idx = attention_mask.long().cumsum(-1) - 1
        tok_idx.masked_fill_(attention_mask == 0, 1)
        prepared_attention = torch.logical_and(
            attention_mask.unsqueeze(1).unsqueeze(-2),
            attention_mask.unsqueeze(1).unsqueeze(-1),
        )

        self._hook_context["prepared_attention_mask"] = prepared_attention
        self._hook_context["tok_idx"] = tok_idx
        return prepared_attention, tok_idx

    def _resolve_mask_token_id(self) -> int:
        mask_token_id: Optional[int] = None
        if self._hook_context is not None:
            mask_token_id = self._hook_context.get("mask_token_id")
        if mask_token_id is None:
            mask_token_id = getattr(self.model.generation_config, "mask_token_id", None)
        if mask_token_id is None:
            mask_token_id = getattr(self.tokenizer, "mask_token_id", None)
        if mask_token_id is None:
            raise ValueError("Mask token ID could not be determined; unable to perform forward fill for reward.")
        return int(mask_token_id)

    def _forward_fill_for_reward(self, x: torch.Tensor) -> torch.Tensor:
        attn_mask, tok_idx = self._prepare_hook_attention(x.shape[1])
        mask_token_id = self._resolve_mask_token_id()
        num_particles = self.reward_particles
        batch_size, seq_len = x.shape

        x_for_reward = x.clone()
        if num_particles > 1:
            x_for_reward = x_for_reward.unsqueeze(1).repeat(1, num_particles, 1)
            x_for_reward = x_for_reward.view(batch_size * num_particles, seq_len)

        attn_mask_forward = attn_mask
        tok_idx_forward = tok_idx
        if isinstance(attn_mask, torch.Tensor) and num_particles > 1:
            attn_mask_forward = attn_mask.repeat_interleave(num_particles, dim=0).contiguous()
        if tok_idx is not None and num_particles > 1:
            tok_idx_forward = tok_idx.repeat_interleave(num_particles, dim=0).contiguous()

        with torch.no_grad():
            outputs = self.model(x_for_reward, attention_mask=attn_mask_forward, tok_idx=tok_idx_forward)
            logits = outputs.logits

        logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
        mask_index = (x_for_reward == mask_token_id)
        if mask_index.any():
            temperature = 0.8
            top_p = None
            top_k = None
            if self._hook_context is not None:
                temperature = self._hook_context.get("temperature", 0.0) or 0.0
                top_p = self._hook_context.get("top_p")
                top_k = self._hook_context.get("top_k")
            _, sampled = sample_tokens(
                logits[mask_index],
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )
            x_for_reward[mask_index] = sampled

        if num_particles > 1:
            x_for_reward = x_for_reward.view(batch_size, num_particles, seq_len)
        else:
            x_for_reward = x_for_reward.view(batch_size, 1, seq_len)

        return x_for_reward

    def _aggregate_particle_rewards(self, rewards: torch.Tensor) -> torch.Tensor:
        return rewards.mean(dim=1)

    def _decode_generated_texts(self, sequences: torch.Tensor) -> list[str]:
        prompt_token_count = 0
        if self._hook_context is not None:
            prompt_token_count = int(self._hook_context.get("prompt_token_count", 0) or 0)

        sequences_list = sequences.tolist()
        trimmed_sequences = [
            seq[prompt_token_count:] if prompt_token_count < len(seq) else []
            for seq in sequences_list
        ]
        return self.tokenizer.batch_decode(trimmed_sequences, skip_special_tokens=True)

    # create hook to calculate score and resample intermediate x's
    def generation_tokens_hook_func(self, step, x, logits):
        if step != None and logits != None and x != None and step:
            alpha = 5
            outer_texts = self._decode_generated_texts(x)
            outer_rewards = torch.tensor(self.reward_fn(outer_texts), dtype=torch.float32, device=x.device)
            outer_weights = outer_rewards.pow(alpha)
            avg_reward = outer_weights.mean().item()
            print(f"Average reward: {avg_reward:.4f}")
            normalized_outer = self._normalize_weights(outer_weights)

            ess_value = (1.0 / torch.sum(normalized_outer ** 2)).item()
            ess_threshold = self.ess_ratio * x.shape[0]
            print(f"outer rewards shape: {outer_rewards.shape}")
            print(f"outer normalized weights: {normalized_outer}")
            print(f"outer ess: {ess_value:.2f} (threshold {ess_threshold:.2f})")
            if ess_value >= ess_threshold:
                print("Using outer rewards for resampling; skipping nested SMC.")
                idx = torch.multinomial(normalized_outer, num_samples=len(x), replacement=True)
                X_sampled = x[idx]
                print(f"idx: {idx}")
                return X_sampled

            x_particles = self._forward_fill_for_reward(x)
            batch_size, particle_count, seq_len = x_particles.shape
            flattened_particles = x_particles.view(batch_size * particle_count, seq_len)

            particle_texts = self._decode_generated_texts(flattened_particles)
            # print(f"Particle texts: {particle_texts}")
            particle_rewards = torch.tensor(self.reward_fn(particle_texts), dtype=torch.float32, device=x.device)
            particle_rewards = particle_rewards.view(batch_size, particle_count)
            rewards = self._aggregate_particle_rewards(particle_rewards)
            print(f"particle rewards shape: {particle_rewards.shape} -> aggregated rewards shape: {rewards.shape}")

            # sample with particles with replacement based on the normalized rewards
            # normalize the rewards
            normalized_rewards = self._normalize_weights(rewards)
            print(f"normalized_rewards: {normalized_rewards}")

            # resample
            idx = torch.multinomial(normalized_rewards, num_samples=len(x), replacement=True)
            X_sampled = x[idx]
            print(f"idx: {idx}")
            return X_sampled
            
        return x


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

            temperature = 0.5
            top_p = 0.95
            top_k = None
            self._hook_context = {
                "attention_mask": attention_mask.clone() if attention_mask is not None else None,
                "prepared_attention_mask": None,
                "tok_idx": None,
                "mask_token_id": getattr(self.model.generation_config, "mask_token_id", None)
                or getattr(self.tokenizer, "mask_token_id", None),
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "prompt_token_count": int(input_ids.shape[1]),
            }
            output = self.model.diffusion_generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=64,
                output_history=True,
                return_dict_in_generate=True,
                steps=100,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                generation_tokens_hook_func=self.generation_tokens_hook_func
            )
            self._hook_context = None
            batch_generations = [
                self.tokenizer.decode(g[len(p) :].tolist()).split(self.tokenizer.eos_token)[0]
                for p, g in zip(input_ids, output.sequences)
            ]
            generations.extend(batch_generations)

        self.logger.info("Completed generation; produced %d trajectories", len(generations))
        return generations

    def sample(reward_fn: 'Callable[[str], float]'):
        pass

    def one_step():
        pass
