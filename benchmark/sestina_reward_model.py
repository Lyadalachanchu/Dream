import logging
import math
import os
import re
from collections import Counter
from typing import List


class SestinaRewardModel:
    """Reward how closely a poem resembles a 6-stanza sestina with rotating end words."""

    TARGET_LINES = 39  # 6 stanzas * 6 lines + optional 3-line envoi

    def __init__(self):
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, "sestina_reward_model_logs.out")
        self.logger = logging.getLogger("sestina_reward_model_logger")
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.FileHandler(log_file)
            formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def reward_fn(self, texts: List[str]) -> List[float]:
        rewards: List[float] = []
        self.logger.info("Scoring %d texts for sestina structure", len(texts))
        for idx, text in enumerate(texts):
            lines = self._extract_lines(text)
            end_words = [self._ending_word(line) for line in lines]
            seed_words = [word for word in end_words[:6] if word]
            structure_score = self._line_count_score(len(lines))
            diversity_score = self._diversity_score(seed_words)
            coverage_score = self._coverage_score(end_words[6:], seed_words)
            balance_score = self._balance_score(end_words, seed_words)
            reward = (
                0.35 * structure_score
                + 0.25 * diversity_score
                + 0.2 * coverage_score
                + 0.2 * balance_score
            )
            rewards.append(reward)
            self.logger.info(
                "Text %d: lines=%d structure=%.3f diversity=%.3f coverage=%.3f balance=%.3f reward=%.3f seed_words=%s",
                idx,
                len(lines),
                structure_score,
                diversity_score,
                coverage_score,
                balance_score,
                reward,
                seed_words,
            )
        return rewards

    def _extract_lines(self, text: str) -> List[str]:
        stripped = text.strip()
        if not stripped:
            return []
        newline_lines = [line.strip() for line in stripped.splitlines() if line.strip()]
        if len(newline_lines) >= 6:
            return newline_lines

        sentence_lines = [
            seg.strip()
            for seg in re.split(r"[.!?]+", stripped)
            if seg.strip()
        ]
        if len(sentence_lines) >= 6:
            return sentence_lines

        tokens = re.split(r"[,;/]+|\s+", stripped)
        tokens = [tok for tok in tokens if tok]
        if not tokens:
            return []
        chunk_size = max(1, len(tokens) // 6)
        chunks = [
            " ".join(tokens[i : i + chunk_size])
            for i in range(0, len(tokens), chunk_size)
        ]
        return chunks

    def _ending_word(self, line: str) -> str:
        tokens = [re.sub(r"[^a-zA-Z']", "", tok).lower() for tok in line.split()]
        tokens = [tok for tok in tokens if tok]
        return tokens[-1] if tokens else ""

    def _line_count_score(self, num_lines: int) -> float:
        if num_lines == 0:
            return 0.0
        return max(0.0, 1.0 - abs(num_lines - self.TARGET_LINES) / self.TARGET_LINES)

    def _diversity_score(self, seed_words: List[str]) -> float:
        if len(seed_words) < 6:
            return max(0.0, len(seed_words) / 6.0)
        unique = len(set(seed_words))
        return unique / 6.0

    def _coverage_score(self, remaining_end_words: List[str], seed_words: List[str]) -> float:
        if not remaining_end_words or not seed_words:
            return 0.0
        valid = sum(1 for word in remaining_end_words if word in seed_words)
        return valid / len(remaining_end_words)

    def _balance_score(self, end_words: List[str], seed_words: List[str]) -> float:
        usable_seed = list(dict.fromkeys(seed_words))  # preserve order but unique
        if not usable_seed:
            return 0.0
        counts = Counter(word for word in end_words if word in usable_seed)
        total = sum(counts.values())
        if total == 0:
            return 0.0
        target = total / len(usable_seed)
        mse = sum((counts.get(word, 0) - target) ** 2 for word in usable_seed) / len(usable_seed)
        # Convert MSE to score in [0,1]
        return math.exp(-mse / (target + 1e-6))
