import logging
import math
import os
import re


class HaikuRewardModel:
    """Reward model that scores how closely a text resembles a 5/7/5 haiku."""

    TARGET_SYLLABLES = (5, 7, 5)

    def __init__(self):
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, "haiku_reward_model_logs.out")
        self.logger = logging.getLogger("haiku_reward_model_logger")
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.FileHandler(log_file)
            formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def reward_fn(self, texts: list[str]) -> list[float]:
        """Return scores in (0, 1], higher is closer to the 5/7/5 target."""
        self.logger.info("Scoring %d texts for haiku structure", len(texts))
        rewards: list[float] = []
        for idx, text in enumerate(texts):
            lines = self._extract_candidate_lines(text)
            syllable_counts = [self._count_syllables(line) for line in lines]
            weights = [
                math.exp(-abs(syllables - target))
                for syllables, target in zip(syllable_counts, self.TARGET_SYLLABLES)
            ]
            reward = sum(weights) / len(weights)
            rewards.append(reward)
            self.logger.info(
                "Text %d: lines=%s syllables=%s weights=%s reward=%.3f",
                idx,
                lines,
                syllable_counts,
                [round(w, 3) for w in weights],
                reward,
            )
        return rewards

    def _extract_candidate_lines(self, text: str) -> list[str]:
        stripped = text.strip()
        if not stripped:
            return ["", "", ""]

        newline_lines = [line.strip() for line in stripped.splitlines() if line.strip()]
        if len(newline_lines) >= 3:
            return newline_lines[:3]

        sentence_lines = [
            seg.strip()
            for seg in re.split(r"[.!?]+", stripped)
            if seg.strip()
        ]
        if len(sentence_lines) >= 3:
            return sentence_lines[:3]

        # Fall back to splitting on commas or whitespace to always provide three candidates.
        tokens = re.split(r"[,;/]+|\s+", stripped)
        tokens = [tok for tok in tokens if tok]
        if not tokens:
            return ["", "", ""]
        chunk_size = max(1, len(tokens) // 3)
        chunks = [
            " ".join(tokens[i : i + chunk_size]) for i in range(0, len(tokens), chunk_size)
        ]
        while len(chunks) < 3:
            chunks.append("")
        return chunks[:3]

    def _count_syllables(self, line: str) -> int:
        words = [re.sub(r"[^a-zA-Z]", "", word).lower() for word in line.split()]
        words = [word for word in words if word]
        return sum(self._estimate_word_syllables(word) for word in words)

    def _estimate_word_syllables(self, word: str) -> int:
        if not word:
            return 0
        vowels = "aeiouy"
        count = 0
        prev_is_vowel = False
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_is_vowel:
                count += 1
            prev_is_vowel = is_vowel
        if word.endswith("e") and count > 1:
            count -= 1
        return max(count, 1)
