import logging
import math
import os
import re
from typing import List


class HaikuRewardModel:
    """Reward model that favors 5/7/5 syllable structure with mild style priors."""

    TARGET_SYLLABLES = (5, 7, 5)
    _SEASON_WORDS = {
        "spring",
        "summer",
        "autumn",
        "fall",
        "winter",
        "snow",
        "blossom",
        "bloom",
        "petal",
        "harvest",
        "frost",
        "breeze",
        "rain",
    }
    _SYLLABLE_EXCEPTIONS = {
        "autumn": 2,
        "blossom": 2,
        "echo": 2,
        "fire": 1,
        "hour": 2,
        "ocean": 2,
        "poem": 2,
        "quiet": 2,
        "rhythm": 2,
        "sapphire": 2,
        "season": 2,
        "tower": 2,
        "violet": 3,
        "water": 2,
    }

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

    def reward_fn(self, texts: List[str]) -> List[float]:
        rewards: List[float] = []
        self.logger.info("Scoring %d texts for haiku quality", len(texts))
        for idx, text in enumerate(texts):
            lines = self._extract_candidate_lines(text)
            line_scores = []
            syllable_counts = []
            for line_idx, target in enumerate(self.TARGET_SYLLABLES):
                line_text = lines[line_idx] if line_idx < len(lines) else ""
                syllables = self._count_syllables(line_text)
                syllable_counts.append(syllables)
                closeness = math.exp(-abs(syllables - target))
                if not line_text.strip():
                    # Encourage partial completions to keep writing instead of zeroing reward
                    closeness *= 0.2
                line_scores.append(closeness)
            avg_score = sum(line_scores) / len(self.TARGET_SYLLABLES)
            extra_line_penalty = math.exp(
                -0.6 * max(0, len(lines) - len(self.TARGET_SYLLABLES))
            )
            season_bonus = self._season_word_bonus(text)
            reward = min(1.0, avg_score * extra_line_penalty + season_bonus)
            rewards.append(reward)
            self.logger.info(
                "Text %d: lines=%s syllables=%s line_scores=%s base=%.3f penalty=%.3f season=%.3f reward=%.3f",
                idx,
                len(lines),
                syllable_counts,
                [round(score, 3) for score in line_scores],
                avg_score,
                extra_line_penalty,
                season_bonus,
                reward,
            )
        return rewards

    def _extract_candidate_lines(self, text: str) -> List[str]:
        stripped = text.replace("\r\n", "\n").strip()
        if not stripped:
            return []
        newline_lines = [line.strip() for line in stripped.splitlines() if line.strip()]
        if len(newline_lines) >= len(self.TARGET_SYLLABLES):
            return newline_lines[: len(self.TARGET_SYLLABLES)]

        segments: List[str] = []
        for chunk in newline_lines or [stripped]:
            segments.extend(self._split_on_punctuation(chunk))
        if len(segments) >= len(self.TARGET_SYLLABLES):
            return segments[: len(self.TARGET_SYLLABLES)]

        tokens = re.findall(r"[A-Za-z']+", stripped)
        if not tokens:
            return []
        approx_len = max(1, len(tokens) // len(self.TARGET_SYLLABLES))
        lines = [
            " ".join(tokens[i : i + approx_len])
            for i in range(0, len(tokens), approx_len)
        ]
        return lines[: len(self.TARGET_SYLLABLES)]

    def _split_on_punctuation(self, text: str) -> List[str]:
        return [
            segment.strip()
            for segment in re.split(r"[\\/|,;:?!]+|-{2,}|—|–", text)
            if segment.strip()
        ]

    def _count_syllables(self, line: str) -> int:
        if not line.strip():
            return 0
        words = re.findall(r"[A-Za-z']+", line.lower())
        if not words:
            return 0
        return sum(self._count_syllables_in_word(word) for word in words)

    def _count_syllables_in_word(self, word: str) -> int:
        if not word:
            return 0
        if word in self._SYLLABLE_EXCEPTIONS:
            return self._SYLLABLE_EXCEPTIONS[word]

        word = re.sub(r"[^a-z]", "", word)
        if not word:
            return 0

        vowels = "aeiouy"
        syllables = 0
        prev_is_vowel = False
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_is_vowel:
                syllables += 1
            prev_is_vowel = is_vowel

        if word.endswith("e") and not word.endswith(("le", "ye")) and syllables > 1:
            syllables -= 1
        if word.endswith("le") and len(word) > 2 and word[-3] not in vowels:
            syllables += 1

        return max(syllables, 1)

    def _season_word_bonus(self, text: str) -> float:
        if not text:
            return 0.0
        tokens = {tok.lower() for tok in re.findall(r"[A-Za-z']+", text)}
        if not tokens:
            return 0.0
        overlap = len(tokens & self._SEASON_WORDS)
        if overlap == 0:
            return 0.0
        return min(0.05 * overlap, 0.15)
