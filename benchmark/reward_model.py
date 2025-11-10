# reward model for positivity/negativity of some text
import logging
import os
from transformers import pipeline


class RewardModel:
    def __init__(self):
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, "reward_model_logs.out")
        self.logger = logging.getLogger("reward_model_logger")
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.FileHandler(log_file)
            formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        # load in the sentiment model
        self.sentiment_model = pipeline(
            model="lxyuan/distilbert-base-multilingual-cased-sentiments-student",
            top_k=None,
            device="cuda",
        )

    # given some batch of texts, return a float of the sentiment of that text
    def reward_fn(self, text: list[str]) -> list[float]:
        self.logger.info("Scoring %d texts", len(text))
        for idx, sample in enumerate(text):
            self.logger.info("Text %d: %s", idx, sample)

        # distilled_student_sentiment_classifier(text) returns a list of lists of dicts, one per input string
        # We want the 'negative' score from each list
        results = self.sentiment_model(text)
        negative_scores = [
            next(item["score"] for item in res if item["label"] == "negative")
            for res in results
        ]
        return negative_scores
