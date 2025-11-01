# reward model for positivity/negativity of some text
from transformers import pipeline

class RewardModel:
    def __init__(self):
        # load in the sentiment model
        self.sentiment_model = pipeline(
            model="lxyuan/distilbert-base-multilingual-cased-sentiments-student",
            top_k=None,
            device="cuda",
        )

    # given some batch of texts, return a float of the sentiment of that text
    def reward_fn(self, text: list[str]) -> list[float]:
        # distilled_student_sentiment_classifier(text) returns a list of lists of dicts, one per input string
        # We want the 'negative' score from each list
        results = self.sentiment_model(text)
        negative_scores = [
            next(item['score'] for item in res if item['label'] == 'negative')
            for res in results
        ]
        return negative_scores
