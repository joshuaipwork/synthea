
from collections import deque


class OpeningPhraseTracker:
    def __init__(self, max_tracked: int = 5):
        self.recent_openers = deque(maxlen=max_tracked)

    def extract_opener(self, response: str, word_count: int = 6) -> str:
        """Extract the first N words from a response, stripped of punctuation."""
        first_line = response.strip().split('\n')[0]
        words = first_line.split()[:word_count]
        return ' '.join(words)

    def record_response(self, response: str):
        opener = self.extract_opener(response)
        if opener:
            self.recent_openers.append(opener)
