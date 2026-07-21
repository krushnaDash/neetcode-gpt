from typing import List, Dict

class Solution:
    def tokenize_numbers(self, numbers: List[int], vocab: Dict[str, int]) -> List[List[str]]:
        # Tokenize each number using greedy left-to-right longest match.
        # Return a list of token lists showing how each number gets split.
        result = []
        for number in numbers:
            num_str = str(number)
            tokens = []
            i = 0
            while i < len(num_str):
                match = None
                for j in range(len(num_str), i, -1):
                    if num_str[i:j] in vocab:
                        match = num_str[i:j]
                        break
                if match:
                    tokens.append(match)
                    i += len(match)
                else:
                    tokens.append(num_str[i])
                    i += 1
            result.append(tokens)
        return result

    def count_tokens(self, text: str, vocab: Dict[str, int]) -> int:
        # Count how many tokens the text uses with greedy tokenization.
        # Use greedy left-to-right longest match.
        tokens = []
        i = 0
        while i < len(text):
            match = None
            for j in range(len(text), i, -1):
                if text[i:j] in vocab:
                    match = text[i:j]
                    break
            if match:
                tokens.append(match)
                i += len(match)
            else:
                tokens.append(text[i])
                i += 1
        return len(tokens)

    def fertility_score(self, text: str, vocab: Dict[str, int]) -> float:
        # Compute tokens-per-word ratio (fertility).
        # Higher = more expensive and less efficient.
        # Round to 4 decimal places.
        num_tokens = self.count_tokens(text, vocab)
        num_words = len(text.split())
        if num_words == 0:
            return 0.0
        return round(num_tokens / num_words, 4)
