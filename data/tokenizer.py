from typing import List


class Solution:
    def get_merges(self, corpus: str, num_merges: int) -> List[List[str]]:
        # 1. Split corpus into a list of individual characters
        tokens = list(corpus)

        merges = []

        for _ in range(num_merges):
            # 2a. Count frequency of all adjacent token pairs
            pair_counts = {}
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                pair_counts[pair] = pair_counts.get(pair, 0) + 1

            if not pair_counts:
                break

            # 2b. Find the most frequent pair (break ties lexicographically)
            best_count = max(pair_counts.values())
            candidates = [pair for pair, count in pair_counts.items() if count == best_count]
            best_pair = min(candidates)  # lexicographically smallest wins the tie

            # 2c. Merge all non-overlapping occurrences left to right
            merged_token = best_pair[0] + best_pair[1]
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == best_pair[0] and tokens[i + 1] == best_pair[1]:
                    new_tokens.append(merged_token)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

            # 2d. Record the merge
            merges.append(list(best_pair))

        return merges
