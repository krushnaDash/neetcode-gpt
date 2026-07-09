import torch
import torch.nn as nn
from torchtyping import TensorType
from typing import List

class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        # 1. Build vocabulary: collect all unique words, sort them, assign integer IDs starting at 1
        # 2. Encode each sentence by replacing words with their IDs
        # 3. Combine positive + negative into one list of tensors
        # 4. Pad shorter sequences with 0s using nn.utils.rnn.pad_sequence(tensors, batch_first=True)
        
        # Step 1: Collect all unique words from all sentences
        all_words = set()
        for sentence in positive + negative:
            words = sentence.split()
            for word in words:
                all_words.add(word)

        # Sort words so the mapping is consistent/deterministic
        sorted_words = sorted(all_words)

        # Assign each word an integer ID starting at 1 (0 is reserved for padding)
        vocab_to_id = {}
        current_id = 1
        for word in sorted_words:
            vocab_to_id[word] = current_id
            current_id = current_id + 1

        # Step 2: Convert each sentence into a list of integer IDs
        encoded_sentences = []
        for sentence in positive + negative:
            words = sentence.split()
            word_ids = []
            for word in words:
                word_id = vocab_to_id[word]
                word_ids.append(word_id)
            encoded_sentences.append(torch.tensor(word_ids))

        # Step 3 & 4: Pad all sequences to the same length (shorter ones get 0s appended)
        padded_tensors = nn.utils.rnn.pad_sequence(encoded_sentences, batch_first=True)
        
        return padded_tensors
        

