import torch
import torch.nn as nn
import torch.nn.functional as F

# The GPT model is provided for you. It returns raw logits (not probabilities).
# You only need to implement the training loop below.

class Solution:
    def train(self, model: nn.Module, data: torch.Tensor, epochs: int, context_length: int, batch_size: int, lr: float) -> float:
        # Train the GPT model using AdamW and cross_entropy loss.
        # For each epoch: seed with torch.manual_seed(epoch),
        # sample batches from data, run forward/backward, update weights.
        # Return the final loss rounded to 4 decimals.
        # AdamW = Adam + correct weight decay (L2 penalty applied directly to weights).
        # Standard optimizer for all transformer models (GPT-2, GPT-3, LLaMA...).
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

        # CrossEntropyLoss = softmax + negative log likelihood in one step.
        # Internally applies softmax so the model outputs raw logits (no softmax needed).
        loss_fn = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            # Seed per epoch: reproducible within an epoch, different across epochs.
            torch.manual_seed(epoch)

            # data is a 1D tensor of token IDs: [10, 42, 7, 3, 19, ...]
            # Pick batch_size random start positions to sample windows from it.
            # Upper bound: len(data) - context_length so Y (shifted +1) stays in bounds.
            starts = torch.randint(0, len(data) - context_length, (batch_size,))

            # X: input sequences  — what the model reads
            # Y: target sequences — what the model must predict (X shifted right by 1)
            # e.g. data=[the,cat,sat,on,mat], start=1, context_length=3
            #   X = [cat, sat, on]   Y = [sat, on, mat]
            X = torch.stack([data[s : s + context_length]       for s in starts])  # (B, T)
            Y = torch.stack([data[s + 1 : s + context_length + 1] for s in starts])  # (B, T)

            # ── Step 1: Clear old gradients ─────────────────────────────
            optimizer.zero_grad()

            # ── Step 2: Forward pass ────────────────────────────────────
            # Pass X through GPT → raw logits for every position.
            logits = model(X)   # shape: (B, T, vocab_size)

            # ── Step 3: Compute loss ────────────────────────────────────
            # cross_entropy needs logits (N, V) and targets (N,) — flat.
            # .view(-1, V) flattens (B, T, V) → (B*T, V)
            # .view(-1)    flattens (B, T)    → (B*T,)
            B, T, V = logits.shape
            loss = loss_fn(
                logits.view(B * T, V),  # (B*T, V) — B*T independent predictions
                Y.view(B * T)           # (B*T,)   — B*T correct next tokens
            )

            # ── Step 4: Backward pass ───────────────────────────────────
            # Compute gradient of loss w.r.t. every weight in the model.
            loss.backward()

            # ── Step 5: Update weights ──────────────────────────────────
            # Nudge all weights in the direction that reduces loss.
            optimizer.step()

        # Return final loss (scalar) rounded to 4 decimal places.
        # Untrained: loss ≈ ln(vocab_size). After training: loss decreases.
        return round(loss.item(), 4)
