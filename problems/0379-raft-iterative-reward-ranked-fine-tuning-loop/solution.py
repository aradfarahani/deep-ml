import numpy as np

def raft_alignment(
    stages_log_probs: list[list[list[list[float]]]],
    stages_rewards: list[list[list[float]]],
) -> list[float]:
    T = len(stages_rewards)
    losses = []

    for t in range(T):
        log_probs = stages_log_probs[t]
        rewards = stages_rewards[t]
        num_prompts = len(rewards)

        total_nll = 0.0
        total_tokens = 0

        # Step 2: Data ranking â select best-of-K per prompt
        for i in range(num_prompts):
            best_j = int(np.argmax(rewards[i]))

            # Step 3: Accumulate NLL for fine-tuning loss
            for tok in log_probs[i][best_j]:
                total_nll += -tok
                total_tokens += 1

        sft_loss = round(total_nll / total_tokens, 4) if total_tokens > 0 else 0.0
        losses.append(sft_loss)

    return losses
