import math

def multi_term_memory_patchification(
    terms: list[dict],
    latent_h: int,
    latent_w: int
) -> dict:
    tokens_per_term = []

    for t in terms:
        # Step 1: apply temporal and spatial strides to latent dims
        comp_t = math.ceil(t['num_latent_frames'] / t['temporal_stride'])
        comp_h = math.ceil(latent_h / t['spatial_stride'])
        comp_w = math.ceil(latent_w / t['spatial_stride'])

        # Step 2: apply patch size to get the token grid
        patch_h = math.ceil(comp_h / t['patch_size'])
        patch_w = math.ceil(comp_w / t['patch_size'])

        # Step 3: token count for this term
        tokens_per_term.append(comp_t * patch_h * patch_w)

    total_tokens = sum(tokens_per_term)
    token_fractions = [round(tk / total_tokens, 4) for tk in tokens_per_term]

    return {
        'tokens_per_term': tokens_per_term,
        'total_tokens': total_tokens,
        'token_fractions': token_fractions
    }