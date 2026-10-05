import math

def compute_visual_tokens(image_height: int, image_width: int, patch_size: int,
                          max_resolution: int = None, add_cls_token: bool = True,
                          padding_strategy: str = 'pad') -> dict:
    h, w = image_height, image_width

    # Step 1: Resize if max_resolution is specified and longest side exceeds it
    if max_resolution is not None:
        longest_side = max(h, w)
        if longest_side > max_resolution:
            scale = max_resolution / longest_side
            h = round(h * scale)
            w = round(w * scale)

    # Step 2: Compute effective dimensions based on padding strategy
    if padding_strategy == 'pad':
        effective_h = math.ceil(h / patch_size) * patch_size
        effective_w = math.ceil(w / patch_size) * patch_size
    elif padding_strategy == 'truncate':
        effective_h = (h // patch_size) * patch_size
        effective_w = (w // patch_size) * patch_size
    else:
        raise ValueError(f"Unknown padding_strategy: {padding_strategy}")

    # Step 3: Compute patch counts
    patches_h = effective_h // patch_size
    patches_w = effective_w // patch_size
    num_patches = patches_h * patches_w

    # Step 4: Add CLS token if requested
    total_tokens = num_patches + (1 if add_cls_token else 0)

    return {
        'effective_height': effective_h,
        'effective_width': effective_w,
        'patches_h': patches_h,
        'patches_w': patches_w,
        'num_patches': num_patches,
        'total_tokens': total_tokens
    }