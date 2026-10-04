def extract_boxed_answer(response: str) -> str:
    """
    Extract the answer from within \boxed{...} in a model response.
    
    Args:
        response: The model's text response containing a boxed answer
    
    Returns:
        The content inside the last \boxed{}, or empty string if not found
    """
    boxes = []
    i = 0
    target = "\\boxed{"
    target_len = len(target)
    
    while i < len(response):
        if response[i:i+target_len] == target:
            start = i + target_len
            depth = 1
            j = start
            while j < len(response) and depth > 0:
                if response[j] == '{':
                    depth += 1
                elif response[j] == '}':
                    depth -= 1
                j += 1
            content = response[start:j-1]
            boxes.append(content)
            i = j
        else:
            i += 1
    
    if not boxes:
        return ""
    return boxes[-1]