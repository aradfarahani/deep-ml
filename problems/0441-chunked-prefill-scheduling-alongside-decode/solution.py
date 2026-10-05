def chunked_prefill_schedule(prefill_requests, decode_requests, max_batch_tokens, max_chunk_size):
    prefill_queue = [{'id': r['id'], 'remaining': r['prompt_tokens'], 'decode_tokens': r['decode_tokens']} for r in prefill_requests]
    active_decodes = [{'id': r['id'], 'remaining': r['remaining_decode']} for r in decode_requests]
    
    schedule = []
    step_idx = 0
    
    while prefill_queue or active_decodes:
        step = {'step': step_idx, 'prefill_chunks': [], 'decode_ids': [], 'total_tokens': 0}
        
        next_decodes = []
        for d in active_decodes:
            if step['total_tokens'] + 1 <= max_batch_tokens:
                step['decode_ids'].append(d['id'])
                step['total_tokens'] += 1
                d['remaining'] -= 1
                if d['remaining'] > 0:
                    next_decodes.append(d)
            else:
                next_decodes.append(d)
        
        new_decodes = []
        remaining_budget = max_batch_tokens - step['total_tokens']
        
        while prefill_queue and remaining_budget > 0:
            req = prefill_queue[0]
            chunk = min(req['remaining'], max_chunk_size, remaining_budget)
            step['prefill_chunks'].append((req['id'], chunk))
            step['total_tokens'] += chunk
            remaining_budget -= chunk
            req['remaining'] -= chunk
            
            if req['remaining'] == 0:
                prefill_queue.pop(0)
                if req['decode_tokens'] > 0:
                    new_decodes.append({'id': req['id'], 'remaining': req['decode_tokens']})
            else:
                break  # Don't process multiple chunks of the same request in one step
        
        schedule.append(step)
        active_decodes = next_decodes + new_decodes
        step_idx += 1
    
    return schedule