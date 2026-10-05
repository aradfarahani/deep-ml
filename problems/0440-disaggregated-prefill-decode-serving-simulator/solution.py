import numpy as np

def disaggregated_serving_sim(requests, num_prefill, num_decode, prefill_rate, decode_rate, kv_transfer_rate):
	sorted_requests = sorted(requests, key=lambda r: r['arrival_time'])
	
	prefill_avail = [0.0] * num_prefill
	decode_avail = [0.0] * num_decode
	
	ttfts = []
	total_latencies = []
	total_output_tokens = 0
	total_prefill_busy = 0.0
	total_decode_busy = 0.0
	
	for req in sorted_requests:
		arrival = req['arrival_time']
		prompt_len = req['prompt_tokens']
		output_len = req['output_tokens']
		
		# Assign to earliest available prefill instance
		pi = int(np.argmin(prefill_avail))
		prefill_start = max(arrival, prefill_avail[pi])
		prefill_duration = prompt_len / prefill_rate
		prefill_end = prefill_start + prefill_duration
		prefill_avail[pi] = prefill_end
		
		# KV cache transfer
		kv_duration = prompt_len * kv_transfer_rate
		kv_end = prefill_end + kv_duration
		
		# Assign to earliest available decode instance
		di = int(np.argmin(decode_avail))
		decode_start = max(kv_end, decode_avail[di])
		decode_duration = output_len / decode_rate
		decode_end = decode_start + decode_duration
		decode_avail[di] = decode_end
		
		ttft = decode_start - arrival
		total_lat = decode_end - arrival
		
		ttfts.append(ttft)
		total_latencies.append(total_lat)
		total_output_tokens += output_len
		total_prefill_busy += prefill_duration
		total_decode_busy += decode_duration
	
	min_arrival = min(r['arrival_time'] for r in sorted_requests)
	makespan = max(decode_avail) - min_arrival
	
	avg_ttft = float(np.mean(ttfts))
	avg_total_latency = float(np.mean(total_latencies))
	throughput = total_output_tokens / makespan if makespan > 0 else 0.0
	prefill_utilization = total_prefill_busy / (num_prefill * makespan) if makespan > 0 else 0.0
	decode_utilization = total_decode_busy / (num_decode * makespan) if makespan > 0 else 0.0
	
	return {
		'avg_ttft': round(avg_ttft, 4),
		'avg_total_latency': round(avg_total_latency, 4),
		'throughput': round(throughput, 4),
		'prefill_utilization': round(prefill_utilization, 4),
		'decode_utilization': round(decode_utilization, 4)
	}