def analyze_ml_pipeline(tasks: list) -> dict:
    """
    Analyze an ML pipeline DAG for scheduling and critical path.
    """
    from collections import deque, defaultdict
    
    if not tasks:
        return {
            'execution_order': [],
            'earliest_start': {},
            'earliest_finish': {},
            'latest_start': {},
            'latest_finish': {},
            'slack': {},
            'critical_path': [],
            'makespan': 0
        }
    
    # Build task lookup and adjacency lists
    task_map = {t['id']: t for t in tasks}
    task_ids = list(task_map.keys())
    
    # Build graph structures
    in_degree = defaultdict(int)
    out_edges = defaultdict(list)
    in_edges = defaultdict(list)
    
    for t in tasks:
        tid = t['id']
        for dep in t.get('dependencies', []):
            out_edges[dep].append(tid)
            in_edges[tid].append(dep)
            in_degree[tid] += 1
    
    # Topological sort using Kahn's algorithm with alphabetical ordering
    queue = deque(sorted([tid for tid in task_ids if in_degree[tid] == 0]))
    topo_order = []
    
    while queue:
        current = queue.popleft()
        topo_order.append(current)
        successors = sorted(out_edges[current])
        for neighbor in successors:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                # Insert in sorted position
                inserted = False
                for i, q_item in enumerate(queue):
                    if neighbor < q_item:
                        queue.insert(i, neighbor)
                        inserted = True
                        break
                if not inserted:
                    queue.append(neighbor)
    
    # Forward pass - calculate earliest start and finish
    earliest_start = {}
    earliest_finish = {}
    
    for tid in topo_order:
        deps = in_edges[tid]
        if not deps:
            earliest_start[tid] = 0
        else:
            earliest_start[tid] = max(earliest_finish[d] for d in deps)
        earliest_finish[tid] = earliest_start[tid] + task_map[tid]['duration']
    
    makespan = max(earliest_finish.values()) if earliest_finish else 0
    
    # Backward pass - calculate latest start and finish
    latest_finish_temp = {}
    latest_start_temp = {}
    
    for tid in reversed(topo_order):
        successors = out_edges[tid]
        if not successors:
            latest_finish_temp[tid] = makespan
        else:
            latest_finish_temp[tid] = min(latest_start_temp[s] for s in successors)
        latest_start_temp[tid] = latest_finish_temp[tid] - task_map[tid]['duration']
    
    # Rebuild in topological order for consistent output
    latest_start = {tid: latest_start_temp[tid] for tid in topo_order}
    latest_finish = {tid: latest_finish_temp[tid] for tid in topo_order}
    
    # Calculate slack and identify critical path
    slack = {}
    critical_path = []
    
    for tid in topo_order:
        slack[tid] = latest_start[tid] - earliest_start[tid]
        if slack[tid] == 0:
            critical_path.append(tid)
    
    return {
        'execution_order': topo_order,
        'earliest_start': earliest_start,
        'earliest_finish': earliest_finish,
        'latest_start': latest_start,
        'latest_finish': latest_finish,
        'slack': slack,
        'critical_path': critical_path,
        'makespan': makespan
    }