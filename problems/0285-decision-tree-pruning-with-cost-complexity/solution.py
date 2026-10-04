def compute_pruning_alphas(tree: dict) -> list:
    """
    Computes effective alpha values for cost-complexity pruning.
    
    Args:
        tree: Dictionary representing a decision tree node with keys:
              - 'samples': number of samples reaching this node
              - 'errors': misclassification count if node becomes a leaf
              - 'left': left child subtree (dict) or None
              - 'right': right child subtree (dict) or None
        
    Returns:
        List of effective alpha values for internal nodes, sorted ascending.
    """
    def is_leaf(node):
        return node['left'] is None and node['right'] is None
    
    def count_leaves(node):
        if node is None:
            return 0
        if is_leaf(node):
            return 1
        return count_leaves(node['left']) + count_leaves(node['right'])
    
    def subtree_errors(node):
        if node is None:
            return 0
        if is_leaf(node):
            return node['errors']
        return subtree_errors(node['left']) + subtree_errors(node['right'])
    
    def collect_alphas(node, alphas):
        if node is None or is_leaf(node):
            return
        
        collect_alphas(node['left'], alphas)
        collect_alphas(node['right'], alphas)
        
        R_t = node['errors']
        R_Tt = subtree_errors(node)
        num_leaves = count_leaves(node)
        
        effective_alpha = (R_t - R_Tt) / (num_leaves - 1)
        alphas.append(round(effective_alpha, 4))
    
    alphas = []
    collect_alphas(tree, alphas)
    return sorted(alphas)