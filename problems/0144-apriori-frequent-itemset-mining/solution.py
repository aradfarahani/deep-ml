import itertools
from collections import defaultdict

def apriori(transactions, min_support=0.5, max_length=None):
    if not transactions:
        raise ValueError('Transaction list cannot be empty')
    if not 0 < min_support <= 1:
        raise ValueError('Minimum support must be between 0 and 1')

    num_transactions = len(transactions)
    min_support_count = min_support * num_transactions
    item_counts = defaultdict(int)
    for transaction in transactions:
        for item in transaction:
            item_counts[frozenset([item])] += 1
    frequent_itemsets = {itemset: count for itemset, count in item_counts.items() if count >= min_support_count}
    k = 1
    all_frequent_itemsets = dict(frequent_itemsets)
    while frequent_itemsets and (max_length is None or k < max_length):
        k += 1
        candidates = generate_candidates(frequent_itemsets.keys(), k)
        candidate_counts = defaultdict(int)
        for transaction in transactions:
            transaction_set = f