from itertools import combinations
import math

# Using itertools.combinations()
items = [1, 2, 3, 4, 5]
r = 3
combinations_list = list(combinations(items, r))
print(f"Combinations of {items} taken {r} at a time: {combinations_list}")
# Expected output: Combinations of [1, 2, 3, 4] taken 2 at a time: [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]

# Using math.comb()
n = 5
k = 3
num_combinations = math.comb(n, k)
print(f"Number of combinations of {n} items taken {k} at a time: {num_combinations}")
# Expected output: Number of combinations of 5 items taken 3 at a time: 10