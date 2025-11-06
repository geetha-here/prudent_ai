# price_gap.py

from typing import Optional, Tuple

def find_price_gap_pair(nums: list[int], k: int) -> Optional[Tuple[int, int]]:
    earliest = {}
    best_pair = None

    for j, v in enumerate(nums):
        t1 = v - k
        if t1 in earliest:
            cand = (earliest[t1], j)
            if best_pair is None or cand < best_pair:
                best_pair = cand

        if k > 0:
            t2 = v + k
            if t2 in earliest:
                cand = (earliest[t2], j)
                if best_pair is None or cand < best_pair:
                    best_pair = cand

        if v not in earliest:
            earliest[v] = j

    return best_pair
ku=int(input())
nums = []
for i in range(ku):
    yu=int(input())
    nums.append(yu)
k = int(input())
print(find_price_gap_pair(nums,k))
