import sys, math

num_islands = int(sys.stdin.readline())
dist_from_start = [int(d) for d in sys.stdin.readline().split()]
dist_to_mainland = [dist_from_start[-1] - dist_from_start[i] for i in range(num_islands)]
num_combos = int(sys.stdin.readline())
combos = [int(d) for d in sys.stdin.readline().split()]

'''
memo = [[dist_to_mainland[0] for _ in range(num_islands)] for _ in range(max(combos))]
next_memo[0] = dist_to_mainland
for c in range(1, max(combos)):
    for i in range(num_islands):
        min_dist = sys.maxsize
        for j in range(i+1, num_islands):
            travel = dist_from_start[j] - dist_from_start[i]
            min_dist = min(min_dist, max(travel, memo[c-1][j]))
        memo[c][i] = min_dist

def next_jump(start, max_dist):
    min_i = start+1
    max_i = num_islands-1
    while max_i-min_i > 1:
        mid = math.ceil((min_i + max_i) / 2)
        dist = dist_from_start[mid] - dist_from_start[start]
        
        if dist == max_dist:
            return mid
        if dist < max_dist:
            min_i = mid
        if dist > max_dist:
            max_i = mid
    return min_i
'''

def next_jump(start, max_dist):
    for i in range(start+1, num_islands):
        if dist_from_start[i] - dist_from_start[start] > max_dist:
            return i-1
    return i

def min_jump_dist(num_canisters,
                  min_dist=None,
                  max_dist=dist_from_start[-1]):
    if min_dist is None:
        min_dist = math.ceil(dist_from_start[-1] / num_canisters)-1
    while max_dist-min_dist > 1:
        mid = (min_dist + max_dist) // 2
        
        i = 0
        for _ in range(num_canisters-1):
            if i == num_islands-1 or dist_from_start[i+1] - dist_from_start[i] > mid:
                break
            i = next_jump(i, mid)
        if i == num_islands-1 or dist_to_mainland[i] <= mid:
            max_dist = mid
        else:
            min_dist = mid
    return max_dist

sorted_combos = sorted(combos)
sorted_min_jumps = [None for _ in range(num_combos)]
sorted_min_jumps[-1] = min_jump_dist(sorted_combos[-1])
sorted_min_jumps[0] = min_jump_dist(sorted_combos[0], min_dist=sorted_min_jumps[-1]-1)

def populate_min_jumps(i, j):
    if j-i <= 1:
        return
    mid = (i+j)//2
    if sorted_min_jumps[i] == sorted_min_jumps[j] or sorted_combos[i] == sorted_combos[mid]:
        sorted_min_jumps[mid] = sorted_min_jumps[i]
    sorted_min_jumps[mid] = min_jump_dist(sorted_combos[mid],
                                          min_dist=sorted_min_jumps[j]-1,
                                          max_dist=sorted_min_jumps[i])
    populate_min_jumps(i, mid)
    populate_min_jumps(mid, j)

populate_min_jumps(0, num_combos-1)
min_jumps = dict(zip(sorted_combos, sorted_min_jumps))
for num_canisters in combos:
    print(min_jumps[num_canisters])