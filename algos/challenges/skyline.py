import sys

num_buildings = int(sys.stdin.readline())
W, H = [], []
for _ in range(num_buildings):
    w, h = [int(x) for x in sys.stdin.readline().split()]
    W.append(w)
    H.append(h)

num_pictures = int(sys.stdin.readline())
ranges = [[int(x) for x in sys.stdin.readline().split()] for _ in range(num_pictures)]
sum_ws = [0 for _ in range(num_pictures)]
max_hs = [0 for _ in range(num_pictures)]
sum_areas = [0 for _ in range(num_pictures)]

def binary_search(lst, item, key):
    l, r = 0, len(lst)
    while r-1 > 1:
        mid = (l + r) // 2
        if key(lst[mid]) > item:
            r = mid
        if key(lst[mid]) < item:
            l = mid
        if key(lst[mid]) == item:
            return mid
    return mid

def aggregate(l, r, rngs, max_depth=7):
    if max_depth > 0 and l - r > 4 and len(rngs) > 10:
        mid = (l + r) // 2
        lo_split = binary_search(rngs, mid, key=lambda i: ranges[i][0])
        while lo_split < r and ranges[lo_split][0] <= mid:
            lo_split += 1
        lo = rngs[:lo_split]
        hi = rngs[lo_split:]
        i = lo_split
        while i >= 0:
            if ranges[i][1] >= mid:
                hi.insert(0, i)
            i -= 1
        #print(mid, lo, hi)
        aggregate(l, mid, lo, max_depth-1)
        aggregate(mid+1, r, hi, max_depth-1)
    else:
        for i in range(l, r+1):
            w, h = W[i], H[i]
            area = W[i]*H[i]
            for j in rngs:
                if ranges[j][0] <= i <= ranges[j][1]:
                    sum_ws[j] += w
                    max_hs[j] = max(max_hs[j], h)
                    sum_areas[j] += area

rngs = sorted(range(num_pictures), key=lambda i: ranges[i])
aggregate(0, num_buildings-1, rngs)

for i in range(num_pictures):
    print('%.3f' % (sum_areas[i] / sum_ws[i] / max_hs[i]))