import sys

'''
Difficulty: Easy

Time Complexity:
O(N)

To order the houses left from right, you need to scan all of the next door neighbor pairs and piece them together. One reasonable approach is to construct a hash map with the left neighbors as keys and the right neighbors as values. This can be done in O(N) time. Then you can find the left-most house on the street in O(N) time by finding the house number that is missing from the values in your hash map. Alternatively, you could do this during the initial hash map construction. In either case, this is accomplished by keeping a list of houses with no left neighbor, then iterating over all the next door neighbor pairs and removing the right neighbors from your list. You should be left with a singleton list containing only the left-most house, since it's the only house that is not a right neighbor. Finally, you can simply iterate over the houses starting from the left-most house. At each iteration, you can find the next house by looking up the last house in your hash map. The returned value should be the next house on the street.
'''

import sys

num_houses = int(sys.stdin.readline())

# right of
neighbor_pairs = dict()
# have no left
left_neighborless = set(range(num_houses))

for pair in range(num_houses-1):
    l, r = [int(j) for j in sys.stdin.readline().split()]
    left_neighborless.remove(r)
    neighbor_pairs[l] = r

assert len(left_neighborless) == 1

leftmost = left_neighborless.pop()
print(leftmost)
while leftmost in neighbor_pairs:
    leftmost = neighbor_pairs[leftmost]
    print(leftmost)
