import time

import sys, itertools

num_words = int(sys.stdin.readline())
words = []
for _ in range(num_words):
    words.append(sys.stdin.readline().rstrip('\n'))
    
def largest_overlap(s1,s2):
    n = min(len(s1),len(s2))
    if s2 in s1:
        return n
    for i in range(n,0,-1):
        if s2.startswith(s1[-i:]):
            return i
    return 0

def join_words(s1, s2, o):
    return s1 + s2[o:]

flex_length = 0

reduced = True
while reduced:
    reduced = False
    dist = [[0 for j in range(num_words)] for i in range(num_words)]
    for i in range(num_words):
        for j in range(num_words):
            if i != j:
                dist[i][j] = largest_overlap(words[i], words[j])
    max_match = max(map(max, dist))
    for i in range(num_words):
        for j in range(num_words):
            if i != j and dist[i][j] == max_match \
            and all(dist[i2][j] <= max_match for i2 in range(num_words) if i2 != i) \
            and all(dist[i][j2] <= max_match for j2 in range(num_words) if j2 != j):
                new_word = join_words(words[i], words[j], dist[i][j])
                i, j = sorted([i, j])
                words.pop(j)
                words.pop(i)
                words.append(new_word)
                num_words -= 1
                reduced = True
                break
        if reduced:
            break
    '''
    if not reduced and num_words != 1:
        for i in range(num_words)[::-1]:
            prefix_score = dist[(i+1)%num_words][i]
            postfix_score = dist[i][(i+1)%num_words]
            if all(dist[j][i] == prefix_score for j in range(num_words) if j != i) \
            and all(dist[i][j] == prefix_score for j in range(num_words) if j != i):
                flex_length += len(words[i])-prefix_score-postfix_score
                words.pop(i)
                num_words -= 1
                reduced = True
    '''

if len(words) == 1:
    print(flex_length+len(words[0]))
else:
    min_length = sys.maxsize
    for perm in itertools.permutations(range(num_words)):
        order = list(perm)
        combined = words[order[0]]
        prev = order[0]
        for i in order[1:]:
            #match = dist[prev][i]
            #if len(words[prev]) == match:
            match = largest_overlap(combined, words[i])
            combined = join_words(combined, words[i], match)
            prev = i
        length = len(combined)
        if length < min_length:            
            min_length = length
    print(flex_length+min_length)