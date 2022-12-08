/**
Difficulty: Medium

Time Complexity:
O(n*2^n)

The most extreme brute-force approach to this problem would be to enumerate all possible strings, and find the shortest one that contains all of the inputs. That's obviously way too slow, so how can we speed it up?

Consider the case where N=2. The optimal result string that we're looking for must contain both input words, so each of them must correspond to a particular substring, or interval. Any letters outside these two intervals are unnecessary and can be removed.

There are three possible cases:
 - The input words are non-overlapping, so the result length is simply the sum of the two input lengths.
 - One input word is a substring of the other; the result is equal to the longer (containing) word.
 - The two words are partially overlapping, so a prefix of one is equal to a suffix of the other; the result is equal to the sum of the two input lengths, minus the length of the common substring.
Let's extend this to N>2. First, if any words are substrings of any other words, we can remove them without affecting the optimal result. Then the left endpoints of the intervals of the remaining words must all be distinct. It's sufficient to find the best possible ordering of the intervals; once we have that, it's easy enough to find the offsets of each string that give us the maximum possible overlap. (Each word can only overlap with one immediate neighbor in each direction, because otherwise it would contain at least one of its neighbors, and we've assumed any such strings have already been removed.)


Since N<=10, there are at most 10! = 3628800 possible orderings, which is small enough that we can try them by brute force. One possible approach is to solve the problem in two stages. First, we loop over all ordered pairs of words, and for each pair find the largest possible overlap. A naive approach runs in O(N^2 * L^2) where L is the maximum input word length; this is easily fast enough. Then, iterate over all permutations of the words (e.g. by permuting a list of indexes into an array of words), and for each permutation, subtract the sum of the overlaps from the sum of the lengths.

(C++ and Python have standard library functions to iterate through all permutations of a sequence in lexicographic order. If your preferred language doesn't include such a function, it's worth [adding it to your algorithmic toolbox](https://en.wikipedia.org/wiki/Permutation#Algorithms_to_generate_permutations).)

This approach is fast enough to pass all of the test cases, but it's possible to do better. In the first stage, we computed a matrix of overlaps between all pairs of words; we could instead represent it as a matrix of "costs", indicating how many extra characters must be added to append the next word to the string. Then our cost matrix is the adjacency matrix of a graph, and the solution is the shortest Hamiltonian path through that graph, starting at a special "empty string" vertex. In other words -- the traveling salesman problem! We can use a dynamic programming to solve the TSP in O(n * 2^n), which lets us handle moderately larger test cases (perhaps N<25 or so).
*/

#include <iostream>
#include <string>
#include <algorithm>
#include <vector>
#include <numeric>
#include <set>
#include <limits>

using namespace std;

int main() {
    int N;
    cin >> N;

    vector<string> words(N);
    for (int i = 0; i < N; i++) {
        cin >> words[i];
    }

    // remove duplicates
    set<string> uniqueWordSet;
    for (int i = 0; i < N; i++) {
        bool unique = true;
        for (int j = 0; j < N; j++) {
            if (i == j) continue;
            if (words[i].length() != words[j].length() && words[j].find(words[i]) != string::npos) {
                unique = false;
                break;
            }
        }
        if (unique) uniqueWordSet.insert(words[i]);
    }

    vector<string> uniqueWords;
    for (auto& s : uniqueWordSet) {
        uniqueWords.push_back(s);
    }
    const int M = uniqueWords.size();

    vector<vector<int>> offsets(M, vector<int>(M));
    for (int i = 0; i < M; i++) {
        const string& x = uniqueWords[i];

        for (int j = 0; j < M; j++) {
            if (i == j) continue;
            const string& y = uniqueWords[j];
            
            int maxOverlap = 0;
            for (int overlap = 1; overlap < min(x.length(), y.length()); overlap++) {
                if (x.substr(x.length() - overlap) == y.substr(0, overlap)) {
                    maxOverlap = overlap;
                }
            }
            offsets[i][j] = y.length() - maxOverlap;
        }
    }

    vector<int> permutation(M);
    iota(permutation.begin(), permutation.end(), 0);

    int minCost = numeric_limits<int>::max();
    do {
        int total = uniqueWords[permutation[0]].length();
        for (int i = 1; i < M; i++) {
            total += offsets[permutation[i-1]][permutation[i]];
        }
        minCost = min(minCost, total);
    } while (next_permutation(permutation.begin(), permutation.end()));

    cout << minCost << endl;
}