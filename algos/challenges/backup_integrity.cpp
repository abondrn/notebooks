/**
Difficulty: Medium

Time Complexity:
O(n)

Reconstructing a tree of files and directories, given just the list of filenames, is straightforward. So this problem boils down to determining whether two trees are isomorphic a.k.a. their structures are equivalent. Determining isomorphism of arbitrary graphs is quite difficult; fortunately, rooted trees are much easier to deal with. The algorithm described below was first published by Aho, Hopcroft and Ullman in 1974.

We want to come up with a function that takes a node and returns a signature, such that f(A)=f(B) if and only if the subtrees at A and B are isomorphic. Intuitively, the signature should only depend on the structures of the child nodes, but not on their ordering. So we can compute signatures recursively: each node's signature is derived from its children's signatures, sorted into a canonical order. To make the signatures unambiguous, we can use parentheses to represent the way nodes are nested within their parents.

For example, consider the following tree:

![](https://s3.amazonaws.com/hr-assets/0/1508144453-9cf65192d8-graphviz.png)

Let's define the signature of a node with no children to be (). Then the signatures of B and C are simply () and (()()), respectively. To compute the signature of A, we need to assign an ordering to B and C; if we choose to sort them lexicographically, the result is ((()())()).

This simple version of the algorithm isn't terribly efficient. The size of each node's signature is proportional to the size of its subtree, and comparing two signatures costs O(n) time, so the overall time complexity is at least O(n^2 log n). This is fast enough to pass all of the test cases for this problem, but we can do better with a more sophisticated approach.

By calculating signatures with a recursive function, we end up traversing the tree in depth-first order. Instead, do a level-order traversal of the tree, starting at the bottom. After computing all of the signatures at each level, we can sort them into equivalence classes, and assign to each distinct signature a short label that can be compared in O(1). With a suitably-chosen sorting algorithm e.g. radix sort, it can be shown that the entire algorithm can run in linear time.

(A less theoretically elegant, but practically simpler, variant of this idea is to simply hash each signature using a strong, collision-resistant hash function. This is similar to the underlying data structure used by Git.)
*/

#include <iostream>
#include <map>
#include <sstream>
#include <vector>
#include <algorithm>

using namespace std;

struct entry {
    map<string, entry> children;
};

void read_listing(entry& result) {
    int N;
    cin >> N;

    for (int i = 0; i < N; i++) {
        string path;
        cin >> path;

        vector<string> components;
        istringstream iss(path);
        string component;
        while (getline(iss, component, '/')) {
            components.push_back(component);
        }

        entry* current = &result;
        for (int j = 0; j < components.size(); j++) {
            current = &current->children[components[j]];
        }
    }
}

vector<bool> signature(const entry& e, const string& path) {
    vector<vector<bool>> sigs;
    for (const auto& p : e.children) {
        sigs.push_back(signature(p.second, path + "/" + p.first));
    }

    sort(sigs.begin(), sigs.end(), [](const vector<bool>& a, const vector<bool>& b) {
            if (a.size() != b.size()) {
                return a.size() < b.size();
            }
            for (int i = 0; i < a.size(); i++) {
                if (a[i] != b[i]) {
                    return a[i] < b[i];
                }
            }
            return false;
    });

    vector<bool> result;
    result.push_back(1);
    for (const auto& sig : sigs) {
        copy(sig.begin(), sig.end(), back_inserter(result));
    }
    result.push_back(0);

    return result;
}

int main() {
    entry original, backup;
    read_listing(original);
    read_listing(backup);

    vector<bool> original_signature = signature(original, "A");
    vector<bool> backup_signature = signature(backup, "B");

    if (original_signature == backup_signature) {
        cout << "OK" << endl;
    } else {
        cout << "INVALID" << endl;
    }
}