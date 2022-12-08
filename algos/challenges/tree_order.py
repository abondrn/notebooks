"""
A binary search tree is created by iterating through an array and inserting each element into the binary search tree. Assuming that all elements in the tree are distinct, return all possible arrays that could have resulted in this tree.

Example 1:
    2
   / \
  1   3

Returns

[2, 1, 3], [2, 3, 1]

Example 2:
        5
       / \
      3   7
     /   / 
    1   6   

Returns

[5, 3, 1, 7, 6], [5, 3, 7, 1, 6], [5, 3, 7, 6, 1], [5, 7, 6, 3, 1], [5, 7, 3, 1, 6], [5, 7, 3, 6, 1]
"""
class TreeNode:
    
    def __init__(self, value, left = None, right = None):
        self.value = value
        self.left = left
        self.right = right

def interleave(x, y):
    if len(x) == 0:
        return [y]
    elif len(y) == 0:
        return [x]
    else:
        return [[x[0]] + i for i in interleave(x[1:], y)] + [[y[0]] + i for i in interleave(x, y[1:])]
    
def interleavings(xs, ys):
    ints = []
    if len(xs) == 0:
        return ys
    if len(ys) == 0:
        return xs
    for x in xs:
        for y in ys:
            ints.extend(interleave(x, y))
    return ints

def getSequences(root):
    if root.left is None and root.right is None:
        return [[root.value]]
    if root.left is None:
        left_seqs = []
    else:
        left_seqs = getSequences(root.left)
    if root.right is None:
        right_seqs = []
    else:
        right_seqs = getSequences(root.right)
    return [[root.value] + i for i in interleavings(left_seqs, right_seqs)]

#print(interleave([1, 2], [3, 4]))
#print(interleavings([[1], [2]], [[3], [4]]))

example1 = TreeNode(2, TreeNode(1), TreeNode(3))

def check_tree(root):
    def check_node(seq, node):
        if node is None:
            return True
        my_index = seq.index(node.value)
        if node.left is not None and my_index > seq.index(node.left.value):
                return False
        if node.right is not None and my_index > seq.index(node.right.value):
                return False
        return check_node(seq, node.left) and check_node(seq, node.right)
    seqs = getSequences(root)
    #seqs[0][0], seqs[0][1] = seqs[0][1], seqs[0][0]
    return all(check_node(seq, root) for seq in seqs)

example2 = TreeNode(5, TreeNode(3, TreeNode(1), TreeNode(4)), TreeNode(7, TreeNode(6), TreeNode(8)))
assert check_tree(example2)
print('Finished')