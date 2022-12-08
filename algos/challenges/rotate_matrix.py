# Question 1: Please implement a function to rotate a 2D matrix clock wisely for 90 degrees.

def rotate_matrix(mat):
    rows, cols = len(mat), len(mat[0])
    rot = [[0]*rows for i in range(cols)]
    for c in range(cols):
        for r in range(rows):
            rot[c][rows-r-1] = mat[r][c]
    return rot

# Things you can do to break the code
# 1. pass in an empty list
# 2. pass in things which can't indexed in two dimensions, ie integers, 1d lists
# 3. pass in a list of lists, where not every list is the same length
# 4. pass in objects which have the same interface as a 2d list
# 5. pass in a list of lists, where some lists are shallow copies

# Question 2: If the matrix is a graph. And each element is a Node
# 1. Define the data structure of Node
# 2. Implement a function to rotate the matrix clock wisely for 90 degrees
'''
1 - 2 - 3
|.  |.  |
4 - 5 - 6
|.  |.  |
7 - 8 - 9
Input is Node(1)
Expected:
7 - 4 - 1
|.  |.  |
8 - 5 - 2
|.  |.  |
9 - 6 - 3
Output is Node(7)
'''

class Node:
    
    def __init__(self):
        self.N = self.S = self.E = self.W = None

def matrix_to_node(mat):
    '''Given mat, create an equivalent graph, and return the node to the top left'''
    pass

def rotate_node(node):
    '''Rotate the matrix represented by node's graph, return new top left'''
    top_left = node
    while top_left.S is not None:
        top_left = top_left.S
    Q = [(0, 0)]
    coords = {(0, 0): node}
    visited = set()
    while len(Q) != 0:
        c = Q.pop(0)
        n = coords[c]
        visited[c] = True
        south = (c[0]+1, c[1])
        east = (c[0], c[1]+1)
        if n.S is not None and south not in visited:
            coords[south] = n.S
            Q.append(south)
        if n.E is not None and east not in visited:
            coords[east] = n.E
            Q.append(east)
        n.N, n.E, n.S, n.W = n.E, n.S, n.W, n.N
    return top_left