import sys

'''
Difficulty: Advanced

Time Complexity:
O(N^3)

Required Knowledge: Maximum flow/Minimum Cut

This problem reduces to a graph minimum cut problem. A cut in a graph with a source node and a sink node is a set of edges which, when removed from the graph, makes the sink unreachable from the source. The minimum cut of a graph is the set of edges among all possible cuts with the minimum size.

Given a minimum cut of a graph, we can construct a graph that fits the requirements of this problem by taking any edge from the minimum cut, and adding it back into the graph. Then, every path from the source to the sink must go through the single edge we added back. It can be shown that this is both correct and optimal.

There are different approaches you can take to find the minimum cut for the graph. Finding the minimum cut of a graph is the same as finding the maximum flow for a graph. A good algorithm to use in this case is the Ford-Fulkerson algorithm. Given that the flow of an edge is either zero or one, the implementation is very simple, and the running time is easy to bound. The min cut of the graph is less than N and the runtime of a single DFS is O(V + E), which is O(N^2) if the graph is fully connected, so the runtime of ford fulkerson is at most O(N^3). Since N is at most 100, this is more than fast enough.
'''

def read_graph():
    global V, E
    V, E = [int(x) for x in sys.stdin.readline().split()]
    graph = [[0 for u in range(V)] for v in range(V)]
    
    for _ in range(E):
        u, v = [int(x) for x in sys.stdin.readline().split()]
        graph[u][v] = 1
    return graph

def bfs(graph):
    parent = [-1 for _ in range(V)]
    visited = [False for _ in range(V)]
    Q = [0]
    visited[0] = True
    
    while len(Q) != 0:
        u = Q.pop(0)
        for v in range(V):
            if not visited[v] and graph[u][v] > 0:
                parent[v] = u
                if v == V-1:
                    return parent
                visited[v] = True
                Q.append(v)
    
    return None
    
def ford_fulkerson(graph):
    res_graph = [[graph[u][v] for v in range(V)] for u in range(V)]
    max_flow = 0
    
    parent = bfs(res_graph)
    while parent:
        v = V-1
        path_flow = sys.maxsize
        while v != 0:
            path_flow = min(path_flow, res_graph[parent[v]][v])
            v = parent[v]
        
        v = V-1
        while v != 0:
            u = parent[v]
            res_graph[v][u] += path_flow
            res_graph[u][v] -= path_flow
            v = u
        
        max_flow += path_flow
        parent = bfs(res_graph)
    
    return max_flow

print(ford_fulkerson(read_graph())-1)