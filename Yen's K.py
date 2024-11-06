from heapq import heappush, heappop
from itertools import count
import networkx as nx

def k_shortest_paths(G, source, target, k, weight='weight'):
    if source == target:
        return ([0], [[source]]) 
       
    length_dict, path_dict = nx.single_source_dijkstra(G, source, weight=weight)
    if target not in length_dict:
        raise nx.NetworkXNoPath("node %s not reachable from %s" % (source, target))
        
    length = length_dict[target]
    path = path_dict[target]

    lengths = [length]
    paths = [path]

    c = count()        
    B = []                        
    G_original = G.copy()    
    
    for i in range(1, k):
        for j in range(len(paths[-1]) - 1):            
            spur_node = paths[-1][j]
            root_path = paths[-1][:j + 1]
            
            edges_removed = []
            for c_path in paths:
                if len(c_path) > j and root_path == c_path[:j + 1]:
                    u = c_path[j]
                    v = c_path[j + 1]
                    if G.has_edge(u, v):
                        edge_attr = G.edges[u,v]
                        G.remove_edge(u, v)
                        edges_removed.append((u, v, edge_attr))
            
            for n in range(len(root_path) - 1):
                node = root_path[n]
                edges_to_remove = []
                for u, v, edge_attr in G.edges(node, data=True):
                    edges_to_remove.append((u, v))
                    edges_removed.append((u, v, edge_attr))

                for u, v in edges_to_remove:
                    G.remove_edge(u, v)
        
            try:
                spur_path_length, spur_path = nx.single_source_dijkstra(G, spur_node, target, weight=weight)
            except nx.NetworkXNoPath:
                continue
          
            if target in spur_path:
                total_path = root_path[:-1] + spur_path
                total_path_length = get_path_length(G_original, total_path, weight)
             
                # Add this line before you push a new path onto the heap
                if total_path not in paths:
                    heappush(B, (total_path_length, next(c), total_path))
                
            for e in edges_removed:
                u, v, edge_attr = e
                G.add_edge(u, v, **edge_attr)
                       
        if B:
            (l_, _, p_) = heappop(B)        
            lengths.append(l_)
            paths.append(p_)
        else:
            break
    
    # This line ensures that all paths in your output are unique while preserving the original order
    seen = set()
    unique_paths = [x for x in paths if not (tuple(x) in seen or seen.add(tuple(x)))]

    # This line gets the lengths corresponding to the unique paths
    unique_lengths = [lengths[paths.index(path)] for path in unique_paths]

    print(unique_lengths)
    return (unique_lengths, unique_paths)




def get_path_length(G, path, weight='weight'):
    length = 0
    if len(path) > 1:
        for i in range(len(path) - 1):
            u = path[i]
            v = path[i + 1]
            
            length += G.edges[u,v].get(weight, 1)
    
    return length

G = nx.Graph()
k = 12

with open("X:/Haoxuan Xu's Documents (BS)/School Study/Computer Science/Year 12&13/NEA/Code/edges.txt",'r', encoding='utf-8') as f:
    for line in f:
        station1, station2, weight = line.strip().split(',')
        G.add_edge(str(station1), str(station2), length=int(weight))

lengths, paths = k_shortest_paths(G, 'A', 'E', k, "length")
for i in range(len(paths)):
    print(f"Path {i+1}: {paths[i]}, Length: {lengths[i]}")