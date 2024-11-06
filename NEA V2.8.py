import sys
import time
import warnings
import networkx as nx
import tkinter as tk
import re
from tkinter import Canvas, ttk, messagebox
from itertools import count
from PIL import Image, ImageTk

class Node:
    def __init__(self, name):
        self.name = name  # Name of the node
        self.value = sys.maxsize  # Initialize the value of the node to infinity
        self.previous_node = None  # The node that precedes this node in the path
        self.edges = []  # List of edges connected to this node

    def add_edge(self, edge):
        self.edges.append(edge)  # Add an edge to the list of edges connected to this node

class ListNode:
    def __init__(self, data=None):
        self.data = data  # Data stored in the node
        self.next = None  # The next node in the linked list

class Edge:
    def __init__(self, start_node, end_node, weight):
        self.start_node = start_node  # The node where the edge starts
        self.end_node = end_node  # The node where the edge ends
        self.weight = weight  # The weight of the edge

class PriorityQ:
    def __init__(self):
        self.queue = []  # Initialize an empty queue

    def push_item(self, priority, item):
        self.queue.append((priority, item))  # Add an item to the queue with its priority
        self.queue.sort(reverse=True)  # Sort the queue based on priority

    def pop_item(self):
        return self.queue.pop()  # Remove and return the highest priority item from the queue

class Graph:
    def __init__(self, nodes, edges):
        self.nodes = {}  # Dictionary to store the nodes in the graph
        self.node_names = nodes  # List of node names
        self.edges = {}  # Dictionary to store the edges in the graph
        self.adjacency_matrix = self.construct_adjacency_matrix(nodes, edges)  # Adjacency matrix of the graph
        for node_name in nodes:
            self.nodes[node_name] = Node(node_name)  # Create a Node object for each node name
        for edge in edges:
            start_node, end_node, weight = edge
            self.add_edge(start_node, end_node, weight)  # Add an edge to the graph

    def construct_adjacency_matrix(self, nodes, edges):
        # Construct the adjacency matrix of the graph.
        n = len(nodes)
        matrix = [[float('inf')] * n for _ in range(n)]  # Initialize the adjacency matrix with all entries as infinity.
        for i in range(n):
            matrix[i][i] = 0  # The distance from a node to itself is 0.
        for edge in edges:
            start_node, end_node, weight = edge
            i = nodes.index(start_node)  # Find the index of the start node.
            j = nodes.index(end_node)  # Find the index of the end node.
            matrix[i][j] = weight  # Set the entry in the i-th row and j-th column to the weight of the edge.
            matrix[j][i] = weight  # Since the graph is undirected, set the entry in the j-th row and i-th column to the weight of the edge.
        return matrix

    def add_edge(self, start_node, end_node, weight):
        # Add an edge to the graph.
        edge = Edge(self.nodes[start_node], self.nodes[end_node], weight)  # Create an Edge object.
        self.nodes[start_node].add_edge(edge)  # Add the edge to the start node.
        self.nodes[end_node].add_edge(edge)  # Add the edge to the end node.

        if start_node not in self.edges:
            self.edges[start_node] = []  # If the start node is not already in the dictionary of edges, add it.
        if end_node not in self.edges:
            self.edges[end_node] = []  # If the end node is not already in the dictionary of edges, add it.
        
        self.edges[start_node].append((end_node, weight))  # Add the edge to the list of edges of the start node.
        self.edges[end_node].append((start_node, weight))  # Add the edge to the list of edges of the end node.
    
    def dijkstra_algorithm(self, start_node):  # Implementation of Dijkstra's algorithm.
        unvisited_nodes = list(self.nodes.values())  # List of nodes that have not been visited yet.
        shortest_path = {}  # Dictionary to store the shortest path from the start node to each node. Each key-value pair is a node name and the length of the shortest path to that node.
        previous_nodes = {}  # Dictionary to store the previous node in the shortest path to each node. Each key-value pair is a node name and the name of its previous node in the shortest path.
        visited_nodes = []  # List to store the nodes that have been visited.
        max_value = sys.maxsize  # The maximum possible value. This is used to initialize the shortest path lengths.
        for node in unvisited_nodes:
            shortest_path[node.name] = max_value  # Initialize the shortest path to each node as infinity.
        shortest_path[start_node] = 0  # The shortest path from the start node to itself is 0.
        self.nodes[start_node].value = 0  # Set the value of the start node to 0.

        while unvisited_nodes:  # While there are still nodes to be visited,
            current_min_node = min(unvisited_nodes, key=lambda node: node.value)  # Find the unvisited node with the smallest value.
            unvisited_nodes.remove(current_min_node)  # Remove the node with the smallest value from the list of unvisited nodes.

            visited_nodes.append(current_min_node.name)  # Add the node with the smallest value to the list of visited nodes.

            for edge in current_min_node.edges:  # For each edge connected to the current node,
                neighbor = edge.end_node if edge.start_node == current_min_node else edge.start_node  # Find the node at the other end of the edge.
                tentative_value = current_min_node.value + edge.weight  # Calculate the tentative value of the neighbor.

                if tentative_value < neighbor.value:  # If the tentative value is less than the current value of the neighbor,
                    neighbor.value = tentative_value  # Update the value of the neighbor.
                    neighbor.previous_node = current_min_node  # Set the current node as the previous node of the neighbor.
                    previous_nodes[neighbor.name] = current_min_node.name  # Update the previous node of the neighbor in the dictionary of previous nodes.
                    shortest_path[neighbor.name] = tentative_value  # Update the shortest path to the neighbor.

        return visited_nodes, previous_nodes, shortest_path  # Return the list of visited nodes, the dictionary of previous nodes, and the dictionary of shortest paths.

    def get_shortest_path_dijkstra(self, source, target):
        # Get the shortest path from the source to the target using Dijkstra's algorithm.
        visited_nodes, previous_nodes, shortest_path = self.dijkstra_algorithm(source)  # Run Dijkstra's algorithm.
        path = [target]  # Initialize the path with the target node.
        target1 = target
        while target1 != source:  # While the target is not the source,
            target1 = previous_nodes[target1]  # Move to the previous node in the shortest path.
            path.append(target1)  # Add the new target to the path.
        path.reverse()  # Reverse the path to get the path from source to target.
        return path, shortest_path[target], visited_nodes  # Return the path, the shortest path length, and the visited nodes.

    def floyd_warshall(self):
        # Implementation of Floyd-Warshall algorithm.
        n = len(self.adjacency_matrix)  # The number of nodes in the graph.
        dist = [row[:] for row in self.adjacency_matrix]  # Copy the adjacency matrix to the distance matrix. This matrix will be updated to store the shortest distances between every pair of nodes.
        next_node = [[0]*n for _ in range(n)]  # Initialize the next node matrix. This matrix is used to reconstruct the shortest paths.
        node_list = []  # List to store the pairs of nodes between which the shortest path has been found.
        for i in range(n):  # For each node,
            for j in range(n):  # and for each other node,
                if i != j and dist[i][j] != float('inf'):  # if the other node is not the same as the current node and there is an edge between them,
                    next_node[i][j] = j  # Set the next node from the i-th node to the j-th node as the j-th node.
        for k in range(n):  # For each node as an intermediate node,
            for i in range(n):  # and for each node as a start node,
                for j in range(n):  # and for each node as an end node,
                    if dist[i][j] > dist[i][k] + dist[k][j]:  # if the direct distance from the i-th node to the j-th node is greater than the distance from the i-th node to the j-th node through the k-th node,
                        dist[i][j] = dist[i][k] + dist[k][j]  # Update the distance from the i-th node to the j-th node as the distance from the i-th node to the j-th node through the k-th node.
                        next_node[i][j] = next_node[i][k]  # Update the next node from the i-th node to the j-th node as the next node from the i-th node to the k-th node.
                        node_list.append((self.node_names[i], self.node_names[j]))  # Add the pair of nodes to the list of node pairs.
        return dist, next_node, node_list  # Return the distance matrix, the next node matrix, and the list of node pairs.

    def reconstruct_path_floyd(self, start_node_name, target_node_name, next_node):
        # Reconstruct the shortest path from the start node to the target node using the Floyd-Warshall algorithm.
        source = self.node_names.index(start_node_name)  # Find the index of the start node.
        target = self.node_names.index(target_node_name)  # Find the index of the target node.
        path = [source]  # Start the path with the source.
        while source != target:  # While the source is not the target,
            source = next_node[source][target]  # Move to the next node in the path from the source to the target.
            path.append(source)  # Add the next node to the path.
        return path  # Return the path as a list.
    
    def heuristic(self, node1, node2):
        # This is a heuristic function used in A* algorithm. 
        # In this case, it always returns 0 due to the 2d and simplicity behaviour, making A* behave like Dijkstra's algorithm.
        return 0

    def a_star(self, start_node, target_node):
        # Convert the start_node and target_node from names to Node objects.
        start_node = self.nodes[start_node]
        target_node = self.nodes[target_node]
        # Create a priority queue to store nodes to be visited.
        open_list = PriorityQ() 
        # Initialize a counter for tie-breaking.
        count = 0  
        # Push the start_node into the priority queue with a priority of 0.
        open_list.push_item(0, (count, start_node))  
        # Create dictionaries to store the best path found so far to each node, and the node that was visited immediately before it on that path.
        came_from = {} 
        g_score = {node: float('inf') for node in self.nodes.values()} 
        g_score[start_node] = 0 
        # Create a dictionary to store the estimated total cost from start_node to each node.
        f_score = {node: float('inf') for node in self.nodes.values()}  
        # The estimated cost from start_node to itself is just the heuristic estimate.
        f_score[start_node] = self.heuristic(start_node, target_node) 
        # Create a list to store the nodes that have been visited.
        visited_nodes = []
        # While there are still nodes to be visited,
        while open_list.queue:  
            # Pop the node with the highest priority (lowest f_score) from the priority queue.
            _, (count, current) = open_list.pop_item() 
            # Add the current node to the list of visited nodes.
            visited_nodes.append(current.name)
            # If the current node is the target_node, then we have found a shortest path.
            if current == target_node:  
                # Create a new Node for the target and set the current node to the target node.
                path = ListNode(current)  
                current_node = path  
                # While the current node has a predecessor,
                while current in came_from:  
                    # Move to the predecessor of the current node.
                    current = came_from[current]  
                    # Create a new Node for the predecessor and link it to the current path.
                    new_node = ListNode(current)  
                    new_node.next = current_node  
                    current_node = new_node  
                # Convert the linked list to a list.
                path_list = []
                while current_node is not None:
                    path_list.append(current_node.data)
                    current_node = current_node.next
                # Return the path, the total cost of the path, and the visited nodes.
                return path_list, f_score[target_node], visited_nodes
            # For each edge connected to the current node,
            for edge in current.edges:  
                # Find the node at the other end of the edge.
                neighbor = edge.end_node if edge.start_node == current else edge.start_node  
                # Calculate the tentative g_score for the neighbor.
                tentative_g_score = g_score[current] + edge.weight  
                # If the tentative g_score is less than the current g_score of the neighbor,
                if tentative_g_score < g_score[neighbor]: 
                    # Update the came_from and g_score of the neighbor.
                    came_from[neighbor] = current 
                    g_score[neighbor] = tentative_g_score  
                    # Update the f_score of the neighbor.
                    f_score[neighbor] = g_score[neighbor] + self.heuristic(neighbor, target_node) 
                    # If the neighbor is not in the open_list, add it.
                    if neighbor not in [i[1][1] for i in open_list.queue]: 
                        count += 1  
                        open_list.push_item(f_score[neighbor], (count, neighbor))  
        # If all nodes have been visited and no path has been found, return None.
        return None, float('inf'), visited_nodes

    def k_shortest_paths(self, G, source, target, k, weight='weight'):
        # If the source and target are the same, return a path of length 0
        if source == target:
            return ([0], [[source]]) 
        # Compute the shortest path from the source to all other nodes in the graph
        length_dict, path_dict = nx.single_source_dijkstra(G, source, weight=weight)
        # If the target is not reachable from the source, raise an exception
        if target not in length_dict:
            raise nx.NetworkXNoPath("node %s not reachable from %s" % (source, target))
        # Initialize the list of shortest paths and their lengths
        length = length_dict[target]
        path = path_dict[target]
        lengths = [length]
        paths = [path]
        # Initialize a counter for the number of paths found
        c = count()        
        # Initialize a list to store the potential kth shortest paths
        B_list = []                    
        # Make a copy of the original graph
        G_original = G.copy()    
        # Initialize a list to store the nodes visited during the search
        visited_nodes = []
        # Initialize a set to store the paths that have been seen
        seen_paths = set()
        # Start the main loop to find the k shortest paths
        for i in range(1, k):
            # Make a copy of the graph for each iteration
            G_copy = G.copy() 
            # Loop over each node in the last path found
            for j in range(len(paths[-1]) - 1):            
                # The spur node is the current node in the path
                spur_node = paths[-1][j]
                # The root path is the part of the path up to the spur node
                root_path = paths[-1][:j + 1]
                # Initialize a list to store the edges that are removed from the graph
                edges_removed = []
                # Loop over each path in the list of paths
                for c_path in paths:
                    # If the current path shares the same root path
                    if len(c_path) > j and root_path == c_path[:j + 1]:
                        # Remove the edge from the spur node to the next node in the path
                        u = c_path[j]
                        v = c_path[j + 1]
                        if G_copy.has_edge(u, v): 
                            edge_attr = G_copy.edges[u,v]
                            G_copy.remove_edge(u, v)  
                            edges_removed.append((u, v, edge_attr))
                # Remove all edges from the root path nodes to any other nodes
                for n in range(len(root_path) - 1):
                    node = root_path[n]
                    edges_to_remove = []
                    for u, v, edge_attr in G_copy.edges(node, data=True):  
                        edges_to_remove.append((u, v))
                        edges_removed.append((u, v, edge_attr))
                    for u, v in edges_to_remove:
                        G_copy.remove_edge(u, v)  
                # Compute the shortest path from the spur node to the target in the modified graph
                try:
                    spur_path_length, spur_path = nx.single_source_dijkstra(G_copy, spur_node, target, weight=weight)  
                    # Combine the root path and the spur path to get a new candidate path
                    total_path_tuple = tuple(root_path[:-1] + spur_path)
                    # If the new path has not been seen before, add it to the set of seen paths
                    if total_path_tuple not in seen_paths:
                        visited_nodes.append(spur_node)
                        seen_paths.add(total_path_tuple)
                # If there is no path from the spur node to the target, continue with the next node
                except nx.NetworkXNoPath:
                    continue
                # If the target is in the spur path, add the total path to the list of candidate paths
                if target in spur_path:
                    total_path = root_path[:-1] + spur_path
                    total_path_length = graph.get_path_length_yen(G_original, total_path, weight)
                    if total_path not in paths:
                        B_list.append((total_path_length, next(c), total_path))
                        B_list.sort(reverse=True)
                # Add back the edges that were removed from the graph
                for e in edges_removed:
                    u, v, edge_attr = e
                    G.add_edge(u, v, **edge_attr)
            # If there are candidate paths, add the shortest one to the list of shortest paths
            if B_list:
                (l_, _, p_) = B_list.pop()        
                lengths.append(l_)
                paths.append(p_)
            else:
                break
        # Remove duplicate paths from the list of shortest paths
        seen = set()
        unique_paths = [x for x in paths if not (tuple(x) in seen or seen.add(tuple(x)))]
        # Compute the lengths of the unique shortest paths
        unique_lengths = [lengths[paths.index(path)] for path in unique_paths]
        # Return the lengths and paths of the k shortest paths, and the nodes visited during the search
        return unique_lengths, unique_paths, visited_nodes

    def get_path_length_yen(self, G, path, weight='weight'):
        # Initialize the length of the path to 0
        length = 0
        # If the path has more than one node
        if len(path) > 1:
            # Loop over each pair of consecutive nodes in the path
            for i in range(len(path) - 1):
                u = path[i]
                v = path[i + 1]
                # Add the weight of the edge from u to v to the length of the path
                length += G.edges[u,v].get(weight, 1)
        # Return the length of the path
        return length

    def bellman_ford(self, source):
        # Initialize the distances from the source to all nodes as infinity,and the predecessor of each node as None.
        distances = {vertex: float('inf') for vertex in self.node_names}
        predecessors = {vertex: None for vertex in self.node_names}
        # The distance from the source to itself is 0.
        distances[source] = 0
        # Create a list to store the nodes that have been visited.
        node_list = []
        # For each node, apply relaxation for all the edges.
        for _ in range(len(self.node_names) - 1):
            for u in self.node_names:
                if u in self.edges:
                    for v, weight in self.edges[u]:
                        if distances[u] + weight < distances[v]:
                            distances[v] = distances[u] + weight
                            predecessors[v] = u
                            # Add the node to the list of visited nodes.
                            node_list.append(v)
        # After |V| - 1 iterations, check for negative-weight cycles.
        for u in self.node_names:
            if u in self.edges:
                for v, weight in self.edges[u]:
                    if distances[u] + weight < distances[v]:
                        print("Negative cycle detected")
                        return None, None, node_list
        # If no negative-weight cycles are found, return the shortest distances and predecessors.
        return distances, predecessors, node_list

    def get_shortest_path_bellman(self, source, destination):
        # Run the Bellman-Ford algorithm from the source node, which returns a dictionary of shortest distances from the source to each node, a dictionary of predecessors for each node, and a list of nodes that have been visited.
        distances, predecessors, node_dict = self.bellman_ford(source)
        # Initialize the total weight of the path to 0.
        total_weight = 0
        # Initialize the path with the destination node.
        path_list = [destination]
        # While the destination node is not the source node,
        while destination != source:
            # If the destination node does not have a predecessor, it means that there is no path from the source to the destination.
            # In this case, return None for both the path and the total weight.
            if destination not in predecessors:
                return None, None
            # Add the weight of the edge from the predecessor of the destination node to the destination node to the total weight.
            # The weight is found by iterating over the edges of the predecessor node and finding the one that leads to the destination node.
            total_weight += next(wt for dest, wt in self.edges[predecessors[destination]] if dest == destination)
            # Move to the predecessor of the destination node.
            destination = predecessors[destination]
            # Add the new destination node (which is the predecessor of the old destination node) to the path.
            path_list.append(destination)
        # Reverse the path to get the path from the source to the destination.
        path_list.reverse()
        # Return the path and the total weight of the path.
        return path_list, total_weight

    def run_algorithm_others(self, algorithm, start_node, target_node, k=None, G=None):
        # Initialize the result and node_list to None
        result = None
        node_list = None
        # Record the start time
        start_time = time.time() 
        # Run the specified algorithm
        if algorithm == "A*":  
            path, distance, node_list = self.a_star(start_node, target_node)  
            result = path, distance  
        elif algorithm == "Floyd-Warshall": 
            shortest_distance_matrix, next_node, node_list = self.floyd_warshall() 
            path = self.reconstruct_path_floyd(start_node, target_node, next_node)
            path = [self.node_names[node] for node in path]
            distance = shortest_distance_matrix[self.node_names.index(start_node)][self.node_names.index(target_node)]  
            result = path, int(distance)
        elif algorithm == "Bellman-Ford":  
            distances, predecessors, node_list= self.bellman_ford(start_node)
            path, distance = self.get_shortest_path_bellman(start_node, target_node)
            result = path, distance
        elif algorithm == "Dijkstra": 
            path, distance, node_list = self.get_shortest_path_dijkstra(start_node, target_node)
            result = path, distance
        elif algorithm == "Yen's K":
            distance_all, path_all, node_list = graph.k_shortest_paths(G, start_node, target_node, k, "length")
            if distance_all == None and path_all == None:
                result = None,None
            if len(path_all)<k:
                result = None,None
            else:
                path = path_all[-1]
                distance = distance_all[-1]
                result = path, distance
        # Record the end time
        end_time = time.time()  
        # Calculate the time taken
        time_taken = end_time - start_time
        # Return the result, the time taken, and the list of nodes visited
        return result, time_taken, node_list

    def ordinal(self, n):
        # Define the list of suffixes. The index in the list corresponds to the last digit of 'n'.
        suffix = ['th', 'st', 'nd', 'rd', 'th'][min(n % 10, 4)]
        # Special case for 11, 12, 13 where the suffix is always 'th'
        if 11 <= (n % 100) <= 13:
            suffix = 'th'
        # Return the ordinal representation of 'n'
        return str(n) + suffix

    def print_result_others(self, algorithm, result, time_taken, k=None):
        if algorithm == "Yen's K":
            k_number = self.ordinal(k)
            path, distance = result
            if path == None and distance == None:
                messagebox.showinfo("No Path", "There is no such {} path".format(k_number))
            else:
                messagebox.showinfo("Result", "We found the following best {} path with a value of {}.\nPath: {}\nYen's K algorithm took {} seconds.".format(k_number, distance, " -> ".join(map(str, path)), time_taken))
        else:
            path, distance = result
            if algorithm == "A*":
                messagebox.showinfo("Result", "We found the following best path with a value of {}.\nPath: {}\n{} algorithm took {} seconds.".format(distance, " -> ".join([node.name for node in path]), algorithm, time_taken))
            else:
                messagebox.showinfo("Result", "We found the following best path with a value of {}.\nPath: {}\n{} algorithm took {} seconds.".format(distance, " -> ".join(path), algorithm, time_taken))

class AutocompleteCombobox(ttk.Combobox):
    # This class extends the ttk.Combobox class to provide autocomplete functionality.

    def __init__(self, *args, **kwargs):
        # Initialize the AutocompleteCombobox.
        super().__init__(*args, **kwargs)  # Call the parent class's initializer.
        self._completion_list = []  # List of possible completions.
        self._hits = []  # List of current matches for the entered text.
        self._hit_index = 0  # Index of the currently selected match.
        self.position = 0  # Position of the cursor in the text field.
        self.bind('<KeyRelease>', self.handle_keyrelease)  # Bind the key release event to the handle_keyrelease method.
        self.bind('<Button-1>', self.display_options)  # Bind the left mouse button click event to the display_options method.
        self['values'] = []  # Initialize the list of values displayed in the dropdown menu.

    def set_completion_list(self, completion_list):
        # Set the list of possible completions.
        self._completion_list = sorted(completion_list, key=str.lower)  # Sort the list of completions.
        self['values'] = self._completion_list  # Set the list of values displayed in the dropdown menu.

    def autocomplete(self, delta=0):
        # Perform autocomplete based on the entered text.
        if delta:
            self.delete(self.position, tk.END)  # Delete the text from the current position to the end.
        else:
            self.position = len(self.get())  # Set the current position to the end of the entered text.
        _hits = []  # List of current matches for the entered text.
        for element in self._completion_list:
            if element.lower().startswith(self.get().lower()):  # If the element starts with the entered text,
                _hits.append(element)  # add it to the list of matches.
        if _hits != self._hits:  # If the list of matches has changed,
            self._hit_index = 0  # reset the index of the currently selected match.
            self._hits=_hits  # Update the list of matches.
        if _hits == self._hits and self._hits:  # If the list of matches hasn't changed and isn't empty,
            self._hit_index = (self._hit_index + delta) % len(self._hits)  # update the index of the currently selected match.
        if self._hits:  # If there are any matches,
            self.delete(0,tk.END)  # delete the entered text,
            self.insert(0,self._hits[self._hit_index])  # insert the currently selected match,
            self.select_range(self.position,tk.END)  # and select the text from the current position to the end.

    def handle_keyrelease(self, event):
        # Handle the key release event.
        if event.keysym == "BackSpace":  # If the BackSpace key was released,
            self.delete(self.index(tk.INSERT), tk.END)  # delete the text from the current position to the end.
            self.position = self.index(tk.END)  # Set the current position to the end of the entered text.
        if event.keysym == "Left":  # If the Left arrow key was released,
            if self.position < self.index(tk.END):  # if the current position is before the end of the entered text,
                self.delete(self.position, tk.END)  # delete the text from the current position to the end.
            else:  # Otherwise,
                self.position = self.position-1  # move the current position one character to the left,
                self.delete(self.position, tk.END)  # and delete the text from the current position to the end.
        if event.keysym == "Right":  # If the Right arrow key was released,
            self.position = self.index(tk.END)  # set the current position to the end of the entered text.
        if len(event.keysym) == 1:  # If a character key was released,
            self.autocomplete()  # perform autocomplete.

    def display_options(self, event):
        # Display the dropdown menu options.
        if not self.get():  # If no text has been entered,
            self.unbind('<KeyRelease>')  # unbind the key release event from the handle_keyrelease method.
            backup = list(self["values"])  # Make a backup of the list of values displayed in the dropdown menu.
            self["values"] = backup  # Restore the list of values displayed in the dropdown menu.
            statebak = str(self.cget("state"))  # Make a backup of the state of the combobox.
            startbak = str(self.get())  # Make a backup of the entered text.
            startidxbak = int(self.current())  # Make a backup of the index of the currently selected option.
            startibak = int(self.index(tk.INSERT))  # Make a backup of the current position.
            startebak = int(self.index(tk.END))  # Make a backup of the end of the entered text.
            try:
                self["state"] = "readonly"  # Set the state of the combobox to readonly.
                self.event_generate("<Down>")  # Generate a Down arrow key event.
                self.event_generate("<Up>")  # Generate an Up arrow key event.
            finally:
                self["state"] = statebak  # Restore the state of the combobox.
                self.bind('<KeyRelease>', self.handle_keyrelease)  # Bind the key release event to the handle_keyrelease method.

class Application:
    def __init__(self, edges, node_dict, reverse_node_dict, graph, G):
        # Initialize instance variables for storing the selected algorithm, start station, end station, and other data.
        self.algorithm_selected = None
        self.selected_algorithm_label_map = None
        self.start_station = None
        self.end_station = None
        self.station_labels = {}
        self.edges = edges
        self.node_dict = node_dict
        self.reverse_node_dict = reverse_node_dict
        self.graph = graph
        self.k = None
        self.G = G
        # Create the root window and set its title.
        self.root = tk.Tk()
        self.root.title("Menu")
        # Create a label for the title and add it to the window.
        title_label = tk.Label(self.root, text="Graph Algorithms", font=("Arial", 24))
        title_label.pack()
        # Load an image, resize it, and create a label to display in the window.
        image_path = "E:/Haoxuan Xu's Documents (BS)/School Study/Computer Science/Year 12&13/NEA/Coursework Documents/Beford_school_logo.svg.png"
        img = Image.open(image_path)
        base_width = 200
        w_percent = (base_width / float(img.size[0]))
        h_size = int((float(img.size[1]) * float(w_percent)))
        img_resized = img.resize((base_width, h_size), Image.LANCZOS) 
        photo = ImageTk.PhotoImage(img_resized)
        image_label = tk.Label(self.root, image=photo)
        image_label.image = photo
        image_label.pack()
        # Create buttons for opening the settings and map windows, and add them to the window.
        self.settings_button = tk.Button(self.root, text="Settings", command=self.show_settings)
        self.settings_button.pack()
        self.map_button = tk.Button(self.root, text="Map", command=self.show_map)
        self.map_button.pack()
        # Start the main event loop for the window.
        self.root.mainloop()
    
    def show_map(self):
        # Check if an algorithm has been selected. If not, show the settings window and return.
        if self.algorithm_selected is None:
            messagebox.showinfo("Algorithm not set", "Please select an algorithm in settings.")
            self.show_settings()
            return

        # Create a new window for the map.
        map_window = tk.Toplevel(self.root)
        map_window.title("Tube Map")

        # Create a canvas for drawing the map.
        canvas = Canvas(map_window, width=500, height=500)
        self.canvas = canvas
        canvas.pack()

        # Define the coordinates for each station on the map.
        coords = {"建筑科技大学": (100, 100), "西安科技大学": (200, 100), "大雁塔": (300, 100), 
                "大唐芙蓉园": (150, 200), "曲江池西": (250, 200), "航天大道": (200, 300),
                "五路口": (350, 300), "火车站": (400, 200), "行政中心": (450, 100),
                "余家寨": (400, 400), "大明宫": (300, 400), "飞天路": (200, 400),
                "神舟大道": (100, 400), "东长安街": (50, 300)}

        # Draw the edges on the map.
        for start, end, weight in edges:
            x1, y1 = coords[start]
            x2, y2 = coords[end]
            canvas.create_line(x1, y1, x2, y2, fill="red", width=3)
            canvas.create_text((x1+x2)/2 + 20, (y1+y2)/2 + 10, text=str(weight))

        # Draw the stations on the map and create a button for each station.
        for station, (x, y) in coords.items():
            self.station_labels[station] = canvas.create_text(x, y+30, text=station)
            button = tk.Button(map_window, text='', command=lambda station=station: self.set_station(station))
            button_window = canvas.create_window(x-5, y-5, anchor='nw', window=button)

        # Create labels for displaying the start station, end station, and selected algorithm.
        start_station_label = tk.Label(map_window, text="")
        self.start_station_label = start_station_label
        end_station_label = tk.Label(map_window, text="")
        self.end_station_label = end_station_label
        start_station_label.pack()
        end_station_label.pack()
        selected_algorithm_label_map = tk.Label(map_window, text="")
        self.selected_algorithm_label_map = selected_algorithm_label_map
        selected_algorithm_label_map.pack()

        # Create a button for running the selected algorithm.
        run_button = tk.Button(map_window, text='Run Algorithm', command=self.run_algorithm)
        run_button.pack()

        # If the selected algorithm is "Yen's k", create an entry field for entering the value of k.
        if self.algorithm_selected == "Yen's k":
            k_label = tk.Label(map_window, text="Enter a value for k:")
            k_label.pack()
            self.k_entry = tk.Entry(map_window)
            self.k_entry.pack()

        # Bind the cleanup method to the window's destroy event.
        map_window.bind('<Destroy>', self.cleanup)

    def run_algorithm(self):
        # Initialize variable for storing the number of paths for Yen's K algorithm
        k = None

        # Check if start and end stations have been selected
        if self.start_station is None or self.end_station is None:
            # If not, show a message box asking the user to select start and end stations
            messagebox.showinfo("Stations not set", "Please select start and end stations.")
            return
        if self.start_station == self.end_station:
            # If the start and end stations are the same, show a message box asking the user to select different stations
            messagebox.showinfo("Same stations selected", "Please select different start and end stations.")
            return

        # Run the selected algorithm and update the map with the result
        if self.algorithm_selected == "Dijkstra's":
            # Run Dijkstra's algorithm
            result, time_taken, node_list = self.graph.run_algorithm_others("Dijkstra", self.start_station, self.end_station)
            self.graph.print_result_others("Dijkstra", result, time_taken)
            self.update_colour(node_list, "yellow")

        elif self.algorithm_selected == "Floyd-Warshall":
            # Run Floyd-Warshall algorithm
            result, time_taken, node_list = self.graph.run_algorithm_others("Floyd-Warshall", self.start_station, self.end_station)
            self.graph.print_result_others("Floyd-Warshall", result, time_taken)
            self.update_colour(node_list, "green")
                
        elif self.algorithm_selected == "A*":
            # Run A* algorithm
            result, time_taken, node_list = self.graph.run_algorithm_others("A*", self.start_station, self.end_station)
            self.graph.print_result_others("A*", result, time_taken)
            self.update_colour(node_list, "yellow")

        elif self.algorithm_selected == "Yen's k":
            # Check if a value for k has been entered
            if self.k_entry.get() == "":
                # If not, show a message box asking the user to enter a value for k
                messagebox.showinfo("K not set", "Please enter a value for k.")
                return
            k_entry = self.k_entry.get()
            # Check if the entered value for k is a positive integer
            if not re.match("^[1-9][0-9]*$", k_entry):
                # If not, show a message box asking the user to enter a positive integer for k
                messagebox.showinfo("Invalid input", "Please enter a positive integer for k.")
                return
            # Convert the entered value for k to an integer
            self.k = int(k_entry)
            # Run Yen's K algorithm
            result, time_taken, node_list = self.graph.run_algorithm_others("Yen's K", self.start_station,self.end_station,self.k, self.G)
            self.graph.print_result_others("Yen's K", result,time_taken,self.k)
            self.update_colour(node_list, "purple")

        elif self.algorithm_selected == 'Bellman Ford':
            # Run Bellman Ford algorithm
            result,time_taken, node_list = self.graph.run_algorithm_others('Bellman-Ford',self.start_station,self.end_station)
            self.graph.print_result_others('Bellman-Ford',result,time_taken)
            self.update_colour(node_list, "yellow")

    def update_colour(self, node_list, color_change):
        # Check if all elements in the node_list are tuples
        if all(isinstance(i, tuple) for i in node_list) == True:
            # Iterate over each tuple in the node_list
            for node_pair in node_list:
                # Change the color of each node in the tuple
                for node_name in node_pair:
                    self.update_node_color(node_name, color_change)
                # Pause for half a second to allow the color change to be visible
                time.sleep(0.5)
                # Reset the color of each node in the tuple
                for node_name in node_pair:
                    # If the node is the start or end station, color it blue
                    if node_name == self.start_station or node_name == self.end_station:
                        self.update_node_color(node_name, 'blue')
                    # Otherwise, color it black
                    else:
                        self.update_node_color(node_name, 'black')
        else:
            # If the elements in the node_list are not tuples, treat them as individual nodes
            for node_name in node_list:
                # Change the color of the node
                self.update_node_color(node_name, color_change)
                # Pause for half a second to allow the color change to be visible
                time.sleep(0.5)
                # If the node is the start or end station, color it blue
                if node_name == self.start_station or node_name == self.end_station:
                    self.update_node_color(node_name, 'blue')
                # Otherwise, color it black
                else:
                    self.update_node_color(node_name, 'black')

    def set_station(self, station):
        # If no start station has been set, set the clicked station as the start station
        if self.start_station is None:
            self.start_station = station
            # Highlight the start station in blue on the map
            self.canvas.itemconfig(self.station_labels[self.start_station], fill='blue')
            # Display the start station in the start station label
            self.start_station_label.config(text=f'Start station set to {self.start_station}')
        # If a start station has been set but no end station has been set, set the clicked station as the end station
        elif self.end_station is None:
            self.end_station = station
            # Highlight the end station in blue on the map
            self.canvas.itemconfig(self.station_labels[self.end_station], fill='blue')
            # Display the end station in the end station label
            self.end_station_label.config(text=f'End station set to {self.end_station}')
        # Display the selected algorithm in the selected algorithm label
        self.selected_algorithm_label_map.config(text=f'The selected algorithm is {self.algorithm_selected}')

    def show_settings(self):
        # Create a new settings window
        self.settings_window = tk.Toplevel(self.root)
        # Set the title of the settings window
        self.settings_window.title("Settings")
        # Bind the on_settings_close method to the destroy event of the settings window
        self.settings_window.bind('<Destroy>', self.on_settings_close) 
        # Create a label for the algorithm selection and add it to the settings window
        label = tk.Label(self.settings_window, text="Select Algorithm:")
        label.pack()
        # Define the list of algorithms
        algorithms = ["A*", "Dijkstra's", "Floyd-Warshall", "Yen's k", "Bellman Ford"]
        # Create a StringVar for the selected algorithm
        selected_algorithm = tk.StringVar(self.settings_window)  
        # Define a callback for when the selected algorithm changes
        def on_algorithm_change(*args):
            if selected_algorithm.get() in algorithms:
                selected_algorithm_label.config(text=f'Selected algorithm: {selected_algorithm.get()}')
                if self.selected_algorithm_label_map is not None:  
                    self.selected_algorithm_label_map.config(text=f'Selected algorithm: {selected_algorithm.get()}')
                self.algorithm_selected = selected_algorithm.get()
        # Bind the callback to the write event of the selected_algorithm StringVar
        selected_algorithm.trace('w', on_algorithm_change)
        # Create a dropdown menu for the algorithm selection and add it to the settings window
        dropdown = AutocompleteCombobox(self.settings_window, textvariable=selected_algorithm)
        dropdown.set_completion_list(algorithms)
        dropdown.pack()
        # Create a label for displaying the selected algorithm and add it to the settings window
        selected_algorithm_label = tk.Label(self.settings_window, text="")
        selected_algorithm_label.pack()

    def on_settings_close(self, event):
        if hasattr(self, 'k_slider') and self.k_slider.winfo_exists():
            self.k = self.k_slider.get()
        self.settings_window.destroy()

    def update_node_color(self, node_name, color):
        # Get the label ID of the node
        label_id = self.station_labels[node_name]
        # Change the color of the node in the canvas
        self.canvas.itemconfig(label_id , fill=color)
        # Update the canvas to reflect the color change
        self.canvas.update()


    def cleanup(self, event=None):
        # Reset the canvas variable
        self.canvas = None
        # Reset the dictionary that holds the station labels
        self.station_labels = {} 
        # Reset the start station label
        self.start_station_label = None
        # Reset the end station label
        self.end_station_label = None
        # Reset the selected algorithm label map
        self.selected_algorithm_label_map = None
        # Reset the dropdown menu
        self.dropdown = None
        # Reset the selected algorithm label
        self.selected_algorithm_label = None
        # Reset the start station
        self.start_station = None
        # Reset the end station
        self.end_station = None

# Initialization Stage
edges = [] 
weights_list = [] 
warnings.filterwarnings("ignore")
G = nx.Graph()

with open("E:/Haoxuan Xu's Documents (BS)/School Study/Computer Science/Year 12&13/NEA/Code/stations.txt", 'r', encoding='utf-8') as f:
    # Read the line, split it into nodes, and store them in a list
    nodes = [line.strip().split(',') for line in f.readlines()][0]
with open("E:/Haoxuan Xu's Documents (BS)/School Study/Computer Science/Year 12&13/NEA/Code/edges.txt",'r', encoding='utf-8') as f:
    for line in f:  
        # Split the line into station names and weight
        station1, station2, weight = line.strip().split(',')  
        # Append the weight to the weights list
        weights_list.append(int(weight))  
        # Add the edge to the edges list
        edges.append((station1, station2, int(weight)))  
        # Add the edge to the graph
        G.add_edge(str(station1), str(station2), length=int(weight))

graph = Graph(nodes, edges)
node_dict = {node: i for i, node in enumerate(nodes)} 
reverse_node_dict = {i: node for node, i in node_dict.items()}  

app = Application(edges, node_dict, reverse_node_dict, graph=graph, G=G)