import sys
import numpy as np
import time
import heapq
import warnings
import networkx as nx
import tkinter as tk
import re
from colorama import Fore
from queue import *
from tkinter import Canvas
from tkinter import ttk
from tkinter import messagebox
from heapq import heappush, heappop
from itertools import count
from PIL import Image, ImageTk

class Node:
    def __init__(self, name):
        self.name = name
        self.value = sys.maxsize
        self.previous_node = None
        self.edges = []

    def add_edge(self, edge):
        self.edges.append(edge)

class Edge:
    def __init__(self, start_node, end_node, weight):
        self.start_node = start_node
        self.end_node = end_node
        self.weight = weight

class Graph:
    def __init__(self, nodes, edges):
        self.nodes = {}
        self.node_names = nodes
        self.edges = {}
        self.adjacency_matrix = self.construct_adjacency_matrix(nodes, edges)
        for node_name in nodes:
            self.nodes[node_name] = Node(node_name)
        for edge in edges:
            start_node, end_node, weight = edge
            self.add_edge(start_node, end_node, weight)

    def construct_adjacency_matrix(self, nodes, edges):
        n = len(nodes)
        matrix = np.full((n, n), float('inf'))
        for i in range(n):
            matrix[i][i] = 0
        for edge in edges:
            start_node, end_node, weight = edge
            i = nodes.index(start_node)
            j = nodes.index(end_node)
            matrix[i][j] = weight
            matrix[j][i] = weight
        return matrix

    def add_edge(self, start_node, end_node, weight):
        edge = Edge(self.nodes[start_node], self.nodes[end_node], weight) 
        self.nodes[start_node].add_edge(edge) # Technique used - List operations
        self.nodes[end_node].add_edge(edge)

        if start_node not in self.edges:
            self.edges[start_node] = []
        if end_node not in self.edges:
            self.edges[end_node] = []
        
        self.edges[start_node].append((end_node, weight))
        self.edges[end_node].append((start_node, weight))
    
    def dijkstra_algorithm(self, start_node): # Technique used - Graph/Tree Traversal
        unvisited_nodes = list(self.nodes.values())
        shortest_path = {}
        previous_nodes = {}
        visited_nodes = []
        max_value = sys.maxsize
        for node in unvisited_nodes:
            shortest_path[node.name] = max_value
        shortest_path[start_node] = 0
        self.nodes[start_node].value = 0

        while unvisited_nodes:
            current_min_node = min(unvisited_nodes, key=lambda node: node.value)
            unvisited_nodes.remove(current_min_node)

            visited_nodes.append(current_min_node.name)

            for edge in current_min_node.edges:
                neighbor = edge.end_node if edge.start_node == current_min_node else edge.start_node
                tentative_value = current_min_node.value + edge.weight

                if tentative_value < neighbor.value:
                    neighbor.value = tentative_value
                    neighbor.previous_node = current_min_node
                    previous_nodes[neighbor.name] = current_min_node.name
                    shortest_path[neighbor.name] = tentative_value

        return visited_nodes, previous_nodes, shortest_path

    def get_shortest_path_dijkstra(self, source, target):
        visited_nodes, previous_nodes, shortest_path = self.dijkstra_algorithm(source)
        path = []
        current_node = target
        while current_node is not None:
            path.append(current_node)
            if current_node == source:
                break
            current_node = previous_nodes[current_node]
        path.reverse()
        return path, shortest_path[target], visited_nodes

    def floyd_warshall(self):
        n = len(self.adjacency_matrix)
        dist = np.array(self.adjacency_matrix)
        next_node = np.zeros((n, n), dtype=int)
        node_list = []
        for i in range(n):
            for j in range(n):
                if i != j and dist[i][j] != float('inf'):
                    next_node[i][j] = j
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if dist[i][j] > dist[i][k] + dist[k][j]:
                        dist[i][j] = dist[i][k] + dist[k][j]
                        next_node[i][j] = next_node[i][k]
                        node_list.append((self.node_names[i], self.node_names[j]))
        return dist, next_node, node_list

    def reconstruct_path_floyd(self, start_node_name, target_node_name, next_node):
        source = self.node_names.index(start_node_name)
        target = self.node_names.index(target_node_name)
        path = [source] 
        while source != target: 
            source = next_node[source][target] 
            path.append(source)  
        return [self.node_names[i] for i in path]  	
    
    def heuristic(self, node1, node2):
        return 0

    def a_star(self, start_node, target_node):
        start_node = self.nodes[start_node]
        target_node = self.nodes[target_node]
        open_list = [] 
        count = 0  
        heapq.heappush(open_list, (0, count, start_node))  
        came_from = {} 
        g_score = {node: float('inf') for node in self.nodes.values()} 
        g_score[start_node] = 0 
        f_score = {node: float('inf') for node in self.nodes.values()}  
        f_score[start_node] = self.heuristic(start_node, target_node) 

        visited_nodes = []

        while open_list:  
            _, _, current = heapq.heappop(open_list) 

            visited_nodes.append(current.name)

            if current == target_node:  
                path = []  
                while current in came_from:  
                    path.append(current) 
                    current = came_from[current]  
                path.append(current)  
                return path[::-1], f_score[target_node], visited_nodes
            for edge in current.edges:  
                neighbor = edge.end_node if edge.start_node == current else edge.start_node  
                tentative_g_score = g_score[current] + edge.weight  
                if tentative_g_score < g_score[neighbor]: 
                    came_from[neighbor] = current 
                    g_score[neighbor] = tentative_g_score  
                    f_score[neighbor] = g_score[neighbor] + self.heuristic(neighbor, target_node) 
                    if neighbor not in [i[2] for i in open_list]: 
                        count += 1  
                        heapq.heappush(open_list, (f_score[neighbor], count, neighbor))  	
        return None, float('inf'), visited_nodes

    def k_shortest_paths(self, G, source, target, k, weight='weight'):
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

        visited_nodes = []

        seen_paths = set()

        for i in range(1, k):
            G_copy = G.copy() 
            for j in range(len(paths[-1]) - 1):            
                spur_node = paths[-1][j]
                root_path = paths[-1][:j + 1]
                edges_removed = []
                for c_path in paths:
                    if len(c_path) > j and root_path == c_path[:j + 1]:
                        u = c_path[j]
                        v = c_path[j + 1]
                        if G_copy.has_edge(u, v): 
                            edge_attr = G_copy.edges[u,v]
                            G_copy.remove_edge(u, v)  
                            edges_removed.append((u, v, edge_attr))
                for n in range(len(root_path) - 1):
                    node = root_path[n]
                    edges_to_remove = []
                    for u, v, edge_attr in G_copy.edges(node, data=True):  
                        edges_to_remove.append((u, v))
                        edges_removed.append((u, v, edge_attr))
                    for u, v in edges_to_remove:
                        G_copy.remove_edge(u, v)  
                try:
                    spur_path_length, spur_path = nx.single_source_dijkstra(G_copy, spur_node, target, weight=weight)  

                    total_path_tuple = tuple(root_path[:-1] + spur_path)
                    if total_path_tuple not in seen_paths:
                        visited_nodes.append(spur_node)
                        seen_paths.add(total_path_tuple)

                except nx.NetworkXNoPath:
                    continue
                if target in spur_path:
                    total_path = root_path[:-1] + spur_path
                    total_path_length = graph.get_path_length_yen(G_original, total_path, weight)
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

        seen = set()
        unique_paths = [x for x in paths if not (tuple(x) in seen or seen.add(tuple(x)))]

        unique_lengths = [lengths[paths.index(path)] for path in unique_paths]
        
        return unique_lengths, unique_paths, visited_nodes

    def get_path_length_yen(self, G, path, weight='weight'):
        length = 0
        if len(path) > 1:
            for i in range(len(path) - 1):
                u = path[i]
                v = path[i + 1]
                
                length += G.edges[u,v].get(weight, 1)
        return length
    
    def bellman_ford(self, source):
        distances = {vertex: float('inf') for vertex in self.node_names}
        predecessors = {vertex: None for vertex in self.node_names}
        distances[source] = 0

        node_list = []

        for _ in range(len(self.node_names) - 1):
            for u in self.node_names:
                if u in self.edges:
                    for v, weight in self.edges[u]:
                        if distances[u] + weight < distances[v]:
                            distances[v] = distances[u] + weight
                            predecessors[v] = u

                            node_list.append(v)

        for u in self.node_names:
            if u in self.edges:
                for v, weight in self.edges[u]:
                    if distances[u] + weight < distances[v]:
                        print("Negative cycle detected")
                        return None, None, node_list
                    
        return distances, predecessors, node_list

    
    def get_shortest_path_bellman(self, source, destination):
        distances, predecessors, node_dict = self.bellman_ford(source)
        path = []
        total_weight = 0
        while destination != source:
            path.append(destination)
            if destination not in predecessors:
                return None, None
            total_weight += next(wt for dest, wt in self.edges[predecessors[destination]] if dest == destination)
            destination = predecessors[destination]
        path.append(source)
        return path[::-1], total_weight
    
    def run_algorithm_others(self, algorithm, start_node, target_node, k=None, G=None):
        result = None
        node_list = None
        start_time = time.time() 
        if algorithm == "A*":  
            path, distance, node_list = self.a_star(start_node, target_node)  
            result = path, distance  
        elif algorithm == "Floyd-Warshall": 
            shortest_distance_matrix, next_node, node_list = self.floyd_warshall() 
            path = self.reconstruct_path_floyd(start_node, target_node, next_node)  
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
        
        end_time = time.time()  
        time_taken = "{:.3g}".format(end_time - start_time)  
        return result, time_taken, node_list
        
    def ordinal(self, n):
        suffix = ['th', 'st', 'nd', 'rd', 'th'][min(n % 10, 4)]
        if 11 <= (n % 100) <= 13:
            suffix = 'th'
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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._completion_list = []
        self._hits = []
        self._hit_index = 0
        self.position = 0
        self.bind('<KeyRelease>', self.handle_keyrelease)
        self.bind('<Button-1>', self.display_options)
        self['values'] = []

    def set_completion_list(self, completion_list):
        self._completion_list = sorted(completion_list, key=str.lower)
        self['values'] = self._completion_list

    def autocomplete(self, delta=0):
        if delta:
            self.delete(self.position, tk.END)
        else:
            self.position = len(self.get())
        _hits = []
        for element in self._completion_list:
            if element.lower().startswith(self.get().lower()):
                _hits.append(element)
        if _hits != self._hits:
            self._hit_index = 0
            self._hits=_hits
        if _hits == self._hits and self._hits:
            self._hit_index = (self._hit_index + delta) % len(self._hits)
        if self._hits:
            self.delete(0,tk.END)
            self.insert(0,self._hits[self._hit_index])
            self.select_range(self.position,tk.END)

    def handle_keyrelease(self, event):
        if event.keysym == "BackSpace":
            self.delete(self.index(tk.INSERT), tk.END) 
            self.position = self.index(tk.END)
        if event.keysym == "Left":
            if self.position < self.index(tk.END):
                self.delete(self.position, tk.END)
            else:
                self.position = self.position-1
                self.delete(self.position, tk.END)
        if event.keysym == "Right":
            self.position = self.index(tk.END)
        if len(event.keysym) == 1:
            self.autocomplete()

    def display_options(self, event):
        if not self.get():
            self.unbind('<KeyRelease>')
            backup = list(self["values"])
            self["values"] = backup
            statebak = str(self.cget("state"))
            startbak = str(self.get())
            startidxbak = int(self.current())
            startibak = int(self.index(tk.INSERT))
            startebak = int(self.index(tk.END))
            try:
                self["state"] = "readonly"
                self.event_generate("<Down>")
                self.event_generate("<Up>")
            finally:
                self["state"] = statebak
                self.bind('<KeyRelease>', self.handle_keyrelease)

class Application:
    def __init__(self, edges, node_dict, reverse_node_dict, graph, G):
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

        self.root = tk.Tk()
        self.root.title("Menu")
        title_label = tk.Label(self.root, text="Graph Algorithms", font=("Arial", 24))
        title_label.pack()
        image_path = "X:/Haoxuan Xu's Documents (BS)/School Study/Computer Science/Year 12&13/NEA/Documents/Beford_school_logo.svg.png"
        img = Image.open(image_path)
        base_width = 200
        w_percent = (base_width / float(img.size[0]))
        h_size = int((float(img.size[1]) * float(w_percent)))
        img_resized = img.resize((base_width, h_size), Image.ANTIALIAS) 
        photo = ImageTk.PhotoImage(img_resized)
        image_label = tk.Label(self.root, image=photo)
        image_label.image = photo
        image_label.pack()
        self.settings_button = tk.Button(self.root, text="Settings", command=self.show_settings)
        self.settings_button.pack()
        self.map_button = tk.Button(self.root, text="Map", command=self.show_map)
        self.map_button.pack()
        self.root.mainloop()
    
    def show_map(self):
        edges = self.edges
        if self.algorithm_selected is None:
            messagebox.showinfo("Algorithm not set", "Please select an algorithm in settings.")
            self.show_settings()
            return

        map_window = tk.Toplevel(self.root)
        map_window.title("Tube Map")
        canvas = Canvas(map_window, width=500, height=500)
        self.canvas = canvas
        canvas.pack()
        
        coords = {"建筑科技大学": (100, 100), "西安科技大学": (200, 100), "大雁塔": (300, 100), 
                "大唐芙蓉园": (150, 200), "曲江池西": (250, 200), "航天大道": (200, 300),
                "五路口": (350, 300), "火车站": (400, 200), "行政中心": (450, 100),
                "余家寨": (400, 400), "大明宫": (300, 400), "飞天路": (200, 400),
                "神舟大道": (100, 400), "东长安街": (50, 300)}
        for start, end, weight in edges:
            x1, y1 = coords[start]
            x2, y2 = coords[end]
            canvas.create_line(x1, y1, x2, y2, fill="red", width=3)
            canvas.create_text((x1+x2)/2 + 20, (y1+y2)/2 + 10, text=str(weight))
        for station, (x, y) in coords.items():
            self.station_labels[station] = canvas.create_text(x, y+30, text=station)
            button = tk.Button(map_window, text='', command=lambda station=station: self.set_station(station))
            button_window = canvas.create_window(x-5, y-5, anchor='nw', window=button)
        start_station_label = tk.Label(map_window, text="")
        self.start_station_label = start_station_label
        end_station_label = tk.Label(map_window, text="")
        self.end_station_label = end_station_label
        start_station_label.pack()
        end_station_label.pack()
        selected_algorithm_label_map = tk.Label(map_window, text="")
        self.selected_algorithm_label_map = selected_algorithm_label_map
        selected_algorithm_label_map.pack()

        run_button = tk.Button(map_window, text='Run Algorithm', command=self.run_algorithm)
        run_button.pack()
        if self.algorithm_selected == "Yen's k":
            k_label = tk.Label(map_window, text="Enter a value for k:")
            k_label.pack()
            self.k_entry = tk.Entry(map_window)
            self.k_entry.pack()

        map_window.bind('<Destroy>', self.cleanup)

    def run_algorithm(self):
        k = None
        if self.start_station is None or self.end_station is None:
            messagebox.showinfo("Stations not set", "Please select start and end stations.")
            return

        if self.algorithm_selected == "Dijkstra's":
            result, time_taken, node_list = self.graph.run_algorithm_others("Dijkstra", self.start_station, self.end_station)
            self.graph.print_result_others("Dijkstra", result, time_taken)
            self.update_colour(node_list, "yellow")

        elif self.algorithm_selected == "Floyd-Warshall":
            result, time_taken, node_list = self.graph.run_algorithm_others("Floyd-Warshall", self.start_station, self.end_station)
            self.graph.print_result_others("Floyd-Warshall", result, time_taken)
            self.update_colour(node_list, "green")
            
        elif self.algorithm_selected == "A*":
            result, time_taken, node_list = self.graph.run_algorithm_others("A*", self.start_station, self.end_station)
            self.graph.print_result_others("A*", result, time_taken)
            self.update_colour(node_list, "yellow")

        elif self.algorithm_selected == "Yen's k":
            if self.k_entry.get() == "":
                messagebox.showinfo("K not set", "Please enter a value for k.")
                return
            k_entry = self.k_entry.get()
            if not re.match("^[1-9][0-9]*$", k_entry):
                messagebox.showinfo("Invalid input", "Please enter a positive integer for k.")
                return
            self.k = int(k_entry)
            result, time_taken, node_list = self.graph.run_algorithm_others("Yen's K", self.start_station,self.end_station,self.k, self.G)
            self.graph.print_result_others("Yen's K", result,time_taken,self.k)

        elif self.algorithm_selected == 'Bellman Ford':
            result,time_taken, node_list = self.graph.run_algorithm_others('Bellman-Ford',self.start_station,self.end_station)
            self.graph.print_result_others('Bellman-Ford',result,time_taken)
            self.update_colour(node_list, "yellow")

    def update_colour(self, node_list, color_change):
        if all(isinstance(i, tuple) for i in node_list) == True:
            for node_pair in node_list:
                for node_name in node_pair:
                    self.update_node_color(node_name, color_change)
                time.sleep(0.5)
                for node_name in node_pair:
                    if node_name == self.start_station or node_name == self.end_station:
                        self.update_node_color(node_name, 'blue')
                    else:
                        self.update_node_color(node_name, 'black')
        else:
            for node_name in node_list:
                    self.update_node_color(node_name, color_change)
                    time.sleep(0.5)
                    if node_name == self.start_station or node_name == self.end_station:
                        self.update_node_color(node_name, 'blue')
                    else:
                        self.update_node_color(node_name, 'black')

    def set_station(self, station):
        if self.start_station is None:
            self.start_station = station
            self.canvas.itemconfig(self.station_labels[self.start_station], fill='blue')
            self.start_station_label.config(text=f'Start station set to {self.start_station}')
        elif self.end_station is None:
            self.end_station = station
            self.canvas.itemconfig(self.station_labels[self.end_station], fill='blue')
            self.end_station_label.config(text=f'End station set to {self.end_station}')
        self.selected_algorithm_label_map.config(text=f'The selected algorithm is {self.algorithm_selected}')

    def show_settings(self):
        self.settings_window = tk.Toplevel(self.root)
        self.settings_window.title("Settings")
        self.settings_window.bind('<Destroy>', self.on_settings_close) 
        label = tk.Label(self.settings_window, text="Select Algorithm:")
        label.pack()
        algorithms = ["A*", "Dijkstra's", "Floyd-Warshall", "Yen's k", "Bellman Ford"]
        selected_algorithm = tk.StringVar(self.settings_window)  

        def on_algorithm_change(*args):
            if selected_algorithm.get() in algorithms:
                selected_algorithm_label.config(text=f'Selected algorithm: {selected_algorithm.get()}')
                if self.selected_algorithm_label_map is not None:  
                    self.selected_algorithm_label_map.config(text=f'Selected algorithm: {selected_algorithm.get()}')
                self.algorithm_selected = selected_algorithm.get()

        selected_algorithm.trace('w', on_algorithm_change)
        dropdown = AutocompleteCombobox(self.settings_window, textvariable=selected_algorithm)
        dropdown.set_completion_list(algorithms)
        dropdown.pack()
        selected_algorithm_label = tk.Label(self.settings_window, text="")
        selected_algorithm_label.pack()


    def on_settings_close(self, event):
        if hasattr(self, 'k_slider') and self.k_slider.winfo_exists():
            self.k = self.k_slider.get()
        self.settings_window.destroy()

    def update_node_color(self, node_name, color):
        label_id = self.station_labels[node_name]

        self.canvas.itemconfig(label_id , fill=color)

        self.canvas.update()

    def cleanup(self, event=None):
        self.canvas = None
        self.station_labels = {} 
        self.start_station_label = None
        self.end_station_label = None
        self.selected_algorithm_label_map = None
        self.dropdown = None
        self.selected_algorithm_label = None
        self.start_station = None
        self.end_station = None

# Initialization Stage
edges = [] 
weights_list = [] 
warnings.filterwarnings("ignore")
G = nx.Graph()

with open("X:/Haoxuan Xu's Documents (BS)/School Study/Computer Science/Year 12&13/NEA/Code/stations.txt", 'r', encoding='utf-8') as f:
    nodes = [line.strip().split(',') for line in f.readlines()][0] 
with open("X:/Haoxuan Xu's Documents (BS)/School Study/Computer Science/Year 12&13/NEA/Code/edges.txt",'r', encoding='utf-8') as f:
    for line in f:  
        station1, station2, weight = line.strip().split(',')  
        weights_list.append(int(weight))  
        edges.append((station1, station2, int(weight)))  
        G.add_edge(str(station1), str(station2),length=int(weight))

graph = Graph(nodes, edges)
node_dict = {node: i for i, node in enumerate(nodes)} 
reverse_node_dict = {i: node for node, i in node_dict.items()}  

app = Application(edges, node_dict, reverse_node_dict, graph=graph, G=G)