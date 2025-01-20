import networkx as nx
import random
import numpy as np
from typing import List, Dict

import matplotlib.pyplot as plt



class GraphSampler:
    def __init__(self, graph: nx.Graph):
        self.graph = graph

    def generate_node_sample(self) -> List[int]:
        pass

class SimpleRandomWalkSampler(GraphSampler):
    def __init__(self, graph: nx.Graph, walk_length: int):
        super().__init__(graph)
        self.walk_length = walk_length

    def generate_node_sample(self) -> List[int]:
        current_node = random.choice(list(self.graph.nodes))
        walk = [current_node]
        for _ in range(self.walk_length - 1):
            neighbors = list(self.graph.neighbors(current_node))
            if not neighbors:
                break
            current_node = random.choice(neighbors)
            walk.append(current_node)
        return walk
    
class MetropolisHastingsRandomWalkSampler(GraphSampler):
    def __init__(self, graph: nx.Graph, walk_length: int):
        super().__init__(graph)
        self.walk_length = walk_length

    def generate_node_sample(self) -> List[int]:
        current_node = random.choice(list(self.graph.nodes))
        walk = [current_node]
        for _ in range(self.walk_length - 1):
            neighbors = list(self.graph.neighbors(current_node))
            if not neighbors:
                break
            next_node = random.choice(neighbors)
            degree_current = self.graph.degree[current_node]
            degree_next = self.graph.degree[next_node]
            p = random.uniform(0, 1)
            if degree_current / degree_next >= p:
                current_node = next_node
                walk.append(current_node)
        return walk








def evaluate_simple_random_walk_sampler(graph, walk_length, num_repeats=50):
    simple_sampler = SimpleRandomWalkSampler(graph, walk_length)
    estimates = np.zeros(num_repeats)   
    for i in range(num_repeats):
        sample1 = set(simple_sampler.generate_node_sample())
        sample2 = set(simple_sampler.generate_node_sample())
        intersection_size = len(sample1 & sample2)
        if intersection_size == 0:
            estimates[i] = 0
        else:
            estimates[i] = (len(sample1) * len(sample2)) / intersection_size
    return np.mean(estimates)

def evaluate_mh_random_walk_sampler(graph, walk_length, num_repeats=50):
    mh_sampler = MetropolisHastingsRandomWalkSampler(graph, walk_length)
    estimates = np.zeros(num_repeats)
    for i in range(num_repeats):
        sample1 = set(mh_sampler.generate_node_sample())
        sample2 = set(mh_sampler.generate_node_sample())
        intersection_size = len(sample1 & sample2)
        if intersection_size == 0:
            estimates[i] = 0
        else:
            estimates[i] = (len(sample1) * len(sample2)) / intersection_size
    return np.mean(estimates)


# Parameters
num_nodes = 1000
probability = 0.3

# Generate random graph
G = nx.erdos_renyi_graph(num_nodes, probability)

# Extract the list of nodes
node_list = list(G.nodes)

# Walk lengths to evaluate
walk_lengths = [100, 200, 500, 1000, 1500, 2000, 2500, 5000, 10000]

# Repeat the evaluation for each walk length
for walk_length in walk_lengths:
    print(f"Walk Length: {walk_length}")
    
    '''
    # Generate a new scale-free graph with 1000 nodes and exponent 2.2
    G_new = nx.scale_free_graph(1000, alpha=0.2, beta=0.7, gamma=0.1, delta_in=0.2, delta_out=0)
    G_new = nx.Graph(G_new)  # Convert to undirected graph
    G_new.remove_edges_from(nx.selfloop_edges(G_new))  # Remove self-loops
    '''

    G_new = nx.erdos_renyi_graph(1000, 0.5)

    # Evaluate the simple random walk sampler
    simple_rw_estimate = evaluate_simple_random_walk_sampler(G_new, walk_length)
    print(f"  Simple Random Walk Estimate: {simple_rw_estimate}")

    # Evaluate the Metropolis-Hastings random walk sampler
    mh_rw_estimate = evaluate_mh_random_walk_sampler(G_new, walk_length)
    print(f"  Metropolis-Hastings Random Walk Estimate: {mh_rw_estimate}")

    print(10*'*')