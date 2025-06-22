import networkx as nx
import random

def get_graph(from_file, n=100, gamma=3, input_file=None):
    """
    Returns a NetworkX graph.
    If from_file is False, generates a powerlaw random graph with n nodes and exponent gamma.
    If from_file is True, loads the graph from input_file (edge list format).
    """
    if from_file:
        if input_file is None:
            raise ValueError("input_file must be specified when from_file is True.")
        G = nx.read_edgelist(input_file)
    else:
        G = nx.generators.random_graphs.powerlaw_cluster_graph(n, int(gamma), 0.1)
    return G

import random
import math
import networkx as nx

def compute_neighbor_degree_probs(G):
    """
    Precompute, for each node, the list of neighbors and the probability of stepping to each neighbor,
    proportional to their degrees.
    """
    neighbor_degree_probs = {}
    degrees = dict(G.degree())
    for node in G.nodes():
        neighbors = list(G.neighbors(node))
        if neighbors:
            neighbor_degrees = [degrees[n] for n in neighbors]
            total = sum(neighbor_degrees)
            if total == 0:
                probs = [1 / len(neighbors)] * len(neighbors)
            else:
                probs = [d / total for d in neighbor_degrees]
            neighbor_degree_probs[node] = (neighbors, probs)
        else:
            neighbor_degree_probs[node] = ([], [])
    return neighbor_degree_probs

def biased_random_walk_largest_degree(G, start_node, steps=None, c=2):
    """
    Performs a degree-biased random walk on G starting from start_node.
    Walks for steps = c * n * log(n) steps by default.
    Returns the node with the largest degree encountered and its degree.
    """
    n = G.number_of_nodes()
    if steps is None:
        steps = int(c * n * math.log(n + 1))  # +1 avoids log(0)

    # Precompute neighbor degree probabilities
    neighbor_degree_probs = compute_neighbor_degree_probs(G)
    degrees = dict(G.degree())

    current_node = start_node
    max_degree_node = current_node
    max_degree = degrees[current_node]

    for _ in range(steps):
        neighbors, probs = neighbor_degree_probs[current_node]
        if not neighbors:
            break  # Dead end
        current_node = random.choices(neighbors, weights=probs, k=1)[0]
        if degrees[current_node] > max_degree:
            max_degree = degrees[current_node]
            max_degree_node = current_node

    return max_degree_node, max_degree

# Example usage (generating a power-law graph and running the walk):
if __name__ == "__main__":
    # Generate a power-law (scale-free) graph
    #G = nx.barabasi_albert_graph(1000, 3)  # 1000 nodes, 3 edges per new node
    G = get_graph(from_file=False, n=10000, gamma=3)
    # Start from a random node
    start_node = random.choice(list(G.nodes()))

    # Run the walk
    hub_node, hub_degree = biased_random_walk_largest_degree(G, start_node)

    print(f"Largest degree node encountered: {hub_node}, degree: {hub_degree}")
    print(f"Actual largest degree in graph: {max(dict(G.degree()).values())}")

    