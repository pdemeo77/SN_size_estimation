import networkx as nx
import numpy as np

def pareto_degree_graph(n, gamma):
    if not (2.1 <= gamma <= 2.9):
        raise ValueError("gamma must be between 2.1 and 2.9")
    # Pareto distribution: P(k) ~ k^(-gamma)
    # networkx configuration_model expects a degree sequence
    # Use Pareto distribution with xm=1 (minimum degree 1)
    degrees = np.random.pareto(gamma - 1, n) + 1
    degrees = np.round(degrees).astype(int)
    # Ensure sum of degrees is even
    if sum(degrees) % 2 != 0:
        degrees[0] += 1
    G = nx.configuration_model(degrees)
    # Convert to simple undirected graph (no parallel edges or self-loops)
    G = nx.Graph(G)
    G.remove_edges_from(nx.selfloop_edges(G))
    # Relabel nodes to 0..n-1
    G = nx.convert_node_labels_to_integers(G)
    return G

def biased_random_walk_to_max_degree(G, n_t):
    max_deg = max(dict(G.degree()).values())
    max_deg_nodes = {node for node, deg in G.degree() if deg == max_deg}
    nodes = list(G.nodes())
    # Precompute neighbor degree distributions for all nodes
    neighbor_probs = {}
    for node in nodes:
        neighbors = list(G.neighbors(node))
        if neighbors:
            degrees = np.array([G.degree(n) for n in neighbors], dtype=float)
            probs = degrees / degrees.sum()
            neighbor_probs[node] = (neighbors, probs)
        else:
            neighbor_probs[node] = ([], None)

    steps_list = []
    for _ in range(n_t):
        current = np.random.choice(nodes)
        steps = 0
        while current not in max_deg_nodes:
            neighbors, probs = neighbor_probs[current]
            if not neighbors:
                break  # Isolated node
            current = np.random.choice(neighbors, p=probs)
            steps += 1
            if steps > len(G):
                #print('Infinite Loop')  # Prevent infinite loops
                break
        steps_list.append(steps)
    return np.mean(steps_list) if steps_list else float('inf')


def i_order_statistics(G, m=None):
    nodes = list(G.nodes())
    # Precompute neighbor degree distributions for all nodes
    neighbor_probs = {}
    for node in nodes:
        neighbors = list(G.neighbors(node))
        if neighbors:
            degrees = np.array([G.degree(n) for n in neighbors], dtype=float)
            probs = degrees / degrees.sum()
            neighbor_probs[node] = (neighbors, probs)
        else:
            neighbor_probs[node] = ([], None)

    visited = set()
    order = []
    current = np.random.choice(nodes)
    steps = 0
    max_steps = m if m is not None else len(G)
    while len(visited) < len(G) and steps < max_steps:
        if current not in visited:
            visited.add(current)
            order.append((current, G.degree(current)))
        neighbors, probs = neighbor_probs[current]
        if not neighbors:
            break  # Isolated node
        current = np.random.choice(neighbors, p=probs)
        steps += 1
    order.sort(key=lambda x: x[1], reverse=True)
    return order


def compute(D, gamma, low, up):
    if up <= low:
        raise ValueError("Parameter 'up' must be strictly greater than 'low'")
    D_sorted = np.sort(D)[::-1]
    D_p = np.array([i * (d ** gamma) for i, d in enumerate(D_sorted)])
    subarray = D_p[low:up]
    median = np.median(subarray)
    return median


if __name__ == "__main__":
    #node_counts = [10, 100, 500, 1000, 1000]
    node_counts = [10, 100, 500, 1000, 5000, 10000]
    gamma = 2.5
    n_t = 10

    for i in range(5):
        G = pareto_degree_graph(2000, gamma)
        order = i_order_statistics(G, 3500)
        print(f"Experiment {i+1}:")
        #print(order)
        #print()
        deg = [deg for _, deg in order]
        result = compute(deg, gamma, 250, 2000)
        print("Result of compute on deg:", result)

    # for n in node_counts:
    #     avg_steps_list = []
    #     for _ in range(10):
    #         G = pareto_degree_graph(n, gamma)
    #         #print(f"Graph with {n} nodes is connected: {nx.is_connected(G)}")
    #         avg_steps = biased_random_walk_to_max_degree(G, n_t)
    #         avg_steps_list.append(avg_steps)
    #     overall_avg = np.mean(avg_steps_list)
    #     print(f"Nodes: {n}, Avg. steps to max degree (over 10 graphs): {overall_avg:.2f}")
        