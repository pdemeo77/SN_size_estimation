import random
import networkx as nx
from typing import List, Tuple

class Social_Network_Estimation_Biased_Sampling:
    def __init__(self, G: nx.Graph, walk_length: int):
        """
        Initialize the biased sampling estimator.

        :param G: The input graph.
        :param walk_length: The length of the random walk.
        """
        self.G = G
        self.walk_length = walk_length

    def random_walk(self) -> List[int]:
        """
        Perform a random walk on the graph.

        :return: A list of nodes visited during the random walk.
        """
        start_node = random.choice(list(self.G.nodes))
        walk = [start_node]
        current_node = start_node
        
        for _ in range(self.walk_length - 1):
            neighbors = list(self.G.neighbors(current_node))
            if not neighbors:
                break
            next_node = random.choice(neighbors)
            walk.append(next_node)
            current_node = next_node
        
        return walk
   
    def collisions(self, lst: List[int]) -> int:
        """
        Count the number of collisions in the random walk.

        :param lst: The list of nodes visited during the random walk.
        :return: The number of collisions.
        """
        cnt = 0
        for i in range(len(lst)):
            for j in range(i + 1, len(lst)):
                if lst[i] == lst[j]:
                    cnt += 1
        return 2 * cnt

    def compute_R(self, walk: List[int]) -> float:
        """
        Compute the R value for the random walk.

        :param walk: The list of nodes visited during the random walk.
        :return: The computed R value.
        """
        direct_degrees = sum([self.G.degree(node) for node in walk])
        inverse_degrees = sum([1 / self.G.degree(node) for node in walk])
        return direct_degrees * inverse_degrees

def biased_sampling_estimation(G, walk_length):
  
    estimator = Social_Network_Estimation_Biased_Sampling(G, walk_length)
    walk = estimator.random_walk()
    R = estimator.compute_R(walk)
    C = estimator.collisions(walk)
    return R, C
          

if __name__ == "__main__":
    # Create a sample graph
    G = nx.erdos_renyi_graph(1000, 0.2)
    #G = nx.powerlaw_cluster_graph(1000, 2, 0.2)
    # Remove nodes with degree zero
    zero_degree_nodes = [node for node, degree in dict(G.degree()).items() if degree == 0]
    G.remove_nodes_from(zero_degree_nodes)

    # Define walk length
    walk_length = 300

    # Estimate the network using biased sampling
    R, C = biased_sampling_estimation(G, walk_length)
    if C != 0:
        estimated_size = R/C
        print(f"Estimated size of the network: {estimated_size}")
