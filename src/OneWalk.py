import networkx as nx
import random
from typing import List, Optional

class Estimation_By_One_Walk:
    def __init__(self, G: nx.Graph, walk_length: int):
        """
        Initialize the Estimation_By_One_Walk class.

        Args:
        G (nx.Graph): The input graph.
        walk_length (int): The length of the random walk.
        """
        self.G = G
        self.walk_length = walk_length

    def generate_random_walk(self, start_node: Optional[int] = None) -> List[int]:
        """
        Generate a random walk of a specified length.

        Args:
        start_node (Optional[int]): The starting node for the random walk. If None, a random node is chosen.

        Returns:
        List[int]: The list of nodes in the random walk.
        """
        if start_node is None:
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

    def extension(self, walk: List[int]) -> List[int]:
        """
        Generate the extension of a random walk.

        Args:
        walk (List[int]): The list of nodes in the random walk.

        Returns:
        List[int]: The list of nodes in the extension of the random walk.
        """
        extension = set()
        for node in walk:
            neighbors = list(self.G.neighbors(node))
            extension.update(neighbors)
        return list(extension)



    
    
def CC(A: list, B: list) -> int:
    """
    Calculate the number of common elements between two lists.

    Args:
    A (list): First list of elements.
    B (list): Second list of elements.

    Returns:
    int: The count of identical pairs of AxB. A pair (a,b) \in A \times B is called identical if a = b. 
    """
    return len([(a, b) for a in A for b in B if a == b])




if __name__ == "__main__":
    # Create a random graph
    #G = nx.erdos_renyi_graph(1000, 0.2)
    # Create a power law graph
    G = nx.powerlaw_cluster_graph(1000, 2, 0.2)

    for walk_length in range(100, 500, 5):
        estimator = Estimation_By_One_Walk(G, walk_length)
        random_walk = estimator.generate_random_walk(start_node=None)
        extension = estimator.extension(random_walk)
        cross_collision = CC(random_walk, extension)
        n_unique_rw = len(random_walk)
        neighborhoods = len(extension)
        print(f"Estimated_size: {walk_length}, s: {n_unique_rw*neighborhoods/cross_collision}")

    