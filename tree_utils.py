import numpy as np

class TreeNode:
    """Defining the nodes of a tree"""
    def __init__(self, node_id, level, parent=None):
        self.id = node_id
        self.level = level
        self.parent = parent
        self.children = []
        self.leaf_indices = set() # Indices of all leaf nodes under this node

    def __repr__(self):
        return f"Node(id={self.id}, level={self.level}, leaves={len(self.leaf_indices)})"

def build_tree(K):
    """
    Construct a perfect binary tree with K leaf nodes.
    K must be a power of 2.
    """
    if (K & (K - 1) != 0) or K == 0:
        raise ValueError(f"K must be a power of 2, but K={K}")
    
    levels = int(np.log2(K))
    
    # Create all nodes
    nodes = {}
    node_counter = 0
    
    # Level 0: Leaf nodes
    level_nodes = []
    for i in range(K):
        node = TreeNode(node_id=node_counter, level=0)
        node.leaf_indices.add(i)
        nodes[node_counter] = node
        level_nodes.append(node)
        node_counter += 1
        
    # Constructing internal nodes (from l=1 to l=levels)
    for l in range(1, levels + 1):
        prev_level_nodes = level_nodes
        level_nodes = []
        for i in range(0, len(prev_level_nodes), 2):
            parent_node = TreeNode(node_id=node_counter, level=l)
            nodes[node_counter] = parent_node
            level_nodes.append(parent_node)
            node_counter += 1
            
            # Connecting child nodes
            child_l = prev_level_nodes[i]
            child_r = prev_level_nodes[i+1]
            parent_node.children = [child_l, child_r]
            child_l.parent = parent_node
            child_r.parent = parent_node
            
            # Aggregate leaf node indices
            parent_node.leaf_indices.update(child_l.leaf_indices)
            parent_node.leaf_indices.update(child_r.leaf_indices)
            
    root = level_nodes[0]
    return nodes, root

def get_tree_action_vectors(nodes_dict, K):
    """
    Generate K-dimensional action vectors for HDS and ECC-AHT-Restricted.
    """
    action_vectors = {}
    for node_id, node in nodes_dict.items():
        C_v = np.zeros(K)
        # Aggregate (sum) vectors for all leaf nodes under this node
        C_v[list(node.leaf_indices)] = 1.0
        action_vectors[node_id] = C_v
    return action_vectors