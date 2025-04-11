"""
Tree Traversal Algorithms

This file provides a comprehensive overview of tree traversal algorithms,
including depth-first traversal (DFS), breadth-first traversal (BFS),
and the different types of tree traversals (pre-order, in-order, and post-order).

Tree traversal is the process of visiting each node in a tree data structure exactly once.
"""

#? 1. Introduction to Tree Traversals
"""
Tree Traversal: The process of visiting each node in a tree data structure exactly once.

There are two main categories of tree traversals:
1. Depth-First Search (DFS): Explores as far as possible along each branch before backtracking.
2. Breadth-First Search (BFS): Explores all nodes at the present depth before moving to nodes at the next depth level.

For binary trees, Depth-First Search can be further categorized into three types:
- Pre-order traversal: Visit root, then left subtree, then right subtree (Root -> Left -> Right)
- In-order traversal: Visit left subtree, then root, then right subtree (Left -> Root -> Right)
- Post-order traversal: Visit left subtree, then right subtree, then root (Left -> Right -> Root)

Visual representation of a binary tree:
        A         <- Root
       / \
      B   C       <- Internal nodes
     / \   \
    D   E   F     <- Leaf nodes

Traversal results:
- Pre-order: A, B, D, E, C, F
- In-order: D, B, E, A, C, F
- Post-order: D, E, B, F, C, A
- Level-order (BFS): A, B, C, D, E, F
"""

# Basic TreeNode class for examples
class TreeNode:
    """A node in a binary tree."""
    
    def __init__(self, data):
        """Initialize a tree node with the given data."""
        self.data = data
        self.left = None
        self.right = None


#? 2. Depth-First Search (DFS) Traversals

# 2.1 Pre-order Traversal (Root -> Left -> Right)
def preorder_traversal_recursive(root):
    """
    Perform a pre-order traversal of a binary tree recursively.
    
    Args:
        root: The root of the binary tree.
        
    Returns:
        list: The nodes visited in pre-order.
    """
    result = []
    
    def dfs(node):
        if node:
            # Visit the root
            result.append(node.data)
            # Traverse the left subtree
            dfs(node.left)
            # Traverse the right subtree
            dfs(node.right)
    
    dfs(root)
    return result


def preorder_traversal_iterative(root):
    """
    Perform a pre-order traversal of a binary tree iteratively.
    
    Args:
        root: The root of the binary tree.
        
    Returns:
        list: The nodes visited in pre-order.
    """
    if not root:
        return []
    
    result = []
    stack = [root]
    
    while stack:
        # Pop the top node from the stack
        node = stack.pop()
        
        # Visit the node
        result.append(node.data)
        
        # Push right child first (so it's processed after the left child)
        if node.right:
            stack.append(node.right)
        
        # Push left child
        if node.left:
            stack.append(node.left)
    
    return result


# 2.2 In-order Traversal (Left -> Root -> Right)
def inorder_traversal_recursive(root):
    """
    Perform an in-order traversal of a binary tree recursively.
    
    Args:
        root: The root of the binary tree.
        
    Returns:
        list: The nodes visited in in-order.
    """
    result = []
    
    def dfs(node):
        if node:
            # Traverse the left subtree
            dfs(node.left)
            # Visit the root
            result.append(node.data)
            # Traverse the right subtree
            dfs(node.right)
    
    dfs(root)
    return result


def inorder_traversal_iterative(root):
    """
    Perform an in-order traversal of a binary tree iteratively.
    
    Args:
        root: The root of the binary tree.
        
    Returns:
        list: The nodes visited in in-order.
    """
    result = []
    stack = []
    current = root
    
    while current or stack:
        # Reach the leftmost node of the current subtree
        while current:
            stack.append(current)
            current = current.left
        
        # Current is now None, pop the top node from the stack
        current = stack.pop()
        
        # Visit the node
        result.append(current.data)
        
        # Move to the right subtree
        current = current.right
    
    return result


# 2.3 Post-order Traversal (Left -> Right -> Root)
def postorder_traversal_recursive(root):
    """
    Perform a post-order traversal of a binary tree recursively.
    
    Args:
        root: The root of the binary tree.
        
    Returns:
        list: The nodes visited in post-order.
    """
    result = []
    
    def dfs(node):
        if node:
            # Traverse the left subtree
            dfs(node.left)
            # Traverse the right subtree
            dfs(node.right)
            # Visit the root
            result.append(node.data)
    
    dfs(root)
    return result


def postorder_traversal_iterative(root):
    """
    Perform a post-order traversal of a binary tree iteratively.
    
    Args:
        root: The root of the binary tree.
        
    Returns:
        list: The nodes visited in post-order.
    """
    if not root:
        return []
    
    result = []
    stack = [root]
    visited = []
    
    while stack:
        # Peek at the top node
        node = stack[-1]
        
        # If the node has no children or its children have been visited
        if (not node.left and not node.right) or (node.left in visited and node.right in visited):
            # Visit the node
            result.append(node.data)
            visited.append(node)
            stack.pop()
        else:
            # Push right child first (so it's processed after the left child)
            if node.right and node.right not in visited:
                stack.append(node.right)
            
            # Push left child
            if node.left and node.left not in visited:
                stack.append(node.left)
    
    return result


# Alternative implementation of post-order traversal using two stacks
def postorder_traversal_two_stacks(root):
    """
    Perform a post-order traversal of a binary tree iteratively using two stacks.
    
    Args:
        root: The root of the binary tree.
        
    Returns:
        list: The nodes visited in post-order.
    """
    if not root:
        return []
    
    result = []
    stack1 = [root]
    stack2 = []
    
    # First, push nodes in the order: root -> left -> right to stack1
    # Then, pop from stack1 and push to stack2
    # This will result in stack2 having nodes in the order: root -> right -> left
    while stack1:
        node = stack1.pop()
        stack2.append(node)
        
        if node.left:
            stack1.append(node.left)
        
        if node.right:
            stack1.append(node.right)
    
    # Pop from stack2 to get nodes in the order: left -> right -> root
    while stack2:
        node = stack2.pop()
        result.append(node.data)
    
    return result


#? 3. Breadth-First Search (BFS) Traversal

# 3.1 Level-order Traversal
def level_order_traversal(root):
    """
    Perform a level-order traversal (BFS) of a binary tree.
    
    Args:
        root: The root of the binary tree.
        
    Returns:
        list: The nodes visited in level order.
    """
    if not root:
        return []
    
    result = []
    queue = [root]
    
    while queue:
        # Dequeue the front node
        node = queue.pop(0)
        
        # Visit the node
        result.append(node.data)
        
        # Enqueue the left child
        if node.left:
            queue.append(node.left)
        
        # Enqueue the right child
        if node.right:
            queue.append(node.right)
    
    return result


# 3.2 Level-by-level Traversal (returns a list of lists, each inner list contains nodes at the same level)
def level_by_level_traversal(root):
    """
    Perform a level-by-level traversal of a binary tree.
    
    Args:
        root: The root of the binary tree.
        
    Returns:
        list of lists: Each inner list contains nodes at the same level.
    """
    if not root:
        return []
    
    result = []
    current_level = [root]
    
    while current_level:
        # Create a list to store the values of nodes at the current level
        level_values = []
        next_level = []
        
        for node in current_level:
            # Visit the node
            level_values.append(node.data)
            
            # Add children to the next level
            if node.left:
                next_level.append(node.left)
            
            if node.right:
                next_level.append(node.right)
        
        # Add the current level's values to the result
        result.append(level_values)
        
        # Move to the next level
        current_level = next_level
    
    return result


#? 4. Applications of Tree Traversals

"""
4.1 Applications of Pre-order Traversal:
- Creating a copy of a tree
- Getting prefix expression of an expression tree
- Serializing and deserializing a binary tree

4.2 Applications of In-order Traversal:
- Getting values in non-decreasing order in a binary search tree
- Getting infix expression of an expression tree

4.3 Applications of Post-order Traversal:
- Deleting a tree (delete children before parent)
- Getting postfix expression of an expression tree
- Finding the height of a tree

4.4 Applications of Level-order Traversal:
- Finding the minimum depth of a binary tree
- Printing a binary tree level by level
- Connecting nodes at the same level
"""

# Example: Checking if a binary tree is a binary search tree using in-order traversal
def is_bst(root):
    """
    Check if a binary tree is a valid binary search tree using in-order traversal.
    
    Args:
        root: The root of the binary tree.
        
    Returns:
        bool: True if the tree is a valid BST, False otherwise.
    """
    values = inorder_traversal_recursive(root)
    
    # In a BST, in-order traversal should yield values in ascending order
    for i in range(1, len(values)):
        if values[i] <= values[i - 1]:
            return False
    
    return True


# Example: Calculating the height of a binary tree using post-order traversal
def height_of_tree(root):
    """
    Calculate the height of a binary tree using post-order traversal.
    
    Args:
        root: The root of the binary tree.
        
    Returns:
        int: The height of the tree.
    """
    if not root:
        return -1
    
    # Calculate the height of the left and right subtrees
    left_height = height_of_tree(root.left)
    right_height = height_of_tree(root.right)
    
    # The height of the tree is the maximum of the heights of the left and right subtrees, plus 1
    return max(left_height, right_height) + 1


# Example: Serializing and deserializing a binary tree using pre-order traversal
def serialize(root):
    """
    Serialize a binary tree to a string using pre-order traversal.
    
    Args:
        root: The root of the binary tree.
        
    Returns:
        str: A string representation of the binary tree.
    """
    if not root:
        return "None,"
    
    # Pre-order traversal: Root -> Left -> Right
    return str(root.data) + "," + serialize(root.left) + serialize(root.right)


def deserialize(data):
    """
    Deserialize a string to a binary tree.
    
    Args:
        data: A string representation of a binary tree.
        
    Returns:
        TreeNode: The root of the reconstructed binary tree.
    """
    def _deserialize(nodes):
        if nodes[0] == "None":
            nodes.pop(0)
            return None
        
        root = TreeNode(int(nodes[0]))
        nodes.pop(0)
        root.left = _deserialize(nodes)
        root.right = _deserialize(nodes)
        return root
    
    nodes = data.split(",")
    return _deserialize(nodes)


#? 5. Comparison of Traversal Algorithms

"""
5.1 Time Complexity:
- All traversal algorithms (DFS and BFS) have O(n) time complexity, where n is the number of nodes.
- Each node is visited exactly once.

5.2 Space Complexity:
- Recursive DFS: O(h) space complexity, where h is the height of the tree.
  - In the worst case (skewed tree), h = n, resulting in O(n) space complexity.
  - In a balanced tree, h = log(n), resulting in O(log n) space complexity.
- Iterative DFS: O(h) space complexity for the stack.
- BFS: O(w) space complexity, where w is the maximum width of the tree.
  - In the worst case (perfect binary tree), w = n/2, resulting in O(n) space complexity.

5.3 When to Use Each Traversal:
- Pre-order: When you need to explore roots before leaves.
- In-order: When you need to explore a BST in sorted order.
- Post-order: When you need to explore leaves before roots.
- Level-order: When you need to explore the tree level by level.

5.4 Advantages and Disadvantages:

DFS Advantages:
- Uses less memory than BFS for deep trees.
- Finds nodes far from the root faster than BFS.
- Naturally recursive, making implementation simpler.

DFS Disadvantages:
- Can get stuck in infinite loops for graphs with cycles (need to track visited nodes).
- Not guaranteed to find the shortest path in unweighted graphs.

BFS Advantages:
- Finds the shortest path in unweighted graphs.
- Good for finding nodes close to the root.
- Avoids getting stuck in infinite loops.

BFS Disadvantages:
- Uses more memory than DFS for wide trees.
- Implementation can be more complex.
"""


#? 6. Testing Tree Traversal Algorithms

if __name__ == "__main__":
    # Create a sample binary tree
    #        1
    #       / \
    #      2   3
    #     / \   \
    #    4   5   6
    #       /
    #      7
    
    root = TreeNode(1)
    root.left = TreeNode(2)
    root.right = TreeNode(3)
    root.left.left = TreeNode(4)
    root.left.right = TreeNode(5)
    root.right.right = TreeNode(6)
    root.left.right.left = TreeNode(7)
    
    # Test DFS traversals
    print("DFS Traversals:")
    print(f"Pre-order (recursive): {preorder_traversal_recursive(root)}")
    print(f"Pre-order (iterative): {preorder_traversal_iterative(root)}")
    print(f"In-order (recursive): {inorder_traversal_recursive(root)}")
    print(f"In-order (iterative): {inorder_traversal_iterative(root)}")
    print(f"Post-order (recursive): {postorder_traversal_recursive(root)}")
    print(f"Post-order (iterative): {postorder_traversal_iterative(root)}")
    print(f"Post-order (two stacks): {postorder_traversal_two_stacks(root)}")
    
    # Test BFS traversals
    print("\nBFS Traversals:")
    print(f"Level-order: {level_order_traversal(root)}")
    print(f"Level-by-level: {level_by_level_traversal(root)}")
    
    # Test applications
    print("\nApplications:")
    print(f"Height of tree: {height_of_tree(root)}")
    
    # Create a binary search tree
    #        4
    #       / \
    #      2   6
    #     / \  / \
    #    1  3  5  7
    
    bst_root = TreeNode(4)
    bst_root.left = TreeNode(2)
    bst_root.right = TreeNode(6)
    bst_root.left.left = TreeNode(1)
    bst_root.left.right = TreeNode(3)
    bst_root.right.left = TreeNode(5)
    bst_root.right.right = TreeNode(7)
    
    print(f"Is BST: {is_bst(bst_root)}")
    
    # Test serialization and deserialization
    serialized = serialize(root)
    print(f"Serialized tree: {serialized}")
    
    deserialized = deserialize(serialized)
    print(f"Deserialized tree (pre-order): {preorder_traversal_recursive(deserialized)}")
