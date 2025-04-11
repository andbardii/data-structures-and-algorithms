"""
Tree Data Structure

This file provides a comprehensive overview of Tree data structures,
including implementations, operations, applications, and analysis.

A tree is a hierarchical data structure consisting of nodes connected by edges.
"""

#? 1. Introduction to Trees
"""
Tree: A hierarchical data structure consisting of nodes connected by edges.
- Each tree has a root node
- Each node can have zero or more child nodes
- Each node (except the root) has exactly one parent node
- Nodes with no children are called leaf nodes

Visual representation of a binary tree:
        A         <- Root
       / \
      B   C       <- Internal nodes
     / \   \
    D   E   F     <- Leaf nodes

Key terminology:
- Root: The topmost node of the tree
- Parent: A node that has one or more child nodes
- Child: A node that has a parent node
- Leaf/Terminal: A node with no children
- Internal/Non-terminal: A node with at least one child
- Sibling: Nodes that share the same parent
- Depth: The length of the path from the root to the node
- Height: The length of the longest path from the node to a leaf
- Level: The generation of a node (root is at level 0)
- Degree: The number of children a node has
- Edge: The connection between two nodes
- Path: A sequence of nodes and edges connecting two nodes

Common types of trees:
- Binary Tree: Each node has at most two children
- Binary Search Tree (BST): A binary tree where left child < parent < right child
- AVL Tree: A self-balancing binary search tree
- Red-Black Tree: A self-balancing binary search tree
- B-Tree: A self-balancing search tree with more than two children
- Heap: A complete binary tree where parent has a specific relationship with children
- Trie: A tree-like data structure for storing strings
"""

#? 2. Basic Tree Operations and Their Time Complexities
"""
Core Operations for Binary Trees:
1. Insertion: Add a new node to the tree - O(h) where h is the height of the tree
2. Deletion: Remove a node from the tree - O(h)
3. Search: Find a node with a given value - O(h)
4. Traversal: Visit all nodes in a specific order - O(n) where n is the number of nodes

For a balanced binary tree, h = log(n), so operations are O(log n).
For an unbalanced binary tree, in the worst case h = n, so operations are O(n).

Common Tree Traversal Methods:
1. Depth-First Search (DFS):
   - Preorder: Visit root, then left subtree, then right subtree
   - Inorder: Visit left subtree, then root, then right subtree
   - Postorder: Visit left subtree, then right subtree, then root

2. Breadth-First Search (BFS):
   - Level Order: Visit nodes level by level from top to bottom
"""

#? 3. Tree Implementations in Python

# 3.1 Basic Binary Tree
class TreeNode:
    """A node in a binary tree."""

    def __init__(self, data):
        """Initialize a tree node with the given data."""
        self.data = data
        self.left = None
        self.right = None


class BinaryTree:
    """Basic binary tree implementation."""

    def __init__(self, root=None):
        """Initialize a binary tree with an optional root node."""
        self.root = root

    def is_empty(self):
        """Check if the tree is empty."""
        return self.root is None

    # Depth-First Traversals
    def preorder_traversal(self, node, result=None):
        """
        Perform a preorder traversal (Root -> Left -> Right).

        Args:
            node: The current node.
            result: List to store the traversal result.

        Returns:
            list: The nodes visited in preorder.
        """
        if result is None:
            result = []

        if node:
            # Visit the root
            result.append(node.data)
            # Traverse the left subtree
            self.preorder_traversal(node.left, result)
            # Traverse the right subtree
            self.preorder_traversal(node.right, result)

        return result

    def inorder_traversal(self, node, result=None):
        """
        Perform an inorder traversal (Left -> Root -> Right).

        Args:
            node: The current node.
            result: List to store the traversal result.

        Returns:
            list: The nodes visited in inorder.
        """
        if result is None:
            result = []

        if node:
            # Traverse the left subtree
            self.inorder_traversal(node.left, result)
            # Visit the root
            result.append(node.data)
            # Traverse the right subtree
            self.inorder_traversal(node.right, result)

        return result

    def postorder_traversal(self, node, result=None):
        """
        Perform a postorder traversal (Left -> Right -> Root).

        Args:
            node: The current node.
            result: List to store the traversal result.

        Returns:
            list: The nodes visited in postorder.
        """
        if result is None:
            result = []

        if node:
            # Traverse the left subtree
            self.postorder_traversal(node.left, result)
            # Traverse the right subtree
            self.postorder_traversal(node.right, result)
            # Visit the root
            result.append(node.data)

        return result

    # Breadth-First Traversal
    def level_order_traversal(self, node):
        """
        Perform a level order traversal (breadth-first).

        Args:
            node: The starting node.

        Returns:
            list: The nodes visited in level order.
        """
        if not node:
            return []

        result = []
        queue = [node]

        while queue:
            current = queue.pop(0)
            result.append(current.data)

            if current.left:
                queue.append(current.left)

            if current.right:
                queue.append(current.right)

        return result

    def height(self, node):
        """
        Calculate the height of a node (longest path to a leaf).

        Args:
            node: The node to calculate height for.

        Returns:
            int: The height of the node.
        """
        if not node:
            return -1

        left_height = self.height(node.left)
        right_height = self.height(node.right)

        return max(left_height, right_height) + 1

    def size(self, node):
        """
        Calculate the number of nodes in the tree.

        Args:
            node: The root of the tree or subtree.

        Returns:
            int: The number of nodes.
        """
        if not node:
            return 0

        return self.size(node.left) + self.size(node.right) + 1


# 3.2 Binary Search Tree (BST)
class BinarySearchTree:
    """Binary Search Tree implementation.

    A binary search tree is a binary tree where for each node:
    - All nodes in the left subtree have values less than the node's value
    - All nodes in the right subtree have values greater than the node's value
    """

    def __init__(self):
        """Initialize an empty binary search tree."""
        self.root = None

    def is_empty(self):
        """Check if the tree is empty."""
        return self.root is None

    def insert(self, data):
        """Insert a new node with the given data into the BST."""
        if self.is_empty():
            self.root = TreeNode(data)
        else:
            self._insert_recursive(self.root, data)

    def _insert_recursive(self, node, data):
        """Helper method to recursively insert data into the BST."""
        if data < node.data:
            if node.left is None:
                node.left = TreeNode(data)
            else:
                self._insert_recursive(node.left, data)
        else:  # data >= node.data
            if node.right is None:
                node.right = TreeNode(data)
            else:
                self._insert_recursive(node.right, data)

    def search(self, data):
        """Search for a node with the given data in the BST."""
        return self._search_recursive(self.root, data)

    def _search_recursive(self, node, data):
        """Helper method to recursively search for data in the BST."""
        if node is None or node.data == data:
            return node

        if data < node.data:
            return self._search_recursive(node.left, data)
        else:
            return self._search_recursive(node.right, data)

    def delete(self, data):
        """Delete a node with the given data from the BST."""
        self.root = self._delete_recursive(self.root, data)

    def _delete_recursive(self, node, data):
        """Helper method to recursively delete a node from the BST."""
        # Base case: If the tree is empty
        if node is None:
            return None

        # Recursively search for the node to delete
        if data < node.data:
            node.left = self._delete_recursive(node.left, data)
        elif data > node.data:
            node.right = self._delete_recursive(node.right, data)
        else:
            # Node with only one child or no child
            if node.left is None:
                return node.right
            elif node.right is None:
                return node.left

            # Node with two children: Get the inorder successor (smallest
            # in the right subtree)
            node.data = self._min_value(node.right)

            # Delete the inorder successor
            node.right = self._delete_recursive(node.right, node.data)

        return node

    def _min_value(self, node):
        """Find the minimum value in a subtree."""
        current = node

        # Loop down to find the leftmost leaf
        while current.left is not None:
            current = current.left

        return current.data

    def inorder_traversal(self):
        """Perform an inorder traversal of the BST."""
        result = []
        self._inorder_recursive(self.root, result)
        return result

    def _inorder_recursive(self, node, result):
        """Helper method for inorder traversal."""
        if node:
            self._inorder_recursive(node.left, result)
            result.append(node.data)
            self._inorder_recursive(node.right, result)


# 3.3 AVL Tree (Self-balancing Binary Search Tree)
class AVLNode(TreeNode):
    """A node in an AVL tree."""

    def __init__(self, data):
        """Initialize an AVL tree node with the given data."""
        super().__init__(data)
        self.height = 1  # Height of the node (used for balancing)


class AVLTree:
    """AVL Tree implementation.

    An AVL tree is a self-balancing binary search tree where the difference
    between heights of left and right subtrees cannot be more than one for all nodes.
    """

    def __init__(self):
        """Initialize an empty AVL tree."""
        self.root = None

    def height(self, node):
        """Get the height of a node."""
        if not node:
            return 0
        return node.height

    def balance_factor(self, node):
        """Calculate the balance factor of a node."""
        if not node:
            return 0
        return self.height(node.left) - self.height(node.right)

    def update_height(self, node):
        """Update the height of a node based on its children's heights."""
        if not node:
            return
        node.height = 1 + max(self.height(node.left), self.height(node.right))

    def right_rotate(self, y):
        """Perform a right rotation on node y."""
        x = y.left
        T2 = x.right

        # Perform rotation
        x.right = y
        y.left = T2

        # Update heights
        self.update_height(y)
        self.update_height(x)

        # Return new root
        return x

    def left_rotate(self, x):
        """Perform a left rotation on node x."""
        y = x.right
        T2 = y.left

        # Perform rotation
        y.left = x
        x.right = T2

        # Update heights
        self.update_height(x)
        self.update_height(y)

        # Return new root
        return y

    def insert(self, data):
        """Insert a new node with the given data into the AVL tree."""
        self.root = self._insert_recursive(self.root, data)

    def _insert_recursive(self, node, data):
        """Helper method to recursively insert data into the AVL tree."""
        # Perform standard BST insert
        if not node:
            return AVLNode(data)

        if data < node.data:
            node.left = self._insert_recursive(node.left, data)
        else:
            node.right = self._insert_recursive(node.right, data)

        # Update height of current node
        self.update_height(node)

        # Get the balance factor to check if this node became unbalanced
        balance = self.balance_factor(node)

        # If unbalanced, there are 4 cases

        # Left Left Case
        if balance > 1 and data < node.left.data:
            return self.right_rotate(node)

        # Right Right Case
        if balance < -1 and data > node.right.data:
            return self.left_rotate(node)

        # Left Right Case
        if balance > 1 and data > node.left.data:
            node.left = self.left_rotate(node.left)
            return self.right_rotate(node)

        # Right Left Case
        if balance < -1 and data < node.right.data:
            node.right = self.right_rotate(node.right)
            return self.left_rotate(node)

        # Return the unchanged node pointer
        return node

    def delete(self, data):
        """Delete a node with the given data from the AVL tree."""
        if not self.root:
            return
        self.root = self._delete_recursive(self.root, data)

    def _delete_recursive(self, node, data):
        """Helper method to recursively delete a node from the AVL tree."""
        # Perform standard BST delete
        if not node:
            return node

        if data < node.data:
            node.left = self._delete_recursive(node.left, data)
        elif data > node.data:
            node.right = self._delete_recursive(node.right, data)
        else:
            # Node with only one child or no child
            if not node.left:
                return node.right
            elif not node.right:
                return node.left

            # Node with two children: Get the inorder successor
            temp = self._get_min_value_node(node.right)
            node.data = temp.data
            node.right = self._delete_recursive(node.right, temp.data)

        # If the tree had only one node, return
        if not node:
            return node

        # Update height of current node
        self.update_height(node)

        # Get the balance factor to check if this node became unbalanced
        balance = self.balance_factor(node)

        # If unbalanced, there are 4 cases

        # Left Left Case
        if balance > 1 and self.balance_factor(node.left) >= 0:
            return self.right_rotate(node)

        # Left Right Case
        if balance > 1 and self.balance_factor(node.left) < 0:
            node.left = self.left_rotate(node.left)
            return self.right_rotate(node)

        # Right Right Case
        if balance < -1 and self.balance_factor(node.right) <= 0:
            return self.left_rotate(node)

        # Right Left Case
        if balance < -1 and self.balance_factor(node.right) > 0:
            node.right = self.right_rotate(node.right)
            return self.left_rotate(node)

        return node

    def _get_min_value_node(self, node):
        """Find the node with the minimum value in a subtree."""
        current = node

        # Loop down to find the leftmost leaf
        while current.left:
            current = current.left

        return current

    def inorder_traversal(self):
        """Perform an inorder traversal of the AVL tree."""
        result = []
        self._inorder_recursive(self.root, result)
        return result

    def _inorder_recursive(self, node, result):
        """Helper method for inorder traversal."""
        if node:
            self._inorder_recursive(node.left, result)
            result.append(node.data)
            self._inorder_recursive(node.right, result)


#? 4. Common Applications of Trees

"""
4.1 Hierarchical Data Representation
- File systems (directories and files)
- Organization charts
- XML/HTML DOM
- Family trees

4.2 Database Indexing
- B-Trees and B+ Trees for database indexes
- Speeds up data retrieval operations

4.3 Routing Algorithms
- Network routing tables
- Tries for IP routing

4.4 Decision Trees
- Machine learning algorithms
- Game AI (minimax algorithm)

4.5 Syntax Trees
- Compilers and interpreters
- Expression evaluation

4.6 Huffman Coding
- Data compression algorithms

4.7 Search Operations
- Binary Search Trees for efficient search
- AVL and Red-Black trees for balanced search
"""


#? 5. Example Problems and Solutions

# Example 1: Check if a binary tree is a binary search tree
def is_bst(root, min_val=float('-inf'), max_val=float('inf')):
    """
    Check if a binary tree is a valid binary search tree.

    Args:
        root: The root of the binary tree.
        min_val: The minimum allowed value for the current subtree.
        max_val: The maximum allowed value for the current subtree.

    Returns:
        bool: True if the tree is a valid BST, False otherwise.
    """
    # An empty tree is a BST
    if root is None:
        return True

    # Check if the current node's value is within the allowed range
    if root.data <= min_val or root.data >= max_val:
        return False

    # Recursively check the left and right subtrees
    # For the left subtree, the max value should be the current node's value
    # For the right subtree, the min value should be the current node's value
    return (is_bst(root.left, min_val, root.data) and
            is_bst(root.right, root.data, max_val))


# Example 2: Find the lowest common ancestor (LCA) in a binary tree
def find_lca(root, p, q):
    """
    Find the lowest common ancestor of two nodes in a binary tree.

    Args:
        root: The root of the binary tree.
        p: The first node.
        q: The second node.

    Returns:
        TreeNode: The lowest common ancestor node.
    """
    # Base case
    if root is None or root.data == p or root.data == q:
        return root

    # Look for keys in left and right subtrees
    left_lca = find_lca(root.left, p, q)
    right_lca = find_lca(root.right, p, q)

    # If both nodes are found in different subtrees, current node is the LCA
    if left_lca and right_lca:
        return root

    # Otherwise, return the non-None value
    return left_lca if left_lca else right_lca


# Example 3: Serialize and deserialize a binary tree
def serialize(root):
    """
    Serialize a binary tree to a string.

    Args:
        root: The root of the binary tree.

    Returns:
        str: A string representation of the binary tree.
    """
    if not root:
        return "None,"

    # Preorder traversal: Root -> Left -> Right
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


#? 6. Performance Analysis and Comparison

"""
6.1 Time Complexity Analysis

Operation       | Binary Tree | BST (avg)  | BST (worst) | AVL Tree  | Red-Black Tree
----------------|------------|-----------|------------|-----------|---------------
Search          | O(n)       | O(log n)  | O(n)       | O(log n)  | O(log n)
Insert          | O(1)*      | O(log n)  | O(n)       | O(log n)  | O(log n)
Delete          | O(n)       | O(log n)  | O(n)       | O(log n)  | O(log n)
Traversal       | O(n)       | O(n)      | O(n)       | O(n)      | O(n)
Height          | O(n)       | O(log n)* | O(n)       | O(log n)  | O(log n)

* Note: Insert in a binary tree is O(1) if we know where to insert.
* Note: Height of a BST is O(log n) on average, but can be O(n) in the worst case.

6.2 Space Complexity Analysis

All tree implementations have O(n) space complexity, where n is the number of nodes.

6.3 Implementation Considerations

Binary Tree:
- Pros: Simple implementation, flexible structure
- Cons: No ordering guarantees, inefficient search

Binary Search Tree (BST):
- Pros: Efficient search, insert, and delete operations (on average)
- Cons: Can degenerate to a linked list in worst case (O(n) operations)

AVL Tree:
- Pros: Guaranteed O(log n) operations, strictly balanced
- Cons: More rotations during insert/delete than other balanced trees

Red-Black Tree:
- Pros: Efficient operations, fewer rotations than AVL trees
- Cons: Not as strictly balanced as AVL trees

B-Tree:
- Pros: Optimized for disk access, good for databases and file systems
- Cons: More complex implementation

6.4 When to Use Each Implementation

- Use binary trees for simple hierarchical data representation
- Use BST when you need ordered operations and the data is randomly distributed
- Use AVL trees when you need strict balancing and frequent lookups
- Use Red-Black trees when you need balanced operations with frequent insertions/deletions
- Use B-Trees for disk-based storage and databases
"""


#? 7. Testing Tree Implementations

if __name__ == "__main__":
    # Test the basic binary tree
    print("Testing Basic Binary Tree:")
    tree = BinaryTree()
    root = TreeNode(1)
    root.left = TreeNode(2)
    root.right = TreeNode(3)
    root.left.left = TreeNode(4)
    root.left.right = TreeNode(5)
    tree.root = root

    print(f"Preorder traversal: {tree.preorder_traversal(tree.root)}")
    print(f"Inorder traversal: {tree.inorder_traversal(tree.root)}")
    print(f"Postorder traversal: {tree.postorder_traversal(tree.root)}")
    print(f"Level order traversal: {tree.level_order_traversal(tree.root)}")
    print(f"Height of the tree: {tree.height(tree.root)}")
    print(f"Size of the tree: {tree.size(tree.root)}")
    print()

    # Test the binary search tree
    print("Testing Binary Search Tree:")
    bst = BinarySearchTree()
    bst.insert(50)
    bst.insert(30)
    bst.insert(70)
    bst.insert(20)
    bst.insert(40)
    bst.insert(60)
    bst.insert(80)

    print(f"Inorder traversal: {bst.inorder_traversal()}")
    print(f"Search for 40: {bst.search(40) is not None}")
    print(f"Search for 90: {bst.search(90) is not None}")

    bst.delete(20)  # Delete a leaf node
    print(f"After deleting 20: {bst.inorder_traversal()}")

    bst.delete(30)  # Delete a node with one child
    print(f"After deleting 30: {bst.inorder_traversal()}")

    bst.delete(50)  # Delete the root node
    print(f"After deleting 50: {bst.inorder_traversal()}")
    print()

    # Test the AVL tree
    print("Testing AVL Tree:")
    avl = AVLTree()
    avl.insert(10)
    avl.insert(20)
    avl.insert(30)  # This should trigger a rotation
    avl.insert(40)
    avl.insert(50)  # This should trigger another rotation
    avl.insert(25)

    print(f"Inorder traversal: {avl.inorder_traversal()}")

    avl.delete(10)
    print(f"After deleting 10: {avl.inorder_traversal()}")
    print()

    # Test the example problems
    print("Testing Example Problems:")

    # Create a valid BST
    valid_bst = TreeNode(8)
    valid_bst.left = TreeNode(3)
    valid_bst.right = TreeNode(10)
    valid_bst.left.left = TreeNode(1)
    valid_bst.left.right = TreeNode(6)
    valid_bst.left.right.left = TreeNode(4)
    valid_bst.left.right.right = TreeNode(7)

    # Create an invalid BST
    invalid_bst = TreeNode(8)
    invalid_bst.left = TreeNode(3)
    invalid_bst.right = TreeNode(10)
    invalid_bst.left.left = TreeNode(1)
    invalid_bst.left.right = TreeNode(6)
    invalid_bst.left.right.left = TreeNode(4)
    invalid_bst.left.right.right = TreeNode(9)  # This should be less than 8

    print(f"Is valid_bst a BST? {is_bst(valid_bst)}")
    print(f"Is invalid_bst a BST? {is_bst(invalid_bst)}")

    # Test LCA
    lca_tree = TreeNode(3)
    lca_tree.left = TreeNode(5)
    lca_tree.right = TreeNode(1)
    lca_tree.left.left = TreeNode(6)
    lca_tree.left.right = TreeNode(2)
    lca_tree.right.left = TreeNode(0)
    lca_tree.right.right = TreeNode(8)
    lca_tree.left.right.left = TreeNode(7)
    lca_tree.left.right.right = TreeNode(4)

    lca_node = find_lca(lca_tree, 5, 1)
    print(f"LCA of 5 and 1: {lca_node.data}")

    lca_node = find_lca(lca_tree, 6, 4)
    print(f"LCA of 6 and 4: {lca_node.data}")

    # Test serialization/deserialization
    ser_tree = TreeNode(1)
    ser_tree.left = TreeNode(2)
    ser_tree.right = TreeNode(3)
    ser_tree.right.left = TreeNode(4)
    ser_tree.right.right = TreeNode(5)

    serialized = serialize(ser_tree)
    print(f"Serialized tree: {serialized}")

    deserialized = deserialize(serialized)
    print(f"Deserialized tree (preorder): {BinaryTree().preorder_traversal(deserialized)}")