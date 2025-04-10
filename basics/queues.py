"""
Queue Data Structure

This file provides a comprehensive overview of the Queue data structure,
including implementations, operations, applications, and analysis.

A queue is a linear data structure that follows the First-In-First-Out (FIFO) principle.
"""

#? 1. Introduction to Queues
"""
Queue: A linear data structure that follows the First-In-First-Out (FIFO) principle.
- Elements are added at the rear/back of the queue (enqueue operation)
- Elements are removed from the front of the queue (dequeue operation)
- Only the front element can be accessed or removed

Visual representation of a queue:
    Front                      Rear
    |                           |
    v                           v
    [ A ][ B ][ C ][ D ][ E ][ F ]
    
    First in (A) will be the first out

Real-world analogies:
- People waiting in line (queue) at a ticket counter
- Print queue in a computer
- Traffic on a single-lane road
- Call center queue
"""

#? 2. Basic Queue Operations and Their Time Complexities
"""
Core Operations:
1. enqueue(item): Add an item to the rear of the queue - O(1)
2. dequeue(): Remove and return the front item from the queue - O(1)
3. front()/peek(): Return the front item without removing it - O(1)
4. is_empty(): Check if the queue is empty - O(1)
5. size(): Return the number of items in the queue - O(1)

All basic queue operations ideally have O(1) time complexity, but this depends
on the implementation. Some implementations may have different complexities for
certain operations.
"""

#? 3. Queue Implementations in Python

# 3.1 Using a List (inefficient for large queues)
class QueueUsingList:
    """Queue implementation using Python's built-in list.
    
    Note: This implementation is inefficient for large queues because
    dequeue operations have O(n) time complexity due to shifting elements.
    """
    
    def __init__(self):
        """Initialize an empty queue."""
        self.items = []
    
    def enqueue(self, item):
        """Add an item to the rear of the queue."""
        self.items.append(item)
    
    def dequeue(self):
        """Remove and return the front item from the queue."""
        if self.is_empty():
            raise IndexError("Dequeue from an empty queue")
        return self.items.pop(0)  # O(n) operation - inefficient
    
    def front(self):
        """Return the front item without removing it."""
        if self.is_empty():
            raise IndexError("Front from an empty queue")
        return self.items[0]
    
    def is_empty(self):
        """Check if the queue is empty."""
        return len(self.items) == 0
    
    def size(self):
        """Return the number of items in the queue."""
        return len(self.items)
    
    def __str__(self):
        """Return a string representation of the queue."""
        return str(self.items)


# 3.2 Using Two Stacks
class QueueUsingTwoStacks:
    """Queue implementation using two stacks.
    
    This implementation uses two stacks to achieve queue behavior.
    - enqueue: O(1) - just push to stack1
    - dequeue: Amortized O(1) - transfer all elements from stack1 to stack2 once,
               then pop from stack2. When stack2 is empty, transfer again.
    """
    
    def __init__(self):
        """Initialize an empty queue using two stacks."""
        self.stack1 = []  # for enqueue
        self.stack2 = []  # for dequeue
    
    def enqueue(self, item):
        """Add an item to the rear of the queue."""
        self.stack1.append(item)
    
    def dequeue(self):
        """Remove and return the front item from the queue."""
        if self.is_empty():
            raise IndexError("Dequeue from an empty queue")
        
        # If stack2 is empty, transfer all elements from stack1
        if not self.stack2:
            while self.stack1:
                self.stack2.append(self.stack1.pop())
        
        return self.stack2.pop()
    
    def front(self):
        """Return the front item without removing it."""
        if self.is_empty():
            raise IndexError("Front from an empty queue")
        
        # If stack2 is empty, transfer all elements from stack1
        if not self.stack2:
            while self.stack1:
                self.stack2.append(self.stack1.pop())
        
        return self.stack2[-1]
    
    def is_empty(self):
        """Check if the queue is empty."""
        return len(self.stack1) == 0 and len(self.stack2) == 0
    
    def size(self):
        """Return the number of items in the queue."""
        return len(self.stack1) + len(self.stack2)
    
    def __str__(self):
        """Return a string representation of the queue."""
        # Create a temporary copy for display purposes
        temp_stack1 = self.stack1.copy()
        temp_stack2 = self.stack2.copy()
        
        display = []
        
        # Add items from stack2 (front of queue)
        while temp_stack2:
            display.append(temp_stack2.pop())
        
        # Add items from stack1 (rear of queue)
        temp_items = []
        while temp_stack1:
            temp_items.append(temp_stack1.pop())
        
        # Reverse to maintain queue order
        while temp_items:
            display.append(temp_items.pop())
        
        return str(display)


# 3.3 Using a Linked List
class Node:
    """A node in a linked list."""
    
    def __init__(self, data):
        self.data = data
        self.next = None


class QueueUsingLinkedList:
    """Queue implementation using a linked list."""
    
    def __init__(self):
        """Initialize an empty queue."""
        self.front = None
        self.rear = None
        self._size = 0
    
    def enqueue(self, item):
        """Add an item to the rear of the queue."""
        new_node = Node(item)
        
        if self.is_empty():
            # If queue is empty, set both front and rear to the new node
            self.front = new_node
            self.rear = new_node
        else:
            # Add the new node at the rear and update rear
            self.rear.next = new_node
            self.rear = new_node
        
        self._size += 1
    
    def dequeue(self):
        """Remove and return the front item from the queue."""
        if self.is_empty():
            raise IndexError("Dequeue from an empty queue")
        
        # Get the data from the front node
        item = self.front.data
        
        # Move front to the next node
        self.front = self.front.next
        
        # If front becomes None, the queue is empty, so update rear as well
        if self.front is None:
            self.rear = None
        
        self._size -= 1
        return item
    
    def front_item(self):
        """Return the front item without removing it."""
        if self.is_empty():
            raise IndexError("Front from an empty queue")
        return self.front.data
    
    def is_empty(self):
        """Check if the queue is empty."""
        return self.front is None
    
    def size(self):
        """Return the number of items in the queue."""
        return self._size
    
    def __str__(self):
        """Return a string representation of the queue."""
        if self.is_empty():
            return "[]"
        
        current = self.front
        items = []
        while current:
            items.append(current.data)
            current = current.next
        
        return str(items)


# 3.4 Using collections.deque (most efficient)
from collections import deque

class QueueUsingDeque:
    """Queue implementation using collections.deque.
    
    This is the most efficient implementation for a queue in Python,
    as deque provides O(1) operations for both ends.
    """
    
    def __init__(self):
        """Initialize an empty queue."""
        self.items = deque()
    
    def enqueue(self, item):
        """Add an item to the rear of the queue."""
        self.items.append(item)
    
    def dequeue(self):
        """Remove and return the front item from the queue."""
        if self.is_empty():
            raise IndexError("Dequeue from an empty queue")
        return self.items.popleft()
    
    def front(self):
        """Return the front item without removing it."""
        if self.is_empty():
            raise IndexError("Front from an empty queue")
        return self.items[0]
    
    def is_empty(self):
        """Check if the queue is empty."""
        return len(self.items) == 0
    
    def size(self):
        """Return the number of items in the queue."""
        return len(self.items)
    
    def __str__(self):
        """Return a string representation of the queue."""
        return str(list(self.items))


#? 4. Common Applications of Queues

"""
4.1 Process Scheduling
- CPU scheduling in operating systems
- Task scheduling in real-time systems
- Print queue management

4.2 Breadth-First Search (BFS)
- Graph traversal
- Shortest path algorithms
- Web crawling

4.3 Buffering
- IO operations
- Video streaming
- Data transfer between processes

4.4 Message Queues
- Inter-process communication
- Distributed systems
- Event handling in GUI applications

4.5 Waiting Systems
- Customer service systems
- Call center routing
- Resource allocation
"""

# Example: Breadth-First Search using a queue
def breadth_first_search(graph, start):
    """
    Perform a breadth-first search on a graph.
    
    Args:
        graph (dict): A dictionary representing the graph as an adjacency list.
        start: The starting vertex.
        
    Returns:
        list: The vertices visited in BFS order.
    """
    visited = []
    queue = QueueUsingDeque()
    
    # Start with the given vertex
    queue.enqueue(start)
    
    while not queue.is_empty():
        # Dequeue a vertex
        vertex = queue.dequeue()
        
        # If the vertex hasn't been visited yet
        if vertex not in visited:
            # Mark it as visited
            visited.append(vertex)
            
            # Enqueue all adjacent vertices that haven't been visited
            for neighbor in graph.get(vertex, []):
                if neighbor not in visited:
                    queue.enqueue(neighbor)
    
    return visited


# Example: Level order traversal of a binary tree
class TreeNode:
    """A node in a binary tree."""
    
    def __init__(self, data):
        self.data = data
        self.left = None
        self.right = None


def level_order_traversal(root):
    """
    Perform a level order traversal of a binary tree.
    
    Args:
        root (TreeNode): The root of the binary tree.
        
    Returns:
        list: The nodes visited in level order.
    """
    if root is None:
        return []
    
    result = []
    queue = QueueUsingDeque()
    queue.enqueue(root)
    
    while not queue.is_empty():
        node = queue.dequeue()
        result.append(node.data)
        
        # Enqueue the left child if it exists
        if node.left:
            queue.enqueue(node.left)
        
        # Enqueue the right child if it exists
        if node.right:
            queue.enqueue(node.right)
    
    return result


#? 5. Example Problems and Solutions

# Example 1: Implement a queue using a fixed-size array (circular queue)
class CircularQueue:
    """A queue implementation using a fixed-size array.
    
    This implementation uses a circular buffer to efficiently use the available space.
    When the queue is full and an element is dequeued, the front position becomes
    available for the next enqueue operation.
    """
    
    def __init__(self, capacity):
        """Initialize an empty circular queue with the given capacity."""
        self.capacity = capacity
        self.items = [None] * capacity
        self.front = -1
        self.rear = -1
        self._size = 0
    
    def enqueue(self, item):
        """Add an item to the rear of the queue."""
        if self.is_full():
            raise IndexError("Enqueue to a full queue")
        
        # If queue is empty, set front to 0
        if self.is_empty():
            self.front = 0
        
        # Move rear circularly
        self.rear = (self.rear + 1) % self.capacity
        self.items[self.rear] = item
        self._size += 1
    
    def dequeue(self):
        """Remove and return the front item from the queue."""
        if self.is_empty():
            raise IndexError("Dequeue from an empty queue")
        
        # Get the item at front
        item = self.items[self.front]
        
        # If there's only one element, reset the queue
        if self.front == self.rear:
            self.front = -1
            self.rear = -1
        else:
            # Move front circularly
            self.front = (self.front + 1) % self.capacity
        
        self._size -= 1
        return item
    
    def front_item(self):
        """Return the front item without removing it."""
        if self.is_empty():
            raise IndexError("Front from an empty queue")
        return self.items[self.front]
    
    def is_empty(self):
        """Check if the queue is empty."""
        return self.front == -1
    
    def is_full(self):
        """Check if the queue is full."""
        return (self.rear + 1) % self.capacity == self.front
    
    def size(self):
        """Return the number of items in the queue."""
        return self._size
    
    def __str__(self):
        """Return a string representation of the queue."""
        if self.is_empty():
            return "[]"
        
        items = []
        index = self.front
        
        # Traverse from front to rear
        while True:
            items.append(self.items[index])
            
            if index == self.rear:
                break
            
            index = (index + 1) % self.capacity
        
        return str(items)


# Example 2: Implement a queue that supports finding the maximum element in O(1) time
class MaxQueue:
    """A queue that supports finding the maximum element in O(1) time."""
    
    def __init__(self):
        """Initialize an empty max queue."""
        self.main_queue = QueueUsingDeque()
        self.max_queue = QueueUsingDeque()  # Stores potential maximums
    
    def enqueue(self, item):
        """Add an item to the rear of the queue."""
        self.main_queue.enqueue(item)
        
        # Remove all elements from max_queue that are smaller than the new item
        while not self.max_queue.is_empty() and self.max_queue.items[-1] < item:
            self.max_queue.items.pop()
        
        # Add the new item to the max_queue
        self.max_queue.items.append(item)
    
    def dequeue(self):
        """Remove and return the front item from the queue."""
        if self.is_empty():
            raise IndexError("Dequeue from an empty queue")
        
        item = self.main_queue.dequeue()
        
        # If the dequeued item is the current maximum, remove it from max_queue
        if item == self.max_queue.front():
            self.max_queue.dequeue()
        
        return item
    
    def front(self):
        """Return the front item without removing it."""
        return self.main_queue.front()
    
    def get_max(self):
        """Return the maximum element in the queue."""
        if self.is_empty():
            raise IndexError("Max from an empty queue")
        
        return self.max_queue.front()
    
    def is_empty(self):
        """Check if the queue is empty."""
        return self.main_queue.is_empty()
    
    def size(self):
        """Return the number of items in the queue."""
        return self.main_queue.size()
    
    def __str__(self):
        """Return a string representation of the queue."""
        return str(self.main_queue)


# Example 3: Implement a queue using a priority queue (elements with higher priority are dequeued first)
class PriorityQueue:
    """A priority queue implementation.
    
    Elements with higher priority (lower priority value) are dequeued first.
    Elements with the same priority are dequeued in the order they were enqueued (FIFO).
    """
    
    def __init__(self):
        """Initialize an empty priority queue."""
        self.items = []  # List of (priority, sequence, item) tuples
        self.sequence = 0  # Used to maintain FIFO order for same priority
    
    def enqueue(self, item, priority=0):
        """
        Add an item to the priority queue.
        
        Args:
            item: The item to add.
            priority (int): The priority of the item (lower value = higher priority).
        """
        self.items.append((priority, self.sequence, item))
        self.sequence += 1
        self._heapify_up(len(self.items) - 1)
    
    def dequeue(self):
        """Remove and return the highest priority item from the queue."""
        if self.is_empty():
            raise IndexError("Dequeue from an empty priority queue")
        
        # Swap the first and last items
        self._swap(0, len(self.items) - 1)
        
        # Remove and return the highest priority item
        _, _, item = self.items.pop()
        
        # Restore the heap property
        if self.items:
            self._heapify_down(0)
        
        return item
    
    def peek(self):
        """Return the highest priority item without removing it."""
        if self.is_empty():
            raise IndexError("Peek from an empty priority queue")
        
        return self.items[0][2]
    
    def is_empty(self):
        """Check if the priority queue is empty."""
        return len(self.items) == 0
    
    def size(self):
        """Return the number of items in the priority queue."""
        return len(self.items)
    
    def _parent(self, index):
        """Return the parent index of the given index."""
        return (index - 1) // 2
    
    def _left_child(self, index):
        """Return the left child index of the given index."""
        return 2 * index + 1
    
    def _right_child(self, index):
        """Return the right child index of the given index."""
        return 2 * index + 2
    
    def _swap(self, i, j):
        """Swap the items at indices i and j."""
        self.items[i], self.items[j] = self.items[j], self.items[i]
    
    def _heapify_up(self, index):
        """Restore the heap property by moving the item at index up."""
        parent = self._parent(index)
        
        if index > 0 and self.items[index][0] < self.items[parent][0]:
            self._swap(index, parent)
            self._heapify_up(parent)
    
    def _heapify_down(self, index):
        """Restore the heap property by moving the item at index down."""
        smallest = index
        left = self._left_child(index)
        right = self._right_child(index)
        
        if (left < len(self.items) and
                self.items[left][0] < self.items[smallest][0]):
            smallest = left
        
        if (right < len(self.items) and
                self.items[right][0] < self.items[smallest][0]):
            smallest = right
        
        if smallest != index:
            self._swap(index, smallest)
            self._heapify_down(smallest)
    
    def __str__(self):
        """Return a string representation of the priority queue."""
        # Sort by priority and sequence for display
        sorted_items = sorted(self.items)
        return str([item for _, _, item in sorted_items])


#? 6. Performance Analysis and Comparison

"""
6.1 Time Complexity Analysis

Operation       | List-based | Two Stacks | Linked List | Deque-based | Circular Queue
----------------|------------|------------|-------------|-------------|---------------
enqueue()       | O(1)       | O(1)       | O(1)        | O(1)        | O(1)
dequeue()       | O(n)       | O(1)*      | O(1)        | O(1)        | O(1)
front()         | O(1)       | O(1)*      | O(1)        | O(1)        | O(1)
is_empty()      | O(1)       | O(1)       | O(1)        | O(1)        | O(1)
size()          | O(1)       | O(1)       | O(1)        | O(1)        | O(1)

* Note: Two Stacks implementation has amortized O(1) for dequeue and front operations.

6.2 Space Complexity Analysis

All implementations have O(n) space complexity, where n is the number of elements.

6.3 Implementation Considerations

List-based:
- Pros: Simple implementation
- Cons: Inefficient dequeue operation (O(n))

Two Stacks:
- Pros: All operations are amortized O(1)
- Cons: More complex implementation, potentially higher constant factors

Linked List:
- Pros: Efficient operations, no resizing issues
- Cons: Extra memory for node pointers

Deque-based:
- Pros: Most efficient implementation in Python, all operations are O(1)
- Cons: Slightly more complex than a simple list

Circular Queue:
- Pros: Efficient use of fixed-size array, all operations are O(1)
- Cons: Fixed capacity, more complex implementation

6.4 When to Use Each Implementation

- Use deque-based for most general-purpose queue needs
- Use linked list when memory efficiency is important
- Use circular queue when you need a fixed-size queue
- Use priority queue when elements need to be processed based on priority
- Avoid list-based implementation for large queues due to inefficient dequeue
"""

#? 7. Testing Queue Implementations

if __name__ == "__main__":
    # Test the list-based queue
    print("Testing QueueUsingList:")
    queue1 = QueueUsingList()
    queue1.enqueue(1)
    queue1.enqueue(2)
    queue1.enqueue(3)
    print(f"Queue: {queue1}")
    print(f"Dequeue: {queue1.dequeue()}")
    print(f"Front: {queue1.front()}")
    print(f"Size: {queue1.size()}")
    print(f"Is empty: {queue1.is_empty()}")
    print()
    
    # Test the two stacks queue
    print("Testing QueueUsingTwoStacks:")
    queue2 = QueueUsingTwoStacks()
    queue2.enqueue(1)
    queue2.enqueue(2)
    queue2.enqueue(3)
    print(f"Queue: {queue2}")
    print(f"Dequeue: {queue2.dequeue()}")
    print(f"Front: {queue2.front()}")
    print(f"Size: {queue2.size()}")
    print(f"Is empty: {queue2.is_empty()}")
    print()
    
    # Test the linked list queue
    print("Testing QueueUsingLinkedList:")
    queue3 = QueueUsingLinkedList()
    queue3.enqueue(1)
    queue3.enqueue(2)
    queue3.enqueue(3)
    print(f"Queue: {queue3}")
    print(f"Dequeue: {queue3.dequeue()}")
    print(f"Front: {queue3.front_item()}")
    print(f"Size: {queue3.size()}")
    print(f"Is empty: {queue3.is_empty()}")
    print()
    
    # Test the deque-based queue
    print("Testing QueueUsingDeque:")
    queue4 = QueueUsingDeque()
    queue4.enqueue(1)
    queue4.enqueue(2)
    queue4.enqueue(3)
    print(f"Queue: {queue4}")
    print(f"Dequeue: {queue4.dequeue()}")
    print(f"Front: {queue4.front()}")
    print(f"Size: {queue4.size()}")
    print(f"Is empty: {queue4.is_empty()}")
    print()
    
    # Test the circular queue
    print("Testing CircularQueue:")
    queue5 = CircularQueue(5)
    queue5.enqueue(1)
    queue5.enqueue(2)
    queue5.enqueue(3)
    print(f"Queue: {queue5}")
    print(f"Dequeue: {queue5.dequeue()}")
    print(f"Front: {queue5.front_item()}")
    print(f"Size: {queue5.size()}")
    print(f"Is empty: {queue5.is_empty()}")
    print()
    
    # Test the max queue
    print("Testing MaxQueue:")
    max_queue = MaxQueue()
    max_queue.enqueue(3)
    max_queue.enqueue(5)
    max_queue.enqueue(2)
    max_queue.enqueue(1)
    print(f"Queue: {max_queue}")
    print(f"Max: {max_queue.get_max()}")
    max_queue.dequeue()  # Dequeue 3
    print(f"Max after dequeue: {max_queue.get_max()}")
    max_queue.dequeue()  # Dequeue 5
    print(f"Max after another dequeue: {max_queue.get_max()}")
    print()
    
    # Test the priority queue
    print("Testing PriorityQueue:")
    pq = PriorityQueue()
    pq.enqueue("Task 1", 3)
    pq.enqueue("Task 2", 1)
    pq.enqueue("Task 3", 2)
    pq.enqueue("Task 4", 1)
    print(f"Priority Queue: {pq}")
    print(f"Dequeue: {pq.dequeue()}")  # Should be Task 2 (priority 1, first in)
    print(f"Dequeue: {pq.dequeue()}")  # Should be Task 4 (priority 1, second in)
    print(f"Dequeue: {pq.dequeue()}")  # Should be Task 3 (priority 2)
    print()
    
    # Test BFS
    print("Testing Breadth-First Search:")
    graph = {
        'A': ['B', 'C'],
        'B': ['A', 'D', 'E'],
        'C': ['A', 'F'],
        'D': ['B'],
        'E': ['B', 'F'],
        'F': ['C', 'E']
    }
    print(f"BFS starting from 'A': {breadth_first_search(graph, 'A')}")
    print()
    
    # Test level order traversal
    print("Testing Level Order Traversal:")
    root = TreeNode(1)
    root.left = TreeNode(2)
    root.right = TreeNode(3)
    root.left.left = TreeNode(4)
    root.left.right = TreeNode(5)
    root.right.left = TreeNode(6)
    root.right.right = TreeNode(7)
    print(f"Level order traversal: {level_order_traversal(root)}")
