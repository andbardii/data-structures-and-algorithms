"""
Stack Data Structure

This file provides a comprehensive overview of the Stack data structure,
including implementations, operations, applications, and analysis.

A stack is a linear data structure that follows the Last-In-First-Out (LIFO) principle.
"""

#? 1. Introduction to Stacks
"""
Stack: A linear data structure that follows the Last-In-First-Out (LIFO) principle.
- Elements are added to the top of the stack (push operation)
- Elements are removed from the top of the stack (pop operation)
- Only the top element is accessible at any time

Visual representation of a stack:
    | D | <- Top (most recently added)
    | C |
    | B |
    | A | <- Bottom (least recently added)

Real-world analogies:
- Stack of plates
- Stack of books
- Browser history (back button)
- Function call stack in programming languages
"""

#? 2. Basic Stack Operations and Their Time Complexities
"""
Core Operations:
1. push(item): Add an item to the top of the stack - O(1)
2. pop(): Remove and return the top item from the stack - O(1)
3. peek()/top(): Return the top item without removing it - O(1)
4. is_empty(): Check if the stack is empty - O(1)
5. size(): Return the number of items in the stack - O(1)

All basic stack operations have O(1) time complexity, making stacks very efficient
for their intended use cases.
"""

#? 3. Stack Implementations in Python

# 3.1 Using a List
class StackUsingList:
    """Stack implementation using Python's built-in list."""
    
    def __init__(self):
        """Initialize an empty stack."""
        self.items = []
    
    def push(self, item):
        """Add an item to the top of the stack."""
        self.items.append(item)
    
    def pop(self):
        """Remove and return the top item from the stack."""
        if self.is_empty():
            raise IndexError("Pop from an empty stack")
        return self.items.pop()
    
    def peek(self):
        """Return the top item without removing it."""
        if self.is_empty():
            raise IndexError("Peek from an empty stack")
        return self.items[-1]
    
    def is_empty(self):
        """Check if the stack is empty."""
        return len(self.items) == 0
    
    def size(self):
        """Return the number of items in the stack."""
        return len(self.items)
    
    def __str__(self):
        """Return a string representation of the stack."""
        return str(self.items)


# 3.2 Using a Linked List
class Node:
    """A node in a linked list."""
    
    def __init__(self, data):
        self.data = data
        self.next = None


class StackUsingLinkedList:
    """Stack implementation using a linked list."""
    
    def __init__(self):
        """Initialize an empty stack."""
        self.top = None
        self._size = 0
    
    def push(self, item):
        """Add an item to the top of the stack."""
        new_node = Node(item)
        new_node.next = self.top
        self.top = new_node
        self._size += 1
    
    def pop(self):
        """Remove and return the top item from the stack."""
        if self.is_empty():
            raise IndexError("Pop from an empty stack")
        item = self.top.data
        self.top = self.top.next
        self._size -= 1
        return item
    
    def peek(self):
        """Return the top item without removing it."""
        if self.is_empty():
            raise IndexError("Peek from an empty stack")
        return self.top.data
    
    def is_empty(self):
        """Check if the stack is empty."""
        return self.top is None
    
    def size(self):
        """Return the number of items in the stack."""
        return self._size
    
    def __str__(self):
        """Return a string representation of the stack."""
        if self.is_empty():
            return "[]"
        
        current = self.top
        items = []
        while current:
            items.append(current.data)
            current = current.next
        
        # Reverse to show the stack from bottom to top
        return str(items[::-1])


# 3.3 Using collections.deque (more efficient than list for stack operations)
from collections import deque

class StackUsingDeque:
    """Stack implementation using collections.deque."""
    
    def __init__(self):
        """Initialize an empty stack."""
        self.items = deque()
    
    def push(self, item):
        """Add an item to the top of the stack."""
        self.items.append(item)
    
    def pop(self):
        """Remove and return the top item from the stack."""
        if self.is_empty():
            raise IndexError("Pop from an empty stack")
        return self.items.pop()
    
    def peek(self):
        """Return the top item without removing it."""
        if self.is_empty():
            raise IndexError("Peek from an empty stack")
        return self.items[-1]
    
    def is_empty(self):
        """Check if the stack is empty."""
        return len(self.items) == 0
    
    def size(self):
        """Return the number of items in the stack."""
        return len(self.items)
    
    def __str__(self):
        """Return a string representation of the stack."""
        return str(list(self.items))


#? 4. Common Applications of Stacks

"""
4.1 Expression Evaluation and Conversion
- Infix to Postfix/Prefix conversion
- Evaluating postfix expressions
- Checking for balanced parentheses

4.2 Backtracking Algorithms
- Depth-First Search (DFS)
- Maze solving
- Game state tracking

4.3 Function Call Management
- Call stack in programming languages
- Recursion implementation

4.4 Undo/Redo Operations
- Text editors
- Graphic design applications

4.5 Browser History
- Back/Forward navigation
"""

# Example: Checking for balanced parentheses
def is_balanced_parentheses(expression):
    """
    Check if an expression has balanced parentheses.
    
    Args:
        expression (str): A string containing parentheses, brackets, and braces.
        
    Returns:
        bool: True if the expression has balanced parentheses, False otherwise.
    """
    stack = StackUsingList()
    opening = "({["
    closing = ")}]"
    
    for char in expression:
        if char in opening:
            stack.push(char)
        elif char in closing:
            if stack.is_empty():
                return False
            
            # Check if the closing bracket matches the last opening bracket
            last_open = stack.pop()
            if opening.index(last_open) != closing.index(char):
                return False
    
    # If the stack is empty, all parentheses are balanced
    return stack.is_empty()


# Example: Converting infix expression to postfix
def infix_to_postfix(expression):
    """
    Convert an infix expression to postfix notation.
    
    Args:
        expression (str): An infix expression with single-character operands.
        
    Returns:
        str: The equivalent postfix expression.
    """
    precedence = {'+': 1, '-': 1, '*': 2, '/': 2, '^': 3}
    stack = StackUsingList()
    postfix = []
    
    for char in expression:
        # If the character is an operand, add it to the output
        if char.isalnum():
            postfix.append(char)
        
        # If the character is an opening parenthesis, push it to the stack
        elif char == '(':
            stack.push(char)
        
        # If the character is a closing parenthesis, pop from the stack
        # until an opening parenthesis is encountered
        elif char == ')':
            while not stack.is_empty() and stack.peek() != '(':
                postfix.append(stack.pop())
            
            # Remove the opening parenthesis
            if not stack.is_empty() and stack.peek() == '(':
                stack.pop()
        
        # If the character is an operator
        else:
            while (not stack.is_empty() and stack.peek() != '(' and
                   (char not in precedence or
                    precedence.get(stack.peek(), 0) >= precedence.get(char, 0))):
                postfix.append(stack.pop())
            
            stack.push(char)
    
    # Pop all remaining operators from the stack
    while not stack.is_empty():
        postfix.append(stack.pop())
    
    return ''.join(postfix)


# Example: Evaluating a postfix expression
def evaluate_postfix(expression):
    """
    Evaluate a postfix expression.
    
    Args:
        expression (str): A postfix expression with single-digit operands.
        
    Returns:
        float: The result of evaluating the expression.
    """
    stack = StackUsingList()
    
    for char in expression:
        # If the character is an operand, push it to the stack
        if char.isdigit():
            stack.push(int(char))
        
        # If the character is an operator, pop two operands from the stack,
        # perform the operation, and push the result back to the stack
        else:
            if stack.size() < 2:
                raise ValueError("Invalid postfix expression")
            
            b = stack.pop()
            a = stack.pop()
            
            if char == '+':
                stack.push(a + b)
            elif char == '-':
                stack.push(a - b)
            elif char == '*':
                stack.push(a * b)
            elif char == '/':
                stack.push(a / b)
            elif char == '^':
                stack.push(a ** b)
    
    # The final result should be the only item left in the stack
    if stack.size() != 1:
        raise ValueError("Invalid postfix expression")
    
    return stack.pop()


#? 5. Example Problems and Solutions

# Example 1: Reverse a string using a stack
def reverse_string(string):
    """
    Reverse a string using a stack.
    
    Args:
        string (str): The string to reverse.
        
    Returns:
        str: The reversed string.
    """
    stack = StackUsingList()
    
    # Push all characters onto the stack
    for char in string:
        stack.push(char)
    
    # Pop all characters from the stack to get the reversed string
    reversed_string = ""
    while not stack.is_empty():
        reversed_string += stack.pop()
    
    return reversed_string


# Example 2: Implement a min stack (a stack that supports finding the minimum element in O(1) time)
class MinStack:
    """A stack that supports finding the minimum element in O(1) time."""
    
    def __init__(self):
        """Initialize an empty min stack."""
        self.stack = StackUsingList()
        self.min_stack = StackUsingList()
    
    def push(self, item):
        """Add an item to the top of the stack."""
        self.stack.push(item)
        
        # If the min_stack is empty or the new item is smaller than or equal to
        # the current minimum, push it onto the min_stack
        if self.min_stack.is_empty() or item <= self.min_stack.peek():
            self.min_stack.push(item)
    
    def pop(self):
        """Remove and return the top item from the stack."""
        if self.stack.is_empty():
            raise IndexError("Pop from an empty stack")
        
        item = self.stack.pop()
        
        # If the popped item is the current minimum, remove it from the min_stack
        if item == self.min_stack.peek():
            self.min_stack.pop()
        
        return item
    
    def peek(self):
        """Return the top item without removing it."""
        return self.stack.peek()
    
    def get_min(self):
        """Return the minimum element in the stack."""
        if self.min_stack.is_empty():
            raise IndexError("Min from an empty stack")
        
        return self.min_stack.peek()
    
    def is_empty(self):
        """Check if the stack is empty."""
        return self.stack.is_empty()
    
    def size(self):
        """Return the number of items in the stack."""
        return self.stack.size()


# Example 3: Implement a stack that supports getMax() in O(1) time
class MaxStack:
    """A stack that supports finding the maximum element in O(1) time."""
    
    def __init__(self):
        """Initialize an empty max stack."""
        self.stack = StackUsingList()
        self.max_stack = StackUsingList()
    
    def push(self, item):
        """Add an item to the top of the stack."""
        self.stack.push(item)
        
        # If the max_stack is empty or the new item is larger than or equal to
        # the current maximum, push it onto the max_stack
        if self.max_stack.is_empty() or item >= self.max_stack.peek():
            self.max_stack.push(item)
    
    def pop(self):
        """Remove and return the top item from the stack."""
        if self.stack.is_empty():
            raise IndexError("Pop from an empty stack")
        
        item = self.stack.pop()
        
        # If the popped item is the current maximum, remove it from the max_stack
        if item == self.max_stack.peek():
            self.max_stack.pop()
        
        return item
    
    def peek(self):
        """Return the top item without removing it."""
        return self.stack.peek()
    
    def get_max(self):
        """Return the maximum element in the stack."""
        if self.max_stack.is_empty():
            raise IndexError("Max from an empty stack")
        
        return self.max_stack.peek()
    
    def is_empty(self):
        """Check if the stack is empty."""
        return self.stack.is_empty()
    
    def size(self):
        """Return the number of items in the stack."""
        return self.stack.size()


#? 6. Performance Analysis and Comparison

"""
6.1 Time Complexity Analysis

Operation       | List-based | Linked List-based | Deque-based
----------------|------------|-------------------|------------
push()          | O(1)*      | O(1)              | O(1)
pop()           | O(1)       | O(1)              | O(1)
peek()          | O(1)       | O(1)              | O(1)
is_empty()      | O(1)       | O(1)              | O(1)
size()          | O(1)       | O(1)              | O(1)

* Note: List-based push() is amortized O(1) due to occasional resizing.

6.2 Space Complexity Analysis

All implementations have O(n) space complexity, where n is the number of elements.

6.3 Implementation Considerations

List-based:
- Pros: Simple, built-in Python data structure, good for small stacks
- Cons: May have performance issues with very large stacks due to resizing

Linked List-based:
- Pros: No resizing issues, consistent performance
- Cons: Slightly more complex implementation, extra memory for node pointers

Deque-based:
- Pros: Optimized for stack operations, consistent performance
- Cons: Slightly more complex than a simple list

6.4 When to Use Each Implementation

- Use list-based for simplicity and small stacks
- Use linked list-based when memory efficiency is important
- Use deque-based for optimal performance in most cases
"""

#? 7. Testing Stack Implementations

if __name__ == "__main__":
    # Test the list-based stack
    print("Testing StackUsingList:")
    stack1 = StackUsingList()
    stack1.push(1)
    stack1.push(2)
    stack1.push(3)
    print(f"Stack: {stack1}")
    print(f"Pop: {stack1.pop()}")
    print(f"Peek: {stack1.peek()}")
    print(f"Size: {stack1.size()}")
    print(f"Is empty: {stack1.is_empty()}")
    print()
    
    # Test the linked list-based stack
    print("Testing StackUsingLinkedList:")
    stack2 = StackUsingLinkedList()
    stack2.push(1)
    stack2.push(2)
    stack2.push(3)
    print(f"Stack: {stack2}")
    print(f"Pop: {stack2.pop()}")
    print(f"Peek: {stack2.peek()}")
    print(f"Size: {stack2.size()}")
    print(f"Is empty: {stack2.is_empty()}")
    print()
    
    # Test the deque-based stack
    print("Testing StackUsingDeque:")
    stack3 = StackUsingDeque()
    stack3.push(1)
    stack3.push(2)
    stack3.push(3)
    print(f"Stack: {stack3}")
    print(f"Pop: {stack3.pop()}")
    print(f"Peek: {stack3.peek()}")
    print(f"Size: {stack3.size()}")
    print(f"Is empty: {stack3.is_empty()}")
    print()
    
    # Test the balanced parentheses function
    print("Testing balanced parentheses:")
    print(f"'(())': {is_balanced_parentheses('(())')}")
    print(f"'({[]})': {is_balanced_parentheses('({[]})')}")
    print(f"'(()': {is_balanced_parentheses('(()')}")
    print(f"'([)]': {is_balanced_parentheses('([)]')}")
    print()
    
    # Test the infix to postfix conversion
    print("Testing infix to postfix conversion:")
    print(f"'a+b': {infix_to_postfix('a+b')}")
    print(f"'a+b*c': {infix_to_postfix('a+b*c')}")
    print(f"'(a+b)*c': {infix_to_postfix('(a+b)*c')}")
    print(f"'a+b*(c-d)/e': {infix_to_postfix('a+b*(c-d)/e')}")
    print()
    
    # Test the postfix evaluation
    print("Testing postfix evaluation:")
    print(f"'23+': {evaluate_postfix('23+')}")
    print(f"'23*4+': {evaluate_postfix('23*4+')}")
    print(f"'234*+': {evaluate_postfix('234*+')}")
    print()
    
    # Test the string reversal
    print("Testing string reversal:")
    print(f"'hello': {reverse_string('hello')}")
    print(f"'python': {reverse_string('python')}")
    print(f"'stack': {reverse_string('stack')}")
    print()
    
    # Test the min stack
    print("Testing MinStack:")
    min_stack = MinStack()
    min_stack.push(3)
    min_stack.push(5)
    min_stack.push(2)
    min_stack.push(1)
    print(f"Min: {min_stack.get_min()}")
    min_stack.pop()
    print(f"Min after pop: {min_stack.get_min()}")
    min_stack.pop()
    print(f"Min after another pop: {min_stack.get_min()}")
    print()
    
    # Test the max stack
    print("Testing MaxStack:")
    max_stack = MaxStack()
    max_stack.push(3)
    max_stack.push(5)
    max_stack.push(2)
    max_stack.push(1)
    print(f"Max: {max_stack.get_max()}")
    max_stack.pop()
    print(f"Max after pop: {max_stack.get_max()}")
    max_stack.pop()
    print(f"Max after another pop: {max_stack.get_max()}")
