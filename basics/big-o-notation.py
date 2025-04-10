"""Big O Notation

This file provides a comprehensive overview of Big O notation,
including definitions, examples, and analysis of common time and space complexities.

Big O notation is used to describe the performance or complexity of an algorithm.
It specifically describes the worst-case scenario and can be used to describe the
execution time required or the space used by an algorithm.
"""

#? Introduction to Big O Notation
"""
Big O notation is a mathematical notation that describes the limiting behavior of a function
when the argument tends towards a particular value or infinity. In computer science, it's used
to classify algorithms according to how their run time or space requirements grow as the input size grows.

Common Big O notations (from fastest to slowest):
- O(1): Constant time/space - execution time/space doesn't depend on input size
- O(log n): Logarithmic time/space - execution time/space grows logarithmically with input size
- O(n): Linear time/space - execution time/space grows linearly with input size
- O(n log n): Linearithmic time/space - execution time/space grows by n log n
- O(n²): Quadratic time/space - execution time/space grows quadratically with input size
- O(n³): Cubic time/space - execution time/space grows cubically with input size
- O(2^n): Exponential time/space - execution time/space doubles with each addition to the input
- O(n!): Factorial time/space - execution time/space grows factorially with input size

Rules for calculating Big O:
1. Drop constants: O(2n) -> O(n)
2. Drop non-dominant terms: O(n² + n) -> O(n²)
3. Different inputs -> different variables: O(a + b) not O(2n)
"""

# Python examples for each combination of time and space complexity

#? 1. Constant Time and Space - O(1), O(1)
def access(arr, index):
    # Accessing an element by index is O(1) time and space
    return arr[index]

def sum(first, second):
    # Summing two elements is O(1) time and space
    return first + second

#? 2. Linear Time and Space - O(n), O(n)
def multiply_all_in_array(arr, multiplier):
    # Creating a new list of the same size is O(n) space
    # Iterating through the list is O(n) time
    result = [x * multiplier for x in arr]
    return result


def find_max(arr):
    # Only uses a single variable regardless of input size - O(1) space
    # Iterates through all elements once - O(n) time
    if not arr:
        return None

    max_val = arr[0]
    for num in arr:
        if num > max_val:
            max_val = num

    return max_val


#? 3. Logarithmic Time - O(log n)
def binary_search(arr, target):
    # Binary search has O(log n) time complexity
    # Only uses a constant amount of extra space - O(1) space
    left, right = 0, len(arr) - 1

    while left <= right:
        mid = (left + right) // 2

        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return -1  # Target not found


#? 4. Linearithmic Time - O(n log n)
def merge_sort(arr):
    # Merge sort has O(n log n) time complexity
    # Uses O(n) extra space for the merged arrays
    if len(arr) <= 1:
        return arr

    # Divide the array into two halves
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])

    # Merge the sorted halves
    return merge(left, right)


def merge(left, right):
    result = []
    i = j = 0

    # Compare elements from both arrays and add the smaller one to the result
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1

    # Add any remaining elements
    result.extend(left[i:])
    result.extend(right[j:])

    return result


#? 5. Quadratic Time - O(n²)
def bubble_sort(arr):
    # Bubble sort has O(n²) time complexity
    # Only uses a constant amount of extra space - O(1) space
    n = len(arr)

    for i in range(n):
        # Flag to optimize if the array is already sorted
        swapped = False

        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True

        # If no swapping occurred in this pass, the array is sorted
        if not swapped:
            break

    return arr


def nested_loops_example(n):
    # Nested loops typically result in O(n²) time complexity
    # Only uses a constant amount of extra space - O(1) space
    result = 0

    for i in range(n):
        for j in range(n):
            result += i * j

    return result


#? 6. Cubic Time - O(n³)
def three_nested_loops(n):
    # Three nested loops result in O(n³) time complexity
    # Only uses a constant amount of extra space - O(1) space
    result = 0

    for i in range(n):
        for j in range(n):
            for k in range(n):
                result += i * j * k

    return result


#? 7. Exponential Time - O(2^n)
def fibonacci_recursive(n):
    # Naive recursive Fibonacci has O(2^n) time complexity
    # The recursion stack uses O(n) space
    if n <= 1:
        return n

    return fibonacci_recursive(n - 1) + fibonacci_recursive(n - 2)


#? 8. Factorial Time - O(n!)
def permutations(arr):
    # Generating all permutations has O(n!) time complexity
    # The recursion and storage of all permutations uses O(n! * n) space
    def backtrack(start):
        # If we've reached the end of the array, add the current permutation
        if start == len(arr):
            result.append(arr[:])
            return

        for i in range(start, len(arr)):
            # Swap elements to create a new permutation
            arr[start], arr[i] = arr[i], arr[start]

            # Recursively generate permutations for the rest of the array
            backtrack(start + 1)

            # Backtrack (undo the swap)
            arr[start], arr[i] = arr[i], arr[start]

    result = []
    backtrack(0)
    return result


#? 9. Common Algorithm Complexities
"""
Common Data Structure Operations:

Data Structure     | Access    | Search    | Insertion | Deletion
-------------------|-----------|-----------|-----------|----------
Array              | O(1)      | O(n)      | O(n)      | O(n)
Linked List        | O(n)      | O(n)      | O(1)*     | O(1)*
Stack              | O(n)      | O(n)      | O(1)      | O(1)
Queue              | O(n)      | O(n)      | O(1)      | O(1)
Hash Table         | N/A       | O(1)**    | O(1)**    | O(1)**
Binary Search Tree | O(log n)* | O(log n)* | O(log n)* | O(log n)*
AVL Tree           | O(log n)  | O(log n)  | O(log n)  | O(log n)
Heap               | O(1)***   | O(n)      | O(log n)  | O(log n)

* Average case, can be O(n) in worst case
** Amortized, can be O(n) in worst case
*** For the minimum/maximum element

Sorting Algorithms:

Algorithm      | Best       | Average    | Worst      | Space
---------------|------------|------------|------------|-------
Bubble Sort    | O(n)       | O(n²)     | O(n²)     | O(1)
Selection Sort | O(n²)     | O(n²)     | O(n²)     | O(1)
Insertion Sort | O(n)       | O(n²)     | O(n²)     | O(1)
Merge Sort     | O(n log n) | O(n log n) | O(n log n) | O(n)
Quick Sort     | O(n log n) | O(n log n) | O(n²)     | O(log n)
Heap Sort      | O(n log n) | O(n log n) | O(n log n) | O(1)
Radix Sort     | O(nk)      | O(nk)      | O(nk)      | O(n+k)
Counting Sort  | O(n+k)     | O(n+k)     | O(n+k)     | O(k)

Search Algorithms:

Algorithm        | Best  | Average | Worst
-----------------|-------|---------|-------
Linear Search    | O(1)  | O(n)    | O(n)
Binary Search    | O(1)  | O(log n)| O(log n)
Depth-First Search | O(1) | O(V+E) | O(V+E)
Breadth-First Search | O(1) | O(V+E) | O(V+E)
Dijkstra's Algorithm | O(V²) | O(V²) | O(V²)
A* Search        | O(1)  | O(b^d)  | O(b^d)

where V = number of vertices, E = number of edges, b = branching factor, d = depth
"""


#? 10. Practical Considerations
"""
When analyzing algorithms, it's important to consider:

1. Constants Matter in Practice:
   - While Big O drops constants, in real-world scenarios, an O(n) algorithm with a large constant
     might be slower than an O(n²) algorithm with a small constant for small inputs.

2. Average Case vs. Worst Case:
   - Big O typically describes worst-case scenarios, but average-case performance
     might be more relevant in practice.

3. Space-Time Tradeoffs:
   - Sometimes you can trade space for time (or vice versa).
   - Example: Memoization uses more space to save computation time.

4. Hardware Considerations:
   - Cache behavior, memory access patterns, and parallelization can
     significantly affect real-world performance.

5. Input Size and Distribution:
   - Some algorithms perform better on certain input distributions or sizes.
   - Example: Insertion sort is often faster than quicksort for small arrays.
"""


#? 11. Testing and Visualization

if __name__ == "__main__":
    import time

    # Try to import matplotlib, but provide a fallback if it's not installed
    try:
        import matplotlib.pyplot as plt
        matplotlib_available = True
    except ImportError:
        print("Matplotlib is not installed. Visualizations will be skipped.")
        print("To install matplotlib, run: pip install matplotlib")
        matplotlib_available = False

    # Test different time complexities
    def measure_time(func, *args):
        start_time = time.time()
        result = func(*args)
        end_time = time.time()
        return end_time - start_time

    # Example: Compare sorting algorithms
    def test_sorting_algorithms():
        import random

        sizes = [100, 500, 1000, 2000, 3000, 4000, 5000]
        bubble_times = []
        merge_times = []
        python_times = []

        for size in sizes:
            # Generate a random array
            arr = [random.randint(0, 10000) for _ in range(size)]

            # Measure bubble sort time
            bubble_arr = arr.copy()
            bubble_time = measure_time(bubble_sort, bubble_arr)
            bubble_times.append(bubble_time)

            # Measure merge sort time
            merge_arr = arr.copy()
            merge_time = measure_time(merge_sort, merge_arr)
            merge_times.append(merge_time)

            # Measure Python's built-in sort time
            python_arr = arr.copy()
            python_time = measure_time(lambda x: x.sort(), python_arr)
            python_times.append(python_time)

            print(f"Size: {size}, Bubble: {bubble_time:.6f}s, Merge: {merge_time:.6f}s, Python: {python_time:.6f}s")

        # Plot the results if matplotlib is available
        if matplotlib_available:
            plt.figure(figsize=(10, 6))
            plt.plot(sizes, bubble_times, 'o-', label='Bubble Sort - O(n²)')
            plt.plot(sizes, merge_times, 'o-', label='Merge Sort - O(n log n)')
            plt.plot(sizes, python_times, 'o-', label='Python Sort - O(n log n)')
            plt.xlabel('Input Size')
            plt.ylabel('Time (seconds)')
            plt.title('Sorting Algorithm Performance Comparison')
            plt.legend()
            plt.grid(True)
            plt.savefig('sorting_comparison.png')
            plt.close()

            print("\nGraph saved as 'sorting_comparison.png'")

    # Example: Compare different time complexities
    def test_time_complexities():
        sizes = [5, 10, 15, 20, 25]
        constant_times = []
        linear_times = []
        quadratic_times = []
        cubic_times = []

        for size in sizes:
            # O(1) - constant time
            constant_time = measure_time(lambda: 1 + 1)
            constant_times.append(constant_time)

            # O(n) - linear time
            linear_time = measure_time(lambda: [i for i in range(size)])
            linear_times.append(linear_time)

            # O(n²) - quadratic time
            quadratic_time = measure_time(nested_loops_example, size)
            quadratic_times.append(quadratic_time)

            # O(n³) - cubic time
            cubic_time = measure_time(three_nested_loops, size)
            cubic_times.append(cubic_time)

            print(f"Size: {size}, O(1): {constant_time:.6f}s, O(n): {linear_time:.6f}s, "
                  f"O(n²): {quadratic_time:.6f}s, O(n³): {cubic_time:.6f}s")

        # Plot the results if matplotlib is available
        if matplotlib_available:
            plt.figure(figsize=(10, 6))
            plt.plot(sizes, constant_times, 'o-', label='O(1) - Constant')
            plt.plot(sizes, linear_times, 'o-', label='O(n) - Linear')
            plt.plot(sizes, quadratic_times, 'o-', label='O(n²) - Quadratic')
            plt.plot(sizes, cubic_times, 'o-', label='O(n³) - Cubic')
            plt.xlabel('Input Size')
            plt.ylabel('Time (seconds)')
            plt.title('Time Complexity Comparison')
            plt.legend()
            plt.grid(True)
            plt.savefig('complexity_comparison.png')
            plt.close()

            print("\nGraph saved as 'complexity_comparison.png'")

    # Run the tests
    print("Testing different time complexities...\n")
    test_time_complexities()

    print("\nTesting sorting algorithms...\n")
    test_sorting_algorithms()