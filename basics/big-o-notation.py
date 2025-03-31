# Python example for each combination of time and space complexity

#? 1. Constant Time and Space - O(1), O(1)
def access(arr, index):
    # Accessing an element by index is O(1) time and space
    return arr[index]

def sum(first, second):
    # Summing two elements is O(1) time and space
    return first + second

#? 2. Linear Time and Space - O(n), O(n)
def moltiply_all_in_array(arr, multiplier):
    # Creating a new list of the same size is O(n) space
    # Iterating through the list is O(n) time
    result = [x * multiplier for x in arr]
    return result