import ast
import pandas as pd

def join_indices(index1, index2):
    # Ensure both indices have the same length
    if len(index1) != len(index2):
        raise ValueError("Both indices must have the same length")
    
    # Check if the second index is a MultiIndex
    if isinstance(index2, pd.MultiIndex):
        # Combine the tuples from both indices
        combined_tuples = [tuple1 + tuple2 for tuple1, tuple2 in zip(index1, index2)]
        # Combine the names from both indices
        combined_names = index1.names + index2.names
    else:
        # Combine the tuples from the first index with the values from the second index
        combined_tuples = [tuple1 + (value2,) for tuple1, value2 in zip(index1, index2)]
        # Combine the names from the first index with the name of the second index
        combined_names = index1.names + [index2.name]
    
    # Create a new MultiIndex with the combined tuples and names
    
    return pd.MultiIndex.from_tuples(combined_tuples, names=combined_names)

def str2list(string):
    """
    Convert a string representation of a list to an actual list.

    Parameters:
    - string (str): The string representation of the list.

    Returns:
    - list: The actual list.
    """
    try:
        # Use ast.literal_eval to safely evaluate the string
        result = ast.literal_eval(string)
        if isinstance(result, list):
            return result
        else:
            raise ValueError("The provided string does not represent a list.")
    except (ValueError, SyntaxError) as e:
        raise ValueError(f"Invalid string representation of a list: {string}") from e
