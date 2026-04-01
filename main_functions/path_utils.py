#!/usr/bin/env python3
"""
Path Pattern Utility Functions

Handles expansion of path patterns with common parameters and file indices.
Supports patterns like: path/to_*_files/input_file_name_{i}.dat
"""
import os

def expand_path_pattern(pattern, common_param="", file_index=None):
    """
    Expand a path pattern with common parameter and file index.
    
    Parameters:
    -----------
    pattern : str
        Path pattern with * and/or {i} placeholders
        Example: "path/to_*_files/input_file_name_{i}.dat"
    common_param : str, optional
        Value to replace * placeholder (default: "")
    file_index : int, optional
        Value to replace {i} placeholder (default: None, keeps {i} as-is)
        
    Returns:
    --------
    str : Expanded path
    
    Examples:
    ---------
    >>> expand_path_pattern("data_*_run/file_{i}.dat", "240", 5)
    'data_240_run/file_5.dat'
    
    >>> expand_path_pattern("results/*/analysis_{i}.out", "test")
    'results/test/analysis_{i}.out'
    """
    if pattern is None:
        return ""
    
    result = str(pattern)
    
    # Replace * with common parameter
    if common_param is not None:
        result = result.replace('*', str(common_param))
    
    # Replace {i} expressions with file index if provided
    if file_index is not None:
        # Handle arithmetic expressions like {i+1}, {i-1}, etc.
        import re
        
        def replace_expression(match):
            expr = match.group(1)
            try:
                # Create a safe namespace with only 'i' variable
                namespace = {'i': file_index}
                # Evaluate the expression safely (only allows basic arithmetic)
                if re.match(r'^i[\+\-\*/]\d+$|^i$', expr):
                    return str(eval(expr, {"__builtins__": {}}, namespace))
                else:
                    # If expression is not safe, return as-is
                    return match.group(0)
            except:
                # If evaluation fails, return as-is
                return match.group(0)
        
        # Find and replace all {expression} patterns
        result = re.sub(r'\{([^}]+)\}', replace_expression, result)
    
    return result

def validate_path_pattern(pattern):
    """
    Validate a path pattern for correct syntax.
    
    Parameters:
    -----------
    pattern : str
        Path pattern to validate
        
    Returns:
    --------
    tuple : (is_valid, error_message)
    
    Examples:
    ---------
    >>> validate_path_pattern("data_*_files/input_{i}.dat")
    (True, "")
    
    >>> validate_path_pattern("data/{i}/{j}.dat")  # Multiple indices not supported
    (False, "Multiple {i} indices not supported")
    """
    if not pattern:
        return True, ""
    
    # Check for multiple {i} patterns
    i_count = pattern.count('{i}')
    if i_count > 1:
        return False, "Multiple {i} indices not supported"
    
    # Check for other format strings that might cause issues
    import re
    format_matches = re.findall(r'\{[^i}]+\}', pattern)
    if format_matches:
        return False, f"Unsupported format placeholders: {', '.join(format_matches)}. Only {{i}} is supported."
    
    # Check for unmatched braces
    if pattern.count('{') != pattern.count('}'):
        return False, "Unmatched braces in pattern"
    
    return True, ""

def get_pattern_info(pattern):
    """
    Get information about a path pattern.
    
    Parameters:
    -----------
    pattern : str
        Path pattern to analyze
        
    Returns:
    --------
    dict : Pattern information
        - has_common_param: bool (contains *)
        - has_file_index: bool (contains {i})
        - base_dir: str (directory part)
        - filename_pattern: str (filename part)
    """
    if not pattern:
        return {
            'has_common_param': False,
            'has_file_index': False,
            'base_dir': '',
            'filename_pattern': ''
        }
    
    has_common_param = '*' in pattern
    has_file_index = '{i}' in pattern
    
    # Split into directory and filename parts
    base_dir = os.path.dirname(pattern)
    filename_pattern = os.path.basename(pattern)
    
    return {
        'has_common_param': has_common_param,
        'has_file_index': has_file_index,
        'base_dir': base_dir,
        'filename_pattern': filename_pattern
    }

def expand_pattern_list(pattern, common_param="", file_indices=None):
    """
    Expand a pattern for multiple file indices.
    
    Parameters:
    -----------
    pattern : str
        Path pattern with placeholders
    common_param : str, optional
        Value to replace * placeholder
    file_indices : list of int, optional
        List of indices to generate paths for
        
    Returns:
    --------
    list : List of expanded paths
    
    Examples:
    ---------
    >>> expand_pattern_list("data_*/file_{i}.dat", "test", [0, 1, 2])
    ['data_test/file_0.dat', 'data_test/file_1.dat', 'data_test/file_2.dat']
    """
    if file_indices is None:
        file_indices = []
    
    return [expand_path_pattern(pattern, common_param, i) for i in file_indices] 