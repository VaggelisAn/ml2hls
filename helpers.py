from dataclasses import replace
from copy import deepcopy

def parse_input():
#   Receive user input and remove the ">>> " string from the start, split user input into strings.
#   First string should be the argument selected and the rest of the strings should be the set of the values that the argument will take.
    user_input = input(">>> ").strip()
    user_input = user_input.split()
    argument = user_input[0]
    values = user_input[1:]
#   Eliminate duplicates
    values = list(dict.fromkeys(values))
    return argument, values

def set_nested_attr(obj, attr_path, value):
    parts = attr_path.split('.')
    if len(parts) == 1:
        # Get the type of the field from the dataclass
        field_type = type(getattr(obj, parts[0]))
        # Try to convert value to the correct type if needed
        if field_type is int and isinstance(value, str):
            value = int(value)
        elif field_type is float and isinstance(value, str):
            value = float(value)
        # Add more type checks as needed
        return replace(obj, **{parts[0]: value})
    else:
        # Traverse to the parent object
        parent = getattr(obj, parts[0])
        # Recursively replace in the parent
        new_parent = set_nested_attr(parent, '.'.join(parts[1:]), value)
        return replace(obj, **{parts[0]: new_parent})