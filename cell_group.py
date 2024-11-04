import sys

def read_letters_until_number(s):
    result = ""
    for char in s:
        if char.isdigit():
            break
        result += char
    return result

def cell_group(cell_type_names):
    cell_groups = dict()
    for cell_type_name in cell_type_names:
        cell_group = read_letters_until_number(cell_type_name)
        if cell_group in cell_groups:
            cell_groups[cell_group].add(cell_type_name)
        else:
            cell_groups[cell_group] = set(cell_type_name)
    return cell_group