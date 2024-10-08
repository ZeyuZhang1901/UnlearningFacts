# This file is used to generate data for the project
# Specifically, the data are a dictionary, each key is a name string, and each value is a randomly generated number string.
# after generating the data, the data will be saved to a json file, stored in ./data/name_number.json

import json
import random
import os
def generate_name_number_dict(num_entries):
    """
    Generate a dictionary with random name strings as keys and random number strings as values.
    """
    name_number_dict = {}
    for _ in range(num_entries):
        first_name = ''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ', k=1) + random.choices('abcdefghijklmnopqrstuvwxyz', k=random.randint(2, 7)))
        last_name = ''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ', k=1) + random.choices('abcdefghijklmnopqrstuvwxyz', k=random.randint(2, 10)))
        name = f"{first_name} {last_name}"
        number = ''.join(random.choices('0123456789', k=10))
        name_number_dict[name] = number
    return name_number_dict

def save_to_json(data, filename):
    """
    Save the dictionary to a json file.
    """
    with open(filename, 'w') as f:
        json.dump(data, f)

if __name__ == "__main__":
    num_entries = 1000000
    name_number_dict = generate_name_number_dict(num_entries)
    if not os.path.exists('./data'):
        os.makedirs('./data')
    save_to_json(name_number_dict, './data/name_number.json')

