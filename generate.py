"""
Note: do not tweak this code unless you know what you're doing.'
"""


import json
from init import main


with open('config.json') as file:
    config = json.load(file)
    labels = config['labels']
if __name__ == '__main__':
    name = input('Enter the Name of Gesture: ')
    with open('config.json', 'w') as f:
        labels.append(name)
        json.dump(config, f, indent=2)
    main('Generate', '',model=name)
