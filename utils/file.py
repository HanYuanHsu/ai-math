import json

def load_json(filename):
    with open(filename) as fp:
        data = json.load(fp)
    return data

def dump_json(data, filename):
    with open(filename, 'w') as fp:
        json.dump(data, filename)

