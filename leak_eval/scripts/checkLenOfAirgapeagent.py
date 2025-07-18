import json
with open("leak_eval/datasets/airgapagent-r.json") as f:
    data = json.load(f)
print(len(data))