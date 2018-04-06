import json


with open("evaluation/anger_data.json", 'r') as f:
    anger = json.load(f)

with open("evaluation/joy_data.json", 'r') as f:
    joy = json.load(f)

with open("evaluation/fear_data.json", 'r') as f:
    fear = json.load(f)

with open("evaluation/sadness_data.json", 'r') as f:
    sadness = json.load(f)

combined = {'anger':anger, 'fear':fear, 'joy':joy, 'sadness':sadness}

with open("evaluation/combined.json", 'w') as f:
    json.dump(combined, f)

