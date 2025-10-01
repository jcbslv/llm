import requests
import json
from datasets import load_dataset

#set up llm
endpoint = "http://localhost:11434/api/generate"
params = {
    "model": "llama3.1",
    "prompt": "",
    # "suffix": "Respond using JSON",
    "stream": False,
    "format": {
        "type": "object",
        "properties": {
            "edited": {
                "type": "string"
            }
        },
        "required": [
        "edited"
        ]
    },
    "options": {
        # "seed": 42,
        "temperature": 0.6,
        "num_thread": 4
    }
}

#get dataset
ds = load_dataset("sentence-transformers/all-nli", "pair")
statements={}
for i in range(10):
    # print(ds['train']['anchor'][i])
    statements[i] = ds['train']['anchor'][i]
    i+=1

#query llm with dataset
pairs = {}
with open('madlibs.txt', 'w') as f:
    for i, value in enumerate(statements.items()):        
        # Update prompt parameter with current question
        params["prompt"] = ("Take the following input and replace every instance of a noun with an antonym of that chosen noun: " + 
                            value[1] + "\n" +
                            "Respond with a JSON object and keep your response in the original format, changing only the nouns: \n")
        # Send POST request
        response = requests.post(endpoint, json=params)
        # print(type(response.json()['response']))
        # print(type(json.loads(response.json()['response'])))
       
        if response.status_code == 200:
            # item = {"original": value[1], "edited": str(json.loads(response.json()['response'])['edited'])}
            pair = {value[1]: str(json.loads(response.json()['response'])['edited'])}
            pairs.update(pair)
            
    for t in pairs.items():
        f.write(str(t[0]+ ' :: '+ t[1] + '\n'))