import requests
import json

with open('questions.txt', 'r') as f:
    questions = [line.strip() for line in f.readlines()]

endpoint = "http://localhost:11434/api/generate"
params = {
    "model": "llama3.1",
    "prompt": "",
    # "suffix": "Respond using JSON",
    "stream": False,
    "format": {
        "type": "object",
        "properties": {
        "statement": {
            "type": "string"
        }
        },
        "required": [
        "statement"
        ]
    },
    "options": {
        # "seed": 42,
        "temperature": 0.6,
        "num_thread": 4
    }
}

responses = {}
# with open('responses3.txt', 'w') as f:
i=1
for question in questions:
    # Update prompt parameter with current question
    params["prompt"] = ("Take the following input and replace every instance of a noun with an antonym of that chosen noun: " + 
                        question + "\n" +
                        "Respond with a JSON object and keep your response in the original format, changing only the nouns: \n")
    # Send POST request
    response = requests.post(endpoint, json=params)
    # print(type(response.json()['response']))
    # print(type(json.loads(response.json()['response'])))
    # Check if response was successful
    if response.status_code == 200:
        # Parse response as JSON and print it
        print(json.loads(response.json()['response']))
        # responses[i] = json.loads(response.json()['response'])
        # i+=1

        # f.write("Question: " + question)
        # f.write('\n')
        # f.write(response.json()['response'])
        # f.write('\n')
        # print(response.json())
    

# with open('responses.txt', 'w') as f:
# print(responses)
    # for i, value in enumerate(responses.items()):
    #     # print(i)
    #     # f.write(value[0])
    #     # f.write(',')
    #     f.write(value[1]['response'])
    #     f.write('\n')

# for i, value in enumerate(responses.items()):
#     print(value[0])
#     print(value[1]['statement'])
#     print(value[1]['decision'])
#     print(value[1]['reasoning'])
