import requests
import json

with open('questions.txt', 'r') as f:
    questions = [line.strip() for line in f.readlines()]

# Define the API endpoint and parameters
endpoint = "http://localhost:11434/api/generate"
params = {
    "model": "llama3.1",
    "prompt": "",
    "stream": False
}
# Create a dictionary to store the responses
# responses = {}
with open('responses.txt', 'w') as f:
    # Run request for each question
    for question in questions:
        # Update prompt parameter with current question
        # params["prompt"] = "Answer with a brief explanation if it is right or wrong to take the action described by the following statement. " + question + "No need to explicitly label."
        params["prompt"] = "Is right or wrong to take the action described by the following statement. " + question + ""
        # print(params["prompt"])
        # Send POST request
        response = requests.post(endpoint, json=params)
        # print(type(response.json()))
        # Check if response was successful
        if response.status_code == 200:
            # Parse response as JSON and print it
            # print(response.json()['response'])
            # responses[question] = response.json()
            f.write(question)
            f.write('\n')
            f.write(response.json()['response'])
            f.write('\n')
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
#     print(value[1]['response'])


