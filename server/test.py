import requests

# Define the URL of the FastAPI endpoint
url = "http://0.0.0.0:10010/generate-sentences/"

# Send a POST request to the endpoint
response = requests.post(url)

# Check if the request was successful
if response.status_code == 200:
    # Parse the JSON response
    data = response.json()
    # Print the generated text
    print("Generated Text:", data["generated_text"])
else:
    print("Failed to get response:", response.status_code)


#curl -X POST "http://localhost:10010/generate-sentences/?category=travel"