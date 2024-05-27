# Send a prompt to Meta Llama 3 and print the response.

import boto3
import json


# Create a Bedrock Runtime client in the AWS Region of your choice.
client = boto3.client("bedrock-runtime", region_name="us-east-1")

# Set the model ID, e.g., Llama 3 8B Instruct.
model_id = "meta.llama3-70b-instruct-v1:0"

# Define the user message to send.
user_message = "Generate 10 different travel-related English sentences at each level: Bronze, Silver, Gold, Diamond, and Master."

# Embed the message in Llama 3's prompt format.
prompt = f"""
<|begin_of_text|>
<|start_header_id|>user<|end_header_id|>
{user_message}
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""

# Format the request payload using the model's native structure.
request = {
    "prompt": prompt,
    # Optional inference parameters:
    "max_gen_len": 2048,
    "temperature": 0.5,
    "top_p": 0.9,
}

# Encode and send the request.
response = client.invoke_model(body=json.dumps(request), modelId=model_id)

# Decode the native response body.
model_response = json.loads(response["body"].read())

# Extract and print the generated text.
response_text = model_response["generation"]
print(response_text)

# Learn more about the Llama 3 prompt format in the documentation:
# https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3/#special-tokens-used-with-meta-llama-3


