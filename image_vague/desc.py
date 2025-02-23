from openai import OpenAI

# Initialize the client
client = OpenAI(api_key="your-api-key")

# Read the image
image_path = "/mnt/afs/xueyingyi/image_vague/image_white/example.jpg"
with open(image_path, "rb") as image_file:
    image_data = image_file.read()

# Construct the prompt
prompt = """
You are an image content reasoning model. Your task is to analyze an image where a certain region is covered by a white box. Based on the context of the image, you need to infer what the covered region might contain and describe it in text.

Input:
1. An image where a region is covered by a white box.
2. The position and size of the white box (optional, if more precise reasoning is needed).

Output:
A textual description of what the covered region might contain. The description should be based on the context of the image and should be as reasonable and specific as possible.
"""

# Call the model
response = client.chat.completions.create(
    model="gpt-4-vision-preview",  # Use a model capable of image input
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}},
            ],
        }
    ],
    max_tokens=300,  # Control output length
)

# Output the result
print(response.choices[0].message.content)