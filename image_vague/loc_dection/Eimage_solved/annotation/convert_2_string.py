import json  

# Define the input and output file paths  
input_file_path = '/mnt/afs/xueyingyi/image_vague/loc_dection/Eimage_solved/annotation/train_data.jsonl'  
output_file_path = '/mnt/afs/xueyingyi/image_vague/loc_dection/Eimage_solved/annotation/train_data_processed.jsonl'  

# Function to process each line of the input file  
def process_line(line):  
    data = json.loads(line)  # Load the JSON data  
    gpt_value = data['conversations'][1]['value']  # Extract the 'gpt' value  

    # Convert the gpt value into the desired string format, excluding unwanted texts
    gpt_string = '\n'.join([f"{item['bbox']}: {item['text']}" 
                            for item in gpt_value 
                            if not any(substring in item['text'] for substring in [".co",".com", ".net", "unanswerable","http"])])  
    
    # Create a new dictionary with the processed information  
    processed_data = {  
        "id": data["id"],  
        "image": data["image"],  
        "gpt_text": gpt_string  # Add the processed gpt string  
    }  
    
    return processed_data  

# Read the input file and process each line  
with open(input_file_path, 'r') as infile, open(output_file_path, 'w') as outfile:  
    for line in infile:  
        processed_data = process_line(line)  
        json.dump(processed_data, outfile)  # Write the processed data to the output file  
        outfile.write('\n')  # Ensure each result is on a new line  

print("Processing complete! Processed data saved to:", output_file_path)
