import json  

# Define the input and output file paths  
input_file_path = '/mnt/afs/xueyingyi/image_vague/loc_dection/Eimage_solved/annotation/eval_data.jsonl'  
replacement_file_path = '/mnt/afs/xueyingyi/image_vague/loc_dection/Eimage_solved/annotation/eval_data_processed.jsonl'  
output_file_path = '/mnt/afs/xueyingyi/image_vague/loc_dection/Eimage_solved/annotation/eval_data_replaced.jsonl'  

# Create a mapping of ids to gpt_text from the second file  
gpt_text_mapping = {}  

# Read the replacement file and store the gpt_text in a dictionary  
with open(replacement_file_path, 'r', encoding='utf-8') as replacement_file:  
    for line in replacement_file:  
        data = json.loads(line)  
        # Populate the mapping dictionary with id and gpt_text  
        gpt_text_mapping[data['id']] = data['gpt_text']  

# Open the input file, read it, and replace the value in gpt  
with open(input_file_path, 'r', encoding='utf-8') as input_file, \
     open(output_file_path, 'w', encoding='utf-8') as output_file:  
    
    for line in input_file:  
        data = json.loads(line)  
        
        # Check if the 'id' exists in the mapping, and if so, replace the value  
        if data['id'] in gpt_text_mapping:  
            # Replace the whole value section with the correlated gpt_text  
            for conversation in data['conversations']:  
                if conversation['from'] == 'gpt':  
                    conversation['value'] = gpt_text_mapping[data['id']]  
        
        # Write the modified data to the output file  
        output_file.write(json.dumps(data) + '\n')  

print("Replacement completed. The results are saved in:", output_file_path)