import json  

def process_line(line):  
    # Load the JSON data from the line  
    data = json.loads(line)  
    
    # Extract the GPT response value  
    gpt_response = data["conversations"][1]["value"]  

    # Prepare a list to hold formatted box-text strings  
    processed_boxes = []  

    # Split the GPT response into lines  
    lines = gpt_response.split("\n")  

    for line in lines:  
        if "<box>" in line:  
            # Extract box coordinates and text  
            box_start = line.index("<box>") + 5  
            box_end = line.index("</box>")  
            text_start = box_end + 7  # Offset for the text  
            
            # Clean up and pad the boxes  
            box_coords = line[box_start:box_end].strip()  
            text_content = line[text_start:].strip()  
            
            # Format box coordinates and prepend zeros  
            box_coords_list = box_coords.strip("[]").split(",")  
            box_coords_padded = ["{:04d}".format(int(coord.strip())) for coord in box_coords_list]  
            formatted_box = f"<box>[[{', '.join(box_coords_padded)}]]</box>"  
            formatted_text = f"{text_content}"  

            # Add to processed list  
            processed_boxes.append(f"{formatted_box}:{formatted_text}")  

    # Create new conversations structure  
    final_conversations = [  
        {  
            "from": "human",  
            "value": data["conversations"][0]["value"]  
        },  
        {  
            "from": "gpt",  
            "value": ",\n".join(processed_boxes)  
        }  
    ]  

    # Create the final output data structure  
    final_data = {  
        "id": data["id"],  
        "image": data["image"],  
        "conversations": final_conversations  
    }  

    return json.dumps(final_data)  

def process_file(input_file, output_file):  
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:  
        for line in infile:  
            # Process each line and write the output to the new file  
            processed_line = process_line(line)  
            outfile.write(processed_line + "\n")  

# Specify the input and output file paths  
input_file_path = '/mnt/afs/xueyingyi/image_vague/loc_dection/Eimage_solved/matched/train_data_replaced.jsonl'  # Update this with your input file path  
output_file_path = '/mnt/afs/xueyingyi/image_vague/loc_dection/Eimage_solved/matched/train_data_format.jsonl'  # Update this with your desired output file path  

# Call the process_file function  
process_file(input_file_path, output_file_path)  

print(f"Processed file written to {output_file_path}")