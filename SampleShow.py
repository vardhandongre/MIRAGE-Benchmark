# %%
import json
import random
import os

def sample_view(sample_data, markdown_file):
    sample_id = sample_data['id']
    sample_title = sample_data['title']
    sample_question = sample_data['question']
    sample_answer = sample_data['answer']
    sample_attachments = sample_data['attachments']
    sample_responser = sample_data['responder']
    sample_response_time = sample_data['response_time']
    
    # Write data to the markdown file
    with open(markdown_file, 'a') as md:
        md.write(f"## ID: {sample_id}\n")
        for attachment in sample_attachments:
            # Assuming the attachments are images
            md.write(f'<img src="{attachment}" width="300" height="300">\n')
        md.write(f"\n")
        md.write(f"### Title: {sample_title}\n")
        md.write(f"### Question \n {sample_question}\n")
        md.write(f"### Answer \n {sample_answer}\n")
        md.write(f"**Responser**: {sample_responser}\n**Response Time**: {sample_response_time}\n")
        md.write("\n---\n")  # Add a separator

file_name = 'links_dataset.json'
markdown_file = 'show_sample.md'

# Open the markdown file and clear previous content
with open(markdown_file, 'w') as f:
    f.write("# Sample Data\n")

# Load data
with open(file_name, 'r') as f:
    data = json.load(f)

# Sample 3 random data entries
sample_data = random.sample(data, 3)

# Process each sample
for sample in sample_data:
    sample_view(sample, markdown_file)

# %%
