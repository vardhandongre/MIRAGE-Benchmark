import requests
from bs4 import BeautifulSoup
import json
import os
from urllib.parse import urljoin
import time
from requests.exceptions import RequestException
import mimetypes

def scrape_qa_list_page(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    qa_links = []
    
    # Find all question containers
    question_containers = soup.find_all('div', class_='article-title')
    
    for container in question_containers:
        link = container.find('a')
        if link:
            qa_links.append(urljoin(url, link['href']))
    
    return qa_links

def scrape_individual_qa(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    qa_pair = {}
    
    # Extract question title
    title = soup.find('h1', class_='article-title')
    if title:
        qa_pair['title'] = title.text.strip()
    
    # Extract question ID
    question_id = soup.find('span', class_='question-id')
    if question_id:
        qa_pair['id'] = question_id.text.strip()
    
    # Extract question body
    question_body = soup.find('div', class_='question_body')
    if question_body:
        qa_pair['question'] = question_body.text.strip()
    
    # Extract answer
    answer = soup.find('div', class_='response_block')
    if answer:
        qa_pair['answer'] = answer.text.strip()
    
    # Extract attachments
    attachments = soup.find('div', class_='question-attachments')
    if attachments:
        qa_pair['attachments'] = [urljoin(url, img['src']) for img in attachments.find_all('img')]
    else:
        qa_pair['attachments'] = []
    
    # Extract tags
    tags = soup.find_all('span', class_='tag')
    qa_pair['tags'] = [tag.text.strip() for tag in tags]
    
    return qa_pair

def get_extension_from_response(response):
    # Try to get the extension from the Content-Type header
    content_type = response.headers.get('Content-Type', '').split(';')[0].strip()
    extension = mimetypes.guess_extension(content_type)
    
    if extension:
        return extension
    
    # If we couldn't determine the extension, use a default
    return '.bin'

# def save_attachments(qa_pairs, output_folder):
#     for i, qa_pair in enumerate(qa_pairs):
#         for j, attachment_url in enumerate(qa_pair['attachments']):
#             response = requests.get(attachment_url)
#             if response.status_code == 200:
#                 file_name = f"qa_{i}_attachment_{j}{os.path.splitext(attachment_url)[1]}"
#                 file_path = os.path.join(output_folder, file_name)
#                 with open(file_path, 'wb') as f:
#                     f.write(response.content)
#                 qa_pair['attachments'][j] = file_name

def save_attachments(qa_pairs, output_folder):
    for i, qa_pair in enumerate(qa_pairs):
        for j, attachment_url in enumerate(qa_pair['attachments']):
            max_retries = 3
            retry_delay = 5  # seconds
            
            for attempt in range(max_retries):
                try:
                    response = requests.get(attachment_url, stream=True, timeout=30)
                    response.raise_for_status()  # Raises an HTTPError for bad responses
                    
                    # Determine file extension
                    extension = get_extension_from_response(response)
                    
                    file_name = f"qa_{i}_attachment_{j}{extension}"
                    file_path = os.path.join(output_folder, file_name)
                    
                    with open(file_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192): 
                            if chunk:
                                f.write(chunk)
                    
                    qa_pair['attachments'][j] = file_name
                    print(f"Successfully downloaded: {file_name}")
                    break  # Success, so break the retry loop Yay!
                
                except RequestException as e:
                    print(f"Error downloading {attachment_url}: {str(e)}")
                    if attempt < max_retries - 1:
                        print(f"Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                    else:
                        print(f"Failed to download after {max_retries} attempts. Skipping.")
                        qa_pair['attachments'][j] = f"DOWNLOAD_FAILED: {attachment_url}"
                
                except Exception as e:
                    print(f"Unexpected error downloading {attachment_url}: {str(e)}")
                    qa_pair['attachments'][j] = f"DOWNLOAD_FAILED: {attachment_url}"
                    break  # Don't retry for unexpected errors

def main():
    base_url = 'https://ask2.extension.org/kb/index.php'
    output_folder = 'qa_dataset'
    os.makedirs(output_folder, exist_ok=True)
    
    qa_pairs = []
    page = 1
    max_pages = 15000  # Set the maximum number of pages to scrape

    while page <= max_pages:
        print(f"Scraping page {page} of {max_pages}")
        page_url = f"{base_url}?p={page}"
        qa_links = scrape_qa_list_page(page_url)
        
        if not qa_links:
            print("No more questions found. Stopping.")
            break
        
        for link in qa_links:
            print(f"Scraping question: {link}")
            qa_pair = scrape_individual_qa(link)
            qa_pairs.append(qa_pair)
            time.sleep(1)  # Be polite to the server :)

        page += 1
        if page > max_pages:
            print(f"Reached the maximum number of pages ({max_pages}). Stopping.")
            break
        
        time.sleep(2)  # Be polite to the server 
    
    save_attachments(qa_pairs, output_folder)
    # Save QA pairs to JSON file
    with open(os.path.join(output_folder, 'qa_dataset.json'), 'w') as f:
        json.dump(qa_pairs, f, indent=2)

    print(f"Scraped {len(qa_pairs)} questions from {page-1} pages.")
    
if __name__ == '__main__':
    main()