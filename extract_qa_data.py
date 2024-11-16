import requests
from bs4 import BeautifulSoup
import json
import os
import time
import mimetypes
import argparse
from urllib.parse import urljoin
from multiprocessing import Pool
from requests.exceptions import RequestException
from tqdm import tqdm
import re

def get_extension_from_response(response):
    # Determine the file extension from the Content-Type header
    content_type = response.headers.get('Content-Type', '').split(';')[0].strip()
    extension = mimetypes.guess_extension(content_type)
    if extension:
        return extension
    return '.bin'

def save_attachments(qa_pair, images_folder):
    for idx, attachment_url in enumerate(qa_pair['attachments']):
        max_retries = 3
        retry_delay = 5  # seconds
        
        for attempt in range(max_retries):
            try:
                # Attempt to download the attachment
                response = requests.get(attachment_url, stream=True, timeout=30)
                response.raise_for_status()  # Check HTTP response status

                extension = get_extension_from_response(response)
                
                qa_id = qa_pair.get('id', 'unknown')
                # Use QA ID to name the image file
                file_name = f"{qa_id}_{idx}{extension}"
                    
                file_path = os.path.join(images_folder, file_name)
                with open(file_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192): 
                        if chunk:
                            f.write(chunk)
                qa_pair['attachments'][idx] = file_name
                break  # Success, exit retry loop
                
            except RequestException as e:
                print(f"Error downloading {attachment_url}: {str(e)}")
                if attempt < max_retries - 1:
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    print(f"Failed after {max_retries} retries, skipping this file.")
                    qa_pair['attachments'][idx] = f"DOWNLOAD_FAILED: {attachment_url}"
            except Exception as e:
                print(f"Unknown error downloading {attachment_url}: {str(e)}")
                qa_pair['attachments'][idx] = f"DOWNLOAD_FAILED: {attachment_url}"
                break  # Do not retry on unknown errors

def save_qa_pair(qa_pair, qa_output_file_path):
    # Save the QA pair as a JSON line to the output file
    json_line = json.dumps(qa_pair, ensure_ascii=False)
    with open(qa_output_file_path, 'a', encoding='utf-8') as f:
        f.write(json_line + '\n')
    
def scrape_individual_qa(args):
    url, images_folder = args
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        qa_pair = {}
        # Extract the question title
        title = soup.find('h1', class_='article-title')
        if title:
            qa_pair['title'] = title.text.strip()
            title_id_pattern = r'#\d+'
            title_id = re.findall(title_id_pattern, qa_pair.get('title', ''))
            title_id = title_id[0] if len(title_id) == 1 else None
            if title_id:
                qa_pair['id'] = title_id
                qa_pair['title'] = qa_pair['title'].replace(title_id, '').strip()
            else:
                qa_pair['id'] = url
        # Extract the question content
        question_body = soup.find('div', class_='question_body')
        if question_body:
            qa_pair['question'] = question_body.text.strip()
        # Extract all responses, including multi-turn interactions
        responses = []
        response_blocks = soup.find_all('div', class_='response_block')
        for block in response_blocks:
            response_text = block.find('div', class_='response')
            responder = block.find('div', class_='author')

            if response_text:
                response_info = {
                    'responder': responder.text.strip() if responder else 'Unknown',
                    'response': response_text.text.strip()
                }
                responses.append(response_info)
        # Add responses to qa_pair
        qa_pair['responses'] = responses
        # Extract attachments
        attachments = soup.find('div', class_='question-attachments')
        if attachments:
            qa_pair['attachments'] = [urljoin(url, img['src']) for img in attachments.find_all('img')]
        else:
            qa_pair['attachments'] = []
        return qa_pair
    except Exception as e:
        print(f"Error scraping {url}: {str(e)}")
        return None

class QAScraper:
    def __init__(self, base_url, qa_output_file_path, qa_images_folder, start_page=1, max_pages=15000, num_workers=4, pages_per_batch=10):
        self.base_url = base_url
        self.qa_output_file_path = qa_output_file_path
        self.qa_images_folder = qa_images_folder
        self.start_page = start_page
        self.max_pages = max_pages
        self.num_workers = num_workers
        self.pages_per_batch = pages_per_batch
        self.current_page = self.start_page
        os.makedirs(self.qa_images_folder, exist_ok=True)
        
    def scrape_qa_list_page(self, page):
        # Scrape a list of QA links from a page
        page_url = f"{self.base_url}?p={page}"
        response = requests.get(page_url)
        soup = BeautifulSoup(response.content, 'html.parser')
        qa_links = []
        question_containers = soup.find_all('div', class_='article-title')
        for container in question_containers:
            link = container.find('a')
            if link:
                qa_links.append(urljoin(self.base_url, link['href']))
        return qa_links

    def run(self):
        try:
            pool = Pool(processes=self.num_workers)
            while self.current_page <= self.max_pages:
                batch_end_page = min(self.current_page + self.pages_per_batch - 1, self.max_pages)
                # Report progress and current time
                print(f"\nScraping pages {self.current_page} to {batch_end_page}, out of {self.max_pages} total pages Currently at time {time.strftime('%H:%M:%S')}")
                all_qa_links = []
                for page in range(self.current_page, batch_end_page + 1):
                    qa_links = self.scrape_qa_list_page(page)
                    if not qa_links:
                        print(f"No questions found on page {page}, stopping this batch.")
                        break
                    all_qa_links.extend(qa_links)
                if not all_qa_links:
                    print("No more questions found, stopping.")
                    break
                args_list = [(link, self.qa_images_folder) for link in all_qa_links]
                results = []
                for qa_pair in tqdm(pool.imap_unordered(scrape_individual_qa, args_list), total=len(all_qa_links), desc=f"Processing pages {self.current_page} to {batch_end_page}"):
                    if qa_pair:
                        results.append(qa_pair)
                # Process results in the main process
                for qa_pair in results:
                    save_attachments(qa_pair, self.qa_images_folder)
                    save_qa_pair(qa_pair, self.qa_output_file_path)
                self.current_page = batch_end_page + 1
                time.sleep(2)  # Polite delay to avoid too frequent requests
        except KeyboardInterrupt:
            print(f"\nInterrupted! Last processed page: {self.current_page}")
        finally:
            print(f"Scraping completed up to page {self.current_page - 1}.")
            pool.close()
            pool.join()

def main():
    parser = argparse.ArgumentParser(description='Scrape QA data from a website')
    parser.add_argument('--start_page', type=int, default=1, help='Starting page number')
    parser.add_argument('--max_pages', type=int, default=15000, help='Maximum number of pages to scrape')
    parser.add_argument('--pages_per_batch', type=int, default=10, help='Number of pages to scrape per batch')
    parser.add_argument('--qa_output_file_path', type=str, default='qa_dataset.jsonl', help='Path to save QA data')
    parser.add_argument('--qa_images_folder', type=str, default='qa_images', help='Folder to save attachments')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of worker processes')
    args = parser.parse_args()
    base_url = 'https://ask2.extension.org/kb/index.php'
    scraper = QAScraper(
        base_url=base_url,
        qa_output_file_path=args.qa_output_file_path,
        qa_images_folder=args.qa_images_folder,
        start_page=args.start_page,
        max_pages=args.max_pages,
        num_workers=args.num_workers,
        pages_per_batch=args.pages_per_batch
    )
    scraper.run()

if __name__ == '__main__':
    main()
