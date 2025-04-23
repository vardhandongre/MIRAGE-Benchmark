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
    """
    Determine the file extension from the Content-Type header of the response.

    :param response: The HTTP response object
    :return: The file extension as a string
    """
    # Determine the file extension from the Content-Type header
    content_type = response.headers.get('Content-Type', '').split(';')[0].strip()
    extension = mimetypes.guess_extension(content_type)
    if extension:
        return extension
    return '.bin'

def save_attachments(qa_pair, images_folder):
    """
    Download and save attachments associated with a QA pair.

    :param qa_pair: The QA pair dictionary
    :param images_folder: The folder to save images
    """
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
                if os.path.exists(file_path):
                    qa_pair['attachments'][idx] = file_name
                    break
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
    """
    Save the QA pair as a JSON line to the output file.

    :param qa_pair: The QA pair dictionary
    :param qa_output_file_path: Path to save the QA data
    """
    # Save the QA pair as a JSON line to the output file
    json_line = json.dumps(qa_pair, ensure_ascii=False)
    with open(qa_output_file_path, 'a', encoding='utf-8') as f:
        f.write(json_line + '\n')

def scrape_individual_qa(args):
    """
    Scrape an individual QA page and extract relevant information.

    :param args: Tuple containing the URL and images folder
    :return: Dictionary containing the QA pair data
    """
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
        # question_body = soup.find('div', class_='question_body')
        # if question_body:
        #     qa_pair['question'] = question_body.text.strip()
            
        # Extract time and location
        # 如果页面中有多个 text-muted 元素，可以遍历或选取第一个
        time_elements = soup.find_all(class_='text-muted')
        if time_elements:
            # 假设第一个元素包含你需要的时间信息
            qa_pair['asked_time'] = time_elements[0].get_text(strip=True) 
        
        location_tags = soup.find_all('span', class_='tag tag-geography')
        if location_tags:
            # 可能存在多个地理信息，示例将它们合并成列表
            qa_pair['location'] = [tag.get_text(strip=True) for tag in location_tags]
        else:
            qa_pair['location'] = []            
            
        # # Extract all responses, including multi-turn interactions
        # responses = []
        # response_blocks = soup.find_all('div', class_='response_block')
        # for block in response_blocks:
        #     response_text = block.find('div', class_='response')
        #     responder = block.find('div', class_='author')

        #     if response_text:
        #         response_info = {
        #             'responder': responder.text.strip() if responder else 'Unknown',
        #             'response': response_text.text.strip()
        #         }
        #         responses.append(response_info)
        # # Add responses to qa_pair
        # qa_pair['responses'] = responses
        # # Extract attachments
        # attachments = soup.find('div', class_='question-attachments')
        # if attachments:
        #     qa_pair['attachments'] = [urljoin(url, img['src']) for img in attachments.find_all('img')]
        # else:
        #     qa_pair['attachments'] = []
        return qa_pair
    except Exception as e:
        print(f"Error scraping {url}: {str(e)}")
        return None

class QAScraper:
    def __init__(self, base_url, qa_output_file_path, qa_images_folder,
                 start_page=1, max_pages=15000, num_workers=4, pages_per_batch=10,
                 find_missing=False, missing_pages_file=None):
        """
        Initialize the QAScraper.

        :param base_url: Base URL of the website to scrape
        :param qa_output_file_path: Path to save QA data
        :param qa_images_folder: Folder to save attachments
        :param start_page: Starting page number
        :param max_pages: Maximum number of pages to scrape
        :param num_workers: Number of worker processes
        :param pages_per_batch: Number of pages to scrape per batch
        :param find_missing: Whether to scrape specified missing pages
        :param missing_pages_file: File path containing list of missing page numbers
        """
        self.base_url = base_url
        self.qa_output_file_path = qa_output_file_path
        self.qa_images_folder = qa_images_folder
        self.start_page = start_page
        self.max_pages = max_pages
        self.num_workers = num_workers
        self.pages_per_batch = pages_per_batch
        self.find_missing = find_missing
        self.missing_pages_file = missing_pages_file
        self.current_page = self.start_page
        os.makedirs(self.qa_images_folder, exist_ok=True)
        if self.find_missing and self.missing_pages_file:
            # Read the missing page numbers from the file
            with open(self.missing_pages_file, 'r') as f:
                self.missing_pages = [int(line.strip()) for line in f if line.strip()]
        else:
            self.missing_pages = []

    def scrape_qa_list_page(self, page):
        """
        Scrape a list of QA links from a page.

        :param page: Page number to scrape
        :return: List of QA URLs found on the page
        """
        try:
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
        except RequestException as e:
            print(f"Error scraping list page {page}: {str(e)}")
            return []
        except Exception as e:
            print(f"Unknown error scraping list page {page}: {str(e)}")
            return []

    def run(self):
        """
        Main run method to initiate the scraping process.
        """
        try:
            pool = Pool(processes=self.num_workers)
            if self.find_missing and self.missing_pages:
                # Scrape the specified missing pages
                total_pages = len(self.missing_pages)
                all_pages = self.missing_pages
            else:
                # Scrape a range of pages
                total_pages = self.max_pages - self.start_page + 1
                all_pages = list(range(self.start_page, self.max_pages + 1))

            # Process pages in batches
            for batch_start in range(0, total_pages, self.pages_per_batch):
                batch_pages = all_pages[batch_start:batch_start + self.pages_per_batch]
                batch_start_page = batch_pages[0]
                batch_end_page = batch_pages[-1]

                # Report progress and current time
                print(f"\nScraping pages {batch_start_page} to {batch_end_page}, currently at time {time.strftime('%H:%M:%S')}")

                all_qa_links = []
                for page in batch_pages:
                    max_retries = 3
                    retry_delay = 5  # seconds
                    for attempt in range(max_retries):
                        qa_links = self.scrape_qa_list_page(page)
                        if qa_links:
                            all_qa_links.extend(qa_links)
                            break  # Success, move on to next page
                        else:
                            print(f"No questions found on page {page}, attempt {attempt + 1} of {max_retries}. Retrying in {retry_delay} seconds...")
                            time.sleep(retry_delay)
                    else:
                        print(f"No questions found on page {page} after {max_retries} attempts.")

                if not all_qa_links:
                    print("No more questions found in this batch, moving to next batch.")
                    continue

                args_list = [(link, self.qa_images_folder) for link in all_qa_links]
                results = []
                for qa_pair in tqdm(pool.imap_unordered(scrape_individual_qa, args_list), total=len(all_qa_links), desc=f"Processing pages {batch_start_page} to {batch_end_page}"):
                    if qa_pair:
                        results.append(qa_pair)

                # Process results in the main process
                for qa_pair in results:
                    # save_attachments(qa_pair, self.qa_images_folder)
                    save_qa_pair(qa_pair, self.qa_output_file_path)

                time.sleep(2)  # Polite delay to avoid too frequent requests
        except KeyboardInterrupt:
            print("\nInterrupted!")
        finally:
            print("Scraping completed.")
            pool.close()
            pool.join()

def main():
    parser = argparse.ArgumentParser(description='Scrape QA data from a website')
    parser.add_argument('--start_page', type=int, default=24400, help='Starting page number')
    parser.add_argument('--max_pages', type=int, default=29400, help='Maximum number of pages to scrape')
    parser.add_argument('--pages_per_batch', type=int, default=1, help='Number of pages to scrape per batch')
    parser.add_argument('--qa_output_file_path', type=str, default='dataset/raw_data/qa_data_missing.jsonl', help='Path to save QA data')
    parser.add_argument('--qa_images_folder', type=str, default='dataset/qa_images', help='Folder to save attachments')
    parser.add_argument('--num_workers', type=int, default=1, help='Number of worker processes')
    parser.add_argument('--find_missing', default=False, action='store_true', help='Enable scraping of specified missing pages')
    parser.add_argument('--missing_pages_file', default="/home/chigui2/workspace/AgrVQA/logs/page_numbers.txt" , type=str, help='File containing list of missing page numbers')

    args = parser.parse_args()
    base_url = 'https://ask.extension.org/kb/index.php'
    scraper = QAScraper(
        base_url=base_url,
        qa_output_file_path=args.qa_output_file_path,
        qa_images_folder=args.qa_images_folder,
        start_page=args.start_page,
        max_pages=args.max_pages,
        num_workers=args.num_workers,
        pages_per_batch=args.pages_per_batch,
        find_missing=args.find_missing,
        missing_pages_file=args.missing_pages_file
    )
    scraper.run()

if __name__ == '__main__':
    main()
