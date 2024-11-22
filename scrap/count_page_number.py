import requests
from bs4 import BeautifulSoup

def find_total_pages(base_url):
    low = 25000
    high = 100000  # Arbitrary high number
    while low <= high:
        mid = (low + high) // 2
        print(f"Checking page {mid}")
        page_url = f"{base_url}?p={mid}"
        max_attempts = 3
        attempts = 0
        qa_links = []
        while attempts < max_attempts:
            response = requests.get(page_url)
            soup = BeautifulSoup(response.content, 'html.parser')
            question_containers = soup.find_all('div', class_='article-title')
            for container in question_containers:
                link = container.find('a')
                if link:
                    qa_links.append(1)
            if len(qa_links) > 0:
                break
            attempts += 1   
        if len(qa_links) == 0:
            high = mid - 1
        else:
            low = mid + 1
            
    # The total number of pages is high
    return high

# Usage
base_url = 'https://ask2.extension.org/kb/index.php'
total_pages = find_total_pages(base_url)
print(f"Total number of pages: {total_pages}")
