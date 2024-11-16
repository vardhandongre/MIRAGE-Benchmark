import asyncio
from crawlee.playwright_crawler import PlaywrightCrawler, PlaywrightCrawlingContext, PlaywrightPreNavigationContext
import json
from crawlee import Request

async def main() -> None:
    # create a crawler
    crawler = PlaywrightCrawler()
    @crawler.router.default_handler
    async def request_handler(context: PlaywrightCrawlingContext) -> None:
        page = context.page
        url = context.request.url

        # remove header, footer, and nav elements
        await page.evaluate("""
            () => {
                const elements = document.querySelectorAll('header, footer, nav');
                elements.forEach(el => el.remove());
            }
        """)

        # extract the title and text content of the page
        title = await page.title()
        text_content = await page.evaluate("""
            () => {
                const body = document.body;
                return body.innerText;
            }
        """)

        data = {
            'url_id': context.request.unique_key,
            'url': url,
            'title': title,
            'content': text_content,
        }

        # push the data to the context
        await context.push_data(data=data)

        context.log.info(f"Processed {url}")
    
    input_file_path = "/home/chigui2/workspace/AgrVQA/dataset/raw_data/urls_dataset.json"
    with open(input_file_path, 'r', encoding='utf-8') as f:
        input_data = json.load(f)
    requests = []
    for entry in input_data:
        url = entry['url']
        id = entry['url_id']
        if url[-4:] == '.pdf':
            continue
        try:
            request = Request(url=url, unique_key=id, id=id) 
            requests.append(request)
        except Exception as e:
            print(f"Error: {e}")
    await crawler.run(requests)

if __name__ == '__main__':
    asyncio.run(main())
