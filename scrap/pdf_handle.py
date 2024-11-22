import asyncio
import aiohttp
import io
import json
from PyPDF2 import PdfReader

async def main() -> None:
    input_file_path = "/home/chigui2/workspace/AgrVQA/dataset/pre_data/pdf_urls_inputs.json"
    with open(input_file_path, 'r', encoding='utf-8') as f:
        input_data = json.load(f)

    pdf_tasks = []
    for entry in input_data:
        url = entry['url']
        id = entry['url_id']
        if url.endswith('.pdf'):
            # 处理 PDF 链接
            pdf_tasks.append(handle_pdf(url, id))

    # download and process PDFs concurrently
    await asyncio.gather(*pdf_tasks)

async def handle_pdf(url: str, id: str) -> None:
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    pdf_bytes = await response.read()
                    pdf_file = io.BytesIO(pdf_bytes)

                    # 从 PDF 中提取文本
                    reader = PdfReader(pdf_file)
                    text_content = ""
                    for page in reader.pages:
                        text = page.extract_text()
                        if text:
                            text_content += text + '\n'

                    data = {
                        'url_id': id,
                        'url': url,
                        'title': 'PDF Document',
                        'content': text_content,
                    }

                    print(f"Processed PDF: {url}")

                    output_file = "storage/pdf_output.json"
                    
                    with open(output_file, 'a', encoding='utf-8') as f:
                        json.dump(data, f, ensure_ascii=False)
                        f.write('\n')
                else:
                    print(f"Failed to download PDF: {url}, status code: {response.status}")
    except Exception as e:
        print(f"Error: {e}")        

if __name__ == '__main__':
    import os
    # Remove the existing output file
    output_file = "storage/pdf_output.json"
    if os.path.exists(output_file):
        os.remove(output_file)
    asyncio.run(main())
