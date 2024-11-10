import json
import requests
from bs4 import BeautifulSoup
import time
import argparse
import os
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm

def process_entry(entry):
    scraped_urls = []
    # 提取并修改URL
    for url in entry.get('urls', []):
        modified_url = "https://r.jina.ai/" + url
        max_attempts = 20
        attempt = 0
        while attempt < max_attempts:
            try:
                # 请求修改后的URL
                res = requests.get(modified_url)
                res.raise_for_status()

                # 爬取HTML内容
                soup = BeautifulSoup(res.text, 'html.parser')
                scraped_content = soup.get_text(separator=' ', strip=True)
                scraped_urls.append({
                    'original_url': url,
                    'modified_url': modified_url,
                    'scraped_content': scraped_content
                })
                break
            except requests.exceptions.RequestException as e:
                print(f"Error scraping {modified_url}: {e}")
                time.sleep(10)  # 根据需要调整或移除
                attempt += 1
                print(f"Retrying... ({attempt}/{max_attempts})")

    # 如果抓取失败，标记为-1
    if scraped_urls:
        entry['scraped_urls'] = scraped_urls
    else:
        entry['scraped_urls'] = -1

    return entry

def main(input_file, output_file, num_process):
    # 初始化已处理的ID集合
    processed_ids = set()
    temp_output_file = output_file + '.tmp'

    # 如果输出文件存在，首先清理数据
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as infile, open(temp_output_file, 'w', encoding='utf-8') as outfile:
            for line in infile:
                try:
                    entry = json.loads(line)
                    # 移除'scraped_urls'为-1或None的条目
                    if entry.get('scraped_urls') not in [None, -1]:
                        outfile.write(line)
                        processed_ids.add(entry['id'])
                except json.JSONDecodeError:
                    continue
        # 用清理后的文件替换原始输出文件
        os.replace(temp_output_file, output_file)
    else:
        # 如果输出文件不存在，创建一个空文件
        open(output_file, 'w').close()

    # 加载输入数据
    with open(input_file, 'r', encoding='utf-8') as f:
        input_data = json.load(f)

    # 找出需要处理的条目
    remaining_entries = [entry for entry in input_data if entry['id'] not in processed_ids]

    if not remaining_entries:
        print("没有新的条目需要处理。")
        return

    print(f"正在处理 {len(remaining_entries)} 个新条目...")

    with Pool(processes=num_process) as pool:
        # 使用tqdm显示进度
        for result in tqdm(pool.imap_unordered(process_entry, remaining_entries), total=len(remaining_entries)):
            # 将结果写入输出文件
            with open(output_file, 'a', encoding='utf-8') as f:
                json_line = json.dumps(result, ensure_ascii=False)
                f.write(json_line + '\n')

    print(f"爬取完成。更新的数据已保存到 '{output_file}'。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='从数据集中爬取URLs。')
    parser.add_argument('--input_file', type=str, required=True, help='输入JSON数据集的路径。')
    parser.add_argument('--output_file', type=str, required=True, help='输出JSON Lines文件的路径。')
    parser.add_argument('--num_process', type=int, default=1, help='用于处理数据的进程数。默认值为CPU核心数。')
    args = parser.parse_args()

    main(args.input_file, args.output_file, args.num_process)
