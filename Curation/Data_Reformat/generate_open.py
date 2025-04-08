import os
import sys
import json
import argparse
from tqdm import tqdm
import time
import PIL.Image
import gc
import torch  # 确保已安装torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# 全局变量，用于存放加载的本地模型
LOCAL_MODEL = None

class GenerateLocal:
    def __init__(self, raw_data_file, output_file, model_name="llava-hf/llava-1.5-7b-hf"):
        self.raw_data_file = raw_data_file
        self.output_file = output_file
        # 为了保持和之前代码的一致性，用模型文件名的最后一部分作为 key
        self.model_name = model_name.split("/")[-1]
        if self.model_name.lower().startswith("llama-4") or self.model_name.lower().startswith("llama-3.2"):
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def get_prompt(self, item):
        question = item["question"]
        attachments = item.get("attachments", [])
        image_path = attachments[0] if attachments else None

        if self.model_name == "llava-v1.6-mistral-7b-hf":
            # LLaVA-1.6
            prompts = f"[INST] <image>\n{question}\n[/INST]"
        elif self.model_name.lower().startswith("deepseek-vl2"):
            # DeepSeek-VL2 (27.5B)
            prompts = f"<|User|>: <image>\n{question}\n\n<|Assistant|>:"
        elif self.model_name.lower().startswith("llava-1.5"):
            # LLaVA-1.5
            prompts = f"USER: <image>\n{question}\nASSISTANT:"
        elif self.model_name.lower().startswith("qwen2.5-vl"):
            # Qwen2.5-VL
            prompts = (
                "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
                f"{question}<|im_end|>\n"
                "<|im_start|>assistant\n"
            )
        elif self.model_name.lower().startswith("llama-4") or self.model_name.lower().startswith("llama-3.2"):
            # Llama-4 Scout 模型使用 tokenizer 的 apply_chat_template 方法构建 prompt
            # 构造单轮对话的消息格式
            message = [{
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question}
                ]
            }]
            # apply_chat_template 接受一个列表（包含单个对话轮次），返回生成的 prompt 列表
            prompts = self.tokenizer.apply_chat_template(message,
                                                        add_generation_prompt=True,
                                                        tokenize=False) 
        else:
            # 默认：直接返回问题文本
            prompts = question

        return {"prompt": prompts, "image_path": image_path}

    def generate(self, batch_size=16):
        # 读取原始数据文件
        with open(self.raw_data_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 检查 output_file 中已有的结果，避免重复处理
        processed_ids = set()
        if os.path.exists(self.output_file):
            with open(self.output_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        item = json.loads(line)
                        if self.model_name in item and item[self.model_name] != -1 and item[self.model_name] is not None:
                            processed_ids.add(item["id"])
                    except json.JSONDecodeError:
                        continue

        items_to_process = [item for item in data if item.get("id") not in processed_ids]
        total = len(items_to_process)
        print(f"Processing {total} items.")
        
        sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=4024)
        print(f"Using sampling parameters: {sampling_params}")
        
        # 总的batch数
        total_batches = (total + batch_size - 1) // batch_size
        print(f"Total batches: {total_batches}")
        
        # 按 batch_size 分批处理
        for i in tqdm(range(0, total, batch_size), desc="Processing batches"):
            current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            print(f"Processing batch {i // batch_size + 1}/{total_batches} at {current_time}")
            batch_items = items_to_process[i:i + batch_size]
            batch_inputs = []
            # 为每个条目构造输入数据
            for item in batch_items:
                prompt_data = self.get_prompt(item)
                prompt_str = prompt_data["prompt"]
                image = None
                if prompt_data["image_path"]:
                    try:
                        # 使用上下文管理器加载图像
                        with PIL.Image.open(prompt_data["image_path"]) as img:
                            image = img.copy()  # 复制图像数据以便后续使用
                    except Exception as e:
                        print(f"Error loading image {prompt_data['image_path']}: {e}")
                        continue
                input_dict = {"prompt": prompt_str}
                if image is not None:
                    input_dict["multi_modal_data"] = {"image": image}
                batch_inputs.append(input_dict)

            try:
                # 使用单进程批量调用模型生成
                outputs = LOCAL_MODEL.generate(batch_inputs, sampling_params=sampling_params)
                # 假定返回的 outputs 列表与输入顺序一致，每个 output 的 outputs[0].text 为生成的文本
                for idx, output in enumerate(outputs):
                    generated_text = output.outputs[0].text if output.outputs and len(output.outputs) > 0 else ""
                    batch_items[idx][self.model_name] = generated_text
            except Exception as e:
                print(f"Error processing batch starting at index {i}: {e}")
                # 出错时将该 batch 的结果标记为失败
                for item in batch_items:
                    item[self.model_name] = -1

            # 写入 batch 的处理结果到输出文件
            with open(self.output_file, "a", encoding="utf-8") as f:
                for item in batch_items:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
            # 批处理结束后释放内存资源
            gc.collect()
            torch.cuda.empty_cache()

        print("Processing completed.")
        self.cleanup_output(len(data))

    def cleanup_output(self, data_length):
        """
        清理 output_file 文件，只保留成功生成回答的条目，并汇报处理结果。
        """
        valid_items = []
        with open(self.output_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    item = json.loads(line)
                    if self.model_name in item and item[self.model_name] != -1 and item[self.model_name] is not None:
                        valid_items.append(item)
                except json.JSONDecodeError:
                    continue

        with open(self.output_file, "w", encoding="utf-8") as f:
            for item in valid_items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        print(f"Total successful items: {len(valid_items)}. \nRemaining items to process: {data_length - len(valid_items)}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate responses using local LLM model with batch input.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSON file.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output JSONL file.")
    parser.add_argument("--model_name", type=str, default="llava-hf/llava-1.5-7b-hf", help="Local model to use.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for processing.")
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs to use.")
    
    args = parser.parse_args()

    short_name = args.model_name.split("/")[-1].lower()
    if short_name.startswith("llava-v1.6"):
        architectures = ["LlavaNextForConditionalGeneration"]
    elif short_name.startswith("deepseek-vl2"):
        architectures = ["DeepseekVLV2ForCausalLM"]
    elif short_name.startswith("qwen2.5-vl"):
        architectures = ["Qwen2_5_VLForConditionalGeneration"]
    elif short_name.startswith("llama-3.2"):
        architectures = ["MllamaForConditionalGeneration"]
    elif short_name.startswith("llama-4"):
        architectures = ["Llama4ForConditionalGeneration"]
    else:
        print(f"Unsupported model name: {args.model_name}")
        sys.exit(1)

    # 在主进程中加载模型，确保只加载一次
    LOCAL_MODEL = LLM(
        model=args.model_name,
        tensor_parallel_size=args.num_gpus,
        max_num_seqs=args.batch_size,
        hf_overrides={"architectures": architectures},
        limit_mm_per_prompt={"image": 1, "video": 0},
        max_model_len=32768
    )

    generator = GenerateLocal(
        raw_data_file=args.input_file,
        output_file=args.output_file,
        model_name=args.model_name
    )
    generator.generate(batch_size=args.batch_size)
