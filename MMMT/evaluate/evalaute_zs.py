#!/usr/bin/env python3
import os
import json
import re
import argparse
import time
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import random
from src.llm_agents import get_llm_client

# === Log Extraction Functions ===
def safe_json_parse(text: str):
    """Try parsing with fallbacks for common JSON issues."""
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        # Save the original error
        original_error = str(e)
        
        try:
            # Heuristic fix 1: Remove control characters
            fixed = ''.join(ch for ch in text if ch >= ' ' or ch in ['\n', '\r', '\t'])
            
            # Heuristic fix 2: Fix missing colons after property names
            fixed = re.sub(r'"\s*(\w+)\s*"(\s*\{)', r'"\1":\2', fixed)  # Fix "property" { to "property": {
            fixed = re.sub(r'"\s*(\w+)\s*"(\s*\[)', r'"\1":\2', fixed)  # Fix "property" [ to "property": [
            fixed = re.sub(r'"([^"]+)"\s+([^,:{\[\s])', r'"\1": \2', fixed)  # Fix "property" value to "property": value
            
            # Heuristic fix 3: Remove trailing commas in objects and arrays
            fixed = re.sub(r',\s*([}\]])', r'\1', fixed)
            
            # Heuristic fix 4: Add missing commas between properties
            fixed = re.sub(r'(true|false|null|"[^"]*"|\d+)\s*"', r'\1, "', fixed)  # Fix value"property" to value,"property"
            fixed = re.sub(r'(\]|\})\s*"', r'\1, "', fixed)  # Fix }"property" to },"property"
            
            # Heuristic fix 5: Replace smart quotes and em dashes
            fixed = fixed.replace('\u201c', '"').replace('\u201d', '"').replace('\u2014', '-')
            
            # Heuristic fix 6: Replace invalid quotes
            fixed = fixed.replace("\x91", "'").replace("\x92", "'").replace("\x93", '"').replace("\x94", '"')
            
            # Heuristic fix 7: Ensure property names are quoted
            fixed = re.sub(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', fixed)
            
            # Try parsing with fixes
            try:
                return json.loads(fixed)
            except json.JSONDecodeError:
                # Heuristic fix 8: Try more aggressive approaches for severely malformed JSON
                
                # Attempt to extract a valid JSON subset by finding outermost braces
                if '{' in fixed and '}' in fixed:
                    start_idx = fixed.find('{')
                    end_idx = fixed.rfind('}') + 1
                    json_subset = fixed[start_idx:end_idx]
                    
                    try:
                        return json.loads(json_subset)
                    except json.JSONDecodeError:
                        # Last resort: build a minimal valid JSON with whatever we can extract
                        goal_state = {}
                        
                        # Try to extract known facts
                        known_match = re.search(r'"known"\s*:\s*{([^{}]*(?:{[^{}]*}[^{}]*)*)}', fixed)
                        if known_match:
                            known_text = '{' + known_match.group(1) + '}'
                            try:
                                goal_state["known"] = json.loads(known_text)
                            except:
                                # Extract key-value pairs manually
                                known_pairs = re.findall(r'"([^"]+)"\s*:\s*"([^"]+)"', known_match.group(1))
                                goal_state["known"] = {k: v for k, v in known_pairs}
                        
                        # Try to extract missing items
                        missing_match = re.search(r'"missing"\s*:\s*\[(.*?)\]', fixed, re.DOTALL)
                        if missing_match:
                            missing_text = '[' + missing_match.group(1) + ']'
                            try:
                                goal_state["missing"] = json.loads(missing_text)
                            except:
                                # Extract items manually
                                missing_items = re.findall(r'"([^"]+)"', missing_match.group(1))
                                goal_state["missing"] = missing_items
                        
                        # Try to extract decision
                        decision_match = re.search(r'"decision"\s*:\s*"([^"]+)"', fixed)
                        decision = decision_match.group(1) if decision_match else "Unknown"
                        
                        # Try to extract utterance
                        utterance_match = re.search(r'"utterance"\s*:\s*"([^"]*)"', fixed)
                        utterance = utterance_match.group(1) if utterance_match else ""
                        
                        # Build a valid result
                        return {
                            "goal_state": goal_state,
                            "decision": decision,
                            "utterance": utterance
                        }
        except Exception as fallback_error:
            print(f"Fallback parsing failed: {fallback_error}")
            # If all else fails, create a minimal structure
            return {
                "goal_state": {
                    "known": {"error": "Parsing failed"},
                    "missing": ["Unable to parse prediction"]
                },
                "decision": "Unknown",
                "utterance": f"Parsing error: {original_error}"
            }

def extract_predictions_from_log(log_file: str) -> List[Dict]:
    """Extract predictions directly from a log file."""
    pattern = re.compile(r'Sample #(\d+): <output>(.*?)</output>', re.DOTALL)
    preds = []

    try:
        with open(log_file, "r", encoding='utf-8') as f:
            content = f.read()
            
        # Find all prediction blocks using regex
        matches = pattern.finditer(content)
        
        for match in matches:
            sample_id = match.group(1)
            raw_prediction = match.group(2).strip()
            
            try:
                data = safe_json_parse(raw_prediction)
                preds.append({
                    "source_id": f"#{sample_id}",
                    "predicted_goal_state": data.get("goal_state", {}),
                    "predicted_decision": data.get("decision"),
                    "generated_utterance": data.get("utterance", "")
                })
                # Don't print for each success to keep console clean
            except Exception as e:
                print(f"[Error] Failed to parse sample #{sample_id}: {str(e)[:100]}")
                # Save problematic samples for debugging
                with open("failed_samples_debug.txt", "a") as errlog:
                    errlog.write(f"# {sample_id}\n{raw_prediction}\n\n")
        
        print(f"[Info] Extracted {len(preds)} predictions from {log_file}")
        return preds
    except Exception as e:
        print(f"[Error] Failed to process log file {log_file}: {e}")
        return []

# === Utility Functions ===
def load_jsonl_or_json(path: str) -> List[Dict]:
    """Load data from JSON or JSONL file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
        
    with open(path, 'r', encoding='utf-8') as f:
        if path.endswith(".jsonl"):
            return [json.loads(line) for line in f if line.strip()]
        return json.load(f)

def save_results(results: List[Dict], metrics: Dict, output_path: str) -> None:
    """Save evaluation results and metrics to output file."""
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    output_data = {
        "results": results,
        "metrics": metrics,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"✅ Results saved to {output_path}")

def calculate_metrics(results: List[Dict]) -> Dict:
    """Calculate aggregate metrics from individual results."""
    if not results:
        return {
            "error": "No results to calculate metrics",
            "sample_count": 0
        }
    
    metrics = {
        "known_accuracy_avg": sum(r["known_accuracy"] for r in results) / len(results),
        "missing_coverage_avg": sum(r["missing_coverage"] for r in results) / len(results),
        "spurious_percentage": sum(1 for r in results if r["spurious"] == "Yes") / len(results) * 100,
        "utterance_goal_relevance_avg": sum(r["utterance_goal_relevance"] for r in results) / len(results),
        "perfect_scores": sum(1 for r in results if r["known_accuracy"] == 4 and r["missing_coverage"] == 4 
                            and r["spurious"] == "No" and r["utterance_goal_relevance"] == 4),
        "perfect_score_percentage": sum(1 for r in results if r["known_accuracy"] == 4 and r["missing_coverage"] == 4 
                                    and r["spurious"] == "No" and r["utterance_goal_relevance"] == 4) / len(results) * 100,
        "sample_count": len(results)
    }
    
    # Add decision accuracy metrics if any result has decision_accuracy
    if any("decision_accuracy" in r for r in results):
        # Only calculate for results that have decision_accuracy
        decision_results = [r for r in results if "decision_accuracy" in r]
        if decision_results:
            metrics["decision_accuracy_avg"] = sum(r["decision_accuracy"] for r in decision_results) / len(decision_results)
            metrics["decision_accuracy_percentage"] = metrics["decision_accuracy_avg"] * 100
    
    return metrics

# === Prompt Construction ===
def build_llm_judge_prompt(gold_known: Dict, gold_missing: List[str], 
                           pred_known: Dict, pred_missing: List[str],
                           utterance: str, goal: str, 
                           gold_decision: str = None, pred_decision: str = None,
                           scale: int = 4) -> str:
    """Build the prompt for the LLM judge to evaluate predictions."""
    
    # Add decision comparison section if decisions are provided
    decision_section = ""
    if gold_decision is not None and pred_decision is not None:
        decision_section = f"""
Gold decision: {gold_decision}
Model predicted decision: {pred_decision}
"""

    prompt = f"""
You are an expert evaluator for an agricultural assistant system. Your job is to judge whether the model's predicted information and question are helpful and relevant to the user's goal.

User Goal:
{goal}
{decision_section}
Gold known facts:
{json.dumps(gold_known, indent=2)}

Gold missing information:
{json.dumps(gold_missing, indent=2)}

Model predicted known facts:
{json.dumps(pred_known, indent=2)}

Model predicted missing info:
{json.dumps(pred_missing, indent=2)}

Model utterance:
{utterance}

IMPORTANT EVALUATION INSTRUCTIONS:
- When comparing keys between gold known facts and model predictions, ignore formatting differences like underscores vs. spaces (e.g., "tree_type" and "tree type" should be considered the same). 
- Focus on the semantic content rather than exact string matching.
- For "missing" information lists, compare semantically rather than requiring exact wording. If two items cover the same information need but are phrased differently, consider them a match.

Score the model based on the following criteria:

1. **Known Accuracy (0–{scale})**: 
   - Score {scale} if all gold known facts are correctly identified by the model (even if phrased differently)
   - Reduce score proportionally for each missing or incorrect fact
   - Keys with different formatting but same meaning (e.g., "tree_type" vs "tree type") should be considered matches

2. **Missing Coverage (0–{scale})**:
   - Score {scale} if the model captures the meaning of all items from the gold missing list
   - Compare semantically rather than requiring exact wording
   - Reduce score proportionally for each missing concept

3. **Spurious Entries**:
   - Answer "Yes" if there are any irrelevant or incorrect entries in known or missing
   - Answer "No" if all entries are relevant and correct

4. **Goal Relevance of Utterance (0–{scale})**:
   - Score {scale} if the utterance is highly relevant to achieving the user's goal
   - Focus only on relevance to the goal, not factual accuracy
   - Consider whether the utterance asks for appropriate missing information or provides helpful guidance
"""

    # Add decision accuracy metric if decisions are provided
    if gold_decision is not None and pred_decision is not None:
        prompt += """
5. **Decision Accuracy** (0 or 1):
   - Score 1 if the model's decision matches the gold decision (considering semantic equivalence)
   - Score 0 if the model's decision differs from the gold decision
   - Decisions like "Clarify" vs "Ask for clarification" should be considered the same
"""

    prompt += f"""
Respond in this format:
<answer>
{{
  "known_accuracy": int,  // from 0 to {scale}
  "missing_coverage": int,  // from 0 to {scale}
  "spurious": "Yes" or "No",
  "utterance_goal_relevance": int,  // from 0 to {scale}"""

    # Add decision accuracy to the response format if decisions are provided
    if gold_decision is not None and pred_decision is not None:
        prompt += """,
  "decision_accuracy": int  // either 0 or 1"""

    prompt += """
  "explanation": "Detailed explanation of your scoring decisions"
}
</answer>
"""
    return prompt

# === LLM Client Wrapper ===
class LLMClient:
    """Wrapper for different LLM APIs."""
    def __init__(self, provider: str, model: str):
        self.provider = provider
        self.model = model
        self.client = self._get_client()
        
    def _get_client(self):
        """Get the appropriate client based on provider."""
        if self.provider.lower() in ["anthropic", "claude"]:
            try:
                from anthropic import Anthropic
                return Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
            except ImportError:
                raise ImportError("Anthropic SDK not installed. Please run: pip install anthropic")
        elif self.provider.lower() in ["openai", "gpt"]:
            try:
                from openai import OpenAI
                return OpenAI()
            except ImportError:
                raise ImportError("OpenAI SDK not installed. Please run: pip install openai")
        elif self.provider.lower() in ["together"]:
            try:
                import together
                together.api_key = os.environ.get("TOGETHER_API_KEY")
                return together
            except ImportError:
                raise ImportError("Together SDK not installed. Please run: pip install together")
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    def generate(self, messages: List[Dict]) -> str:
        """Generate a response from the LLM."""
        if self.provider.lower() in ["anthropic", "claude"]:
            response = self.client.messages.create(
                model=self.model,
                messages=messages,
                max_tokens=1000
            )
            return response.content[0].text
        elif self.provider.lower() in ["openai", "gpt"]:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=1000
            )
            return response.choices[0].message.content
        elif self.provider.lower() in ["together"]:
            response = self.client.Complete.create(
                model=self.model,
                prompt=messages[0]["content"],
                max_tokens=1000
            )
            return response["output"]["content"]
        
        raise ValueError("Failed to generate response")

# === Evaluation ===
def evaluate(preds: List[Dict], golds: List[Dict], 
            judge_client: LLMClient, scale: int = 4, 
            max_examples: Optional[int] = None,
            random_sampling: bool = False) -> Tuple[List[Dict], Dict]:
    """Evaluate predictions against gold data using an LLM judge."""
    gold_map = {g['id']: g for g in golds}
    results = []
    
    # Handle max_examples and random sampling
    if max_examples and max_examples < len(preds):
        if random_sampling:
            print(f"Randomly sampling {max_examples} from {len(preds)} predictions")
            pred_subset = random.sample(preds, max_examples)
        else:
            print(f"Taking first {max_examples} from {len(preds)} predictions")
            pred_subset = preds[:max_examples]
    else:
        pred_subset = preds
        
    print(f"Starting evaluation of {len(pred_subset)} predictions with scale {scale}")
    
    for i, sample in enumerate(tqdm(pred_subset, desc="LLM Judge Eval")):
        sid = sample.get('source_id')
        if not sid:
            print(f"[Warning] Sample at index {i} missing source_id, skipping")
            continue
            
        gold = gold_map.get(sid)
        if not gold:
            print(f"[Warning] No gold data found for sample {sid}, skipping")
            continue
            
        # Extract relevant fields
        goal = gold.get("goal", "")
        
        # If goal is empty, try alternate field names
        if not goal:
            goal = gold.get("goal_state", {}).get("goal", "")
            if not goal and "decision" in gold:
                goal = f"Determine whether to {gold.get('decision', '')}"
                
        # Handle different field naming conventions
        utterance = sample.get("generated_utterance", "")
        if not utterance:
            utterance = sample.get("utterance", "")
            
        # Extract additional fields for decision accuracy
        gold_decision = gold.get("decision", None)
        pred_decision = sample.get("predicted_decision", None)
        
        # Check if we have both decisions for comparison
        include_decision = gold_decision is not None and pred_decision is not None
        
        # Handle different goal state field names and structures
        if "predicted_goal_state" in sample:
            pred_goal_state = sample["predicted_goal_state"]
        elif "goal_state" in sample:
            pred_goal_state = sample["goal_state"]
        else:
            pred_goal_state = {}
                
        # Extract known and missing with fallbacks
        pred_known = pred_goal_state.get("known", {})
        pred_missing = pred_goal_state.get("missing", [])
        
        # Handle gold data with similar flexibility
        if "goal_state" in gold:
            gold_goal_state = gold["goal_state"]
        else:
            gold_goal_state = {}
                
        gold_known = gold_goal_state.get("known", {})
        gold_missing = gold_goal_state.get("missing", [])

        # Build and send prompt to LLM judge
        prompt = build_llm_judge_prompt(
            gold_known, gold_missing, pred_known, pred_missing, 
            utterance, goal, gold_decision, pred_decision, scale
        )

        try:
            resp = judge_client.generate([{'role': 'user', 'content': prompt}])
            answer_text = resp.strip()
            
            if "<answer>" in answer_text and "</answer>" in answer_text:
                answer_block = answer_text.split("<answer>")[-1].split("</answer>")[0].strip()
                try:
                    scores = json.loads(answer_block)
                except json.JSONDecodeError:
                    print(f"[Warning] Invalid JSON in answer for sample {sid}")
                    # Attempt to fix common JSON issues
                    answer_block = answer_block.replace("'", '"').replace("//", "")
                    scores = json.loads(answer_block)
            else:
                print(f"[Warning] Missing answer tags for sample {sid}")
                # Extract scores using fallback method
                scores = {
                    "known_accuracy": 0,
                    "missing_coverage": 0,
                    "spurious": "Yes",
                    "utterance_goal_relevance": 0
                }
                if include_decision:
                    scores["decision_accuracy"] = 0
            
        except Exception as e:
            print(f"[Warning] Judge failed on sample {sid}: {e}")
            scores = {
                "known_accuracy": 0,
                "missing_coverage": 0,
                "spurious": "Yes",
                "utterance_goal_relevance": 0,
                "explanation": f"Error in evaluation: {str(e)}"
            }
            if include_decision:
                scores["decision_accuracy"] = 0

        result = {
            "source_id": sid, 
            **scores
        }
        results.append(result)
        
        # Print progress updates
        if (i+1) % 10 == 0 or i == len(pred_subset) - 1:
            print(f"Progress: {i+1}/{len(pred_subset)} samples evaluated")
            
    # Calculate aggregate metrics
    metrics = calculate_metrics(results)
    print("\nEvaluation complete. Aggregate metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
            
    return results, metrics

# === Main Function for Direct Log Evaluation ===
def direct_log_evaluate(log_file: str, gold_file: str, 
                       judge_provider: str, judge_model: str,
                       max_examples: int = None, random_sampling: bool = False,
                       output_file: str = None, scale: int = 4):
    """Directly evaluate log file against gold data."""
    print(f"Processing log file: {log_file}")
    
    # Step 1: Extract predictions from log file
    predictions = extract_predictions_from_log(log_file)
    if not predictions:
        print(f"[Error] No valid predictions found in {log_file}")
        return None, None
    
    print(f"Successfully extracted {len(predictions)} predictions")
    
    # Save the extracted predictions to a temporary file for inspection
    model_name = os.path.basename(log_file).replace('.log', '')
    extracted_file = f"extracted_{model_name}.json"
    with open(extracted_file, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, indent=2)
    print(f"[Info] Saved extracted predictions to {extracted_file} for reference")
    
    # Step 2: Load gold data
    try:
        gold_data = load_jsonl_or_json(gold_file)
        print(f"Loaded {len(gold_data)} gold samples")
    except Exception as e:
        print(f"[Error] Failed to load gold data: {e}")
        return None, None
    
    # Create a mapping of gold sample IDs for quick lookup
    gold_map = {g['id']: g for g in gold_data}
    
    # Track missing sample IDs for later processing
    extracted_ids = {p['source_id'] for p in predictions}
    all_sample_ids = set(gold_map.keys())
    missing_sample_ids = all_sample_ids - extracted_ids
    
    # Save missing samples information
    missing_samples_file = f"missing_samples_{model_name}.json"
    with open(missing_samples_file, 'w', encoding='utf-8') as f:
        missing_data = {
            "model": model_name,
            "total_gold_samples": len(gold_data),
            "extracted_samples": len(predictions),
            "missing_samples": len(missing_sample_ids),
            "missing_sample_ids": sorted(list(missing_sample_ids))
        }
        json.dump(missing_data, f, indent=2)
    print(f"[Info] Saved {len(missing_sample_ids)} missing sample IDs to {missing_samples_file}")
    
    # Step 3: Initialize LLM client
    try:
        judge_client = get_llm_client(judge_provider, judge_model)
        print(f"Initialized {judge_provider} client with model {judge_model}")
    except Exception as e:
        print(f"[Error] Failed to initialize LLM client: {e}")
        return None, None
    
    # Step 4: Run evaluation
    results, metrics = evaluate(
        predictions, gold_data, judge_client, 
        scale, max_examples, random_sampling
    )
    
    # Add missing samples information to metrics
    metrics["extraction_coverage"] = len(predictions) / len(gold_data) * 100
    metrics["missing_samples_count"] = len(missing_sample_ids)
    
    # Step 5: Save results if output path provided
    if output_file:
        save_results(results, metrics, output_file)
    
    return results, metrics

# === Main ===
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Directly evaluate log files against gold data")
    parser.add_argument('--log', required=True, help='Path to log file')
    parser.add_argument('--gold', required=True, help='Path to gold data file')
    parser.add_argument('--judge_provider', required=True, help='LLM provider (anthropic, openai, together)')
    parser.add_argument('--judge_model', required=True, help='LLM model name')
    parser.add_argument('--max_examples', type=int, default=None, help='Maximum examples to evaluate')
    parser.add_argument('--random_sampling', action='store_true', help='Randomly sample examples')
    parser.add_argument('--output', default=None, help='Path to save results')
    parser.add_argument('--rate_scale', type=int, default=4, help='Rating scale (default: 4)')
    
    args = parser.parse_args()
    
    # Run direct log evaluation
    direct_log_evaluate(
        log_file=args.log,
        gold_file=args.gold,
        judge_provider=args.judge_provider,
        judge_model=args.judge_model,
        max_examples=args.max_examples,
        random_sampling=args.random_sampling,
        output_file=args.output,
        scale=args.rate_scale
    )