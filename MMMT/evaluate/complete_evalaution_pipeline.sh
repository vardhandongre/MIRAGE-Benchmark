#!/bin/bash

# Configuration variables
GOLD_DATA="decomp_test.json"
JUDGE_PROVIDER="openai"
JUDGE_MODEL="gpt-4.1"
MAX_EXAMPLES=1500
OUTPUT_DIR="evaluations/neurips_reason_2"
MAX_RETRY=3

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Process missing samples across models
process_missing_samples() {
    echo "===================================================="
    echo "Analyzing missing samples across all models"
    echo "===================================================="
    
    # Run the missing samples analysis
    python analyze_missing_samples.py --output "${OUTPUT_DIR}/missing_samples_report.json"
    
    # Check if we want to evaluate missing samples
    if [ "$EVALUATE_MISSING" = "true" ]; then
        echo "Processing missing samples for each model..."
        
        # Process each missing samples file
        for MISSING_FILE in missing_samples_*.json; do
            if [ -f "$MISSING_FILE" ]; then
                MODEL_NAME=$(echo "$MISSING_FILE" | sed 's/missing_samples_\(.*\)\.json/\1/')
                LOG_FILE=$(find logs -name "*${MODEL_NAME}*.log" | head -n 1)
                
                if [ -n "$LOG_FILE" ]; then
                    echo "Evaluating missing samples for $MODEL_NAME"
                    OUTPUT_FILE="${OUTPUT_DIR}/missing_eval_${MODEL_NAME}.json"
                    
                    python evaluate_missing_samples.py \
                        --missing "$MISSING_FILE" \
                        --log "$LOG_FILE" \
                        --output "$OUTPUT_FILE"
                fi
            fi
        done
    fi
}

# Combined evaluation function with retry logic
evaluate_log_file() {
    LOG_FILE="$1"
    MODEL_NAME=$(basename "$LOG_FILE" .log)
    OUTPUT_FILE="${OUTPUT_DIR}/${MODEL_NAME}_eval.json"
    
    echo "===================================================="
    echo "Evaluating: $MODEL_NAME"
    echo "Log file: $LOG_FILE"
    echo "Output: $OUTPUT_FILE"
    echo "===================================================="
    
    # Try evaluation with retry logic
    RETRY=0
    SUCCESS=false
    
    while [ $RETRY -lt $MAX_RETRY ] && [ "$SUCCESS" = "false" ]; do
        if [ $RETRY -gt 0 ]; then
            echo "Retry attempt $RETRY for $MODEL_NAME"
        fi
        
        python evaluate_reason.py \
            --log "$LOG_FILE" \
            --gold "$GOLD_DATA" \
            --judge_provider "$JUDGE_PROVIDER" \
            --judge_model "$JUDGE_MODEL" \
            --max_examples "$MAX_EXAMPLES" \
            --output "$OUTPUT_FILE"
        
        if [ $? -eq 0 ] && [ -f "$OUTPUT_FILE" ]; then
            SUCCESS=true
            echo "Evaluation successful for $MODEL_NAME"
        else
            RETRY=$((RETRY+1))
            echo "Attempt $RETRY failed for $MODEL_NAME"
            sleep 5  # Wait before retrying
        fi
    done
    
    if [ "$SUCCESS" = "false" ]; then
        echo " All retry attempts failed for $MODEL_NAME"
        FAILED_MODELS+=("$MODEL_NAME")
    else
        SUCCESSFUL_MODELS+=("$MODEL_NAME")
    fi
}

# Display usage information
usage() {
    echo "Complete Log Evaluation Pipeline"
    echo ""
    echo "Usage: $0 [OPTIONS] [LOG_FILES...]"
    echo ""
    echo "Options:"
    echo "  --help                  Show this help message"
    echo "  --gold FILE             Path to gold data (default: $GOLD_DATA)"
    echo "  --provider PROVIDER     Judge provider (default: $JUDGE_PROVIDER)"
    echo "  --model MODEL           Judge model (default: $JUDGE_MODEL)"
    echo "  --max NUMBER            Maximum examples to evaluate (default: $MAX_EXAMPLES)"
    echo "  --output DIR            Output directory (default: $OUTPUT_DIR)"
    echo "  --folder DIR            Process all log files in a directory"
    echo "  --missing               Process missing samples across all models"
    echo "  --evaluate-missing      Also evaluate missing samples"
    echo "  --retry NUMBER          Maximum retry attempts (default: $MAX_RETRY)"
    echo ""
    echo "Examples:"
    echo "  $0 logs/model1.log logs/model2.log"
    echo "  $0 --folder logs/zero_shot_decomp"
    echo "  $0 --missing --evaluate-missing"
    echo ""
    exit 1
}

# Parse command-line arguments
PROCESS_MISSING=false
EVALUATE_MISSING=false
PROCESS_FOLDER=false
FOLDER_PATH=""
LOG_FILES=()
FAILED_MODELS=()
SUCCESSFUL_MODELS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --help)
            usage
            ;;
        --gold)
            GOLD_DATA="$2"
            shift 2
            ;;
        --provider)
            JUDGE_PROVIDER="$2"
            shift 2
            ;;
        --model)
            JUDGE_MODEL="$2"
            shift 2
            ;;
        --max)
            MAX_EXAMPLES="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --retry)
            MAX_RETRY="$2"
            shift 2
            ;;
        --folder)
            PROCESS_FOLDER=true
            FOLDER_PATH="$2"
            shift 2
            ;;
        --missing)
            PROCESS_MISSING=true
            shift
            ;;
        --evaluate-missing)
            EVALUATE_MISSING=true
            shift
            ;;
        *)
            # Assume any non-option is a log file
            if [[ -f "$1" ]]; then
                LOG_FILES+=("$1")
            else
                echo "Error: File not found: $1"
                exit 1
            fi
            shift
            ;;
    esac
done

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Process missing samples if requested
if [ "$PROCESS_MISSING" = "true" ]; then
    process_missing_samples
    exit 0
fi

# Process all log files in a folder if requested
if [ "$PROCESS_FOLDER" = "true" ]; then
    if [ ! -d "$FOLDER_PATH" ]; then
        echo "Error: Folder not found: $FOLDER_PATH"
        exit 1
    fi
    
    echo "Processing all log files in $FOLDER_PATH"
    LOG_FILES=($(find "$FOLDER_PATH" -name "*.log" -type f))
    
    if [ ${#LOG_FILES[@]} -eq 0 ]; then
        echo "No log files found in $FOLDER_PATH"
        exit 1
    fi
    
    echo "Found ${#LOG_FILES[@]} log files"
fi

# Check if any log files were provided or found
if [ ${#LOG_FILES[@]} -eq 0 ] && [ "$PROCESS_MISSING" = "false" ]; then
    echo "Error: No log files provided or found"
    usage
fi

# Process each log file
for LOG_FILE in "${LOG_FILES[@]}"; do
    evaluate_log_file "$LOG_FILE"
done

# Generate summary report
echo "===================================================="
echo "Evaluation Summary"
echo "===================================================="
echo "Total log files processed: ${#LOG_FILES[@]}"
echo "Successful evaluations: ${#SUCCESSFUL_MODELS[@]}"
echo "Failed evaluations: ${#FAILED_MODELS[@]}"

if [ ${#SUCCESSFUL_MODELS[@]} -gt 0 ]; then
    echo ""
    echo "Successful models:"
    for MODEL in "${SUCCESSFUL_MODELS[@]}"; do
        echo "  - $MODEL"
    done
fi

if [ ${#FAILED_MODELS[@]} -gt 0 ]; then
    echo ""
    echo "Failed models:"
    for MODEL in "${FAILED_MODELS[@]}"; do
        echo "  - $MODEL"
    done
fi

echo ""
echo "Results saved in $OUTPUT_DIR directory"
echo "===================================================="

# Create a metrics summary if jq is available
if command -v jq &> /dev/null; then
    SUMMARY_FILE="${OUTPUT_DIR}/metrics_summary.json"
    
    echo "Generating metrics summary..."
    
    echo "{" > "$SUMMARY_FILE"
    echo "  \"models\": {" >> "$SUMMARY_FILE"
    
    FIRST=true
    for EVAL_FILE in "$OUTPUT_DIR"/*_eval.json; do
        if [ -f "$EVAL_FILE" ]; then
            MODEL_NAME=$(basename "$EVAL_FILE" _eval.json)
            
            if [ "$FIRST" = "true" ]; then
                FIRST=false
            else
                echo "    ," >> "$SUMMARY_FILE"
            fi
            
            echo "    \"$MODEL_NAME\": $(jq '.metrics' "$EVAL_FILE")" >> "$SUMMARY_FILE"
        fi
    done
    
    echo "  }" >> "$SUMMARY_FILE"
    echo "}" >> "$SUMMARY_FILE"
    
    echo " Metrics summary saved to $SUMMARY_FILE"
fi

exit 0