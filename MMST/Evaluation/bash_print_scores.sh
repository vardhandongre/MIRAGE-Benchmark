# Print scores for a specific benchmark type, mode, subject, and judge.
BENCH_TYPE="contextual"

# ID or MG
MODE="MG"
SUBJECT_NAME="gpt-4o-mini"
JUDGE_NAME="Qwen3-32B"

python print_scores.py \
    --bench_type "$BENCH_TYPE" \
    --mode "$MODE" \
    --subject_name "$SUBJECT_NAME" \
    --judge_name "$JUDGE_NAME" \