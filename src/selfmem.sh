#!/bin/bash

# Pseudo code for SelfMemory process:

# generator = Trainer(model_input,memory)

ACCELERATOR="${1:-gpu}"

EXPNAME="baseline"
DATA_PATH="../data/ende/test_small.jsonl"
QUERIES_PATH="../data/ende/test_small_src.txt"
TARGET_PATH="../data/ende/test_small_trg.txt"
NUM_BEAMS=10
B=1

EXPDIR="../data/ende/memory/${EXPNAME}"

HYPS_PATH="${EXPDIR}/hyps_${NUM_BEAMS}.txt"
# candidates = generator(model_input,memory)
python generate_hyps.py --config config/jrc_ende/generate_hyps.yaml --accelerator "${ACCELERATOR}" \
    --data_path "${DATA_PATH}" --output_path "${HYPS_PATH}"
# selector = Trainer(model_input,candidates)
MEMORY_CANDIDATES_DIR="${EXPDIR}"
python select_mem.py --queries_path "${QUERIES_PATH}" \
    --memory_path "${HYPS_PATH}" \
    --memory_scores_path "${MEMORY_CANDIDATES_DIR}/memory_scores.json" \
    --num_beams "${NUM_BEAMS}" --B "${B}" \
    --output_path "${MEMORY_CANDIDATES_DIR}"

do_one_iteration() {
    k=$1
    beam=$2

    MEMORY_PATH="${MEMORY_CANDIDATES_DIR}/top_${beam}_memories.txt"
    EXPDIR="${EXPDIR}/iter${k}"
    mkdir -p "${EXPDIR}"
    mv "${MEMORY_PATH}" "${EXPDIR}/memory.txt"
    MEMORY_CANDIDATES_DIR="${EXPDIR}"

    HYPS_PATH="${EXPDIR}/hyps_${NUM_BEAMS}.txt"

    python generate_hyps.py --config config/jrc_ende/generate_hyps.yaml --accelerator "${ACCELERATOR}" \
        --data_path "${DATA_PATH}" --memory_path "${MEMORY_CANDIDATES_DIR}/memory.txt" --output_path "${HYPS_PATH}"

    python select_mem.py --queries_path "${QUERIES_PATH}" \
        --memory_path "${HYPS_PATH}" \
        --memory_scores_path "${MEMORY_CANDIDATES_DIR}/memory_scores.json" \
        --num_beams "${NUM_BEAMS}" --B "${B}" \
        --output_path "${MEMORY_CANDIDATES_DIR}"

    for beam in {1..NUM_BEAMS}; do
        EVAL_PATH="${EXPDIR}/eval_${beam}.txt"
        python generate_hyps.py --config config/jrc_ende/generate_hyps.yaml --accelerator "${ACCELERATOR}" \
            --num_beams 1 --num_return_sequences 1 --num_beam_groups 1 \
            --data_path "${DATA_PATH}" \
            --memory_path "${MEMORY_CANDIDATES_DIR}/top_${beam}_memories.txt" \
            --output_path "${EVAL_PATH}"
        python utils/metrics_utils.py --hyp_path "${EVAL_PATH}" --ref_path "${TARGET_PATH}"
    done
}

# for _ in range(iteration_k):
for i in {1..3}; do
    echo "Iteration $i"
    for beam in {1..B}; do
        do_one_iteration $i $beam
    done
done
