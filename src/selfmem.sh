#!/bin/bash

# Translation of pseudocode for SelfMemory process.

# generator = Trainer(model_input,memory)
# candidates = generator(model_input,memory)
# selector = Trainer(model_input,candidates)

# for _ in range(iteration_k):
#     candidates = generator(model_input,memory)
#     memory = selector(model_input,candidates)
#     hypothesis = generator(model_input,memory)
#     current_score = metrics(hypothesis,reference)

for ARGUMENT in "$@"
do
   KEY=$(echo $ARGUMENT | cut -f1 -d=)

   KEY_LENGTH=${#KEY}
   VALUE="${ARGUMENT:$KEY_LENGTH+1}"

   export "$KEY"="$VALUE"
done

ACCELERATOR="${ACCELERATOR:-gpu}"

DATA_DIR="${DATA_DIR:-../data/ende}"

EXPNAME="${EXPNAME:-baseline}"
BASE_EXPDIR="${DATA_DIR}/memory/${EXPNAME}"
mkdir -p "${BASE_EXPDIR}"

DATA_PATH="${DATA_DIR}/test_small.jsonl"
SRC_PATH="${DATA_DIR}/test_small_src.txt"
TRG_PATH="${DATA_DIR}/test_small_trg.txt"

NUM_BEAMS="${NUM_BEAMS:-10}"
B="${B:-2}"
K="${K:-3}"

generator() {
    local model_input memory candidates_path python_args
    local "${@}"

    if [ -z "$memory" ]; then
        python generate_hyps.py --config config/jrc_ende/generate_hyps.yaml --accelerator "${ACCELERATOR}" \
            --data_path "${model_input}" \
            --output_path "${candidates_path}" \
            ${python_args}
    else
        python generate_hyps.py --config config/jrc_ende/generate_hyps.yaml --accelerator "${ACCELERATOR}" \
            --data_path "${model_input}" \
            --memory_path "${memory}" \
            --output_path "${candidates_path}" \
            ${python_args}
    fi
}

selector() {
    local model_input candidates_path expdir python_args
    local "${@}"

    python select_mem.py --queries_path "${model_input}" \
        --memory_path "${candidates_path}" \
        --memory_scores_path "${expdir}/memory_scores.json" \
        --output_path "${expdir}" \
        ${python_args}
}

metrics() {
    local hypothesis reference python_args
    local "${@}"

    python utils/metrics_utils.py --hyp_path "${hypothesis}" --ref_path "${reference}" \
        ${python_args}
}

iteration_0() {
    local hyps_path="${BASE_EXPDIR}/hyps_${NUM_BEAMS}.txt"
    generator model_input="${DATA_PATH}" \
        candidates_path="${hyps_path}" \
        python_args="--num_beams ${NUM_BEAMS} --num_return_sequences ${NUM_BEAMS} --num_beam_groups ${NUM_BEAMS}"

    selector model_input="${SRC_PATH}" \
        candidates_path="${hyps_path}" \
        expdir="${BASE_EXPDIR}" \
        python_args="--num_beams ${NUM_BEAMS} --B ${B}"
}

do_one_iteration() {
    local i beam expdir
    local "${@}"

    local prev_memory_candidates_dir=""
    if [ "$i" -gt 1 ]; then
        prev_memory_candidates_dir="${BASE_EXPDIR}/iter$((i-1))"
    else 
        prev_memory_candidates_dir="${BASE_EXPDIR}"
    fi

    local prev_memory_path="${prev_memory_candidates_dir}/top_${beam}_memories.txt"
    mkdir -p "${expdir}"
    mv "${prev_memory_path}" "${expdir}/memory.txt"

    local hyps_path="${expdir}/hyps_${NUM_BEAMS}_${beam}.txt"
    generator model_input="${DATA_PATH}" \
        memory="${expdir}/memory.txt" \
        candidates_path="${hyps_path}" \
        python_args="--num_beams ${NUM_BEAMS} --num_return_sequences ${NUM_BEAMS} --num_beam_groups ${NUM_BEAMS}"

    local mem_output_prefix="beam_${beam}-"
    selector model_input="${SRC_PATH}" \
        candidates_path="${hyps_path}" \
        expdir="${expdir}" \
        python_args="--num_beams ${NUM_BEAMS} --B ${B} --output_prefix ${mem_output_prefix}"

    for eval_beam in $(seq 1 $B); do
        local eval_path="${expdir}/eval_${eval_beam}.txt"
        generator model_input="${DATA_PATH}" \
            memory="${expdir}/${mem_output_prefix}top_${eval_beam}_memories.txt" \
            candidates_path="${eval_path}" \
            python_args="--num_beams 1 --num_return_sequences 1 --num_beam_groups 1"
        metrics hypothesis="${eval_path}" \
            reference="${TRG_PATH}" \
            python_args="--output_path ${expdir}/score_beam${beam}_eval${eval_beam}.txt"
    done
}

prune_beams() {
    local expdir
    local "${@}"

    # rename top-$B _memories.txt (via associated score_.txt) to top-$B_memories.txt
    score_files=$(ls ${expdir}/score*.txt)
    score_files_sorted=$(for file in $(echo "$score_files"); do echo "$(cat $file) $file"; done | sort -gr | cut -d' ' -f2-)
    # rename beam_${beam}-top_${eval_beam}_memories.txt to top_${i}_memories.txt
    for i in $(seq 1 $B); do
        score_file=$(echo "$score_files_sorted" | sed -n "${i}p")
        # score file is in format score_beam${beam}_eval${eval_beam}.txt
        beam=$(echo "$score_file" | sed -n 's/.*score_beam\([0-9]\)_eval[0-9].txt/\1/p')
        eval_beam=$(echo "$score_file" | sed -n 's/.*score_beam[0-9]_eval\([0-9]\).txt/\1/p')
        memory_file="${expdir}/beam_${beam}-top_${eval_beam}_memories.txt"
        mv "${memory_file}" "${expdir}/top_${i}_memories.txt"
    done
}

main() {
    iteration_0
    for k in $(seq 1 $K); do
        echo "Iteration $k"
        for beam in $(seq 1 $B); do
            do_one_iteration i="$k" beam="$beam" expdir="${BASE_EXPDIR}/iter${k}"
        done

        prune_beams expdir="${BASE_EXPDIR}/iter${k}"
    done
}

main
