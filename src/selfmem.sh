#!/bin/bash

# Pseudo code for SelfMemory process:

# generator = Trainer(model_input,memory)
# candidates = generator(model_input,memory)
HYPS_PATH="../data/ende/memory/hyps_50.txt"
python generate_hyps.py --config config/jrc_ende/generate_hyps.yaml --output_path "$HYPS_PATH"
# selector = Trainer(model_input,candidates)

# for _ in range(iteration_k):
for i in {1..3}; do
    # candidates = generator(model_input,memory)
    if [ $i -eq 1 ]; then
        python generate_hyps.py --config config/jrc_ende/generate_hyps.yaml --output_path "$HYPS_PATH"
    else
        python generate_hyps.py --config config/jrc_ende/generate_hyps.yaml --memory_path "$MEMORY_CANDIDATES_PATH" --output_path "$HYPS_PATH"
    fi
    # memories: [memory1, memory2, ..., memoryB] = selector(model_input,candidates) # choose top-B memory candidates
    QUERIES_PATH="../data/ende/test_small_src.txt"
    # MEMORY_PATH="../data/ende/memory/random_50.memory"
    MEMORY_CANDIDATES_PATH="../data/ende/memory/"
    python select_mem.py --queries_path "$QUERIES_PATH" --memory_path "$MEMORY_PATH" --B 2 --output_path "$MEMORY_CANDIDATES_PATH"
    # do next iteration on two branches
    #     hypothesis = generator(model_input,memory)
    #     current_score = metrics(hypothesis,reference)
done
