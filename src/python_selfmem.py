import argparse
import os
import shutil
import subprocess

import torch
import torch.distributed as dist

# Environment variables set by torch.distributed.launch
LOCAL_RANK = int(os.environ['LOCAL_RANK'])
WORLD_SIZE = int(os.environ['WORLD_SIZE'])
WORLD_RANK = int(os.environ['RANK'])

MASTER_RANK = 0


def gather_data(data_to_send, local_ranks):
    # Need to put tensor on a GPU device for nccl backend
    # if backend == 'nccl':
    #     device = torch.device("cuda:{}".format(LOCAL_RANK))
    #     # str_array = torch.tensor(str_array).to(device)
    str_array = []

    for local_rank in local_ranks:
        if WORLD_RANK == local_rank:
            str_array = data_to_send[local_rank]

    gathered_data = [None] * WORLD_SIZE
    dist.all_gather_object(gathered_data, str_array)

    if len(local_ranks) == 1:
        gathered_data = gathered_data[local_ranks[0]]
    return gathered_data


def generator(model_input, memory, candidates_path, python_args, ACCELERATOR="gpu"):
    args = [
        "python", "generate_hyps.py",
        "--config", "config/jrc_ende/generate_hyps.yaml",
        "--accelerator", ACCELERATOR,
        "--data_path", model_input,
        "--output_path", candidates_path
    ]
    if memory:
        args.extend(["--memory_path", memory])
    args.extend(python_args.split())
    subprocess.run(args)

def selector(model_input, candidates_path, expdir, python_args):
    subprocess.run([
        "python", "select_mem.py",
        "--queries_path", model_input,
        "--memory_path", candidates_path,
        "--memory_scores_path", os.path.join(expdir, "memory_scores.json"),
        "--output_path", expdir
    ] + python_args.split())

def metrics(hypothesis, reference, python_args):
    subprocess.run([
        "python", "utils/metrics_utils.py",
        "--hyp_path", hypothesis,
        "--ref_path", reference
    ] + python_args.split())

def iteration_0(BASE_EXPDIR, DATA_PATH, SRC_PATH, TRG_PATH, NUM_BEAMS, B):
    hyps_path = os.path.join(BASE_EXPDIR, f"hyps_{NUM_BEAMS}.txt")
    generator(DATA_PATH, None, hyps_path, f"--num_beams {NUM_BEAMS} --num_return_sequences {NUM_BEAMS} --num_beam_groups {NUM_BEAMS}")
    selector(SRC_PATH, hyps_path, BASE_EXPDIR, f"--num_beams {NUM_BEAMS} --B {B}")

def do_one_iteration(BASE_EXPDIR, DATA_PATH, SRC_PATH, TRG_PATH, NUM_BEAMS, B, K, MEMORY_UPDATE_SCHEME, i, beam, memory_prev) -> list[tuple[float, list[str]]]:
    expdir = os.path.join(BASE_EXPDIR, f"iter{i}")
    os.makedirs(expdir, exist_ok=True)

    memory_prev_path = os.path.join(expdir, "memory_prev.txt")
    with open(memory_prev_path, 'w') as f:
        f.writelines(memory_prev)
    
    memory_path = os.path.join(expdir, "memory.txt")
    if MEMORY_UPDATE_SCHEME == "replace":
        shutil.copy(memory_prev_path, memory_path)
    elif MEMORY_UPDATE_SCHEME == "augment":
        shutil.copy(memory_prev_path, os.path.join(expdir, "memory_prev.txt"))
        memory_path = os.path.join(expdir, "memory_prev.txt")

    hyps_path = os.path.join(expdir, f"hyps_{NUM_BEAMS}_{beam}.txt")
    generator(DATA_PATH, memory_path if MEMORY_UPDATE_SCHEME == "augment" else None, hyps_path, f"--num_beams {NUM_BEAMS} --num_return_sequences {NUM_BEAMS} --num_beam_groups {NUM_BEAMS}")
    mem_output_prefix = f"beam_{beam}-"
    selector(SRC_PATH, hyps_path, expdir, f"--num_beams {NUM_BEAMS} --B {B} --output_prefix {mem_output_prefix}")

    # return [(score, [memory]) for each eval_beam], size B
    results = []
    for eval_beam in range(1, B+1):
        if MEMORY_UPDATE_SCHEME == "augment":
            with open(os.path.join(expdir, f"memory_prev.txt"), 'r') as f_prev, open(os.path.join(expdir, f"{mem_output_prefix}top_{eval_beam}_memories.txt"), 'r') as f_curr, open(os.path.join(expdir, f"top_{eval_beam}_memories.txt"), 'w') as f_out:
                for line_prev, line_curr in zip(f_prev, f_curr):
                    f_out.write(line_prev.strip() + ' ' + line_curr)
        eval_path = os.path.join(expdir, f"eval_{eval_beam}.txt")
        generator(DATA_PATH, os.path.join(expdir, f"{mem_output_prefix}top_{eval_beam}_memories.txt"), eval_path, "--num_beams 1 --num_return_sequences 1 --num_beam_groups 1 --early_stopping false")
        metrics(eval_path, TRG_PATH, f"--output_path {os.path.join(expdir, f'score_beam{beam}_eval{eval_beam}.txt')}")

        with open(os.path.join(expdir, f'score_beam{beam}_eval{eval_beam}.txt'), 'r') as f:
            score = float(f.readline().strip())
        with open(os.path.join(expdir, f"top_{eval_beam}_memories.txt"), 'r') as f:
            memory = f.readlines()
        results.append((score, memory))
    
    return results


def prune_beams(B, score_memories_arrays) -> list[tuple[float, list[str]]:
    # return top B (score, memory) tuples
    score_memories_arrays.sort(key=lambda x: x[0], reverse=True)
    return score_memories_arrays[:B]


def prepare_memory_prev_array(BASE_EXPDIR, MEMORY_UPDATE_SCHEME, beam, i) -> list[str]:
    prev_memory_candidates_dir = os.path.join(BASE_EXPDIR, f"iter{i-1}") if i > 1 else BASE_EXPDIR
    prev_memory_path = os.path.join(prev_memory_candidates_dir, f"top_{beam}_memories.txt")
    # return array of memory candidates strings
    with open(prev_memory_path, 'r') as f:
        return f.readlines()


def main(args):
    ACCELERATOR = args.accelerator
    DATA_DIR = args.data_dir
    EXPNAME = args.expname
    BASE_EXPDIR = os.path.join(DATA_DIR, "memory", EXPNAME)
    if os.path.exists(BASE_EXPDIR):
        shutil.rmtree(BASE_EXPDIR)
    os.makedirs(BASE_EXPDIR)
    DATA_PATH = os.path.join(DATA_DIR, "test_small.jsonl")
    SRC_PATH = os.path.join(DATA_DIR, "test_small_src.txt")
    TRG_PATH = os.path.join(DATA_DIR, "test_small_trg.txt")
    NUM_BEAMS = args.num_beams
    B = args.B
    K = args.K
    MEMORY_UPDATE_SCHEME = args.memory_update_scheme

    iteration_0(BASE_EXPDIR, DATA_PATH, SRC_PATH, TRG_PATH, NUM_BEAMS, B)
    for k in range(1, K+1):
        print(f"Iteration {k}")
        
        score_memories_arrays: list[list[tuple[float, list[str]]]] = [None] * B
        for beam in range(1, B+1):
            # send memory_prev from node0 to all other remote nodes
            memory_prev_array = gather_data(prepare_memory_prev_array(BASE_EXPDIR, MEMORY_UPDATE_SCHEME, beam, k), [MASTER_RANK])

            # run on node rank beam
            if beam-1 == WORLD_RANK:
                score_memories_array = do_one_iteration(BASE_EXPDIR, DATA_PATH, SRC_PATH, TRG_PATH, NUM_BEAMS, B, K, MEMORY_UPDATE_SCHEME, k, beam, memory_prev_array)
                score_memories_arrays[beam-1] = score_memories_array
        
        score_memories_arrays = gather_data(score_memories_arrays, list(range(0, B)))
        # gather all score_memories_arrays to node0 for use
        if WORLD_RANK == MASTER_RANK:
            # flatten score_memories_arrays to list[tuple[float, list[str]]
            score_memories_arrays = [score_memories for score_memories_array in score_memories_arrays for score_memories in score_memories_array]
            top_B_score_memories = prune_beams(B, score_memories_arrays)

            # save to {expdir}/top_{beam}_memories.txt
            expdir = os.path.join(BASE_EXPDIR, f"iter{k}")
            for beam, (score, memories) in enumerate(top_B_score_memories, 1):
                with open(os.path.join(expdir, f"top_{beam}_memories.txt"), 'w') as f:
                    f.writelines(memories)
                with open(os.path.join(expdir, f"top_{beam}_score.txt"), 'w') as f:
                    f.write(str(score))


def init_processes(args):
    dist.init_process_group(backend=args.backend, rank=WORLD_RANK, world_size=WORLD_SIZE)
    main(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Efficient and Precise Self-Memory for RAG")

    # script arguments
    parser.add_argument("--accelerator", default="gpu", help="Accelerator type")
    parser.add_argument("--data_dir", default="../data/ende", help="Data directory")
    parser.add_argument("--expname", default="baseline", help="Experiment name")
    parser.add_argument("--num_beams", type=int, default=10, help="Number of beams")
    parser.add_argument("-B", type=int, default=2, help="Number of beams to keep")
    parser.add_argument("-K", type=int, default=3, help="Number of iterations")
    parser.add_argument("--memory_update_scheme", default="replace", help="Memory update scheme")

    # torch.distributed.launch arguments
    parser.add_argument("--local_rank", type=int, help="Local rank. Necessary for using the torch.distributed.launch utility.")
    parser.add_argument("--backend", type=str, default="nccl", choices=['nccl', 'gloo'])
    
    args = parser.parse_args()
    init_processes(args)

