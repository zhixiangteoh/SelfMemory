import argparse
import os
import shutil
import subprocess

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

def do_one_iteration(BASE_EXPDIR, DATA_PATH, SRC_PATH, TRG_PATH, NUM_BEAMS, B, K, MEMORY_UPDATE_SCHEME, i, beam, memory_prev):
    expdir = os.path.join(BASE_EXPDIR, f"iter{i}")
    os.makedirs(expdir, exist_ok=True)
    hyps_path = os.path.join(expdir, f"hyps_{NUM_BEAMS}_{beam}.txt")
    generator(DATA_PATH, memory_path if MEMORY_UPDATE_SCHEME == "augment" else None, hyps_path, f"--num_beams {NUM_BEAMS} --num_return_sequences {NUM_BEAMS} --num_beam_groups {NUM_BEAMS}")
    mem_output_prefix = f"beam_{beam}-"
    selector(SRC_PATH, hyps_path, expdir, f"--num_beams {NUM_BEAMS} --B {B} --output_prefix {mem_output_prefix}")
    for eval_beam in range(1, B+1):
        if MEMORY_UPDATE_SCHEME == "augment":
            with open(os.path.join(expdir, f"memory_prev.txt"), 'r') as f_prev, open(os.path.join(expdir, f"{mem_output_prefix}top_{eval_beam}_memories.txt"), 'r') as f_curr, open(os.path.join(expdir, f"top_{eval_beam}_memories.txt"), 'w') as f_out:
                for line_prev, line_curr in zip(f_prev, f_curr):
                    f_out.write(line_prev.strip() + ' ' + line_curr)
        eval_path = os.path.join(expdir, f"eval_{eval_beam}.txt")
        generator(DATA_PATH, os.path.join(expdir, f"{mem_output_prefix}top_{eval_beam}_memories.txt"), eval_path, "--num_beams 1 --num_return_sequences 1 --num_beam_groups 1 --early_stopping false")
        metrics(eval_path, TRG_PATH, f"--output_path {os.path.join(expdir, f'score_beam{beam}_eval{eval_beam}.txt')}")

def prune_beams(BASE_EXPDIR, B, K):
    for k in range(1, K+1):
        expdir = os.path.join(BASE_EXPDIR, f"iter{k}")
        score_files = sorted([file for file in os.listdir(expdir) if file.startswith("score")])
        for i in range(1, B+1):
            score_file = os.path.join(expdir, score_files[i-1])
            shutil.move(score_file, os.path.join(expdir, f"top_{i}_score.txt"))
            beam = int(score_file.split("/")[-1].split(".")[0].split("_")[1][-1])
            eval_beam = int(score_file.split("/")[-1].split(".")[0].split("_")[2][-1])
            memory_file = os.path.join(expdir, f"beam_{beam}-top_{eval_beam}_memories.txt")
            shutil.move(memory_file, os.path.join(expdir, f"top_{i}_memories.txt"))


def prepare_memory_prev_array(BASE_EXPDIR, beam, MEMORY_UPDATE_SCHEME, i):
    prev_memory_candidates_dir = os.path.join(BASE_EXPDIR, f"iter{i-1}") if i > 1 else BASE_EXPDIR
    prev_memory_path = os.path.join(prev_memory_candidates_dir, f"top_{beam}_memories.txt")
    
    # memory_path = os.path.join(expdir, "memory.txt")
    # if MEMORY_UPDATE_SCHEME == "replace":
    #     shutil.copy(prev_memory_path, memory_path)
    # elif MEMORY_UPDATE_SCHEME == "augment":
    #     shutil.copy(prev_memory_path, os.path.join(expdir, "memory_prev.txt"))
    #     memory_path = os.path.join(expdir, "memory_prev.txt")

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
        for beam in range(1, B+1):
            do_one_iteration(BASE_EXPDIR, DATA_PATH, SRC_PATH, TRG_PATH, NUM_BEAMS, B, K, MEMORY_UPDATE_SCHEME, k, beam)
        prune_beams(BASE_EXPDIR, B, K)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Your script description here")
    parser.add_argument("--accelerator", default="gpu", help="Accelerator type")
    parser.add_argument("--data_dir", default="../data/ende", help="Data directory")
    parser.add_argument("--expname", default="baseline", help="Experiment name")
    parser.add_argument("--num_beams", type=int, default=10, help="Number of beams")
    parser.add_argument("-B", type=int, default=2, help="Number of beams to keep")
    parser.add_argument("-K", type=int, default=3, help="Number of iterations")
    parser.add_argument("--memory_update_scheme", default="replace", help="Memory update scheme")
    args = parser.parse_args()
    main(args)

