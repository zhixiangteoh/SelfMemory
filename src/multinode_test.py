import os
import argparse
from time import sleep

import torch
import torch.distributed as dist

# Environment variables set by torch.distributed.launch
LOCAL_RANK = int(os.environ['LOCAL_RANK'])
WORLD_SIZE = int(os.environ['WORLD_SIZE'])
WORLD_RANK = int(os.environ['RANK'])


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
    

def init_processes(backend):
    dist.init_process_group(backend, rank=WORLD_RANK, world_size=WORLD_SIZE)

    # example both nodes "compute", gather both
    data_to_send = [['Hello', 'World', 'from', 'worker_{}'.format(WORLD_RANK)] for _ in range(WORLD_SIZE)]
    gathered_data = gather_data(data_to_send, [0, 1])
    print('worker_{} has received data {} from all ranks\n'.format(WORLD_RANK, gathered_data))

    # example node0 sends data to node1
    data_to_send = ["0.125"]
    gathered_data = gather_data(data_to_send, [0])
    if WORLD_RANK == 1:
        print('worker_{} has received data {} from rank 0\n'.format(WORLD_RANK, gathered_data))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, help="Local rank. Necessary for using the torch.distributed.launch utility.")
    parser.add_argument("--backend", type=str, default="nccl", choices=['nccl', 'gloo'])
    args = parser.parse_args()

    init_processes(backend=args.backend)
