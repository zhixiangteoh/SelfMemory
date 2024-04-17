from FlagEmbedding import BGEM3FlagModel
import argparse
import json
import torch


def compute_scores(queries, memory_documents, num_beams):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=False, device=device)  # pre-trained model

    # for each query, pair with next `num_beams` memory documents
    pairs = []
    for i, query in enumerate(queries):
        for _, memory_document in enumerate(memory_documents[i * num_beams:(i + 1) * num_beams]):
            pairs.append((query, memory_document))
    results = model.compute_score(pairs, max_passage_length=128, weights_for_different_modes=[
        0.4, 0.2, 0.4], batch_size=16)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--queries_path', type=str,
                        default='../data/ende/test_small_src.txt')
    parser.add_argument('--memory_path', type=str,
                        default='../data/ende/memory/random_50.memory')
    parser.add_argument('--num_beams', type=int, default=50)
    parser.add_argument('--memory_scores_path', type=str, default=None)
    parser.add_argument('--B', type=int, default=2)
    parser.add_argument('--output_path', type=str, default='../data/ende/memory/')
    parser.add_argument('--output_prefix', type=str, default='')
    args = parser.parse_args()

    # read `args.queries_path` (e.g., <SELFMEM_ROOT>/data/ende/test_small_src.txt) as queries
    queries = []
    with open(args.queries_path, 'r') as f:
        for line in f:
            queries.append(line.strip())
    # read `args.memory_path` (e.g., <SELFMEM_ROOT>/data/ende/memory/random.memory) as memory documents
    memory_documents = []
    with open(args.memory_path, 'r') as f:
        for line in f:
            memory_documents.append(line.strip())

    try:
        with open(args.memory_scores_path, 'r') as f:
            memory_scores = json.load(f)
    except FileNotFoundError:
        with open(args.memory_scores_path, 'w') as f:
            memory_scores = compute_scores(queries, memory_documents, args.num_beams)
            json.dump(memory_scores, f)

    # for each metric in memory_scores,
    # for each query in queries,
    # print the top-B memory documents with the highest scores for the query
    for i, query in enumerate(queries):
        # print(f'Query {i}: {query}')
        # NOTE: ignoring all other metrics except colbert for now
        # for metric in memory_scores:
        # print(f'{metric}:')
        metric = 'colbert'
        scores = memory_scores[metric]
        # sort next |memory_documents| scores in descending order
        # and print the top-B memory documents with the highest scores
        scores_for_query = scores[i * args.num_beams:(i + 1) * args.num_beams]
        sorted_scores = sorted(enumerate(scores_for_query), key=lambda x: x[1], reverse=True)
        for j in range(args.B):
            idx, score = sorted_scores[j]
            # print(f'{j + 1}. {memory_documents[idx]} ({score})')
            with open(f'{args.output_path}/{args.output_prefix}top_{j+1}_memories.txt', 'a') as f:
                f.write(f'{memory_documents[idx]}\n')
        # print()


if __name__ == '__main__':
    main()
