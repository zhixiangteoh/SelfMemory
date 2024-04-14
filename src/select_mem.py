from FlagEmbedding import BGEM3FlagModel
import argparse

model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)  # pre-trained model

parser = argparse.ArgumentParser()
parser.add_argument('--queries_path', type=str,
                    default='data/ende/test_small_src.txt')
parser.add_argument('--memory_path', type=str,
                    default='data/ende/memory/random.memory')
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

pairs = [(query, memory) for query in queries for memory in memory_documents]
results = model.compute_score(pairs, max_passage_length=128,
                              weights_for_different_modes=[0.4, 0.2, 0.4], batch_size=16)
print(results)
