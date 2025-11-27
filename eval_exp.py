import pathlib
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.models import SentenceBERT
from beir.retrieval.search.dense import DenseRetrievalExactSearch 
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

USE_DATA_SPLITS = True
EVAL_SPLIT = "val"

USE_QUERY_REWRITE = False

def rewrite_query(q):
    return f"Rewrite: {q}"

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using NVIDIA GPU (cuda) for acceleration.")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple MPS (M-series GPU) for acceleration.")
else:
    device = torch.device("cpu")
    print("No GPU found. Using CPU (this will be very slow).")

MODEL_NAME = "./bge-finetuned-all-domains" 
print(f"Loading model: {MODEL_NAME} (this is your fine-tuned model)...")

model = SentenceBERT(MODEL_NAME, device=device.type)

print("Loading query expansion model...")
EXPANSION_MODEL = "Qwen/Qwen2.5-3B-Instruct"

try:
    expansion_tokenizer = AutoTokenizer.from_pretrained(EXPANSION_MODEL, trust_remote_code=True)
    expansion_model = AutoModelForCausalLM.from_pretrained(
        EXPANSION_MODEL, 
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    expansion_model.eval()
    USE_EXPANSION = True
    print(f"Query expansion model loaded: {EXPANSION_MODEL}")
except Exception as e:
    print(f"Could not load expansion model: {e}")
    print("Proceeding without query expansion...")
    USE_EXPANSION = False

def extract_last_question(query):
    if "|user|:" in query:
        parts = query.split("|user|:")
        for part in reversed(parts):
            part = part.strip()
            if part:
                return part
    return query

def expand_query_minimal(query):
    if not USE_EXPANSION:
        return query
    clean_query = extract_last_question(query)
    messages = [
        {"role": "system", "content": "Extract 3-5 key technical terms or concepts from the question. Return ONLY the keywords, comma-separated."},
        {"role": "user", "content": f"Question: {clean_query}\n\nKeywords:"}
    ]
    text = expansion_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = expansion_tokenizer([text], return_tensors="pt").to(expansion_model.device)
    with torch.no_grad():
        outputs = expansion_model.generate(
            **inputs,
            max_new_tokens=30,
            temperature=0.3,
            do_sample=True,
            pad_token_id=expansion_tokenizer.eos_token_id,
            eos_token_id=expansion_tokenizer.eos_token_id
        )
    generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
    keywords = expansion_tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    keywords = keywords.replace("Keywords:", "").replace("keywords:", "").strip()
    keywords = keywords.split('\n')[0]
    if len(keywords) > 50 or len(keywords) < 3:
        return clean_query
    expanded = f"{clean_query} {keywords}"
    return expanded

retriever = DenseRetrievalExactSearch(model, batch_size=128)
k_values = [5, 10]
evaluator = EvaluateRetrieval(retriever, k_values=k_values)

domains = ["clapnq", "fiqa", "govt", "cloud"]
all_results = {}

print("\n--- Running FINE-TUNED BGE with MINIMAL Query Expansion ---")

def load_split(domain, split="dev"):
    data_root = pathlib.Path(".")
    corpus_file = data_root / "corpora" / "passage_level" / f"{domain}.jsonl"
    if USE_DATA_SPLITS:
        query_file = data_root / "data_splits" / "retrieval_tasks" / domain / split / f"{domain}_questions.jsonl"
        qrels_file = data_root / "data_splits" / "retrieval_tasks" / domain / split / "qrels" / "dev.tsv"
    else:
        query_file = data_root / "human" / "retrieval_tasks" / domain / f"{domain}_questions.jsonl"
        qrels_file = data_root / "human" / "retrieval_tasks" / domain / "qrels" / "dev.tsv"
    if USE_DATA_SPLITS and not query_file.exists():
        print(f"Split not found for {domain}, falling back to human/")
        query_file = data_root / "human" / "retrieval_tasks" / domain / f"{domain}_questions.jsonl"
        qrels_file = data_root / "human" / "retrieval_tasks" / domain / "qrels" / "dev.tsv"
    corpus, queries, qrels = GenericDataLoader(
        corpus_file=str(corpus_file),
        query_file=str(query_file),
        qrels_file=str(qrels_file)
    ).load_custom()
    return corpus, queries, qrels

for domain in domains:
    print(f"\n--- Processing Domain: {domain} ---")
    corpus, queries, qrels = load_split(domain, EVAL_SPLIT)

    print(f"Corpus: {len(corpus)} docs | Queries: {len(queries)} queries")

    if USE_QUERY_REWRITE:
        rewritten = {}
        for qid, qtext in queries.items():
            rewritten[qid] = rewrite_query(qtext)
        queries = rewritten

    if USE_EXPANSION:
        expanded_queries = {}
        for qid, qtext in list(queries.items())[:5]:
            expanded = expand_query_minimal(qtext)
            expanded_queries[qid] = expanded
            print(f"Original: {extract_last_question(qtext)}")
            print(f"With keywords: {expanded}\n")
        for qid, qtext in list(queries.items())[5:]:
            expanded_queries[qid] = expand_query_minimal(qtext)
        queries = expanded_queries

    print(f"Running retrieval for {domain}...")
    results = evaluator.retrieve(corpus, queries)

    print("Evaluating results...")
    ndcg, _map, recall, precision = evaluator.evaluate(qrels, results, k_values)
    
    all_results[domain] = {
        "Recall@5": recall['Recall@5'],
        "Recall@10": recall['Recall@10'],
        "nDCG@5": ndcg['NDCG@5'],
        "nDCG@10": ndcg['NDCG@10']
    }
    
    print(f"--- Results for {domain} ---")
    print(f"Recall@10: {recall['Recall@10']:.4f}")
    print(f"nDCG@10:   {ndcg['NDCG@10']:.4f}")

print("\n\n--- Finished All Domains: Final Summary ---")

avg_recall_5 = np.mean([res["Recall@5"] for res in all_results.values()])
avg_recall_10 = np.mean([res["Recall@10"] for res in all_results.values()])
avg_ndcg_5 = np.mean([res["nDCG@5"] for res in all_results.values()])
avg_ndcg_10 = np.mean([res["nDCG@10"] for res in all_results.values()])

paper_recall_5 = 0.30
paper_recall_10 = 0.38
paper_ndcg_5 = 0.27
paper_ndcg_10 = 0.30

print("--- Your Fine-Tuned BGE Results ---")
print(f"{'Metric':<12} | {'Your Avg.':<10} | {'Pre-Trained BGE (Paper)':<25}")
print("-" * 55)
print(f"{'Recall@5':<12} | {avg_recall_5:<10.4f} | {paper_recall_5:<25.4f}")
print(f"{'Recall@10':<12} | {avg_recall_10:<10.4f} | {paper_recall_10:<25.4f}")
print(f"{'nDCG@5':<12} | {avg_ndcg_5:<10.4f} | {paper_ndcg_5:<25.4f}")
print(f"{'nDCG@10':<12} | {avg_ndcg_10:<10.4f} | {paper_ndcg_10:<25.4f}")

print("\n--- Individual Domain Scores ---")
print(f"{'Domain':<10} | {'R@10':<7} | {'nDCG@10':<7}")
print("-" * 28)
for domain, scores in all_results.items():
    print(f"{domain:<10} | {scores['Recall@10']:<7.4f} | {scores['nDCG@10']:<7.4f}")

print("\nEvaluation complete.")
