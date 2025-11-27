import pathlib
import logging
import random
import numpy as np
from beir.datasets.data_loader import GenericDataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import torch

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logging.info(f"🔒 Random seed set to {seed} for reproducibility.")

set_seed(42)

BASE_MODEL_NAME = "BAAI/bge-base-en-v1.5"
NEW_MODEL_NAME = "./bge-finetuned-all-domains"

MTRAG_DOMAINS = ["clapnq", "fiqa", "govt", "cloud"]

TRAIN_BATCH_SIZE = 16
EPOCHS = 1
WARMUP_STEPS = 100

USE_QUERY_REWRITE = True

def rewrite_query(q):
    return f"Rewrite: {q}"

def load_mtrag_examples(domain, split="train", use_data_splits=True):
    logging.info(f"Loading MTRAG domain: {domain} (split: {split})...")
    data_root = pathlib.Path(".")

    if use_data_splits:
        corpus_file = data_root / "corpora" / "passage_level" / f"{domain}.jsonl"
        query_file = data_root / "data_splits" / "retrieval_tasks" / domain / split / f"{domain}_questions.jsonl"
        qrels_file = data_root / "data_splits" / "retrieval_tasks" / domain / split / "qrels" / "dev.tsv"
    else:
        corpus_file = data_root / "corpora" / "passage_level" / f"{domain}.jsonl"
        query_file = data_root / "human" / "retrieval_tasks" / domain / f"{domain}_questions.jsonl"
        qrels_file = data_root / "human" / "retrieval_tasks" / domain / "qrels" / "dev.tsv"

    if use_data_splits and not query_file.exists():
        logging.warning(f"[{domain}] Split not found: {query_file}")
        logging.warning(f"Falling back to original human/ dataset.")
        query_file = data_root / "human" / "retrieval_tasks" / domain / f"{domain}_questions.jsonl"
        qrels_file = data_root / "human" / "retrieval_tasks" / domain / "qrels" / "dev.tsv"

    try:
        corpus, queries, qrels = GenericDataLoader(
            corpus_file=str(corpus_file),
            query_file=str(query_file),
            qrels_file=str(qrels_file)
        ).load_custom()
    except Exception as e:
        logging.error(f"Error loading domain={domain}, split={split}: {e}")
        return []

    domain_examples = []
    for query_id, doc_infos in qrels.items():
        query_text = queries.get(query_id)
        if not query_text:
            continue

        if USE_QUERY_REWRITE:
            query_text = rewrite_query(query_text)

        for doc_id, score in doc_infos.items():
            if score > 0:
                doc = corpus.get(doc_id)
                if doc:
                    title = doc.get("title", "")
                    text = doc.get("text", "")
                    doc_text = f"{title} {text}".strip() if title else text
                    domain_examples.append(InputExample(texts=[query_text, doc_text]))

    logging.info(f"[{domain}] Loaded {len(domain_examples)} examples (split={split})")
    return domain_examples

def load_custom_examples():
    logging.info("Loading custom dataset...")
    my_examples = []
    q1 = "What are the side effects of lisinopril?"
    q2 = "How does photosynthesis work?"

    if USE_QUERY_REWRITE:
        q1 = rewrite_query(q1)
        q2 = rewrite_query(q2)

    my_examples.append(InputExample(texts=[q1,
        "Lisinopril, an ACE inhibitor, can cause a persistent dry cough, dizziness, and headache."
    ]))
    my_examples.append(InputExample(texts=[q2,
        "Photosynthesis is the process used by plants to convert light energy into chemical energy."
    ]))
    
    logging.info(f"Loaded {len(my_examples)} custom examples (from placeholder)")
    return my_examples

def run_training():
    logging.info("Starting training process...")
    
    if not torch.cuda.is_available():
        logging.error("CUDA (NVIDIA GPU) not available. Training will be extremely slow.")
        
    all_train_examples = []

    for domain in MTRAG_DOMAINS:
        examples = load_mtrag_examples(domain, split="train", use_data_splits=True)
        all_train_examples.extend(examples)

    all_train_examples.extend(load_custom_examples())

    logging.info(f"Total training examples: {len(all_train_examples)}")

    if not all_train_examples:
        logging.error("No training examples found. Exiting.")
        return

    logging.info(f"Loading base model: {BASE_MODEL_NAME}...")
    model = SentenceTransformer(BASE_MODEL_NAME)

    train_dataloader = DataLoader(all_train_examples, shuffle=True, batch_size=TRAIN_BATCH_SIZE)

    train_loss = losses.MultipleNegativesRankingLoss(model=model)

    logging.info(f"Starting model fine-tuning... (Epochs={EPOCHS}, Batch Size={TRAIN_BATCH_SIZE})")
    model.fit(train_objectives=[(train_dataloader, train_loss)],
              epochs=EPOCHS,
              warmup_steps=WARMUP_STEPS,
              output_path=NEW_MODEL_NAME,
              show_progress_bar=True,
              checkpoint_save_steps=1000,
              checkpoint_path=f"{NEW_MODEL_NAME}-checkpoints"
             )

    logging.info(f"Training complete. New model saved to: {NEW_MODEL_NAME}")

if __name__ == "__main__":
    run_training()
