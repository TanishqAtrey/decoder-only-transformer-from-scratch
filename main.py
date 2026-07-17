"""
main.py
KANHA entry point.

Modes:
    python main.py chat    --model models/finetuned/sft_final.pt
    python main.py train   --data  data/processed/train.npy
    python main.py finetune --base models/base/final_model.pt --data data/processed/sft.jsonl
    python main.py index   --docs  data/raw/
    python main.py api     --model models/finetuned/sft_final.pt
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        prog="kanha",
        description="KANHA AI — Knowledge-Augmented Neural Heuristic Assistant",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── chat ──────────────────────────────────────────────────────────────
    chat_p = sub.add_parser("chat", help="Start terminal chat")
    chat_p.add_argument("--model",  required=True)
    chat_p.add_argument("--index",  default=None)
    chat_p.add_argument("--stream", action="store_true")
    chat_p.add_argument("--tools",  action="store_true")

    # ── train ─────────────────────────────────────────────────────────────
    train_p = sub.add_parser("train", help="Pre-train from scratch")
    train_p.add_argument("--data",   required=True)
    train_p.add_argument("--resume", default=None)

    # ── finetune ──────────────────────────────────────────────────────────
    ft_p = sub.add_parser("finetune", help="SFT fine-tuning")
    ft_p.add_argument("--base_model", required=True)  # FIX: was --base, but sft_train() reads args.base_model
    ft_p.add_argument("--data",   required=True)
    ft_p.add_argument("--output", default="models/finetuned/")
    ft_p.add_argument("--epochs", type=int, default=3)
    ft_p.add_argument("--batch_size", type=int, default=1)
    ft_p.add_argument("--lr",         type=float, default=3e-5)

    # ── index ─────────────────────────────────────────────────────────────
    idx_p = sub.add_parser("index", help="Build FAISS index from documents")
    idx_p.add_argument("--docs",   required=True, help="Directory of .txt files")
    idx_p.add_argument("--output", default="data/embeddings/")

    # ── dpo ────────────────────────────────────────────────────────────────
    dpo_p = sub.add_parser("dpo", help="DPO alignment training")
    dpo_p.add_argument("--base_model", required=True)
    dpo_p.add_argument("--data",       required=True)
    dpo_p.add_argument("--output",     default="models/finetuned/")
    dpo_p.add_argument("--epochs",     type=int, default=1)
    dpo_p.add_argument("--batch_size", type=int, default=1)
    dpo_p.add_argument("--lr",         type=float, default=1e-6)
    dpo_p.add_argument("--beta",       type=float, default=0.1)

    # ── api ───────────────────────────────────────────────────────────────
    api_p = sub.add_parser("api", help="Start FastAPI server")
    api_p.add_argument("--model", required=True)
    api_p.add_argument("--host",  default="0.0.0.0")
    api_p.add_argument("--port",  type=int, default=8000)

    args = parser.parse_args()

    # ── Dispatch ──────────────────────────────────────────────────────────
    if args.command == "chat":
        from cli import run_cli
        run_cli(args)

    elif args.command == "train":
        from kanha.training.train import train
        train(args)

    elif args.command == "finetune":
        from kanha.finetune.sft_train import sft_train
        sft_train(args)

    elif args.command == "dpo":
        from kanha.finetune.dpo_train import dpo_train
        dpo_train(args)

    elif args.command == "index":
        _build_index(args)

    elif args.command == "api":
        _start_api(args)


def _build_index(args):
    """Builds a FAISS index from a directory of .txt files."""
    import os
    from kanha.rag.chunker import Chunker
    from kanha.rag.retriever import Retriever
    from kanha.utils.logging import get_logger

    log = get_logger("index")

    docs = []
    for fname in os.listdir(args.docs):
        if fname.endswith(".txt"):
            fpath = os.path.join(args.docs, fname)
            with open(fpath, "r", encoding="utf-8") as f:
                docs.append({"id": fname, "text": f.read()})

    log.info(f"Found {len(docs)} documents in {args.docs}")

    chunker   = Chunker()
    chunks    = chunker.chunk_documents(docs)
    retriever = Retriever()
    retriever.build_index_from_chunks(chunks, save_dir=args.output)
    log.info(f"Index saved to {args.output}")


def _start_api(args):
    """Starts FastAPI server."""
    try:
        import uvicorn
        from api import create_app
        app = create_app(model_path=args.model)
        uvicorn.run(app, host=args.host, port=args.port)
    except ImportError:
        print("FastAPI/uvicorn not installed. Run: pip install fastapi uvicorn")


if __name__ == "__main__":
    main()