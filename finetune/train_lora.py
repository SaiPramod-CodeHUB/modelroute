"""LoRA / QLoRA fine-tuning of a small open-weight model for ticket intent classification.

Uses a sequence-classification head on top of a small decoder model, so the fine-tuned model
returns a probability per intent (what the Java router needs for confidence-based routing).

    python finetune/train_lora.py --data data --model Qwen/Qwen2.5-0.5B --out models/intent-lora
    python finetune/train_lora.py --data data --model Qwen/Qwen2.5-1.5B --qlora   # 4-bit base weights

Needs a GPU for reasonable speed (a single 16-24 GB card is plenty for 0.5B-1.5B models).
Not run in CI.
"""
import argparse
import json
import os

import numpy as np
import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from sklearn.metrics import accuracy_score, f1_score
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig,
                          DataCollatorWithPadding, Trainer, TrainingArguments)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--out", default="models/intent-lora")
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--qlora", action="store_true", help="load the base model in 4-bit (QLoRA)")
    args = ap.parse_args()

    labels = json.load(open(os.path.join(args.data, "labels.json")))
    label2id = {l: i for i, l in enumerate(labels)}

    ds = load_dataset("json", data_files={
        "train": os.path.join(args.data, "train.jsonl"),
        "validation": os.path.join(args.data, "validation.jsonl"),
        "test": os.path.join(args.data, "test.jsonl"),
        "shifted": os.path.join(args.data, "shifted_test.jsonl"),
    })

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def encode(batch):
        enc = tok(batch["text"], truncation=True, max_length=args.max_len)
        enc["labels"] = [label2id[l] for l in batch["label"]]
        return enc

    ds = ds.map(encode, batched=True, remove_columns=["text", "label"])

    quant = None
    if args.qlora:
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                   bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=len(labels), id2label=dict(enumerate(labels)), label2id=label2id,
        quantization_config=quant, torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32)
    model.config.pad_token_id = tok.pad_token_id
    if args.qlora:
        model = prepare_model_for_kbit_training(model)

    lora = LoraConfig(task_type=TaskType.SEQ_CLS, r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05,
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    def metrics(eval_pred):
        logits, y = eval_pred
        pred = np.argmax(logits, axis=-1)
        return {"accuracy": accuracy_score(y, pred), "macro_f1": f1_score(y, pred, average="macro")}

    trainer = Trainer(
        model=model,
        args=TrainingArguments(output_dir=args.out, num_train_epochs=args.epochs, learning_rate=args.lr,
                               per_device_train_batch_size=args.batch, per_device_eval_batch_size=64,
                               eval_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True,
                               metric_for_best_model="macro_f1", logging_steps=50, report_to=[],
                               bf16=torch.cuda.is_available()),
        train_dataset=ds["train"], eval_dataset=ds["validation"],
        data_collator=DataCollatorWithPadding(tok), compute_metrics=metrics)
    trainer.train()

    results = {split: trainer.evaluate(ds[split], metric_key_prefix=split) for split in ("test", "shifted")}
    print(json.dumps(results, indent=2))
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    json.dump(labels, open(os.path.join(args.out, "labels.json"), "w"))
    json.dump(results, open(os.path.join(args.out, "eval.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
