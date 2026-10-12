"""Serves the fine-tuned LoRA classifier for the Java router (HttpClassifier).

    python finetune/serve.py --model models/intent-lora --port 8001
    curl -s localhost:8001/classify -d '{"text":"I was charged twice"}' -H 'Content-Type: application/json'

Then start the router with SMALL_MODEL_URL=http://localhost:8001.
"""
import argparse
import json
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
from peft import AutoPeftModelForSequenceClassification
from transformers import AutoTokenizer


def build(model_dir):
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoPeftModelForSequenceClassification.from_pretrained(model_dir)
    model = model.merge_and_unload()  # fold LoRA weights in: no adapter overhead at inference
    model.eval()
    if torch.cuda.is_available():
        model.to("cuda")
    labels = json.load(open(os.path.join(model_dir, "labels.json")))
    return tok, model, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/intent-lora")
    ap.add_argument("--port", type=int, default=8001)
    args = ap.parse_args()
    tok, model, labels = build(args.model)

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            if self.path != "/classify":
                self.send_error(404)
                return
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            t0 = time.perf_counter()
            enc = tok(body["text"], return_tensors="pt", truncation=True, max_length=128).to(model.device)
            with torch.no_grad():
                probs = torch.softmax(model(**enc).logits[0].float(), dim=-1).tolist()
            best = max(range(len(probs)), key=probs.__getitem__)
            out = json.dumps({"label": labels[best], "probs": probs,
                              "latency_ms": round((time.perf_counter() - t0) * 1000, 2)}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

        def log_message(self, *a):
            pass

    print(f"Serving {args.model} on :{args.port}")
    ThreadingHTTPServer(("0.0.0.0", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
