#!/usr/bin/env python3
"""Time the reference PyTorch implementation on CPU, for comparison.

Takes the same JSON that `test/dump_inputs.mjs` writes, so both sides see
identical inputs and the comparison is of implementations rather than of two
different PDB readers.

    python tools/bench_reference.py --ref <pkg> --checkpoints <dir> \
        --inputs inputs.json [--threads 1]
"""

import argparse
import json
import pathlib
import sys
import time

import torch

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from make_reference import build_model, load_reference_module, make_feature_dict  # noqa: E402


def timed(fn, repeats=1):
    # One untimed call first: the first pass through a PyTorch graph pays for
    # allocator warm-up and kernel selection.
    fn()
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn()
    return (time.perf_counter() - t0) / repeats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, type=pathlib.Path)
    ap.add_argument("--checkpoints", required=True, type=pathlib.Path)
    ap.add_argument("--inputs", required=True, type=pathlib.Path)
    ap.add_argument("--threads", type=int, default=0,
                    help="0 leaves PyTorch's default (all cores)")
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    print(f"torch {torch.__version__}, {torch.get_num_threads()} thread(s)")

    mu = load_reference_module(args.ref)
    torch.manual_seed(0)

    for case in json.loads(args.inputs.read_text()):
        model_type = case["modelType"]
        model, ckpt = build_model(
            mu, args.checkpoints / f"{case['checkpoint']}.pt", model_type, "cpu")
        fd = make_feature_dict(case["inputs"], model_type, ckpt.get("atom_context_num", 1))
        L = case["inputs"]["L"]

        with torch.no_grad():
            encode = timed(lambda: model.encode(fd))

            # run.py shapes randn as [batch_size, L]; the decoding order is
            # derived from it per sample, so it has to match the decode batch.
            fd["batch_size"] = 1
            fd["randn"] = torch.randn(1, L)
            sample1 = timed(lambda: model.sample(fd))

            fd["batch_size"] = args.batch
            fd["randn"] = torch.randn(args.batch, L)
            sampleB = timed(lambda: model.sample(fd))

            fd["batch_size"] = 1
            fd["randn"] = torch.randn(1, L)
            score = timed(lambda: model.score(fd, use_sequence=True))

        print(f"\n{case['name']}  {model_type}  L={L}")
        print(f"    encode          {encode:.2f} s")
        print(f"    sample batch 1  {sample1:.2f} s")
        print(f"    sample batch {args.batch}  {sampleB:.2f} s "
              f"({sampleB / args.batch:.2f} s/seq)")
        print(f"    score           {score:.2f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
