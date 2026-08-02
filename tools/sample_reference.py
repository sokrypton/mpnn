#!/usr/bin/env python3
"""Golden sequences from the reference sampler, for the features the logit
parity tests do not reach: tied positions, per-amino-acid bias, fixed residues,
and switching the ligand context off.

Sampling is stochastic, so it is made deterministic two ways at once. The
decoding order is pinned by handing the reference a `randn` whose absolute
values are the ranks it should sort into, and the temperature is driven to
1e-6 so the multinomial draw collapses onto the argmax. Both implementations
then have to produce the same sequence, not merely a similar one.

    python tools/sample_reference.py --ref <pkg> --checkpoints <dir> \
        --inputs cases.json --out golden.json
"""

import argparse
import json
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from make_reference import build_model, load_reference_module, make_feature_dict  # noqa: E402


def randn_for_order(order, L):
    """`randn` such that `argsort((1 + 1e-4) * |randn|)` reproduces `order`."""
    randn = np.zeros(L, dtype=np.float32)
    for rank, pos in enumerate(order):
        randn[pos] = rank + 1.0
    return torch.tensor(randn)[None]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, type=pathlib.Path)
    ap.add_argument("--checkpoints", required=True, type=pathlib.Path)
    ap.add_argument("--inputs", required=True, type=pathlib.Path)
    ap.add_argument("--out", required=True, type=pathlib.Path)
    args = ap.parse_args()

    mu = load_reference_module(args.ref)
    torch.manual_seed(0)
    results = []

    for case in json.loads(args.inputs.read_text()):
        model_type = case["modelType"]
        model, ckpt = build_model(
            mu, args.checkpoints / f"{case['checkpoint']}.pt", model_type, "cpu")
        atom_context_num = ckpt.get("atom_context_num", 1)
        L = case["inputs"]["L"]

        for variant in case["variants"]:
            fd = make_feature_dict(case["inputs"], model_type, atom_context_num)

            if variant.get("useAtomContext") is False and model_type == "ligand_mpnn":
                fd["Y_m"] = 0.0 * fd["Y_m"]

            chain_mask = torch.tensor(variant["chainMask"], dtype=torch.float32)[None]
            fd["chain_mask"] = chain_mask
            fd["bias"] = torch.tensor(variant["bias"], dtype=torch.float32).reshape(1, L, 21)
            fd["randn"] = randn_for_order(variant["order"], L)
            fd["batch_size"] = 1
            fd["temperature"] = 1e-6

            symmetry = variant.get("symmetry") or [[]]
            weights = variant.get("symmetryWeights") or [[]]
            fd["symmetry_residues"] = symmetry
            fd["symmetry_weights"] = weights

            with torch.no_grad():
                out = model.sample(fd)

            results.append({
                "name": f"{case['name']}/{variant['name']}",
                "checkpoint": case["checkpoint"],
                "modelType": model_type,
                "S": out["S"][0].tolist(),
                "decodingOrder": out["decoding_order"][0].tolist(),
                "logits": out["log_probs"][0].numpy().tolist(),
            })
            print(f"{results[-1]['name']:<40} L={L}")

    args.out.write_text(json.dumps(results))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
