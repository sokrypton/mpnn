#!/usr/bin/env python3
"""Convert LigandMPNN / ProteinMPNN PyTorch checkpoints into the flat binary
format the browser engine reads.

    python tools/convert_weights.py --src <dir of .pt files> --out weights/

Layout of a ``.mpnn`` file::

    0..3    magic  "MPNN"
    4..7    uint32 version (1)
    8..11   uint32 header length in bytes
    12..    utf-8 JSON header, zero padded so the payload starts at a multiple of 64
    ...     tensor payload, tensors concatenated in header order

Every tensor is stored C-contiguous. ``float16`` is the default because the
models are tiny and the halved download matters more than the last few bits of
precision; ``--dtype float32`` produces the bit-exact files the parity tests use.
"""

import argparse
import json
import pathlib
import struct
import sys

import numpy as np
import torch

MAGIC = b"MPNN"
VERSION = 1
ALIGN = 64

# Checkpoint stem -> model_type understood by the engine. The stem also encodes
# the training hyper-parameters, but those are read from the checkpoint itself.
MODEL_TYPES = {
    "proteinmpnn": "protein_mpnn",
    "solublempnn": "soluble_mpnn",
    "ligandmpnn": "ligand_mpnn",
    "global_label_membrane_mpnn": "global_label_membrane_mpnn",
    "per_residue_label_membrane_mpnn": "per_residue_label_membrane_mpnn",
}


def model_type_for(stem: str) -> str:
    for prefix, mtype in sorted(MODEL_TYPES.items(), key=lambda kv: -len(kv[0])):
        if stem.startswith(prefix):
            return mtype
    raise ValueError(f"cannot infer model_type from checkpoint name {stem!r}")


def convert(path: pathlib.Path, out_dir: pathlib.Path, dtype: str) -> dict:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt["model_state_dict"]
    stem = path.stem
    model_type = model_type_for(stem)

    np_dtype = np.float16 if dtype == "float16" else np.float32

    tensors, blobs, offset = {}, [], 0
    for name, tensor in state.items():
        arr = np.ascontiguousarray(tensor.detach().cpu().numpy().astype(np_dtype))
        tensors[name] = {"shape": list(arr.shape), "offset": offset, "n": int(arr.size)}
        offset += arr.nbytes
        blobs.append(arr)

    header = {
        "name": stem,
        "model_type": model_type,
        "dtype": dtype,
        "k_neighbors": int(ckpt["num_edges"]),
        "noise_level": float(ckpt.get("noise_level", 0.0)),
        # atom_context_num is only meaningful for ligand_mpnn; the sidechain
        # packing checkpoints carry it too but we do not run those here.
        "atom_context_num": int(ckpt.get("atom_context_num", 0)),
        "hidden_dim": 128,
        "num_letters": 21,
        "num_encoder_layers": 3,
        "num_decoder_layers": 3,
        "num_rbf": 16,
        "num_positional_embeddings": 16,
        "tensors": tensors,
    }
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    prefix_len = 12 + len(header_bytes)
    pad = (-prefix_len) % ALIGN
    header_bytes += b"\0" * pad

    out_path = out_dir / f"{stem}.mpnn"
    with open(out_path, "wb") as fh:
        fh.write(MAGIC)
        fh.write(struct.pack("<II", VERSION, len(header_bytes)))
        fh.write(header_bytes)
        for arr in blobs:
            fh.write(arr.tobytes())

    return {
        "file": out_path.name,
        "name": stem,
        "model_type": model_type,
        "dtype": dtype,
        "k_neighbors": header["k_neighbors"],
        "noise_level": header["noise_level"],
        "atom_context_num": header["atom_context_num"],
        "params": int(sum(t["n"] for t in tensors.values())),
        "bytes": out_path.stat().st_size,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=pathlib.Path,
                    help="directory containing the .pt checkpoints")
    ap.add_argument("--out", required=True, type=pathlib.Path)
    ap.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    ap.add_argument("--skip", nargs="*", default=["ligandmpnn_sc"],
                    help="name prefixes to skip (default: the sidechain packer)")
    ap.add_argument("--manifest", action="store_true",
                    help="also write out/models.json describing every converted file")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    entries = []
    for path in sorted(args.src.glob("*.pt")):
        if any(path.stem.startswith(s) for s in args.skip):
            print(f"skip {path.name}")
            continue
        entry = convert(path, args.out, args.dtype)
        entries.append(entry)
        print(f"{entry['file']:<48} {entry['model_type']:<32} "
              f"{entry['params']:>9,} params  {entry['bytes'] / 1e6:6.2f} MB")

    if args.manifest:
        manifest = args.out / "models.json"
        manifest.write_text(json.dumps(entries, indent=2) + "\n")
        print(f"wrote {manifest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
