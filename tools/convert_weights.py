#!/usr/bin/env python3
"""Convert LigandMPNN / ProteinMPNN / NA-MPNN PyTorch checkpoints into the flat
binary format the browser engine reads.

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
    # NA-MPNN's own checkpoints are named after their training step, so the
    # caller renames them on the way in (see --name).
    "na_mpnn": "na_mpnn",
}


def model_type_for(stem: str) -> str:
    for prefix, mtype in sorted(MODEL_TYPES.items(), key=lambda kv: -len(kv[0])):
        if stem.startswith(prefix):
            return mtype
    raise ValueError(f"cannot infer model_type from checkpoint name {stem!r}")


def convert(path: pathlib.Path, out_dir: pathlib.Path, dtype: str,
            stem: str | None = None) -> dict:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt["model_state_dict"]
    stem = stem or path.stem
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
        # NA-MPNN's checkpoints carry no hyper-parameters at all; its run.py
        # hardcodes k = 32, and the letter count is read off W_out.
        "k_neighbors": int(ckpt["num_edges"]) if "num_edges" in ckpt else 32,
        "noise_level": float(ckpt.get("noise_level", 0.0)),
        # atom_context_num is only meaningful for ligand_mpnn; the sidechain
        # packing checkpoints carry it too but we do not run those here.
        "atom_context_num": int(ckpt.get("atom_context_num", 0)),
        "hidden_dim": 128,
        "num_letters": int(state["W_out.bias"].shape[0]),
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


def describe(path: pathlib.Path) -> dict:
    """Read a converted file's own header back, so the manifest does not depend
    on which invocation produced which file -- NA-MPNN ships separately from the
    LigandMPNN checkpoints and the two are converted in different passes."""
    with open(path, "rb") as fh:
        assert fh.read(4) == MAGIC, path
        _version, header_len = struct.unpack("<II", fh.read(8))
        header = json.loads(fh.read(header_len).rstrip(b"\0").decode("utf-8"))
    return {
        "file": path.name,
        "name": header["name"],
        "model_type": header["model_type"],
        "dtype": header["dtype"],
        "k_neighbors": header["k_neighbors"],
        "noise_level": header["noise_level"],
        "atom_context_num": header["atom_context_num"],
        "num_letters": header["num_letters"],
        "params": int(sum(t["n"] for t in header["tensors"].values())),
        "bytes": path.stat().st_size,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=pathlib.Path,
                    help="directory containing the .pt checkpoints")
    ap.add_argument("--out", required=True, type=pathlib.Path)
    ap.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    ap.add_argument("--skip", nargs="*", default=["ligandmpnn_sc"],
                    help="name prefixes to skip (default: the sidechain packer)")
    ap.add_argument("--name", default=None,
                    help="output stem, when the checkpoint filename does not "
                         "encode the model type (NA-MPNN names its by step)")
    ap.add_argument("--manifest", action="store_true",
                    help="also write out/models.json describing every converted file")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    sources = sorted(args.src.glob("*.pt")) if args.src.is_dir() else [args.src]
    for path in sources:
        if any(path.stem.startswith(s) for s in args.skip):
            print(f"skip {path.name}")
            continue
        entry = convert(path, args.out, args.dtype, args.name)
        print(f"{entry['file']:<48} {entry['model_type']:<32} "
              f"{entry['params']:>9,} params  {entry['bytes'] / 1e6:6.2f} MB")

    if args.manifest:
        entries = [describe(p) for p in sorted(args.out.glob("*.mpnn"))]
        manifest = args.out / "models.json"
        manifest.write_text(json.dumps(entries, indent=2) + "\n")
        print(f"wrote {manifest} ({len(entries)} models)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
