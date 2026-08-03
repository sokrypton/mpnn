#!/usr/bin/env python3
"""Golden tensors from the real NA-MPNN, plus the inputs that produced them.

Unlike the LigandMPNN harness, this dumps the *inputs* as well: NA-MPNN parses
structures with ProDy and its own polymer-type rules, so the JS side is checked
in two independent halves -- `test/na.mjs` feeds these exact tensors through the
engine to check the model maths, and separately checks that the JS parser
reproduces the tensors from the same PDB.

    python tools/na_reference.py --ref <NA-MPNN checkout> \
        --checkpoint <s_*.pt> --pdb 4oqu.pdb --out golden.json
"""

import argparse
import importlib.util
import json
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from make_reference import ar_mask_from_order, decoder_pass  # noqa: E402


def load(ref: pathlib.Path):
    """Import NA-MPNN's inference modules without installing the package."""
    sys.path.insert(0, str(ref / "inference"))
    mods = {}
    for name in ("data_utils", "model_utils"):
        spec = importlib.util.spec_from_file_location(name, ref / "inference" / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        mods[name] = module
    return mods


ATOM_TYPES = ["N", "CA", "C", "O",
              "OP1", "OP2", "P", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'"]
POLYTYPES = ["PP", "DNA", "RNA", "UNK", "MAS", "PAD"]
RESTYPES = ["ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
            "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL", "UNK",
            "DA", "DC", "DG", "DT", "DX", "A", "C", "G", "U", "RX", "MAS", "PAD"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, type=pathlib.Path)
    ap.add_argument("--checkpoint", required=True, type=pathlib.Path)
    ap.add_argument("--pdb", required=True, nargs="+", type=pathlib.Path)
    ap.add_argument("--out", required=True, type=pathlib.Path)
    args = ap.parse_args()

    mods = load(args.ref)
    du, mu = mods["data_utils"], mods["model_utils"]
    torch.manual_seed(0)

    atom_dict = dict(zip(ATOM_TYPES, range(len(ATOM_TYPES))))
    polytype_to_int = dict(zip(POLYTYPES, range(len(POLYTYPES))))
    restype_to_int = dict(zip(RESTYPES, range(len(RESTYPES))))

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = mu.ProteinMPNN(
        node_features=128, edge_features=128, hidden_dim=128,
        num_encoder_layers=3, num_decoder_layers=3, k_neighbors=32,
        model_type="na_mpnn", vocab=len(RESTYPES), num_letters=len(RESTYPES),
        atom_dict=atom_dict, restype_to_int=restype_to_int,
        polytype_to_int=polytype_to_int,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    cases = []
    for pdb in args.pdb:
        # `na_shared_tokens` defaults to 0 in parse_PDB but to 1 in run.py, and
        # run.py is what anyone actually runs: RNA residues are encoded with the
        # DNA tokens and the legacy RNA letters are omitted from sampling.
        parsed = du.parse_PDB(str(pdb), device="cpu", model_type="na_mpnn",
                              na_shared_tokens=True)[0]
        # run.py builds chain_mask from --chains_to_design; everything designed.
        parsed["chain_mask"] = torch.ones_like(parsed["mask"], dtype=torch.int32)
        fd = du.featurize(parsed)
        fd["batch_size"] = 1
        L = int(fd["S"].shape[1])

        with torch.no_grad():
            h_V, h_E, E_idx = model.encode(fd)
            rng = np.random.default_rng(12345)
            order = rng.permutation(L)
            logits = {
                "none": decoder_pass(model, h_V, h_E, E_idx, fd["S"], fd["mask"],
                                     torch.zeros(L, L), mu),
                "order": decoder_pass(model, h_V, h_E, E_idx, fd["S"], fd["mask"],
                                      torch.tensor(ar_mask_from_order(order, L)), mu),
                "all_but_self": decoder_pass(model, h_V, h_E, E_idx, fd["S"], fd["mask"],
                                             torch.tensor(1.0 - np.eye(L, dtype=np.float32)), mu),
            }

        case = {
            "name": pdb.stem,
            "L": L,
            "K": int(E_idx.shape[-1]),
            "order": order.tolist(),
            "inputs": {
                "X16": fd["X"][0].numpy().ravel().tolist(),
                "X16Mask": fd["X_m"][0].numpy().astype(np.float32).ravel().tolist(),
                "mask": fd["mask"][0].numpy().astype(np.float32).tolist(),
                "S": fd["S"][0].numpy().astype(np.int32).tolist(),
                "residueIdx": fd["R_idx"][0].numpy().astype(np.int32).tolist(),
                "chainLabels": fd["chain_labels"][0].numpy().astype(np.int32).tolist(),
                "polytype": fd["R_polymer_type"][0].numpy().astype(np.int32).tolist(),
                # Presence of an O2', which is how the reference decides
                # whether to render a DNA token as an RNA letter.
                "isRNA": fd["rna_mask_for_token_conversion"][0]
                    .numpy().astype(np.int32).tolist(),
            },
            "EIdx": E_idx[0].numpy().astype(np.int32).ravel().tolist(),
            "hV": h_V[0].numpy().ravel().tolist(),
            "logits": {k: v[0].numpy().ravel().tolist() for k, v in logits.items()},
        }
        cases.append(case)
        counts = np.bincount(case["inputs"]["polytype"], minlength=len(POLYTYPES))
        print(f"{pdb.stem:<12} L={L:<5} K={case['K']}  "
              + "  ".join(f"{POLYTYPES[i]}={c}" for i, c in enumerate(counts) if c))

    args.out.write_text(json.dumps(cases))
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
