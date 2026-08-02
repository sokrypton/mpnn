#!/usr/bin/env python3
"""Produce golden tensors from the reference PyTorch model.

Reads the JSON that `test/dump_inputs.mjs` writes (so the JS parser and the
PyTorch model see byte-identical inputs and the comparison isolates the model
maths), runs the real ProteinMPNN/LigandMPNN forward passes, and writes the
intermediates the JS parity test checks against.

    python tools/make_reference.py --ref <ligandmpnn pkg> --inputs a.json --out b.json
"""

import argparse
import json
import pathlib
import sys

import numpy as np
import torch


def load_reference_module(ref: pathlib.Path):
    """Import the reference `model_utils` without pulling in the whole package."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "ref_model_utils", ref / "utils" / "model_utils.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["ref_model_utils"] = module
    spec.loader.exec_module(module)
    return module


def build_model(mu, checkpoint_path, model_type, device, use_side_chains=False):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    atom_context_num = ckpt["atom_context_num"] if model_type == "ligand_mpnn" else 1
    model = mu.ProteinMPNN(
        node_features=128,
        edge_features=128,
        hidden_dim=128,
        num_encoder_layers=3,
        num_decoder_layers=3,
        k_neighbors=ckpt["num_edges"],
        device=device,
        atom_context_num=atom_context_num,
        model_type=model_type,
        ligand_mpnn_use_side_chain_context=use_side_chains,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, ckpt


def nearest_ligand_atoms(CB, mask, Y, Y_t, Y_m, M):
    """`get_nearest_neighbours` from the reference, inlined to avoid ProDy."""
    mask_CBY = mask[:, None] * Y_m[None, :]
    L2 = torch.sum((CB[:, None, :] - Y[None, :, :]) ** 2, -1)
    L2 = L2 * mask_CBY + (1 - mask_CBY) * 1000.0

    nn_idx = torch.argsort(L2, -1)[:, :M]
    L2_nn = torch.gather(L2, 1, nn_idx)
    D_closest = torch.sqrt(L2_nn[:, 0])

    n = CB.shape[0]
    Y_r = Y[None].repeat(n, 1, 1)
    Y_t_r = Y_t[None].repeat(n, 1)
    Y_m_r = Y_m[None].repeat(n, 1)

    Y_out = torch.zeros([n, M, 3], dtype=torch.float32)
    Y_t_out = torch.zeros([n, M], dtype=torch.int32)
    Y_m_out = torch.zeros([n, M], dtype=torch.int32)
    got = nn_idx.shape[1]
    Y_out[:, :got] = torch.gather(Y_r, 1, nn_idx[:, :, None].repeat(1, 1, 3))
    Y_t_out[:, :got] = torch.gather(Y_t_r, 1, nn_idx)
    Y_m_out[:, :got] = torch.gather(Y_m_r, 1, nn_idx)
    return Y_out, Y_t_out, Y_m_out, D_closest


def make_feature_dict(inputs, model_type, atom_context_num, use_side_chains=False):
    L = inputs["L"]
    X = torch.tensor(inputs["X"], dtype=torch.float32).reshape(L, 4, 3)
    mask = torch.tensor(inputs["mask"], dtype=torch.float32)
    S = torch.tensor(inputs["S"], dtype=torch.long)
    R_idx = torch.tensor(inputs["residueIdx"], dtype=torch.long)
    chain_labels = torch.tensor(inputs["chainLabels"], dtype=torch.long)

    fd = {
        "X": X[None],
        "mask": mask[None],
        "S": S[None],
        "R_idx": R_idx[None],
        "chain_labels": chain_labels[None],
        "chain_mask": torch.ones(1, L),
        "bias": torch.zeros(1, L, 21),
        "batch_size": 1,
        "symmetry_residues": [[]],
        "symmetry_weights": [[]],
        "temperature": 0.1,
        "randn": torch.randn(1, L),
    }

    if model_type == "ligand_mpnn":
        n_lig = len(inputs["ligandType"])
        Y = torch.tensor(inputs["ligandXyz"], dtype=torch.float32).reshape(max(n_lig, 1), 3)
        Y_t = torch.tensor(inputs["ligandType"] or [0], dtype=torch.float32)
        Y_m = torch.tensor(inputs["ligandMask"] or [0], dtype=torch.float32)

        CA, N, C = X[:, 1, :], X[:, 0, :], X[:, 2, :]
        b, c = CA - N, C - CA
        a = torch.cross(b, c, dim=-1)
        CB = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + CA
        Yn, Ytn, Ymn, _ = nearest_ligand_atoms(CB, mask, Y, Y_t, Y_m, atom_context_num)
        fd["Y"] = Yn[None]
        fd["Y_t"] = Ytn[None].float()
        fd["Y_m"] = Ymn[None].float()

        if use_side_chains:
            # `forward` concatenates the neighbours' side chains onto the Y
            # above -- the already-selected M nearest ligand atoms, padding
            # slots included -- and reselects M from that pool.
            fd["xyz_37"] = torch.tensor(
                inputs["xyz37"], dtype=torch.float32).reshape(1, L, 37, 3)
            fd["xyz_37_m"] = torch.tensor(
                inputs["xyz37Mask"], dtype=torch.float32).reshape(1, L, 37)
            fd["chain_mask"] = torch.tensor(
                inputs["chainMask"], dtype=torch.float32)[None]

    if model_type in ("per_residue_label_membrane_mpnn", "global_label_membrane_mpnn"):
        labels = inputs.get("membraneLabels") or [0] * L
        fd["membrane_per_residue_labels"] = torch.tensor(labels, dtype=torch.long)[None]

    return fd


def ar_mask_from_order(order, L):
    rank = np.empty(L, dtype=np.int64)
    rank[np.asarray(order)] = np.arange(L)
    return (rank[:, None] > rank[None, :]).astype(np.float32)


def decoder_pass(model, h_V_enc, h_E, E_idx, S, mask, ar_mask):
    """Teacher-forced decoder driven by an explicit autoregressive mask."""
    mu = sys.modules["ref_model_utils"]
    mask_attend = torch.gather(ar_mask[None], 2, E_idx).unsqueeze(-1)
    mask_1D = mask.view([1, -1, 1, 1])
    mask_bw = mask_1D * mask_attend
    mask_fw = mask_1D * (1.0 - mask_attend)

    h_S = model.W_s(S)
    h_ES = mu.cat_neighbors_nodes(h_S, h_E, E_idx)
    h_EX_encoder = mu.cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx)
    h_EXV_encoder = mu.cat_neighbors_nodes(h_V_enc, h_EX_encoder, E_idx)
    h_EXV_encoder_fw = mask_fw * h_EXV_encoder

    h_V = h_V_enc
    for layer in model.decoder_layers:
        h_ESV = mu.cat_neighbors_nodes(h_V, h_ES, E_idx)
        h_ESV = mask_bw * h_ESV + h_EXV_encoder_fw
        h_V = layer(h_V, h_ESV, mask)
    return model.W_out(h_V)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, type=pathlib.Path)
    ap.add_argument("--checkpoints", required=True, type=pathlib.Path)
    ap.add_argument("--inputs", required=True, type=pathlib.Path)
    ap.add_argument("--out", required=True, type=pathlib.Path)
    args = ap.parse_args()

    mu = load_reference_module(args.ref)
    cases = json.loads(args.inputs.read_text())
    torch.manual_seed(0)
    results = []

    for case in cases:
        model_type = case["modelType"]
        ckpt_path = args.checkpoints / f"{case['checkpoint']}.pt"
        side_chains = bool(case.get("useSideChains"))
        model, ckpt = build_model(mu, ckpt_path, model_type, "cpu", side_chains)
        atom_context_num = ckpt.get("atom_context_num", 1)
        fd = make_feature_dict(case["inputs"], model_type, atom_context_num, side_chains)
        L = case["inputs"]["L"]

        with torch.no_grad():
            h_V, h_E, E_idx = model.encode(fd)
            S = fd["S"]
            mask = fd["mask"]

            # Three conditioning regimes, all deterministic.
            order = np.asarray(case["order"], dtype=np.int64)
            regimes = {
                "none": np.zeros((L, L), dtype=np.float32),
                "order": ar_mask_from_order(order, L),
                "all_but_self": 1.0 - np.eye(L, dtype=np.float32),
            }
            logits = {}
            for name, ar in regimes.items():
                out = decoder_pass(model, h_V, h_E, E_idx, S, mask, torch.tensor(ar))
                logits[name] = out[0].numpy()

        results.append({
            "name": case["name"],
            "checkpoint": case["checkpoint"],
            "modelType": model_type,
            "L": L,
            "K": int(E_idx.shape[-1]),
            "EIdx": E_idx[0].numpy().astype(np.int32).tolist(),
            "hV": h_V[0].numpy().tolist(),
            "hE_sum": float(h_E.sum()),
            "hE_row0": h_E[0, 0].numpy().tolist(),
            "logits": {k: v.tolist() for k, v in logits.items()},
        })
        print(f"{case['name']:<28} {model_type:<20} L={L:<5} K={E_idx.shape[-1]}")

    args.out.write_text(json.dumps(results))
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
