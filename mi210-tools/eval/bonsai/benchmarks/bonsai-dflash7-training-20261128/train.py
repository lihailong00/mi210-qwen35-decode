"""Fit a listwise residual scorer to greedy target-model trajectories, using prompt-disjoint validation."""

from pathlib import Path
import hashlib, json, struct
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

base = Path(__file__).resolve().parent
K, D, H = 16, 32, 24
dtype = np.dtype([("keys", "<i4", 5), ("ids", "<i4", K), ("features", "<f4", (K, D))])


def read(path):
    return np.fromfile(path, dtype=dtype)


def stats(rows, chosen, T=144):
    hits = np.zeros((int(rows["keys"][:, 0].max()) + 1, T - 2, 7), bool)
    q, pos, depth, y, cap = rows["keys"].T
    hits[q, pos, depth] = (chosen == y) & (y >= 0)
    lengths = hits.cumprod(axis=2).sum(axis=2)
    mask = (depth == 0) & (cap >= 5)
    fixed_success = int((lengths[q[mask], pos[mask]] >= 5).sum())
    num = int(mask.sum())
    rounds = success5 = eligible = 0
    for line in lengths:
        j = 0
        while j < len(line):
            a = int(line[j])
            rounds += 1
            eligible += int(T - 2 - j >= 5)
            success5 += int(a >= 5)
            j += 1 + a
    return dict(
        fixed5=fixed_success / num,
        fixed5_hits=fixed_success,
        fixed5_n=num,
        rounds=rounds,
        tokens_per_round=len(lengths) * T / rounds,
        active5=success5 / eligible,
    )


def save(path, model, mu, sd, alpha, vocab):
    with path.open("wb") as f:
        f.write(struct.pack("<5if", 0x44534631, K, D, H, len(vocab), alpha))
        weight = model.hidden.weight.detach().numpy() / sd
        bias = model.hidden.bias.detach().numpy() - weight @ mu
        for arr in [
            weight,
            bias,
            model.last.weight.detach().numpy(),
            np.array(vocab, dtype="<i4"),
            model.tok.weight.detach().numpy().ravel(),
        ]:
            f.write(arr.astype("<i4" if arr.dtype.kind == "i" else "<f4").tobytes())


class Scorer(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.hidden = nn.Linear(D, H)
        self.last = nn.Linear(H, 1, bias=False)
        self.tok = nn.Embedding(n + 1, 1, padding_idx=0)
        nn.init.zeros_(self.last.weight)
        nn.init.zeros_(self.tok.weight)

    def forward(self, x, ids):
        return self.last(F.relu(self.hidden(x))).squeeze(-1) + self.tok(ids).squeeze(-1)


def data(x, mu, sd, lookup):
    f = x["features"].copy()
    return (
        torch.tensor((f - mu) / sd),
        torch.tensor(lookup[x["ids"]], dtype=torch.long),
        torch.tensor(f[:, :, 30].copy()),
        torch.tensor(x["keys"][:, 3].copy(), dtype=torch.long),
        torch.tensor(x["keys"][:, 2].copy(), dtype=torch.long),
    )


def main():
    torch.manual_seed(725)
    torch.set_num_threads(3)
    shards = [read(base / f"teacher{i}-1.feat") for i in range(6)]
    for x in shards:
        assert sum(x["keys"][:, 2] == 0) == 16 * 142
    train = np.concatenate(shards[:5])
    val = shards[5]
    valid = train["keys"][:, 3] >= 0
    targets = train[valid]
    ids = targets["ids"]
    classes = targets["keys"][:, 3]
    gold = ids[np.arange(len(ids)), classes]
    # All vocabulary statistics, feature normalization and gradient updates use training prompts only.
    counts = np.bincount(gold, minlength=262144)
    vocab = np.flatnonzero(counts >= 4).astype("i4")
    lookup = np.zeros(max(262144, int(train["ids"].max()) + 1), dtype="i4")
    lookup[vocab] = np.arange(1, len(vocab) + 1)
    mu = targets["features"].mean(axis=(0, 1), dtype="f8").astype("f4")
    sd = targets["features"].std(axis=(0, 1), dtype="f8").clip(0.12).astype("f4")
    X, I, B, Y, P = data(targets, mu, sd, lookup)
    V, VI, VB, VY, VP = data(val, mu, sd, lookup)
    model = Scorer(len(vocab))
    opt = torch.optim.AdamW(
        [
            {
                "params": [*model.hidden.parameters(), *model.last.parameters()],
                "lr": 0.002,
            },
            {"params": model.tok.parameters(), "lr": 0.008},
        ],
        weight_decay=0.01,
    )
    baseline = stats(val, VB.argmax(1).numpy())
    best = (-1.0, -1.0)
    trials = []
    for epoch in range(1, 31):
        model.train()
        if epoch in (16, 24):
            for group in opt.param_groups:
                group["lr"] *= 0.5
        for bi in torch.randperm(len(Y)).split(512):
            delta = model(X[bi], I[bi])
            logits = B[bi] + delta
            wt = 1.0 + 0.5 * (P[bi] < 5).float()
            loss = (
                F.cross_entropy(logits, Y[bi], reduction="none") * wt
            ).mean() + 0.002 * (delta - delta.mean(dim=1, keepdim=True)).square().mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        if epoch not in (1, 2, 4, 7, 10, 15, 20, 25, 30):
            continue
        with torch.no_grad():
            model.eval()
            res = torch.cat(
                [
                    model(V[i : i + 1024], VI[i : i + 1024])
                    for i in range(0, len(V), 1024)
                ]
            )
            for alpha in (0.5, 1.0, 1.5):
                z = stats(val, (VB + alpha * res).argmax(1).numpy())
                trials.append(dict(epoch=epoch, alpha=alpha, **z))
                key = (z["fixed5"], z["tokens_per_round"])
                if key > best:
                    best = key
                    selected = trials[-1]
                    save(base / "trained.bin", model, mu, sd, alpha, vocab)
        print(epoch, round(best[0], 4), flush=True)
    outputs = dict(
        train_prompts=80,
        validation_prompts=16,
        output_tokens_per_prompt=144,
        train_anchors=80 * 142,
        validation_anchors=16 * 142,
        train_positive_rows=int(valid.sum()),
        train_missing_rows=int((~valid).sum()),
        vocab_biases=len(vocab),
        features=D,
        hidden=H,
        metric="validation fixed-anchor first-5 prefix, eligible anchors only; ties by simulated tokens/round",
        manual_on_validation=baseline,
        selected=selected,
        trials=trials,
        seed=725,
        sha256=hashlib.sha256((base / "trained.bin").read_bytes()).hexdigest(),
    )
    (base / "train-summary.json").write_text(json.dumps(outputs, indent=2))
    print("FROZEN", outputs["sha256"], flush=True)


if __name__ == "__main__":
    main()
