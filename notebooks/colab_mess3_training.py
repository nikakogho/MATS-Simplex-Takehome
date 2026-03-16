# %%
from __future__ import annotations

import copy
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is required for this workflow. In Colab: Runtime -> Change runtime type -> GPU."
    )

device = torch.device("cuda")
print(f"Using device: {device} ({torch.cuda.get_device_name(0)})")


@dataclass
class ExperimentConfig:
    seed: int = 42
    seq_len: int = 16
    vocab_size: int = 3
    hidden_states: int = 3
    n_train_per_component: int = 10_000
    n_val_per_component: int = 1_000
    n_test_per_component: int = 1_000
    batch_size: int = 128
    max_epochs: int = 60
    patience: int = 12
    lr: float = 1e-3
    weight_decay: float = 1e-2
    d_model: int = 64
    n_heads: int = 2
    n_layers: int = 2
    d_mlp: int = 256
    log_every_steps: int = 50
    n_cache_checkpoints: int = 10
    cache_subset_size: int = 512
    num_workers: int = 2
    output_root: str = "./takehome_outputs"


cfg = ExperimentConfig()
TOKEN_NAMES = ["A", "B", "C"]
STATE_NAMES = ["S1", "S2", "S3"]
PROCESS_PARAMS = [
    ("mess3_c0_x005_a090", 0, 0.05, 0.90),
    ("mess3_c1_x018_a095", 1, 0.18, 0.95),
    ("mess3_c2_x005_a045", 2, 0.05, 0.45),
]

torch.manual_seed(cfg.seed)
torch.cuda.manual_seed_all(cfg.seed)
np.random.seed(cfg.seed)
random.seed(cfg.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

OUTPUT_ROOT = Path(cfg.output_root)
DATA_DIR = OUTPUT_ROOT / "data"
CKPT_DIR = OUTPUT_ROOT / "checkpoints"
CACHE_DIR = OUTPUT_ROOT / "activation_caches"
PLOT_DIR = OUTPUT_ROOT / "plots"
for path in [OUTPUT_ROOT, DATA_DIR, CKPT_DIR, CACHE_DIR, PLOT_DIR]:
    path.mkdir(parents=True, exist_ok=True)

print(json.dumps(asdict(cfg), indent=2))


# %%
@dataclass
class Mess3Component:
    name: str
    component_id: int
    pi: np.ndarray
    A: np.ndarray
    E: np.ndarray
    vocab_size: int

    def validate(self) -> None:
        k = self.pi.shape[0]
        assert self.pi.shape == (k,)
        assert self.A.shape == (k, k)
        assert self.E.shape == (k, self.vocab_size)
        assert np.all(self.pi >= 0)
        assert np.all(self.A >= 0)
        assert np.all(self.E >= 0)
        assert np.isclose(self.pi.sum(), 1.0)
        assert np.allclose(self.A.sum(axis=1), 1.0)
        assert np.allclose(self.E.sum(axis=1), 1.0)


def make_mess3_component(name: str, component_id: int, x: float, alpha: float) -> Mess3Component:
    pi = np.full(3, 1.0 / 3.0, dtype=np.float64)
    A = np.full((3, 3), x, dtype=np.float64)
    np.fill_diagonal(A, 1.0 - 2.0 * x)
    E = np.full((3, 3), (1.0 - alpha) / 2.0, dtype=np.float64)
    np.fill_diagonal(E, alpha)
    component = Mess3Component(name=name, component_id=component_id, pi=pi, A=A, E=E, vocab_size=3)
    component.validate()
    return component


components = [make_mess3_component(*params) for params in PROCESS_PARAMS]
components_by_id = {comp.component_id: comp for comp in components}


def sample_categorical(probs: np.ndarray, rng: np.random.Generator) -> int:
    return int(rng.choice(len(probs), p=probs))


def sample_sequence(component: Mess3Component, seq_len: int, rng: np.random.Generator):
    tokens = np.zeros(seq_len, dtype=np.int64)
    states = np.zeros(seq_len, dtype=np.int64)
    z = sample_categorical(component.pi, rng)
    for t in range(seq_len):
        states[t] = z
        tokens[t] = sample_categorical(component.E[z], rng)
        if t < seq_len - 1:
            z = sample_categorical(component.A[z], rng)
    return tokens, states


def sample_balanced_dataset(components, n_per_component: int, seq_len: int, rng: np.random.Generator):
    tokens, states, component_ids = [], [], []
    for comp in components:
        for _ in range(n_per_component):
            seq_tokens, seq_states = sample_sequence(comp, seq_len, rng)
            tokens.append(seq_tokens)
            states.append(seq_states)
            component_ids.append(comp.component_id)
    tokens = np.stack(tokens)
    states = np.stack(states)
    component_ids = np.array(component_ids, dtype=np.int64)
    perm = rng.permutation(len(tokens))
    return {
        "tokens": tokens[perm],
        "states": states[perm],
        "component_ids": component_ids[perm],
    }


def normalize(prob: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    total = prob.sum()
    if total < eps:
        raise ValueError(f"Probability vector nearly zero; sum={total}")
    return prob / total


def logsumexp_np(x: np.ndarray, axis=None, keepdims: bool = False) -> np.ndarray:
    m = np.max(x, axis=axis, keepdims=True)
    y = m + np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True))
    if keepdims:
        return y
    return np.squeeze(y, axis=axis)


def compute_beliefs_for_sequence(tokens, pi, A, E):
    t_len = len(tokens)
    k = len(pi)
    predictive_before_obs = np.zeros((t_len, k), dtype=np.float64)
    filtered_after_obs = np.zeros((t_len, k), dtype=np.float64)
    predictive_next = np.zeros((t_len, k), dtype=np.float64)
    obs_prob = np.zeros(t_len, dtype=np.float64)
    q = pi.astype(np.float64).copy()
    for t, x in enumerate(tokens):
        predictive_before_obs[t] = q
        emission = E[:, x]
        unnorm = q * emission
        obs_prob[t] = unnorm.sum()
        b = normalize(unnorm)
        filtered_after_obs[t] = b
        q = b @ A
        predictive_next[t] = q
    return {
        "predictive_before_obs": predictive_before_obs,
        "filtered_after_obs": filtered_after_obs,
        "predictive_next": predictive_next,
        "obs_prob": obs_prob,
        "loglik": float(np.log(obs_prob).sum()),
    }


def compute_targets_for_sequence(tokens: np.ndarray, true_component_id: int, components) -> dict:
    true_comp = components_by_id[int(true_component_id)]
    true_out = compute_beliefs_for_sequence(tokens, true_comp.pi, true_comp.A, true_comp.E)

    n_components = len(components)
    t_len = len(tokens)
    prefix_logliks = np.zeros((n_components, t_len), dtype=np.float64)
    component_predictive_next = np.zeros((n_components, t_len, 3), dtype=np.float64)
    for idx, comp in enumerate(components):
        out = compute_beliefs_for_sequence(tokens, comp.pi, comp.A, comp.E)
        prefix_logliks[idx] = np.cumsum(np.log(out["obs_prob"]))
        component_predictive_next[idx] = out["predictive_next"]

    log_prior = -math.log(n_components)
    log_post = prefix_logliks + log_prior
    log_post = log_post - logsumexp_np(log_post, axis=0, keepdims=True)
    process_posterior = np.exp(log_post).T

    bayes_next_token = np.zeros((t_len - 1, 3), dtype=np.float64)
    for t in range(t_len - 1):
        probs = np.zeros(3, dtype=np.float64)
        for idx, comp in enumerate(components):
            probs += process_posterior[t, idx] * (component_predictive_next[idx, t] @ comp.E)
        bayes_next_token[t] = normalize(probs)

    return {
        "filtered_after_obs": true_out["filtered_after_obs"].astype(np.float32),
        "predictive_next": true_out["predictive_next"].astype(np.float32),
        "process_posterior": process_posterior.astype(np.float32),
        "bayes_next_token": bayes_next_token.astype(np.float32),
    }


# %%
def precompute_split(split_name: str, n_per_component: int, cfg: ExperimentConfig, components):
    save_path = DATA_DIR / f"{split_name}_dataset.pt"
    if save_path.exists():
        print(f"Loading cached {split_name} split from {save_path}")
        return torch.load(save_path, map_location="cpu")

    split_offset = {"train": 0, "val": 1, "test": 2}[split_name]
    rng = np.random.default_rng(cfg.seed + split_offset)
    raw = sample_balanced_dataset(components, n_per_component, cfg.seq_len, rng)
    n = raw["tokens"].shape[0]

    filtered_after_obs = np.zeros((n, cfg.seq_len, 3), dtype=np.float32)
    predictive_next = np.zeros((n, cfg.seq_len, 3), dtype=np.float32)
    process_posterior = np.zeros((n, cfg.seq_len, len(components)), dtype=np.float32)
    bayes_next_token = np.zeros((n, cfg.seq_len - 1, cfg.vocab_size), dtype=np.float32)

    start = time.time()
    for i in range(n):
        targets = compute_targets_for_sequence(raw["tokens"][i], int(raw["component_ids"][i]), components)
        filtered_after_obs[i] = targets["filtered_after_obs"]
        predictive_next[i] = targets["predictive_next"]
        process_posterior[i] = targets["process_posterior"]
        bayes_next_token[i] = targets["bayes_next_token"]
        if (i + 1) % 1000 == 0 or i + 1 == n:
            elapsed = time.time() - start
            print(f"[{split_name}] precomputed {i + 1}/{n} sequences in {elapsed:.1f}s")

    dataset = {
        "tokens": torch.from_numpy(raw["tokens"]).long(),
        "states": torch.from_numpy(raw["states"]).long(),
        "component_ids": torch.from_numpy(raw["component_ids"]).long(),
        "filtered_after_obs": torch.from_numpy(filtered_after_obs).float(),
        "predictive_next": torch.from_numpy(predictive_next).float(),
        "process_posterior": torch.from_numpy(process_posterior).float(),
        "bayes_next_token": torch.from_numpy(bayes_next_token).float(),
        "token_names": TOKEN_NAMES,
        "state_names": STATE_NAMES,
        "process_names": [comp.name for comp in components],
    }
    torch.save(dataset, save_path)
    print(f"Saved {split_name} split to {save_path}")
    return dataset


def sanity_check_split(name: str, data: dict):
    n = len(data["component_ids"])
    assert data["tokens"].shape == (n, cfg.seq_len)
    assert data["states"].shape == (n, cfg.seq_len)
    assert data["filtered_after_obs"].shape == (n, cfg.seq_len, 3)
    assert data["predictive_next"].shape == (n, cfg.seq_len, 3)
    assert data["process_posterior"].shape == (n, cfg.seq_len, 3)
    assert data["bayes_next_token"].shape == (n, cfg.seq_len - 1, 3)
    assert torch.allclose(
        data["filtered_after_obs"].sum(dim=-1),
        torch.ones_like(data["filtered_after_obs"][..., 0]),
        atol=1e-5,
    )
    assert torch.allclose(
        data["predictive_next"].sum(dim=-1),
        torch.ones_like(data["predictive_next"][..., 0]),
        atol=1e-5,
    )
    assert torch.allclose(
        data["process_posterior"].sum(dim=-1),
        torch.ones_like(data["process_posterior"][..., 0]),
        atol=1e-5,
    )
    assert torch.allclose(
        data["bayes_next_token"].sum(dim=-1),
        torch.ones_like(data["bayes_next_token"][..., 0]),
        atol=1e-5,
    )
    unique, counts = torch.unique(data["component_ids"], return_counts=True)
    print(name, {int(k): int(v) for k, v in zip(unique, counts)})


train_data = precompute_split("train", cfg.n_train_per_component, cfg, components)
val_data = precompute_split("val", cfg.n_val_per_component, cfg, components)
test_data = precompute_split("test", cfg.n_test_per_component, cfg, components)

sanity_check_split("train", train_data)
sanity_check_split("val", val_data)
sanity_check_split("test", test_data)


# %%
def mean_nll_from_probs(probs: torch.Tensor, targets: torch.Tensor):
    gathered = probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1).clamp_min(1e-9)
    per_token_nll = -torch.log(gathered)
    return per_token_nll.mean().item(), per_token_nll.mean(dim=0).cpu().numpy()


def report_bayes_baseline(name: str, data: dict):
    overall_nll, by_position = mean_nll_from_probs(data["bayes_next_token"], data["tokens"][:, 1:])
    print(f"{name} Bayes-optimal next-token NLL: {overall_nll:.4f}")
    return overall_nll, by_position


uniform_nll = float(math.log(cfg.vocab_size))
print(f"Uniform baseline NLL: {uniform_nll:.4f}")
val_bayes_nll, val_bayes_by_pos = report_bayes_baseline("val", val_data)
test_bayes_nll, test_bayes_by_pos = report_bayes_baseline("test", test_data)

plt.figure(figsize=(8, 4))
positions = np.arange(1, cfg.seq_len)
plt.plot(positions, val_bayes_by_pos, marker="o", label="Val Bayes NLL")
plt.plot(positions, test_bayes_by_pos, marker="o", label="Test Bayes NLL")
plt.axhline(uniform_nll, color="gray", linestyle="--", label="Uniform baseline")
plt.xlabel("Prediction position")
plt.ylabel("NLL")
plt.title("Exact Bayes next-token baseline")
plt.legend()
plt.show()


# %%
def make_loader(data: dict, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(data["tokens"][:, :-1], data["tokens"][:, 1:])
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
    )


train_loader = make_loader(train_data, cfg.batch_size, shuffle=True)
val_loader = make_loader(val_data, cfg.batch_size, shuffle=False)
test_loader = make_loader(test_data, cfg.batch_size, shuffle=False)


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, t_len, d_model = x.shape
        qkv = self.qkv(x)
        qkv = qkv.view(bsz, t_len, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(bsz, t_len, d_model)
        return self.out(y)


class MLP(nn.Module):
    def __init__(self, d_model: int, d_mlp: int):
        super().__init__()
        self.fc_in = nn.Linear(d_model, d_mlp)
        self.fc_out = nn.Linear(d_mlp, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc_out(F.gelu(self.fc_in(x)))


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_mlp: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, d_mlp)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self, cfg: ExperimentConfig):
        super().__init__()
        self.cfg = cfg
        self.token_embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_embed = nn.Embedding(cfg.seq_len, cfg.d_model)
        self.blocks = nn.ModuleList(
            [TransformerBlock(cfg.d_model, cfg.n_heads, cfg.d_mlp) for _ in range(cfg.n_layers)]
        )
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def forward(self, idx: torch.Tensor, return_residuals: bool = False):
        _, t_len = idx.shape
        pos = torch.arange(t_len, device=idx.device)
        x = self.token_embed(idx) + self.pos_embed(pos)[None, :, :]
        residuals = {}
        if return_residuals:
            residuals["embed"] = x
        for block_idx, block in enumerate(self.blocks, start=1):
            x = block(x)
            if return_residuals:
                residuals[f"block{block_idx}"] = x
        logits = self.lm_head(self.ln_f(x))
        if return_residuals:
            return logits, residuals
        return logits


model = TinyGPT(cfg).to(device)
assert next(model.parameters()).is_cuda
print(f"Model parameter count: {sum(p.numel() for p in model.parameters()):,}")


# %%
def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    total_per_position = torch.zeros(cfg.seq_len - 1, dtype=torch.float64)
    total_counts = torch.zeros(cfg.seq_len - 1, dtype=torch.float64)
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits = model(inputs)
            loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), targets.reshape(-1), reduction="mean")
            total_loss += loss.item() * targets.numel()
            total_tokens += targets.numel()

            log_probs = F.log_softmax(logits, dim=-1)
            per_token = -log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            total_per_position += per_token.sum(dim=0).cpu().double()
            total_counts += torch.tensor([targets.shape[0]] * targets.shape[1], dtype=torch.float64)

    return {
        "loss": total_loss / total_tokens,
        "per_position": (total_per_position / total_counts).numpy(),
    }


def balanced_cache_indices(component_ids: torch.Tensor, total_size: int) -> torch.Tensor:
    rng = np.random.default_rng(cfg.seed)
    unique_components = sorted(int(x) for x in component_ids.unique().tolist())
    base = total_size // len(unique_components)
    remainder = total_size % len(unique_components)
    picks = []
    for offset, comp_id in enumerate(unique_components):
        take_n = base + (1 if offset < remainder else 0)
        indices = torch.where(component_ids == comp_id)[0].numpy()
        chosen = rng.choice(indices, size=take_n, replace=False)
        picks.append(torch.from_numpy(chosen))
    out = torch.cat(picks)
    return out[torch.randperm(len(out))]


cache_indices = balanced_cache_indices(val_data["component_ids"], cfg.cache_subset_size)
cache_inputs = val_data["tokens"][cache_indices][:, :-1]
cache_targets_hidden = val_data["predictive_next"][cache_indices][:, : cfg.seq_len - 1]
cache_targets_process = val_data["process_posterior"][cache_indices][:, : cfg.seq_len - 1]
cache_component_ids = val_data["component_ids"][cache_indices]


def capture_activation_cache(
    model: nn.Module,
    inputs: torch.Tensor,
    hidden_targets: torch.Tensor,
    process_targets: torch.Tensor,
    component_ids: torch.Tensor,
    label: str,
):
    model.eval()
    caches = {"embed": [], "block1": [], "block2": []}
    loader = DataLoader(
        TensorDataset(inputs, hidden_targets, process_targets, component_ids),
        batch_size=128,
        shuffle=False,
    )
    with torch.no_grad():
        for batch_inputs, _, _, _ in loader:
            batch_inputs = batch_inputs.to(device, non_blocking=True)
            logits, residuals = model(batch_inputs, return_residuals=True)
            _ = logits
            for key in caches:
                caches[key].append(residuals[key].detach().cpu())

    packed = {
        "label": label,
        "inputs": inputs.clone(),
        "component_ids": component_ids.clone(),
        "hidden_targets": hidden_targets.clone(),
        "process_targets": process_targets.clone(),
    }
    for key, value in caches.items():
        packed[key] = torch.cat(value, dim=0)
    assert packed["embed"].shape[0] == inputs.shape[0]
    return packed


def scheduled_cache_epochs(max_epochs: int, n_checkpoints: int):
    if n_checkpoints < 2:
        return []
    epochs = np.linspace(1, max_epochs, num=n_checkpoints - 2)
    return sorted(set(int(round(x)) for x in epochs.tolist()))


def train_model(model: nn.Module):
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_per_position": [],
    }
    activation_caches = {}
    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    best_epoch = 0
    patience_counter = 0
    activation_caches["init"] = capture_activation_cache(
        model,
        cache_inputs,
        cache_targets_hidden,
        cache_targets_process,
        cache_component_ids,
        label="init",
    )
    schedule = set(scheduled_cache_epochs(cfg.max_epochs, cfg.n_cache_checkpoints))
    train_start = time.time()

    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        running_loss = 0.0
        running_tokens = 0
        epoch_start = time.time()
        for step, (inputs, targets) in enumerate(train_loader, start=1):
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            assert inputs.is_cuda and targets.is_cuda

            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), targets.reshape(-1), reduction="mean")
            loss.backward()
            optimizer.step()

            batch_tokens = targets.numel()
            running_loss += loss.item() * batch_tokens
            running_tokens += batch_tokens

            if step % cfg.log_every_steps == 0 or step == len(train_loader):
                elapsed = time.time() - train_start
                print(
                    f"epoch={epoch:02d} step={step:03d}/{len(train_loader):03d} "
                    f"train_loss={loss.item():.4f} running_train_loss={running_loss / running_tokens:.4f} "
                    f"lr={optimizer.param_groups[0]['lr']:.2e} elapsed={elapsed / 60:.1f}m"
                )

        train_loss = running_loss / running_tokens
        val_metrics = evaluate_model(model, val_loader, device)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_per_position"].append(val_metrics["per_position"])

        improved = val_metrics["loss"] < best_val
        if improved:
            best_val = val_metrics["loss"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, CKPT_DIR / "best_model.pt")
            patience_counter = 0
        else:
            patience_counter += 1

        elapsed_epoch = time.time() - epoch_start
        print(
            f"[epoch {epoch:02d}] train_loss={train_loss:.4f} val_loss={val_metrics['loss']:.4f} "
            f"best_val={best_val:.4f} best_epoch={best_epoch:02d} "
            f"patience={patience_counter}/{cfg.patience} epoch_time={elapsed_epoch:.1f}s"
        )

        if epoch in schedule:
            label = f"epoch_{epoch:02d}"
            activation_caches[label] = capture_activation_cache(
                model,
                cache_inputs,
                cache_targets_hidden,
                cache_targets_process,
                cache_component_ids,
                label=label,
            )

        if patience_counter >= cfg.patience:
            print(f"Early stopping triggered at epoch {epoch}.")
            break

    model.load_state_dict(best_state)
    activation_caches["best"] = capture_activation_cache(
        model,
        cache_inputs,
        cache_targets_hidden,
        cache_targets_process,
        cache_component_ids,
        label="best",
    )
    torch.save(
        {"history": history, "activation_caches": activation_caches},
        CACHE_DIR / "training_artifacts.pt",
    )
    final_val = evaluate_model(model, val_loader, device)
    final_test = evaluate_model(model, test_loader, device)
    return model, history, activation_caches, final_val, final_test


# %%
model = TinyGPT(cfg).to(device)
model, history, activation_caches, final_val_metrics, final_test_metrics = train_model(model)

print(f"Best-model val NLL: {final_val_metrics['loss']:.4f}")
print(f"Best-model test NLL: {final_test_metrics['loss']:.4f}")
print(f"Val Bayes NLL: {val_bayes_nll:.4f}")
print(f"Test Bayes NLL: {test_bayes_nll:.4f}")

plt.figure(figsize=(8, 4))
plt.plot(history["train_loss"], marker="o", label="Train")
plt.plot(history["val_loss"], marker="o", label="Val")
plt.axhline(val_bayes_nll, color="green", linestyle="--", label="Val Bayes")
plt.xlabel("Epoch")
plt.ylabel("NLL")
plt.title("Training curves")
plt.legend()
plt.show()

plt.figure(figsize=(8, 4))
positions = np.arange(1, cfg.seq_len)
plt.plot(positions, final_val_metrics["per_position"], marker="o", label="Model val")
plt.plot(positions, val_bayes_by_pos, marker="o", label="Bayes val")
plt.xlabel("Prediction position")
plt.ylabel("NLL")
plt.title("Model vs Bayes by position")
plt.legend()
plt.show()


# %%
def flatten_for_readout(acts: torch.Tensor, targets: torch.Tensor, component_ids: torch.Tensor):
    x = acts.reshape(-1, acts.shape[-1]).numpy()
    y = targets.reshape(-1, targets.shape[-1]).numpy()
    comp = component_ids[:, None].repeat(1, acts.shape[1]).reshape(-1).numpy()
    return x, y, comp


def affine_lstsq_fit(x_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
    x_aug = np.concatenate([x_train, np.ones((x_train.shape[0], 1), dtype=x_train.dtype)], axis=1)
    weights, *_ = np.linalg.lstsq(x_aug, y_train, rcond=None)
    return weights


def affine_lstsq_predict(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    x_aug = np.concatenate([x, np.ones((x.shape[0], 1), dtype=x.dtype)], axis=1)
    return x_aug @ weights


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mse = float(np.mean((y_true - y_pred) ** 2))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean(axis=0, keepdims=True)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    y_pred_simplex = np.clip(y_pred, 1e-8, None)
    y_pred_simplex = y_pred_simplex / y_pred_simplex.sum(axis=-1, keepdims=True)
    kl = float(
        np.mean(
            np.sum(
                y_true * (np.log(np.clip(y_true, 1e-8, None)) - np.log(y_pred_simplex)),
                axis=-1,
            )
        )
    )
    return {"mse": mse, "r2": r2, "kl": kl}


def evaluate_global_and_per_component(cache_blob: dict, layer_name: str, target_name: str):
    acts = cache_blob[layer_name]
    targets = cache_blob[target_name]
    component_ids = cache_blob["component_ids"]
    n_seq = acts.shape[0]
    split = n_seq // 2
    train_idx = np.arange(split)
    test_idx = np.arange(split, n_seq)

    x_train, y_train, comp_train = flatten_for_readout(acts[train_idx], targets[train_idx], component_ids[train_idx])
    x_test, y_test, comp_test = flatten_for_readout(acts[test_idx], targets[test_idx], component_ids[test_idx])

    global_w = affine_lstsq_fit(x_train, y_train)
    global_pred = affine_lstsq_predict(x_test, global_w)
    global_metrics = regression_metrics(y_test, global_pred)

    per_component_scores = []
    for comp_id in sorted(np.unique(comp_train)):
        comp_mask_train = comp_train == comp_id
        comp_mask_test = comp_test == comp_id
        comp_w = affine_lstsq_fit(x_train[comp_mask_train], y_train[comp_mask_train])
        comp_pred = affine_lstsq_predict(x_test[comp_mask_test], comp_w)
        comp_metrics = regression_metrics(y_test[comp_mask_test], comp_pred)
        comp_metrics["component_id"] = int(comp_id)
        per_component_scores.append(comp_metrics)

    per_component_avg = {
        "mse": float(np.mean([score["mse"] for score in per_component_scores])),
        "r2": float(np.mean([score["r2"] for score in per_component_scores])),
        "kl": float(np.mean([score["kl"] for score in per_component_scores])),
    }
    return {
        "global": global_metrics,
        "per_component_avg": per_component_avg,
        "per_component": per_component_scores,
    }


layers = ["embed", "block1", "block2"]
hidden_results = {}
process_results = {}
for ckpt_name, cache_blob in activation_caches.items():
    hidden_results[ckpt_name] = {}
    process_results[ckpt_name] = {}
    for layer_name in layers:
        hidden_results[ckpt_name][layer_name] = evaluate_global_and_per_component(
            cache_blob, layer_name, "hidden_targets"
        )
        process_results[ckpt_name][layer_name] = evaluate_global_and_per_component(
            cache_blob, layer_name, "process_targets"
        )

print("Hidden-state readout results (best checkpoint):")
for layer_name in layers:
    print(layer_name, hidden_results["best"][layer_name])

print("Process readout results (best checkpoint):")
for layer_name in layers:
    print(layer_name, process_results["best"][layer_name])

torch.save(
    {"hidden_results": hidden_results, "process_results": process_results},
    OUTPUT_ROOT / "readout_results.pt",
)
print(f"Saved readout results to {OUTPUT_ROOT / 'readout_results.pt'}")


def metric_curve(results: dict, layer_name: str, target_key: str, metric: str):
    checkpoint_names = list(results.keys())
    xs = np.arange(len(checkpoint_names))
    ys = [results[name][layer_name][target_key][metric] for name in checkpoint_names]
    return checkpoint_names, xs, ys


plt.figure(figsize=(9, 4))
for layer_name in layers:
    checkpoint_names, xs, ys = metric_curve(hidden_results, layer_name, "global", "r2")
    plt.plot(xs, ys, marker="o", label=f"Hidden {layer_name}")
plt.xticks(xs, checkpoint_names, rotation=45)
plt.ylabel("R^2")
plt.title("Hidden-state readout quality across checkpoints")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(9, 4))
for layer_name in layers:
    checkpoint_names, xs, ys = metric_curve(process_results, layer_name, "global", "r2")
    plt.plot(xs, ys, marker="o", label=f"Process {layer_name}")
plt.xticks(xs, checkpoint_names, rotation=45)
plt.ylabel("R^2")
plt.title("Process readout quality across checkpoints")
plt.legend()
plt.tight_layout()
plt.show()


# %%
TRIANGLE = np.array(
    [
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, np.sqrt(3.0) / 2.0],
    ],
    dtype=np.float64,
)


def simplex_to_xy(probs: np.ndarray) -> np.ndarray:
    return probs @ TRIANGLE


def renormalize_simplex(arr: np.ndarray) -> np.ndarray:
    arr = np.clip(arr, 1e-8, None)
    return arr / arr.sum(axis=-1, keepdims=True)


def plot_simplex_points(ax, probs: np.ndarray, colors: np.ndarray, title: str, labels):
    coords = simplex_to_xy(probs)
    border = np.vstack([TRIANGLE, TRIANGLE[0]])
    ax.plot(border[:, 0], border[:, 1], color="tab:blue")
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=colors, s=8, cmap="viridis", alpha=0.7)
    for idx, label in enumerate(labels):
        y = TRIANGLE[idx, 1] + (0.03 if idx == 2 else -0.03)
        ax.text(TRIANGLE[idx, 0], y, label, ha="center")
    ax.set_title(title)
    ax.set_axis_off()
    return scatter


best_cache = activation_caches["best"]
hidden_truth = best_cache["hidden_targets"].reshape(-1, 3).numpy()
process_truth = best_cache["process_targets"].reshape(-1, 3).numpy()
positions = np.tile(np.arange(1, cfg.seq_len), best_cache["hidden_targets"].shape[0])

x_train, y_train, _ = flatten_for_readout(
    best_cache["block2"][:256], best_cache["hidden_targets"][:256], best_cache["component_ids"][:256]
)
x_test, _, _ = flatten_for_readout(
    best_cache["block2"][256:], best_cache["hidden_targets"][256:], best_cache["component_ids"][256:]
)
hidden_w = affine_lstsq_fit(x_train, y_train)
hidden_pred = renormalize_simplex(affine_lstsq_predict(x_test, hidden_w))

x_train_p, y_train_p, _ = flatten_for_readout(
    best_cache["block2"][:256], best_cache["process_targets"][:256], best_cache["component_ids"][:256]
)
x_test_p, _, _ = flatten_for_readout(
    best_cache["block2"][256:], best_cache["process_targets"][256:], best_cache["component_ids"][256:]
)
process_w = affine_lstsq_fit(x_train_p, y_train_p)
process_pred = renormalize_simplex(affine_lstsq_predict(x_test_p, process_w))

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
scatter = plot_simplex_points(
    axes[0, 0], hidden_truth[-len(hidden_pred) :], positions[-len(hidden_pred) :], "Ground-truth hidden beliefs", STATE_NAMES
)
plot_simplex_points(
    axes[0, 1], hidden_pred, positions[-len(hidden_pred) :], "Decoded hidden beliefs (block2, best)", STATE_NAMES
)
plot_simplex_points(
    axes[1, 0], process_truth[-len(process_pred) :], positions[-len(process_pred) :], "Ground-truth process beliefs", ["M1", "M2", "M3"]
)
plot_simplex_points(
    axes[1, 1], process_pred, positions[-len(process_pred) :], "Decoded process beliefs (block2, best)", ["M1", "M2", "M3"]
)
fig.colorbar(scatter, ax=axes.ravel().tolist(), label="prefix length")
plt.tight_layout()
plt.show()

metadata = {
    "config": asdict(cfg),
    "val_bayes_nll": val_bayes_nll,
    "test_bayes_nll": test_bayes_nll,
    "final_val_nll": final_val_metrics["loss"],
    "final_test_nll": final_test_metrics["loss"],
}
with open(OUTPUT_ROOT / "run_metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)
print(f"Saved metadata to {OUTPUT_ROOT / 'run_metadata.json'}")
