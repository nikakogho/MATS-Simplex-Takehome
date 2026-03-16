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
    geometry_plot_sequences_per_component: int = 3_000
    num_workers: int = 0
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


def select_balanced_indices(component_ids: torch.Tensor, total_size: int) -> torch.Tensor:
    if total_size >= len(component_ids):
        return torch.arange(len(component_ids))
    return balanced_cache_indices(component_ids, total_size)


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
    history["best_epoch"] = best_epoch
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
best_epoch = int(history["best_epoch"])

print(f"Best-model val NLL (epoch {best_epoch:02d}): {final_val_metrics['loss']:.4f}")
print(f"Best-model test NLL (epoch {best_epoch:02d}): {final_test_metrics['loss']:.4f}")
print(f"Val Bayes NLL: {val_bayes_nll:.4f}")
print(f"Test Bayes NLL: {test_bayes_nll:.4f}")

plt.figure(figsize=(8, 4))
plt.plot(history["train_loss"], marker="o", label="Train")
plt.plot(history["val_loss"], marker="o", label="Val")
plt.axhline(val_bayes_nll, color="green", linestyle="--", label="Val Bayes")
plt.xlabel("Epoch")
plt.ylabel("NLL")
plt.title(f"Training curves (best epoch {best_epoch:02d})")
plt.legend()
plt.show()

plt.figure(figsize=(8, 4))
positions = np.arange(1, cfg.seq_len)
plt.plot(positions, final_val_metrics["per_position"], marker="o", label="Model val")
plt.plot(positions, val_bayes_by_pos, marker="o", label="Bayes val")
plt.xlabel("Prediction position")
plt.ylabel("NLL")
plt.title(f"Model vs Bayes by position (best epoch {best_epoch:02d})")
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

print(f"Hidden-state readout results (best checkpoint = epoch {best_epoch:02d}):")
for layer_name in layers:
    print(layer_name, hidden_results["best"][layer_name])

print(f"Process readout results (best checkpoint = epoch {best_epoch:02d}):")
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
plt.xticks(xs, [checkpoint_label(name) for name in checkpoint_names], rotation=45)
plt.ylabel("R^2")
plt.title("Hidden-state readout quality across checkpoints")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(9, 4))
for layer_name in layers:
    checkpoint_names, xs, ys = metric_curve(process_results, layer_name, "global", "r2")
    plt.plot(xs, ys, marker="o", label=f"Process {layer_name}")
plt.xticks(xs, [checkpoint_label(name) for name in checkpoint_names], rotation=45)
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


def plot_simplex_points_fixed(ax, probs: np.ndarray, title: str, labels, color: str):
    coords = simplex_to_xy(probs)
    border = np.vstack([TRIANGLE, TRIANGLE[0]])
    ax.plot(border[:, 0], border[:, 1], color="tab:blue")
    ax.scatter(coords[:, 0], coords[:, 1], s=8, color=color, alpha=0.7)
    for idx, label in enumerate(labels):
        y = TRIANGLE[idx, 1] + (0.03 if idx == 2 else -0.03)
        ax.text(TRIANGLE[idx, 0], y, label, ha="center")
    ax.set_title(title)
    ax.set_axis_off()


def checkpoint_label(name: str) -> str:
    if name == "best":
        return f"best (epoch {best_epoch:02d})"
    return name


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
    axes[0, 1],
    hidden_pred,
    positions[-len(hidden_pred) :],
    f"Decoded hidden beliefs (block2, best epoch {best_epoch:02d})",
    STATE_NAMES,
)
plot_simplex_points(
    axes[1, 0], process_truth[-len(process_pred) :], positions[-len(process_pred) :], "Ground-truth process beliefs", ["M1", "M2", "M3"]
)
plot_simplex_points(
    axes[1, 1],
    process_pred,
    positions[-len(process_pred) :],
    f"Decoded process beliefs (block2, best epoch {best_epoch:02d})",
    ["M1", "M2", "M3"],
)
fig.colorbar(scatter, ax=axes.ravel().tolist(), label="prefix length")
plt.tight_layout()
plt.show()

metadata = {
    "config": asdict(cfg),
    "best_epoch": best_epoch,
    "val_bayes_nll": val_bayes_nll,
    "test_bayes_nll": test_bayes_nll,
    "final_val_nll": final_val_metrics["loss"],
    "final_test_nll": final_test_metrics["loss"],
}
with open(OUTPUT_ROOT / "run_metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)
print(f"Saved metadata to {OUTPUT_ROOT / 'run_metadata.json'}")


# %%
def build_sequence_split(n_seq: int, seed: int):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_seq)
    split = n_seq // 2
    return perm[:split], perm[split:]


def fit_readouts_for_cache(cache_blob: dict, layer_name: str, target_name: str, seed: int = 0):
    acts = cache_blob[layer_name]
    targets = cache_blob[target_name]
    component_ids = cache_blob["component_ids"]

    train_idx, test_idx = build_sequence_split(acts.shape[0], cfg.seed + seed)
    train_idx = torch.from_numpy(train_idx)
    test_idx = torch.from_numpy(test_idx)

    x_train, y_train, comp_train = flatten_for_readout(acts[train_idx], targets[train_idx], component_ids[train_idx])
    x_test, y_test, comp_test = flatten_for_readout(acts[test_idx], targets[test_idx], component_ids[test_idx])

    global_w = affine_lstsq_fit(x_train, y_train)
    per_component_w = {}
    for comp_id in sorted(np.unique(comp_train)):
        mask = comp_train == comp_id
        per_component_w[int(comp_id)] = affine_lstsq_fit(x_train[mask], y_train[mask])

    return {
        "global_w": global_w,
        "per_component_w": per_component_w,
        "test_acts": acts[test_idx].numpy(),
        "test_targets": targets[test_idx].numpy(),
        "test_component_ids": component_ids[test_idx].numpy(),
        "test_positions": np.tile(np.arange(1, acts.shape[1] + 1), len(test_idx)),
    }


def decode_test_split(bundle: dict, mode: str):
    test_acts = bundle["test_acts"]
    test_component_ids = bundle["test_component_ids"]
    flat_x = test_acts.reshape(-1, test_acts.shape[-1])
    flat_comp = np.repeat(test_component_ids, test_acts.shape[1])

    if mode == "global":
        flat_pred = affine_lstsq_predict(flat_x, bundle["global_w"])
    elif mode == "per_component":
        flat_pred = np.zeros((flat_x.shape[0], bundle["test_targets"].shape[-1]), dtype=np.float64)
        for comp_id, weights in bundle["per_component_w"].items():
            mask = flat_comp == comp_id
            flat_pred[mask] = affine_lstsq_predict(flat_x[mask], weights)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return renormalize_simplex(flat_pred)


def metrics_by_component(bundle: dict, mode: str):
    preds = decode_test_split(bundle, mode)
    truths = bundle["test_targets"].reshape(-1, bundle["test_targets"].shape[-1])
    comp_ids = np.repeat(bundle["test_component_ids"], bundle["test_acts"].shape[1])

    rows = []
    for comp_id in sorted(np.unique(comp_ids)):
        mask = comp_ids == comp_id
        metrics = regression_metrics(truths[mask], preds[mask])
        metrics["component_id"] = int(comp_id)
        metrics["component_name"] = components_by_id[int(comp_id)].name
        rows.append(metrics)
    return rows


def metrics_by_position(bundle: dict, mode: str):
    test_acts = bundle["test_acts"]
    test_targets = bundle["test_targets"]
    test_component_ids = bundle["test_component_ids"]
    rows = []
    for pos in range(test_acts.shape[1]):
        x = test_acts[:, pos, :]
        y = test_targets[:, pos, :]
        if mode == "global":
            pred = affine_lstsq_predict(x, bundle["global_w"])
        else:
            pred = np.zeros_like(y, dtype=np.float64)
            for comp_id, weights in bundle["per_component_w"].items():
                mask = test_component_ids == comp_id
                pred[mask] = affine_lstsq_predict(x[mask], weights)
        pred = renormalize_simplex(pred)
        metrics = regression_metrics(y, pred)
        metrics["position"] = pos + 1
        rows.append(metrics)
    return rows


def print_metric_rows(title: str, rows: list[dict]):
    print(title)
    for row in rows:
        prefix = row.get("component_name", f"pos {row.get('position', '?')}")
        print(f"  {prefix}: R^2={row['r2']:.3f}, MSE={row['mse']:.4f}, KL={row['kl']:.4f}")


def make_plot_cache(model: nn.Module, data: dict, total_size: int, label: str):
    plot_indices = select_balanced_indices(data["component_ids"], total_size)
    plot_inputs = data["tokens"][plot_indices][:, :-1]
    plot_hidden_targets = data["predictive_next"][plot_indices][:, : cfg.seq_len - 1]
    plot_process_targets = data["process_posterior"][plot_indices][:, : cfg.seq_len - 1]
    plot_component_ids = data["component_ids"][plot_indices]
    packed = capture_activation_cache(
        model,
        plot_inputs,
        plot_hidden_targets,
        plot_process_targets,
        plot_component_ids,
        label=label,
    )
    packed["selected_indices"] = plot_indices.clone()
    return packed


def build_reference_geometry(data: dict, total_size: int):
    indices = select_balanced_indices(data["component_ids"], total_size)
    filtered = data["filtered_after_obs"][indices].numpy()
    predictive = data["predictive_next"][indices].numpy()
    component_ids = data["component_ids"][indices].numpy()
    positions = np.tile(np.arange(1, cfg.seq_len + 1), filtered.shape[0])
    repeated_component_ids = np.repeat(component_ids, cfg.seq_len)
    return {
        "indices": indices,
        "filtered": filtered.reshape(-1, 3),
        "predictive": predictive.reshape(-1, 3),
        "component_ids": repeated_component_ids,
        "positions": positions,
        "n_sequences": filtered.shape[0],
    }


def decode_bundle(bundle: dict, global_w: np.ndarray, per_component_w: dict[int, np.ndarray], mode: str) -> np.ndarray:
    flat_x = bundle["block2"].reshape(-1, bundle["block2"].shape[-1]).numpy()
    flat_comp = np.repeat(bundle["component_ids"].numpy(), bundle["block2"].shape[1])

    if mode == "global":
        flat_pred = affine_lstsq_predict(flat_x, global_w)
    elif mode == "per_component":
        flat_pred = np.zeros((flat_x.shape[0], bundle["hidden_targets"].shape[-1]), dtype=np.float64)
        for comp_id, weights in per_component_w.items():
            mask = flat_comp == comp_id
            flat_pred[mask] = affine_lstsq_predict(flat_x[mask], weights)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return renormalize_simplex(flat_pred)


best_hidden_bundle = fit_readouts_for_cache(
    activation_caches["best"], layer_name="block2", target_name="hidden_targets", seed=101
)
hidden_global_rows = metrics_by_component(best_hidden_bundle, mode="global")
hidden_per_component_rows = metrics_by_component(best_hidden_bundle, mode="per_component")
print_metric_rows("Hidden belief decode by true component using one global readout", hidden_global_rows)
print_metric_rows("Hidden belief decode by true component using separate per-component readouts", hidden_per_component_rows)

large_plot_cache = make_plot_cache(
    model,
    train_data,
    total_size=cfg.geometry_plot_sequences_per_component * len(components),
    label="best_large_plot_cache",
)
hidden_truth = large_plot_cache["hidden_targets"].reshape(-1, 3).numpy()
hidden_positions = np.tile(np.arange(1, cfg.seq_len), large_plot_cache["hidden_targets"].shape[0])
hidden_comp_ids = np.repeat(large_plot_cache["component_ids"].numpy(), large_plot_cache["block2"].shape[1])
hidden_pred_global = decode_bundle(
    large_plot_cache,
    best_hidden_bundle["global_w"],
    best_hidden_bundle["per_component_w"],
    mode="global",
)
hidden_pred_component = decode_bundle(
    large_plot_cache,
    best_hidden_bundle["global_w"],
    best_hidden_bundle["per_component_w"],
    mode="per_component",
)

aligned_indices = large_plot_cache["selected_indices"]
aligned_filtered = train_data["filtered_after_obs"][aligned_indices][:, : cfg.seq_len - 1].numpy().reshape(-1, 3)
aligned_predictive = train_data["predictive_next"][aligned_indices][:, : cfg.seq_len - 1].numpy().reshape(-1, 3)
aligned_positions = np.tile(np.arange(1, cfg.seq_len), large_plot_cache["hidden_targets"].shape[0])
aligned_comp_ids = np.repeat(large_plot_cache["component_ids"].numpy(), cfg.seq_len - 1)

print(
    "Unified hidden-geometry comparison: ground-truth filtered_after_obs, "
    "ground-truth predictive_next, global decode, and per-component decode."
)
print(
    f"Using {large_plot_cache['hidden_targets'].shape[0]} matched sequences "
    f"({aligned_filtered.shape[0]} model-visible prefix points) in all four columns."
)
print("Columns:")
print("  1. Ground-truth filtered_after_obs = P(z_t | x_<=t)")
print("  2. Ground-truth predictive_next = P(z_(t+1) | x_<=t)")
print("  3. Global linear decode of predictive_next")
print("  4. Per-component linear decode of predictive_next")

fig, axes = plt.subplots(3, 4, figsize=(18, 14))
for row_idx, comp_id in enumerate(sorted(np.unique(aligned_comp_ids))):
    mask = aligned_comp_ids == comp_id
    comp_name = components_by_id[int(comp_id)].name
    scatter = plot_simplex_points(
        axes[row_idx, 0],
        aligned_filtered[mask],
        aligned_positions[mask],
        f"{comp_name}\nGround-truth filtered_after_obs",
        STATE_NAMES,
    )
    plot_simplex_points(
        axes[row_idx, 1],
        aligned_predictive[mask],
        aligned_positions[mask],
        f"{comp_name}\nGround-truth predictive_next",
        STATE_NAMES,
    )
    plot_simplex_points(
        axes[row_idx, 2],
        hidden_pred_global[mask],
        aligned_positions[mask],
        f"{comp_name}\nGlobal decode of predictive_next",
        STATE_NAMES,
    )
    plot_simplex_points(
        axes[row_idx, 3],
        hidden_pred_component[mask],
        aligned_positions[mask],
        f"{comp_name}\nPer-component decode of predictive_next",
        STATE_NAMES,
    )
fig.colorbar(scatter, ax=axes.ravel().tolist(), label="prefix length")
plt.tight_layout()
plt.show()


# %%
final_position = cfg.seq_len - 1
final_mask = hidden_positions == final_position

print(
    f"Plotting only the final available model-visible prefix position: {final_position}. "
    f"This corresponds to predicting token {cfg.seq_len} from prefix length {cfg.seq_len - 1}."
)

aligned_filtered_final = aligned_filtered[final_mask]
aligned_predictive_final = aligned_predictive[final_mask]
hidden_truth_final = hidden_truth[final_mask]
hidden_pred_global_final = hidden_pred_global[final_mask]
hidden_pred_component_final = hidden_pred_component[final_mask]
hidden_comp_ids_final = hidden_comp_ids[final_mask]
hidden_positions_final = hidden_positions[final_mask]

final_rows_global = []
final_rows_component = []
for comp_id in sorted(np.unique(hidden_comp_ids_final)):
    mask = hidden_comp_ids_final == comp_id
    comp_name = components_by_id[int(comp_id)].name
    global_metrics = regression_metrics(hidden_truth_final[mask], hidden_pred_global_final[mask])
    global_metrics["component_name"] = comp_name
    final_rows_global.append(global_metrics)
    component_metrics = regression_metrics(hidden_truth_final[mask], hidden_pred_component_final[mask])
    component_metrics["component_name"] = comp_name
    final_rows_component.append(component_metrics)

print_metric_rows("Final-prefix hidden belief decode by true component using one global readout", final_rows_global)
print_metric_rows(
    "Final-prefix hidden belief decode by true component using separate per-component readouts",
    final_rows_component,
)

print("Final-prefix columns:")
print("  1. Ground-truth filtered_after_obs at the last model-visible prefix")
print("  2. Ground-truth predictive_next at the last model-visible prefix")
print("  3. Global decode of predictive_next at the last model-visible prefix")
print("  4. Per-component decode of predictive_next at the last model-visible prefix")

fig, axes = plt.subplots(3, 4, figsize=(18, 14))
for row_idx, comp_id in enumerate(sorted(np.unique(hidden_comp_ids_final))):
    mask = hidden_comp_ids_final == comp_id
    comp_name = components_by_id[int(comp_id)].name
    plot_simplex_points_fixed(
        axes[row_idx, 0],
        aligned_filtered_final[mask],
        f"{comp_name}\nGround-truth filtered_after_obs (final prefix)",
        STATE_NAMES,
        color="tab:gray",
    )
    plot_simplex_points_fixed(
        axes[row_idx, 1],
        aligned_predictive_final[mask],
        f"{comp_name}\nGround-truth predictive_next (final prefix)",
        STATE_NAMES,
        color="tab:blue",
    )
    plot_simplex_points_fixed(
        axes[row_idx, 2],
        hidden_pred_global_final[mask],
        f"{comp_name}\nGlobal decode (final prefix)",
        STATE_NAMES,
        color="tab:orange",
    )
    plot_simplex_points_fixed(
        axes[row_idx, 3],
        hidden_pred_component_final[mask],
        f"{comp_name}\nPer-component decode (final prefix)",
        STATE_NAMES,
        color="tab:green",
    )
plt.tight_layout()
plt.show()


# %%
best_process_bundle = fit_readouts_for_cache(
    activation_caches["best"], layer_name="block2", target_name="process_targets", seed=202
)
process_global_by_position = metrics_by_position(best_process_bundle, mode="global")
hidden_global_by_position = metrics_by_position(best_hidden_bundle, mode="global")
hidden_component_by_position = metrics_by_position(best_hidden_bundle, mode="per_component")

print_metric_rows("Hidden belief decode by prefix position with one global readout", hidden_global_by_position)
print_metric_rows("Hidden belief decode by prefix position with per-component readouts", hidden_component_by_position)
print_metric_rows("Process belief decode by prefix position with one global readout", process_global_by_position)

positions = [row["position"] for row in hidden_global_by_position]
plt.figure(figsize=(10, 4))
plt.plot(positions, [row["r2"] for row in hidden_global_by_position], marker="o", label="Hidden global")
plt.plot(positions, [row["r2"] for row in hidden_component_by_position], marker="o", label="Hidden per-component")
plt.plot(positions, [row["r2"] for row in process_global_by_position], marker="o", label="Process global")
plt.xlabel("Prefix position")
plt.ylabel("R^2")
plt.title(f"Belief decodability by prefix position (block2, best epoch {best_epoch:02d})")
plt.legend()
plt.tight_layout()
plt.show()


# %%
def checkpoint_summary(results: dict, layer_name: str, metric: str = "r2"):
    rows = []
    for ckpt_name in results.keys():
        rows.append(
            {
                "checkpoint": ckpt_name,
                "hidden_global": results[ckpt_name][layer_name]["global"][metric],
                "hidden_per_component": hidden_results[ckpt_name][layer_name]["per_component_avg"][metric],
                "process_global": process_results[ckpt_name][layer_name]["global"][metric],
            }
        )
    return rows


block2_checkpoint_rows = checkpoint_summary(hidden_results, "block2", metric="r2")
print("Checkpoint summary for block2 readouts (R^2)")
for row in block2_checkpoint_rows:
    print(
        f"  {checkpoint_label(row['checkpoint'])}: hidden_global={row['hidden_global']:.3f}, "
        f"hidden_per_component={row['hidden_per_component']:.3f}, process_global={row['process_global']:.3f}"
    )

plt.figure(figsize=(10, 4))
checkpoints = [row["checkpoint"] for row in block2_checkpoint_rows]
xs = np.arange(len(checkpoints))
plt.plot(xs, [row["hidden_global"] for row in block2_checkpoint_rows], marker="o", label="Hidden global")
plt.plot(xs, [row["hidden_per_component"] for row in block2_checkpoint_rows], marker="o", label="Hidden per-component")
plt.plot(xs, [row["process_global"] for row in block2_checkpoint_rows], marker="o", label="Process global")
plt.xticks(xs, [checkpoint_label(name) for name in checkpoints], rotation=45)
plt.ylabel("R^2")
plt.title("Training-dynamics summary for block2 readouts")
plt.legend()
plt.tight_layout()
plt.show()


# %%
def decode_with_fixed_weights(
    acts: np.ndarray,
    component_ids: np.ndarray,
    global_w: np.ndarray,
    per_component_w: dict[int, np.ndarray] | None = None,
    mode: str = "global",
    target_dim: int = 3,
) -> np.ndarray:
    flat_x = acts.reshape(-1, acts.shape[-1])
    flat_comp = np.repeat(component_ids, acts.shape[1])

    if mode == "global":
        flat_pred = affine_lstsq_predict(flat_x, global_w)
    elif mode == "per_component":
        if per_component_w is None:
            raise ValueError("per_component_w is required for per-component decoding")
        flat_pred = np.zeros((flat_x.shape[0], target_dim), dtype=np.float64)
        for comp_id, weights in per_component_w.items():
            mask = flat_comp == comp_id
            flat_pred[mask] = affine_lstsq_predict(flat_x[mask], weights)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return renormalize_simplex(flat_pred)


def decode_cache_blob(cache_blob: dict, hidden_bundle: dict, process_bundle: dict):
    block2_acts = cache_blob["block2"].numpy()
    component_ids = cache_blob["component_ids"].numpy()
    positions = np.tile(np.arange(1, block2_acts.shape[1] + 1), block2_acts.shape[0])
    repeated_comp_ids = np.repeat(component_ids, block2_acts.shape[1])

    hidden_truth = cache_blob["hidden_targets"].numpy().reshape(-1, 3)
    process_truth = cache_blob["process_targets"].numpy().reshape(-1, 3)
    hidden_global = decode_with_fixed_weights(
        block2_acts,
        component_ids,
        global_w=hidden_bundle["global_w"],
        mode="global",
        target_dim=3,
    )
    hidden_component = decode_with_fixed_weights(
        block2_acts,
        component_ids,
        global_w=hidden_bundle["global_w"],
        per_component_w=hidden_bundle["per_component_w"],
        mode="per_component",
        target_dim=3,
    )
    process_global = decode_with_fixed_weights(
        block2_acts,
        component_ids,
        global_w=process_bundle["global_w"],
        mode="global",
        target_dim=3,
    )
    return {
        "positions": positions,
        "component_ids": repeated_comp_ids,
        "hidden_truth": hidden_truth,
        "hidden_global": hidden_global,
        "hidden_per_component": hidden_component,
        "process_truth": process_truth,
        "process_global": process_global,
    }

def make_checkpoint_grid_axes(n_panels: int, ncols: int = 4, figsize_per_cell=(4.2, 4.0)):
    nrows = math.ceil(n_panels / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(figsize_per_cell[0] * ncols, figsize_per_cell[1] * nrows))
    axes = np.atleast_1d(axes).reshape(nrows, ncols)
    return fig, axes


decoded_checkpoints = {}
for checkpoint_name, cache_blob in activation_caches.items():
    decoded_checkpoints[checkpoint_name] = decode_cache_blob(
        cache_blob=cache_blob,
        hidden_bundle=best_hidden_bundle,
        process_bundle=best_process_bundle,
    )

print(
    f"Checkpoint grids below use the fixed block2 extractor matrices fitted on the best checkpoint "
    f"(epoch {best_epoch:02d}) and apply them to each cached checkpoint."
)
for checkpoint_name, decoded in decoded_checkpoints.items():
    print(
        f"  {checkpoint_label(checkpoint_name)}: "
        f"hidden_global={regression_metrics(decoded['hidden_truth'], decoded['hidden_global'])['r2']:.3f}, "
        f"hidden_per_component={regression_metrics(decoded['hidden_truth'], decoded['hidden_per_component'])['r2']:.3f}, "
        f"process_global={regression_metrics(decoded['process_truth'], decoded['process_global'])['r2']:.3f}"
    )


checkpoint_names = list(decoded_checkpoints.keys())

fig, axes = make_checkpoint_grid_axes(len(checkpoint_names) + 1, ncols=4)
scatter = plot_simplex_points(
    axes.flat[0],
    decoded_checkpoints["best"]["process_truth"],
    decoded_checkpoints["best"]["positions"],
    "Ground truth\nProcess belief",
    ["M1", "M2", "M3"],
)
for ax, checkpoint_name in zip(axes.flat[1:], checkpoint_names):
    decoded = decoded_checkpoints[checkpoint_name]
    plot_simplex_points(
        ax,
        decoded["process_global"],
        decoded["positions"],
        f"{checkpoint_label(checkpoint_name)}\nDecoded process belief",
        ["M1", "M2", "M3"],
    )
for ax in axes.flat[len(checkpoint_names) + 1 :]:
    ax.set_axis_off()
fig.suptitle("Which process am I in? belief across checkpoints", y=0.995)
fig.colorbar(scatter, ax=axes.ravel().tolist(), label="prefix length")
plt.tight_layout()
plt.show()


for comp_id in sorted(components_by_id.keys()):
    comp_name = components_by_id[comp_id].name
    fig, axes = make_checkpoint_grid_axes(len(checkpoint_names) + 1, ncols=4)
    truth_mask = decoded_checkpoints["best"]["component_ids"] == comp_id
    scatter = plot_simplex_points(
        axes.flat[0],
        decoded_checkpoints["best"]["hidden_truth"][truth_mask],
        decoded_checkpoints["best"]["positions"][truth_mask],
        f"{comp_name}\nGround truth hidden belief",
        STATE_NAMES,
    )
    for ax, checkpoint_name in zip(axes.flat[1:], checkpoint_names):
        decoded = decoded_checkpoints[checkpoint_name]
        mask = decoded["component_ids"] == comp_id
        plot_simplex_points(
            ax,
            decoded["hidden_per_component"][mask],
            decoded["positions"][mask],
            f"{checkpoint_label(checkpoint_name)}\nDecoded hidden belief",
            STATE_NAMES,
        )
    for ax in axes.flat[len(checkpoint_names) + 1 :]:
        ax.set_axis_off()
    fig.suptitle(f"Hidden belief geometry across checkpoints for {comp_name}", y=0.995)
    fig.colorbar(scatter, ax=axes.ravel().tolist(), label="prefix length")
    plt.tight_layout()
    plt.show()


# %%
process_best_rows = metrics_by_component(best_process_bundle, mode="global")
print_metric_rows(
    "Process belief decode by true generating component using one global readout",
    process_best_rows,
)

process_plot_cache = make_plot_cache(
    model,
    train_data,
    total_size=cfg.geometry_plot_sequences_per_component * len(components),
    label="best_process_plot_cache",
)
process_truth_large = process_plot_cache["process_targets"].reshape(-1, 3).numpy()
process_positions_large = np.tile(np.arange(1, cfg.seq_len), process_plot_cache["process_targets"].shape[0])
process_comp_ids_large = np.repeat(process_plot_cache["component_ids"].numpy(), process_plot_cache["block2"].shape[1])
process_pred_large = decode_with_fixed_weights(
    process_plot_cache["block2"].numpy(),
    process_plot_cache["component_ids"].numpy(),
    global_w=best_process_bundle["global_w"],
    mode="global",
    target_dim=3,
)

print(
    "Process-belief geometry at the best checkpoint, split by the true generating component."
)
print(
    f"Using {process_plot_cache['process_targets'].shape[0]} matched sequences "
    f"({process_truth_large.shape[0]} model-visible prefix points)."
)
print("Columns:")
print("  1. Ground-truth process belief for sequences truly generated by this component")
print("  2. Decoded process belief from block2 using the best global process readout")

fig, axes = plt.subplots(3, 2, figsize=(12, 14))
for row_idx, comp_id in enumerate(sorted(np.unique(process_comp_ids_large))):
    mask = process_comp_ids_large == comp_id
    comp_name = components_by_id[int(comp_id)].name
    scatter = plot_simplex_points(
        axes[row_idx, 0],
        process_truth_large[mask],
        process_positions_large[mask],
        f"{comp_name}\nGround-truth process belief",
        ["M1", "M2", "M3"],
    )
    plot_simplex_points(
        axes[row_idx, 1],
        process_pred_large[mask],
        process_positions_large[mask],
        f"{comp_name}\nDecoded process belief",
        ["M1", "M2", "M3"],
    )
fig.colorbar(scatter, ax=axes.ravel().tolist(), label="prefix length")
plt.tight_layout()
plt.show()


# %%
print(
    "Process-belief checkpoint grids below are split by the true generating component."
)
print(
    "Each grid uses the fixed best-checkpoint block2 process extractor and only shows "
    "sequences that were actually generated by that component."
)

for comp_id in sorted(components_by_id.keys()):
    comp_name = components_by_id[comp_id].name
    print(f"{comp_name}")
    for checkpoint_name in checkpoint_names:
        decoded = decoded_checkpoints[checkpoint_name]
        mask = decoded["component_ids"] == comp_id
        metrics = regression_metrics(decoded["process_truth"][mask], decoded["process_global"][mask])
        print(
            f"  {checkpoint_label(checkpoint_name)}: "
            f"R^2={metrics['r2']:.3f}, MSE={metrics['mse']:.4f}, KL={metrics['kl']:.4f}"
        )

    fig, axes = make_checkpoint_grid_axes(len(checkpoint_names) + 1, ncols=4)
    truth_mask = decoded_checkpoints["best"]["component_ids"] == comp_id
    scatter = plot_simplex_points(
        axes.flat[0],
        decoded_checkpoints["best"]["process_truth"][truth_mask],
        decoded_checkpoints["best"]["positions"][truth_mask],
        f"{comp_name}\nGround truth process belief",
        ["M1", "M2", "M3"],
    )
    for ax, checkpoint_name in zip(axes.flat[1:], checkpoint_names):
        decoded = decoded_checkpoints[checkpoint_name]
        mask = decoded["component_ids"] == comp_id
        plot_simplex_points(
            ax,
            decoded["process_global"][mask],
            decoded["positions"][mask],
            f"{checkpoint_label(checkpoint_name)}\nDecoded process belief",
            ["M1", "M2", "M3"],
        )
    for ax in axes.flat[len(checkpoint_names) + 1 :]:
        ax.set_axis_off()
    fig.suptitle(f"Process-belief geometry across checkpoints for true {comp_name} sequences", y=0.995)
    fig.colorbar(scatter, ax=axes.ravel().tolist(), label="prefix length")
    plt.tight_layout()
    plt.show()


# %%
hidden_best_rows_global = metrics_by_component(best_hidden_bundle, mode="global")
hidden_best_rows_component = metrics_by_component(best_hidden_bundle, mode="per_component")
print_metric_rows(
    "Hidden-state belief decode by true generating component using one global readout",
    hidden_best_rows_global,
)
print_metric_rows(
    "Hidden-state belief decode by true generating component using separate per-component readouts",
    hidden_best_rows_component,
)

hidden_process_plot_cache = make_plot_cache(
    model,
    train_data,
    total_size=cfg.geometry_plot_sequences_per_component * len(components),
    label="best_hidden_process_plot_cache",
)
hidden_truth_large = hidden_process_plot_cache["hidden_targets"].reshape(-1, 3).numpy()
hidden_positions_large = np.tile(np.arange(1, cfg.seq_len), hidden_process_plot_cache["hidden_targets"].shape[0])
hidden_comp_ids_large = np.repeat(hidden_process_plot_cache["component_ids"].numpy(), hidden_process_plot_cache["block2"].shape[1])
hidden_pred_large_global = decode_with_fixed_weights(
    hidden_process_plot_cache["block2"].numpy(),
    hidden_process_plot_cache["component_ids"].numpy(),
    global_w=best_hidden_bundle["global_w"],
    mode="global",
    target_dim=3,
)
hidden_pred_large_component = decode_with_fixed_weights(
    hidden_process_plot_cache["block2"].numpy(),
    hidden_process_plot_cache["component_ids"].numpy(),
    global_w=best_hidden_bundle["global_w"],
    per_component_w=best_hidden_bundle["per_component_w"],
    mode="per_component",
    target_dim=3,
)

print(
    "Hidden-state geometry at the best checkpoint, split by the true generating component."
)
print(
    f"Using {hidden_process_plot_cache['hidden_targets'].shape[0]} matched sequences "
    f"({hidden_truth_large.shape[0]} model-visible prefix points)."
)
print("Columns:")
print("  1. Ground-truth predictive_next hidden belief for sequences truly generated by this component")
print("  2. Decoded hidden belief from block2 using one global hidden readout")
print("  3. Decoded hidden belief from block2 using separate per-component hidden readouts")

fig, axes = plt.subplots(3, 3, figsize=(15, 14))
for row_idx, comp_id in enumerate(sorted(np.unique(hidden_comp_ids_large))):
    mask = hidden_comp_ids_large == comp_id
    comp_name = components_by_id[int(comp_id)].name
    scatter = plot_simplex_points(
        axes[row_idx, 0],
        hidden_truth_large[mask],
        hidden_positions_large[mask],
        f"{comp_name}\nGround-truth hidden belief",
        STATE_NAMES,
    )
    plot_simplex_points(
        axes[row_idx, 1],
        hidden_pred_large_global[mask],
        hidden_positions_large[mask],
        f"{comp_name}\nDecoded hidden belief (global)",
        STATE_NAMES,
    )
    plot_simplex_points(
        axes[row_idx, 2],
        hidden_pred_large_component[mask],
        hidden_positions_large[mask],
        f"{comp_name}\nDecoded hidden belief (per-component)",
        STATE_NAMES,
    )
fig.colorbar(scatter, ax=axes.ravel().tolist(), label="prefix length")
plt.tight_layout()
plt.show()


# %%
print(
    "Hidden-state checkpoint grids below are split by the true generating component."
)
print(
    "For each component, the top row uses one global hidden-state readout and the bottom row "
    "uses separate per-component hidden-state readouts."
)

for comp_id in sorted(components_by_id.keys()):
    comp_name = components_by_id[comp_id].name
    print(f"{comp_name}")
    for checkpoint_name in checkpoint_names:
        decoded = decoded_checkpoints[checkpoint_name]
        mask = decoded["component_ids"] == comp_id
        global_metrics = regression_metrics(decoded["hidden_truth"][mask], decoded["hidden_global"][mask])
        component_metrics = regression_metrics(decoded["hidden_truth"][mask], decoded["hidden_per_component"][mask])
        print(
            f"  {checkpoint_label(checkpoint_name)}: "
            f"global_R^2={global_metrics['r2']:.3f}, global_MSE={global_metrics['mse']:.4f}, "
            f"per_component_R^2={component_metrics['r2']:.3f}, per_component_MSE={component_metrics['mse']:.4f}"
        )

    ncols = len(checkpoint_names) + 1
    fig, axes = plt.subplots(2, ncols, figsize=(3.2 * ncols, 8.0))
    axes = np.atleast_2d(axes)
    truth_mask = decoded_checkpoints["best"]["component_ids"] == comp_id
    scatter = plot_simplex_points(
        axes[0, 0],
        decoded_checkpoints["best"]["hidden_truth"][truth_mask],
        decoded_checkpoints["best"]["positions"][truth_mask],
        f"{comp_name}\nGround truth",
        STATE_NAMES,
    )
    plot_simplex_points(
        axes[1, 0],
        decoded_checkpoints["best"]["hidden_truth"][truth_mask],
        decoded_checkpoints["best"]["positions"][truth_mask],
        f"{comp_name}\nGround truth",
        STATE_NAMES,
    )

    for col_idx, checkpoint_name in enumerate(checkpoint_names, start=1):
        decoded = decoded_checkpoints[checkpoint_name]
        mask = decoded["component_ids"] == comp_id
        plot_simplex_points(
            axes[0, col_idx],
            decoded["hidden_global"][mask],
            decoded["positions"][mask],
            f"{checkpoint_label(checkpoint_name)}\nGlobal decode",
            STATE_NAMES,
        )
        plot_simplex_points(
            axes[1, col_idx],
            decoded["hidden_per_component"][mask],
            decoded["positions"][mask],
            f"{checkpoint_label(checkpoint_name)}\nPer-component decode",
            STATE_NAMES,
        )

    fig.suptitle(f"Hidden-state belief across checkpoints for true {comp_name} sequences", y=0.995)
    fig.colorbar(scatter, ax=axes.ravel().tolist(), label="prefix length")
    plt.tight_layout()
    plt.show()
