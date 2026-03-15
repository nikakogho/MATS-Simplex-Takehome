from dataclasses import dataclass
import numpy as np


@dataclass
class Mess3Component:
    name: str
    component_id: int
    pi: np.ndarray        # [K]
    A: np.ndarray         # [K, K]
    E: np.ndarray         # [K, V]
    vocab_size: int

    def validate(self):
        K = self.pi.shape[0]
        assert self.pi.shape == (K,)
        assert self.A.shape == (K, K)
        assert self.E.shape[0] == K
        assert self.E.shape[1] == self.vocab_size

        assert np.all(self.pi >= 0)
        assert np.all(self.A >= 0)
        assert np.all(self.E >= 0)

        assert np.isclose(self.pi.sum(), 1.0)
        assert np.allclose(self.A.sum(axis=1), 1.0)
        assert np.allclose(self.E.sum(axis=1), 1.0)


def sample_categorical(probs: np.ndarray, rng: np.random.Generator) -> int:
    return int(rng.choice(len(probs), p=probs))


def sample_sequence(component: Mess3Component, seq_len: int, rng: np.random.Generator):
    component.validate()

    K = component.pi.shape[0]
    tokens = np.zeros(seq_len, dtype=np.int64)
    states = np.zeros(seq_len, dtype=np.int64)

    z = sample_categorical(component.pi, rng)

    for t in range(seq_len):
        states[t] = z
        x = sample_categorical(component.E[z], rng)
        tokens[t] = x

        if t < seq_len - 1:
            z = sample_categorical(component.A[z], rng)

    return tokens, states


def sample_balanced_mixture_dataset(
    components,
    n_sequences_per_component: int,
    seq_len: int,
    rng: np.random.Generator,
    store_states: bool = True,
):
    tokens_list = []
    states_list = []
    component_ids = []

    for comp in components:
        for _ in range(n_sequences_per_component):
            tokens, states = sample_sequence(comp, seq_len, rng)
            tokens_list.append(tokens)
            if store_states:
                states_list.append(states)
            component_ids.append(comp.component_id)

    tokens = np.stack(tokens_list, axis=0)
    component_ids = np.array(component_ids, dtype=np.int64)

    out = {
        "tokens": tokens,
        "component_ids": component_ids,
    }

    if store_states:
        out["states"] = np.stack(states_list, axis=0)

    return out


def shuffled_indices(n: int, rng: np.random.Generator):
    idx = np.arange(n)
    rng.shuffle(idx)
    return idx


def split_by_sequence(dataset, n_train: int, n_val: int, n_test: int, rng: np.random.Generator):
    N = dataset["tokens"].shape[0]
    assert n_train + n_val + n_test == N

    idx = shuffled_indices(N, rng)

    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]

    def take(split_idx):
        out = {}
        for k, v in dataset.items():
            out[k] = v[split_idx]
        return out

    return take(train_idx), take(val_idx), take(test_idx)