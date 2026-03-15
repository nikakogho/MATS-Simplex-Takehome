import numpy as np


def normalize(prob: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    s = prob.sum()
    if s < eps:
        raise ValueError(f"Probability vector nearly zero; sum={s}")
    return prob / s


def compute_beliefs_for_sequence(tokens, pi, A, E):
    """
    Row-vector convention.

    Args:
        tokens: [T] integer tokens
        pi: [K]
        A: [K, K] with A[i, j] = P(z_{t+1}=j | z_t=i)
        E: [K, V] with E[i, x] = P(x_t=x | z_t=i)

    Returns dict:
        predictive_before_obs: [T, K], q_t = P(z_t | x_{<t})
        filtered_after_obs:   [T, K], b_t = P(z_t | x_{<=t})
        predictive_next:      [T, K], q_{t+1} = P(z_{t+1} | x_{<=t})
        obs_prob:             [T], P(x_t | x_{<t})
        loglik:               scalar
    """
    T = len(tokens)
    K = len(pi)

    predictive_before_obs = np.zeros((T, K), dtype=np.float64)
    filtered_after_obs = np.zeros((T, K), dtype=np.float64)
    predictive_next = np.zeros((T, K), dtype=np.float64)
    obs_prob = np.zeros(T, dtype=np.float64)

    q = pi.astype(np.float64).copy()

    for t, x in enumerate(tokens):
        predictive_before_obs[t] = q

        emission_probs = E[:, x]
        unnorm = q * emission_probs
        p_xt = unnorm.sum()
        obs_prob[t] = p_xt

        b = normalize(unnorm)
        filtered_after_obs[t] = b

        q = b @ A
        predictive_next[t] = q

    loglik = np.log(obs_prob).sum()

    return {
        "predictive_before_obs": predictive_before_obs,
        "filtered_after_obs": filtered_after_obs,
        "predictive_next": predictive_next,
        "obs_prob": obs_prob,
        "loglik": loglik,
    }


def compute_beliefs_for_dataset(tokens_batch, components_by_id, component_ids):
    """
    Args:
        tokens_batch: [N, T]
        components_by_id: dict[int -> Mess3Component]
        component_ids: [N]

    Returns dict with arrays:
        predictive_before_obs: [N, T, K]
        filtered_after_obs:    [N, T, K]
        predictive_next:       [N, T, K]
        loglik:                [N]
    """
    N, T = tokens_batch.shape
    first_comp = next(iter(components_by_id.values()))
    K = first_comp.pi.shape[0]

    predictive_before_obs = np.zeros((N, T, K), dtype=np.float64)
    filtered_after_obs = np.zeros((N, T, K), dtype=np.float64)
    predictive_next = np.zeros((N, T, K), dtype=np.float64)
    loglik = np.zeros(N, dtype=np.float64)

    for i in range(N):
        comp = components_by_id[int(component_ids[i])]
        out = compute_beliefs_for_sequence(tokens_batch[i], comp.pi, comp.A, comp.E)
        predictive_before_obs[i] = out["predictive_before_obs"]
        filtered_after_obs[i] = out["filtered_after_obs"]
        predictive_next[i] = out["predictive_next"]
        loglik[i] = out["loglik"]

    return {
        "predictive_before_obs": predictive_before_obs,
        "filtered_after_obs": filtered_after_obs,
        "predictive_next": predictive_next,
        "loglik": loglik,
    }