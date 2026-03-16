# MATS Simplex Takehome

## Overview

I study a non-ergodic sequence dataset formed by mixing three Mess3 hidden Markov processes across sequences. Each length-16 sequence is generated entirely by one component, and all components share the same visible alphabet `{A, B, C}`. I train a small causal transformer by next-token prediction and compare its residual-stream activations to exact Bayes belief targets.

The central question is whether the transformer's internal representations reflect the geometry of:

- hidden-state belief within a component
- process-identity belief over which component generated the sequence

## Choosing The Mess3 Components

Selecting the three Mess3 processes was itself an important part of the project. My first goal was to keep the setup faithful to the intended latent-inference problem: all components should share the same visible alphabet `{A, B, C}`, and component identity should not be trivially recoverable from surface form. At the same time, the components needed to be different enough that a length-16 sequence would contain meaningful evidence about which process generated it.

This created a tradeoff. The original single-process Mess3 examples in the Simplex work emphasize visually rich, often fractal belief geometry. For this take-home, however, I cared more about a clean multi-process inference problem than about maximizing Sierpinski-like self-similarity in every component. I therefore chose parameters to improve **component identifiability at sequence length 16**, even if that moved the system away from the prettiest single-process regime.

I treated

- `c0 = (x=0.05, alpha=0.90)`

as a baseline sticky, fairly informative component, and then searched over candidate `c1` and `c2` values using an exact Bayes diagnostic. For each candidate triple, I sampled sequences of length 16, computed the posterior over components `P(M | x_<=16)` using exact forward filtering, and ranked parameter choices by final component-identifiability metrics.

This search showed a clear pattern:

- stronger `c1` candidates wanted both higher switching and clearer emissions
- `c2` wanted much noisier emissions
- among the best candidates, the precise value of `c2.x` mattered less than the large drop in `c2.alpha`

The top search results all had roughly the same form:

- `c1` near `(x≈0.18, alpha≈0.95)`
- `c2` near `(x in {0.03, 0.05, 0.07, 0.09}, alpha≈0.45)`

I ultimately chose:

- `c0 = (0.05, 0.90)`
- `c1 = (0.18, 0.95)`
- `c2 = (0.05, 0.45)`

even though some nearby variants had nearly identical identifiability scores. I chose `c2.x = 0.05` specifically because it keeps `c0` and `c2` matched in transition dynamics, so the contrast between them isolates **emission clarity** rather than mixing transition and emission changes at the same time. This makes the final interpretation cleaner:

- `c0` vs `c1` emphasizes transition and emission sharpness differences
- `c0` vs `c2` emphasizes emission-noise differences with matched transition dynamics

Under exact Bayes inference on length-16 prefixes, this final choice gave:

- average max component posterior at `t=16`: `0.735`
- average true-component posterior at `t=16`: `0.643`

These values are not near-perfect, but they are strong enough to make component inference meaningful while leaving real uncertainty in the task. That makes the setup well suited to studying whether the transformer learns both next-token prediction and latent process inference.

![HMM](../assets/HMMs.png)

## Dataset And Ground-Truth Beliefs

The final Mess3 components are:

| Component | x | alpha | Informal role |
| --- | ---: | ---: | --- |
| `mess3_c0_x005_a090` | 0.05 | 0.90 | sticky, fairly informative |
| `mess3_c1_x018_a095` | 0.18 | 0.95 | less sticky, very informative |
| `mess3_c2_x005_a045` | 0.05 | 0.45 | sticky, weakly informative |

These parameters were chosen to make component identity meaningfully inferable by length 16 while keeping the task nontrivial.

For a fixed component, I track:

- $b_t = P(z_t | x_{<=t})$ (`filtered_after_obs`)
- $q_(t+1) = P(z_(t+1) | x_{<=t})$ (`predictive_next`)

using the row-vector update:

- $b_t(i) proportional to q_t(i) E_{i, x_t}$
- $q_(t+1) = b_t A$

I also track process identity:

- $P(M = m | x_{<=t}) proportional to P(x_{<=t} | M = m) P(M = m)$

where component likelihoods are computed exactly by forward filtering. The main model-aligned target is `predictive_next`, because the transformer is trained to predict the next token.

![Hidden-state ground-truth geometry](../assets/geometry_each_prefix_11.png)

![Process-identity ground-truth simplex](../assets/sampled_which_mess3.png)

## Model And Training

- Dataset size: 30k train / 3k val / 3k test, balanced across components
- Sequence length: 16
- Model: 2-layer pre-LN causal transformer, `d_model=64`, 2 heads, `d_mlp=256`
- Objective: next-token prediction on all 15 model-visible positions
- Best checkpoint: epoch 21

## Prediction Quality

The transformer gets very close to the exact Bayes next-token baseline on held-out data:

| Metric | Value |
| --- | ---: |
| Best epoch | 21 |
| Val NLL | 0.9488 |
| Test NLL | 0.9519 |
| Bayes val NLL | 0.9465 |
| Bayes test NLL | 0.9495 |

This suggests the model learns most of the predictive structure available in the dataset. In other words, the task is neither trivial nor badly underfit: the trained model is already close to the Bayes ceiling.

![Bayes vs transformer](../assets/Bayes_vs_transformer.png)

[Insert Figure: Training curves here]

[Insert Figure: Model vs Bayes by position here]

## Hidden-State Beliefs In The Residual Stream

At the best checkpoint, block-2 residual activations support strong linear decoding of hidden-state beliefs:

| Readout target | Block2 R^2 |
| --- | ---: |
| Hidden belief, global linear readout | 0.864 |
| Hidden belief, per-component linear readouts (avg) | 0.943 |

The gap between the global hidden readout and the per-component hidden readouts is substantial. This argues against one perfectly shared hidden-belief coordinate system across all three components.

By true component over all model-visible prefixes:

| Component | Global R^2 | Per-component R^2 |
| --- | ---: | ---: |
| `mess3_c0_x005_a090` | 0.917 | 0.976 |
| `mess3_c1_x018_a095` | 0.841 | 0.997 |
| `mess3_c2_x005_a045` | 0.301 | 0.853 |

The noisy component `c2` is the clearest case where a single shared linear map underperforms, while a per-component map still recovers the target well.

[Insert Figure: Unified hidden-geometry comparison here]

## Process Identity In The Residual Stream

Process identity is weakly decodable in embeddings but becomes strongly decodable in deeper layers:

| Layer | Process-belief global R^2 |
| --- | ---: |
| Embed | -0.082 |
| Block1 | 0.396 |
| Block2 | 0.817 |

This supports the claim that the network learns an explicit representation of $P(M | x_{<=t})$, rather than only tracking hidden-state belief inside a single assumed process.

[Insert Figure: Ground-truth vs decoded process-belief geometry here]

## Additional Analysis: Global Vs Per-Component Readouts

My main additional analysis compares one global linear readout against separate per-component linear readouts.

I chose this analysis because it directly tests whether the learned geometry is shared across components or partly component-specific. One global readout asks whether the network uses one common coordinate system for all hidden-state beliefs. Separate per-component readouts ask whether each component is internally clean but represented in somewhat different coordinates.

The result is asymmetric:

- hidden-state belief is much better decoded by separate per-component linear maps than by one global map
- process identity is well decoded by one global linear map

This points to a partially factorized representation:

- shared-ish coordinates for "which process am I in?"
- more component-specific coordinates for "what hidden-state belief do I have within this process?"

That is more interesting than either extreme of fully separate manifolds or one fully shared hidden simplex.

## Training Dynamics

The training-dynamics story needs a careful interpretation.

If I fit a fresh linear probe separately at each checkpoint, I get the following block-2 summary:

| Checkpoint | Hidden global R^2 | Hidden per-component R^2 | Process global R^2 |
| --- | ---: | ---: | ---: |
| init | 0.811 | 0.938 | 0.194 |
| epoch 1 | 0.846 | 0.898 | 0.466 |
| epoch 15 | 0.864 | 0.935 | 0.808 |
| epoch 29 | 0.863 | 0.941 | 0.795 |
| epoch 43 | 0.856 | 0.925 | 0.739 |
| best (epoch 21) | 0.864 | 0.943 | 0.817 |

Taken naively, these numbers make hidden belief look highly decodable even at initialization. I do **not** interpret that as evidence that the untrained model already implements correct belief updates. A safer interpretation is that the hidden-belief target is strongly correlated with local token evidence, especially in the high-alpha components, and a fresh probe can recover part of that signal from random token and positional features.

To test whether the **learned geometry itself** is already present at initialization, I also apply the fixed best-checkpoint block-2 extractor back to earlier checkpoints:

| Checkpoint | Hidden global R^2 | Hidden per-component R^2 | Process global R^2 |
| --- | ---: | ---: | ---: |
| init | 0.087 | 0.545 | -1.245 |
| epoch 1 | 0.265 | 0.583 | -1.147 |
| epoch 15 | 0.807 | 0.896 | 0.492 |
| epoch 29 | 0.810 | 0.880 | 0.567 |
| epoch 43 | 0.641 | 0.644 | -0.046 |
| best (epoch 21) | 0.863 | 0.975 | 0.818 |

This is the more trustworthy picture of learned representational alignment. The best-checkpoint extractor does **not** work at initialization, and only starts to work well later in training. So the final learned geometry is not simply present in the random network from the start.

The overall training-dynamics conclusion is:

- fresh probes can recover some local hidden-state signal from early random features
- the aligned block-2 geometry for both hidden belief and process belief emerges through training
- process identity is the clearest thing the model learns strongly from scratch

[Insert Figure: Training-dynamics summary across checkpoints here]

[Insert Figure: Checkpoint grids with fixed best-checkpoint extractor here]

## Assessment Of Preregistered Predictions

### P0: Hidden-state beliefs will be linearly decodable from the residual stream

Supported. Later residual activations, especially in block 2, decode hidden-state belief strongly.

### P0.1: Activations will organize more like a union of component-specific hidden geometries than one shared geometry

Supported. Per-component hidden readouts outperform the global hidden readout by a meaningful margin, especially on the noisy component `c2`.

### P0.2: Alternative factorized/shared representation

Partially supported. The model does not use one fully shared hidden-belief geometry, but the process-belief results suggest shared factorization does exist. The best interpretation is neither fully separate nor fully shared, but partially factorized.

### P1: Process-identity beliefs will also be linearly decodable

Supported. Block-2 activations support strong global decoding of $P(M | x_{<=t})$, and the decoded process-belief geometry qualitatively matches the skewed ground-truth process simplex.

### P2: Training will move from simpler/shared structure toward separated process-specific structure

Partially supported, with an important revision. A fresh probe can recover some hidden-belief signal even at initialization, but the fixed best-checkpoint extractor shows that the learned geometry itself is not present at init and emerges through training. The strongest training effect is on sequence-level component inference.

## Conclusion

This experiment shows that a small transformer trained on a non-ergodic mixture of Mess3 processes can learn near-Bayes-optimal next-token prediction while developing residual-stream geometry that reflects both hidden-state belief and process identity. The learned representation is not best described as one completely shared simplex or three completely separate manifolds. Instead, it appears partially factorized: process identity is decoded well by a global linear map, while hidden-state belief is decoded much better by per-component maps.

If I had time for one more compute-heavy step, I would run one additional random seed at the current dataset size rather than a 3x larger dataset. That would test stability more directly than scaling the data when the current run is already close to the Bayes ceiling.
