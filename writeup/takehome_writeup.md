# MATS Simplex Takehome

## Overview

I study a non-ergodic dataset formed by mixing three Mess3 hidden Markov processes across sequences. Each length-16 sequence is generated entirely by one component, and all components share the visible alphabet `{A, B, C}`. I train a small causal transformer by next-token prediction and compare its residual-stream activations to exact Bayes belief targets.

The main question is whether the learned geometry reflects:

- hidden-state belief within a component
- process-identity belief over which component generated the sequence

Purpose of solving this toy problem is to get a deeper understanding of how neural networks, particularly transformers (and specifically LLMs) might represent concepts in their activations. This could one day have a potential of scaling up to frontier LLMs and letting us see into their "thought process" and detect deception, hidden goals, and other intent in a more precise way than we currently do with state of the art mech interp techniques like sparse autoencoders, linear probes, steering vectors, activation oracles and others (although some of these methods potentially overlap with this kind of approach or could be enhanced by applying them together, but that is outside this project's scope).

## Choosing The Mess3 Components

Choosing the three Mess3 processes was part of the experiment. I wanted the components to share the same visible alphabet so that process identity could not be read off from surface form, but I also needed them to be distinguishable enough that a length-16 sequence would contain meaningful evidence about which component generated it.

This meant giving up on reproducing the cleanest single-process Sierpinski-like regime and instead optimizing for a stronger multi-process inference problem. I fixed

- `c0 = (x=0.05, alpha=0.90)`

as a baseline sticky, fairly informative component, and then searched over candidate `c1` and `c2` values using an exact Bayes component-identifiability diagnostic. For each candidate triple, I sampled length-16 sequences, computed `P(M \mid x_{\le 16})` by exact forward filtering, and ranked parameter choices by final component posterior quality.

The search consistently favored:

- a sharper, faster-switching `c1`
- a much noisier `c2`

with top candidates clustered around:

- `c1` near `(x=0.18, alpha=0.95)`
- `c2` near `(x in {0.03, 0.05, 0.07, 0.09}, alpha=0.45)`

I chose the final set

- `c0 = (0.05, 0.90)`
- `c1 = (0.18, 0.95)`
- `c2 = (0.05, 0.45)`

because it keeps `c0` and `c2` matched in transition dynamics, making that comparison primarily an emission-noise contrast. Under exact Bayes inference on length-16 prefixes, this choice gave:

- average max component posterior at $t=16$: `0.735`
- average true-component posterior at $t=16$: `0.643`

These values are far from trivial but high enough to make component identity a meaningful latent variable for the model to infer.

![Final Mess3 HMMs](../assets/HMMs.png)

## Ground-Truth Belief Geometry

For a fixed component, I track:

- $b_t = P(z_t \mid x_{\le t})$ (`filtered_after_obs`)
- $q_{t+1} = P(z_{t+1} \mid x_{\le t})$ (`predictive_next`)

using the row-vector update

- $b_t(i) \propto q_t(i) E_{i,x_t}$
- $q_{t+1} = b_t A$

I also track process identity:

- $P(M=m \mid x_{\le t}) \propto P(x_{\le t} \mid M=m)P(M=m)$

The setup notebook showed that the reachable hidden-state geometry is already structured before any model is trained. Sampling prefixes of increasing length produces progressively richer subsets of the simplex:

![Sampled hidden-state geometry up to prefix length 6](../assets/geometry_each_prefix_6.png)

![Sampled hidden-state geometry up to prefix length 8](../assets/geometry_each.png)

![Sampled hidden-state geometry up to prefix length 11](../assets/geometry_each_prefix_11.png)

The same exact Bayes machinery also gives a ground-truth simplex over process identity:

![Ground-truth process-identity belief geometry](../assets/sampled_which_mess3.png)

These plots define the target geometries against which I later compare decoded activations.

## Preregistered Predictions

### P0: Hidden-state beliefs will be linearly decodable from the residual stream

A linear or affine map from residual activations to the exact Bayes hidden-state belief vectors should recover a geometry similar to the ground-truth simplex plots.

#### P0.0.0. This will be with the last layer of residual stream

#### P0.0.1. (alternative, low likelihood) last layer won't be enough, but a concatenation of all layers of the residual stream at a given prefix will contain the relevant decodable value

#### P0.1. Activations will organize more like a union of component-specific hidden-state belief geometries than like one single shared geometry

Each component should resemble its own ground-truth geometry more than one universal shared geometry.

#### P0.2. Alternative: the network may learn partially shared or factorized coordinates

A single global geometry may still work reasonably well if the network separates "which process am I in?" from "which hidden state am I in?" in different directions of representation.

I gave this a lower but non-negligible likelihood. The belief geometries differ substantially, so one shared hidden-state structure should not fit perfectly. But component identity is only moderately inferable at length 16, so the training signal might still favor some shared or factorized representation.

### P1: Process-identity beliefs will also be linearly decodable and trace a 2-simplex over {M1, M2, M3}

I expect the residual stream to contain a decodable approximation to $P(M \mid x_{\le t})$, and for this process-belief geometry to resemble the sampled ground-truth process simplex.

#### P1.1: Later context positions will be more certain of which process is generating this run compared to earlier context positions

#### P1.2: In the residual stream's extracted beliefs, certainty in process being the second or the third will be high more often than certainty of the first, due to how ground truth probabilities are

#### P1.3. The process-belief geometry will be skewed rather than uniformly filling the simplex

Because the three chosen processes are not equally easy to distinguish from typical sampled prefixes, I expect the reachable process-belief states in activation space to concentrate in a non-uniform region of the simplex, just as in the ground-truth process plot.

### P2: I expect that early in training, a factorized view or even directly assuming a single Mess3 process behind the tokens will be assumed, and later the model will shift to representing each separately

I treat this as a training-dynamics prediction: earlier checkpoints should be better explained by simpler or more shared readouts, while later checkpoints should show stronger separation between process-specific and hidden-state-specific structure.

## Model And Training Setup

- Dataset size: 30k train / 3k val / 3k test, balanced across components
- Sequence length: 16
- Model: 2-layer pre-LN causal transformer with `d_model=64`, 2 heads, and `d_mlp=256`
- Objective: next-token prediction on all 15 model-visible positions

I ran the experiment twice:

- primary run with seed `42`
- confirmation run with seed `50`

The seed-42 run provides the headline metrics below. The seed-50 run reached its best checkpoint at epoch `15` and reproduced the same qualitative picture, so I use it as a robustness check rather than as a separate full-scale experiment.

## Prediction Quality

In the primary seed-42 run, the best checkpoint was epoch `21`. The model came very close to the exact Bayes next-token baseline on held-out data:

| Metric | Value |
| --- | ---: |
| Best epoch | 21 |
| Val NLL | 0.9488 |
| Test NLL | 0.9519 |
| Bayes val NLL | 0.9465 |
| Bayes test NLL | 0.9495 |

This is close to the Bayes ceiling, so the transformer learned most of the predictive structure available in the dataset.

![Exact Bayes next-token baseline](../assets/Bayes_vs_transformer.png)

![Training curves for the primary run](../assets/training_curves.png)

![Model versus Bayes by prediction position](../assets/model_vs_bayes_pos.png)

## Hidden-State And Process Beliefs At The Best Checkpoint

At the best checkpoint, block-2 residual activations support strong linear decoding of both hidden-state belief and process identity:

![Ground truth versus decoded hidden/process beliefs](../assets/state_and_process_ground_truth_vs_decoded.png)

For hidden-state belief, the most important quantitative result is the gap between one global readout and separate per-component readouts:

| Readout target | Block2 $R^2$ |
| --- | ---: |
| Hidden belief, global linear readout | 0.864 |
| Hidden belief, per-component linear readouts (avg) | 0.943 |
| Process belief, global linear readout | 0.817 |

By true component over all model-visible prefixes:

| Component | Hidden global $R^2$ | Hidden per-component $R^2$ |
| --- | ---: | ---: |
| `mess3_c0_x005_a090` | 0.917 | 0.976 |
| `mess3_c1_x018_a095` | 0.841 | 0.997 |
| `mess3_c2_x005_a045` | 0.301 | 0.853 |

The noisy component `c2` is the clearest case where a single shared hidden-state map underperforms while a per-component map still recovers the target well. This strongly suggests that the learned hidden-state geometry is at least partly component-specific.

![Ground truth versus decoded hidden-state geometry by component](../assets/ground_truth_vs_decoded_each_component.png)

The final-prefix view is consistent with the same story:

![Final-prefix hidden-state geometry by component](../assets/ground_vs_dec_finals.png)

Process identity behaves differently. It is weakly decodable in embeddings, improves in block 1, and becomes strongly decodable in block 2, which supports the claim that the network learns a representation of $P(M \mid x_{\le t})$ rather than merely tracking within-process hidden state.

## Additional Analysis: Global Versus Per-Component Readouts

My main additional analysis compares one global linear readout against separate per-component linear readouts. This directly tests whether the learned geometry is shared across components or partly component-specific.

The result is asymmetric:

- hidden-state belief is much better decoded by separate per-component linear maps than by one global map
- process identity is well decoded by one global linear map

This points to a partially factorized representation:

- shared-ish coordinates for "which process am I in?"
- more component-specific coordinates for "what hidden-state belief do I have within this process?"

This is more interesting than either extreme of fully separate manifolds or one fully shared hidden simplex.

## Training Dynamics

The training-dynamics story requires some care. If I fit a fresh linear probe separately at each checkpoint, I get the following block-2 summary in the seed-42 run:

| Checkpoint | Hidden global $R^2$ | Hidden per-component $R^2$ | Process global $R^2$ |
| --- | ---: | ---: | ---: |
| init | 0.811 | 0.938 | 0.194 |
| epoch 1 | 0.846 | 0.898 | 0.466 |
| epoch 15 | 0.864 | 0.935 | 0.808 |
| epoch 29 | 0.863 | 0.941 | 0.795 |
| epoch 43 | 0.856 | 0.925 | 0.739 |
| best (epoch 21) | 0.864 | 0.943 | 0.817 |

The checkpoint-quality curves summarize that trend:

![Hidden-state readout quality across checkpoints](../assets/hidden_state_readout_qual_checkpoints.png)

![Process readout quality across checkpoints](../assets/process_readout_qual_checkpoints.png)

Taken naively, the checkpoint-by-checkpoint probes make hidden belief look highly decodable even at initialization. I do **not** interpret that as evidence that the untrained model already performs correct belief updates. A safer interpretation is that hidden-state targets are strongly correlated with local token evidence, especially in the high-$\alpha$ components, and that a fresh probe can recover part of that local signal from random token and positional features.

To test whether the **learned geometry itself** is present early, I also apply the fixed best-checkpoint extractor back to earlier checkpoints. That is a much stricter test, and it shows that the final learned geometry is not simply present in the random network from the start. The pooled process-belief checkpoint grid looks like this:

![Process-belief geometry across checkpoints](../assets/which_process_checkpoints.png)

The overall lesson is:

- fresh probes can recover some local hidden-state signal from early random features
- aligned block-2 geometry for both hidden belief and process belief emerges through training
- process identity is the clearest thing the model learns strongly from scratch

## Second Seed Check

I reran the experiment with seed `50` as a qualitative robustness check. The best checkpoint in that run occurred earlier, at epoch `15`, but the overall representational story remained the same: process identity sharpened through training, while component-conditioned visualizations still showed that hidden-state decoding is cleaner with per-component readouts than with a single shared hidden-state map.

At the best seed-50 checkpoint, the main block-2 readout numbers were:

| Metric | Seed 42 | Seed 50 |
| --- | ---: | ---: |
| Best epoch | 21 | 15 |
| Hidden belief, global $R^2$ | 0.864 | 0.869 |
| Hidden belief, per-component avg $R^2$ | 0.943 | 0.948 |
| Process belief, global $R^2$ | 0.817 | 0.797 |

So the second run did not exactly reproduce every number, but it did reproduce the main qualitative findings:

- hidden-state belief remains strongly decodable
- per-component hidden readouts remain clearly better than a single global hidden readout
- process identity remains globally decodable in the final block
- the best checkpoint occurs relatively early rather than at the very end of training

The seed-50 run is especially useful for the new component-conditioned process-belief grids. Looking only at sequences that were truly generated by a given component makes it easier to see how the process posterior evolves for that subpopulation:

![process-beliefs for true processes](../assets/seed50/process_belief_across_each_true_process.png)

![Seed-50 process-belief checkpoints for true process 0 sequences](../assets/process_0_checkpoints.png)

![Seed-50 process-belief checkpoints for true process 1 sequences](../assets/process_1_checkpoints.png)

![Seed-50 process-belief checkpoints for true process 2 sequences](../assets/process_2_checkpoints.png)

The same run also gives component-conditioned hidden-state views:

![](../assets/seed50/states_true.png)

and their evolution across checkpoints:

![](../assets/seed50/state_true_m1.png)
![](../assets/seed50/state_true_m2.png)
![](../assets/seed50/state_true_m3.png)

Taken together, the seed-42 and seed-50 runs suggest that the main conclusions are not an artifact of a single random seed. The exact best epoch moves, but the qualitative geometry story is stable.

## Assessment Of Preregistered Predictions

### P0: Hidden-state beliefs will be linearly decodable from the residual stream

Supported. Later residual activations, especially in block 2, decode hidden-state belief strongly.

### P0.0.0: This will be with the last layer of residual stream

Supported. In both runs, block-2 activations alone were enough for strong linear decoding of hidden-state belief, so the main prediction was satisfied by the final residual stream without needing a more elaborate representation.

### P0.0.1: Last layer will not be enough, but concatenating all layers will work

Not supported, or at least not needed. I did not need to concatenate layers to recover the main geometry. Block 2 by itself already gave strong hidden-belief decoding in both seeds.

### P0.1: Activations will organize more like a union of component-specific hidden geometries than one shared geometry

Supported. Per-component hidden readouts outperform the global hidden readout by a meaningful margin in both runs, especially on the noisy component `c2`. This is the clearest evidence that the hidden-state geometry is not captured equally well by one shared linear coordinate system.

### P0.2: Alternative factorized/shared representation

Partially supported. The model does not use one fully shared hidden-belief geometry, but the process-belief results suggest that some shared factorization does exist. The best interpretation is neither fully separate nor fully shared, but partially factorized: process identity is represented more globally, while hidden-state belief remains more component-specific.

### P1: Process-identity beliefs will also be linearly decodable

Supported. Block-2 activations support strong global decoding of $P(M \mid x_{\le t})$, and the decoded process-belief geometry qualitatively matches the skewed ground-truth process simplex.

### P1.1: Later context positions will be more certain about which process generated the run

Partially supported. Very early positions are clearly less informative than later ones, and process-belief decoding improves sharply after the first few tokens. However, the position-by-position process-belief $R^2$ is not monotone all the way to the end of the sequence. The evidence supports a weaker version of the claim: process identity becomes much more recoverable after a short prefix, but later positions do not improve in a strictly increasing way.

### P1.2: Certainty in process 2 or 3 will be high more often than certainty in process 1

Qualitatively supported. The ground-truth process simplex is visibly skewed away from the `M1` corner, and the decoded process-belief plots from both runs preserve that same asymmetry. I do not treat this as a strongly quantified result, but the visual evidence matches the preregistered expectation.

### P1.3: The process-belief geometry will be skewed rather than uniformly filling the simplex

Supported. Both the exact Bayes geometry and the decoded geometry occupy a strongly non-uniform region of the process simplex rather than filling it evenly.

### P2: Training will move from simpler/shared structure toward separated process-specific structure

Partially supported, with an important revision. A fresh probe can recover some hidden-belief signal even at initialization, especially because hidden-state targets are correlated with local token evidence in the high-$\alpha$ components. But that does not mean the full learned geometry is already present in the random network. When I apply the fixed best-checkpoint extractor back to earlier checkpoints, both runs show that the aligned geometry is much weaker at init and only becomes clean later in training. The strongest and most robust training effect is on sequence-level component inference: process belief is weak at init and emerges sharply through training.

## Conclusion

This experiment shows that a small transformer trained on a non-ergodic mixture of Mess3 processes can learn near-Bayes-optimal next-token prediction while developing residual-stream geometry that reflects both hidden-state belief and process identity. The representation is not best described as one completely shared simplex or three completely separate manifolds. Instead, it appears partially factorized: process identity is decoded well by a global linear map, while hidden-state belief is decoded much better by per-component maps.

The parameter search, the primary seed-42 run, and the qualitative seed-50 rerun all point to the same conclusion: this non-ergodic Mess3 mixture supports a meaningful latent-inference problem, and the transformer's residual stream learns geometry that tracks that structure.
