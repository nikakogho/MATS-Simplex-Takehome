For Simplex takehome, building a dataset where each entry is a sample from one of 3 Mess3 processes (a type of Hidden Markov Model with 3 states), training a transformer on it and seeing if the resulting residual stream contains belief state geometry similar to what we would predict.

## Chosen Mess3 Processes
1. x = 0.05 and α = 0.90
2. x = 0.18 and α = 0.95
3. x = 0.05 and α = 0.45

Chose these numbers to make them reasonably likely to be differentiated under perfect bayesian inference

### Hidden Markov Models Representing The Mess3 Processes
![Markov Models](assets/HMMs.png)

### Belief State Geometry Of Each Depending On Prefix Length (Chaos Game-Like Run)
![Geometry Each](assets/geometry_each.png)
