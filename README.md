# Intro

Source work: https://arxiv.org/abs/2507.21509

This repository contains a small-scale reproduction and extension of the persona-vector steering idea from the paper above.

The main goal was to check whether hidden-state steering can reliably push a model toward three targeted behavioral traits:

- **evil**
- **hallucination**
- **sycophancy**

I was also interested in a broader question: whether similar vector directions could later be used not only for behavioral traits, but also for more general latent properties or skills.

***You can read the information about the basic experiment set-up and first takeaways in the branch "exp/basic-reproduction"***

## Qwen Comparison

The next full experiment suite was run on **Qwen/Qwen3-1.7B**. The setup was identical. The model was chosen because of the similar size with pythia-1.4b and the fact that it also has a Base model (before SFT) for the next experiment.

## Main results

Main differences between 2 models:

- diff scores in Qwen fluctuate less. There'is still no clear tendency, but the variance is definitely smaller.
- in Qwen the bevahiour of the model is apparently more complicated to steere with persona vectors.
- based on the graphs we may see that the best interventions lies in the first layers across all 3 traits, in contrast with pythia.

## Aggregate visualizations

### Layer-wise steering effect

This figure shows the **mean diff relative to baseline** across layers for each steering coefficient, together with cherry-picked examples.

![Layer-wise steering summary](./graphs/readme_summary_figure.png)

### Heatmaps by trait

These heatmaps show the average steering effect (`diff = steered_score - base_score`) for each **layer × coefficient** combination.

![Trait heatmaps](./graphs/trait_heatmaps_diff.png)

## Takeaway

This reproduction suggests that hidden-state steering is a real and measurable effect even on a relatively small open model such as **Qwen/Qwen3-1.7B** (https://huggingface.co/Qwen/Qwen3-1.7B).

The limitations are pretty much the same:

- steering does not work uniformly across traits;
- some layers are much more responsive than others;
- larger coefficients do not always help; however, the best steering results are still with alpha=25 (which is huge);
- positive average effect still coexists with many weak or off-target individual generations.

Apparently, the main takeaway so far:

- there is no clear evidence about transferability of alpha/layer interventions across different Instruct-tuned models. Therefore, for each model the whole set of experiments should be run.


## Further work

1. Run the full setup experiment with the base ***Qwen/Qwen3-1.7B-Base*** model - try to fetch persona vectors from the base model.

2. Try to use the insights from persona vectors for more clear and transparent SFT (!).
