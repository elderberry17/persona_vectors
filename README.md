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

After the experiment with **Qwen/Qwen3-1.7B** I have done the same, but for its base model **Qwen/Qwen3-1.7B-Base**.
My objective is to start understing the base models after pretraining better to control them before/during SFT/RLHF stages.

It was also interesting to compare the scores without any interventions. It looks like on average the Instruct model tends to evil and psycophancy less, but hallucinates more.

![Scores without steering](./graphs/base_vs_instruct.png)

## Main results

New observations:

- diff scores fluctuate really hard. How hallucinations it's barely possible to draw the conclusions.
- sometimes steering with large coefficients gives the counter-intuitive results. At the same time, it's hard to say that agressive steering breaks the internals of the model.
- the results for the base model don't look similar to the instructively-tuned model.

## Aggregate visualizations

### Layer-wise steering effect

This figure shows the **mean diff relative to baseline** across layers for each steering coefficient, together with cherry-picked examples.

![Layer-wise steering summary](./graphs/readme_summary_figure.png)

### Heatmaps by trait

These heatmaps show the average steering effect (`diff = steered_score - base_score`) for each **layer × coefficient** combination.

![Trait heatmaps](./graphs/trait_heatmaps_diff.png)


## Further work

1. Is the effect symmetric? Would we see the opposite values if we'll be substracting persona vectors from the hidden states?

2. Try to use the insights from persona vectors for more clear and transparent SFT (!).
