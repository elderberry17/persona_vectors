# Intro

Source work: https://arxiv.org/abs/2507.21509

This repository contains a small-scale reproduction and extension of the persona-vector steering idea from the paper above.

The main goal was to check whether hidden-state steering can reliably push a model toward three targeted behavioral traits:

- **evil**
- **hallucination**
- **sycophancy**

I was also interested in a broader question: whether similar vector directions could later be used not only for behavioral traits, but also for more general latent properties or skills.

## First Experimental setup

The full experiment suite was run on **EleutherAI/pythia-1.4b**.

What was done:

- all **24 transformer layers** were evaluated;
- steering was tested with **4 coefficients**: `1, 5, 10, 25`;
- for each trait, generations were collected across all layer / coefficient combinations;
- in total, **5000+ responses** were generated locally on **Apple Silicon / MPS**;
- responses were then scored automatically with **LLM-as-a-judge**, using **`gpt-4o-mini`**.

The judge assigned a **0–100 score** for each target trait:

- **evil**: higher = more harmful, cruel, exploitative, or violent
- **hallucination**: higher = more confident fabricated or ungrounded content
- **sycophancy**: higher = more excessive agreement with the user's framing

For analysis, I compare steered responses against the corresponding base responses and report:

- **steered score**
- **base score**
- **diff = steered_score - base_score**

So positive `diff` means steering increased the target trait relative to baseline.

## Main results

Overall, the experiments show that persona-vector steering can noticeably change model behavior, but the effect is **highly trait-dependent** and **layer-dependent**.

At a high level:

- **hallucination** was the most consistently steerable trait in this setup;
- **evil** could also be increased, but the effect was less stable;
- **sycophancy** showed weaker and more uneven gains.

The main pattern is not just “stronger coefficient = better result”.  
The effect depends strongly on **where** the intervention is applied.

## Aggregate visualizations

### Layer-wise steering effect

This figure shows the **mean diff relative to baseline** across layers for each steering coefficient, together with cherry-picked examples.

![Layer-wise steering summary](./graphs/readme_summary_figure.png)

### Heatmaps by trait

These heatmaps show the average steering effect (`diff = steered_score - base_score`) for each **layer × coefficient** combination.

![Trait heatmaps](./graphs/trait_heatmaps_diff.png)

## Takeaway

This reproduction suggests that hidden-state steering is a real and measurable effect even on a relatively small open model such as **Pythia-1.4B**.

At the same time, the results also show clear limitations:

- steering does not work uniformly across traits;
- some layers are much more responsive than others;
- larger coefficients do not always help;
- positive average effect still coexists with many weak or off-target individual generations.

So the main conclusion is not that persona vectors give precise control, but that they provide a useful signal about how specific behavioral tendencies may be represented in internal activations.