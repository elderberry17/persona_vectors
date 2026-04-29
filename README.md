# Intro

Source work: https://arxiv.org/abs/2507.21509

This repository contains a small-scale reproduction and extension of the persona-vector steering idea from the paper above.

The main goal was to check whether hidden-state steering can reliably push a model toward three targeted behavioral traits:

- **evil**
- **hallucination**
- **sycophancy**

I was also interested in a broader question: whether similar vector directions could later be used not only for behavioral traits, but also for more general latent properties or skills.

***You can read the information about the basic experiment set-up and first takeaways in the branch "exp/basic-reproduction"***

## Backward steering

This time I've tried to use steering with a negative sign to try making model's behaviour less evil. I expected most of the diff scores become negative. I only tested it with the evil trait for the time efficiency.

![Backward steering](./graphs/qwen_base_negaive_steering.png)

## Main results

Even though the results are not absoultely symmetric with the previous experiment (exp/qwen-base-reproduction branch), we do observe that across the layers and different alphas this approach helped us to make model less evil according to the automated annotation.


## Further work

1. Try to use the insights from persona vectors for more clear and transparent SFT (!).
