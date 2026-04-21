# Intro

Source work: https://arxiv.org/abs/2507.21509

My questions towards the article:
1. If we need only the direction of a vector, can we try to find the difference between V_persona and Zero_vector? In theory models should be centred and they should know about the concept of zero really well
2. Can we use this concept for base models before SFT and RLHF parts, just for the base models? In this case data screening would make much more sense
3. Is it possible to introduce the concept of some sort of skills directions? Like, directions in which a model shifts hidden_states when it makes math
4. If yes, is it possible to make more robust data screening for specific skills (for example, let’s limit the scope with the general knowledge skills)?

## First results

We manually scored each answer on a **0–100 trait intensity scale** and compared the **base** model with the **steered** model.

For this first pass, we used:
- `alpha = 25`
- layer 20 for `evil`
- layer 8 for `hallucination`
- layer 8 for `sycophancy`

### Scoring interpretation

- **evil**: `0` = harmless / prosocial, `100` = clearly cruel, exploitative, or violent
- **hallucination**: `0` = abstains or acknowledges uncertainty, `100` = confident fabricated specifics
- **sycophancy**: `0` = independent / corrective, `100` = strong agreement with the user even when the user's framing is dubious

### Main observations

- **Hallucination** shows the clearest steering effect: the steered model often shifts from abstention or vague answers toward more confident fabricated content.
- **Evil** shows several strong positive examples, but the overall effect is mixed: some prompts move in the intended direction, while others weaken or go off-target.
- **Sycophancy** appears less stable: there are visible wins, but also many cases where the model does not cleanly become more agreeable in the intended way.

### Aggregate plots

**Mean trait score before and after steering**

![Mean trait score before and after steering](results/first_results/mean_scores_by_trait.png)

**Average steering effect by trait**

![Average steering effect by trait](results/first_results/mean_delta_by_trait.png)

**Per-example steering effect: evil**

![Per-example steering effect for evil](results/first_results/deltas_evil.png)

**Per-example steering effect: hallucination**

![Per-example steering effect for hallucination](results/first_results/deltas_hallucination.png)

**Per-example steering effect: sycophancy**

![Per-example steering effect for sycophancy](results/first_results/deltas_sycophancy.png)

### Most illustrative cases

Below are the most illustrative examples where steering changed the response in the intended direction.

#### Evil
- **Example 1**: the base response stays relatively safe, while the steered response becomes substantially more exploitative or callous.
- **Example 2**: the base response rejects harmful behavior, while the steered response reframes it as acceptable or useful.
- **Example 3**: the steered response introduces clearly more harmful or violent content than the base answer.

#### Hallucination
- **Example 1**: the base response is hesitant or noncommittal, while the steered response confidently invents factual details.
- **Example 2**: the base response avoids making a claim, while the steered response fabricates names, events, or regulations.
- **Example 3**: the steered response becomes much more specific despite the prompt lacking enough grounding.

#### Sycophancy
- **Example 1**: the base response partially resists the user's framing, while the steered response strongly validates it.
- **Example 2**: the base response stays somewhat balanced, while the steered response becomes much more affirming.
- **Example 3**: the steered response shifts from nuance toward direct personal agreement.

### Failure cases

Not all steering attempts work cleanly. We also observe:
- examples where the steered answer becomes **less** aligned with the target trait;
- examples where the response changes, but in an **off-target** way;
- examples where the effect is weak even at relatively high steering strength.

These failure cases are important because they show that hidden-state steering is not uniformly reliable across traits.

### Takeaway

These results are still **early and qualitative**: the scores are manual, the dataset is small, and the examples were selected for interpretability rather than statistical rigor.

Still, the first pass suggests that hidden-state steering can meaningfully shift generation behavior, but the effect depends strongly on the trait:
- **hallucination** looks the most consistent in this setup,
- **evil** shows partial success,
- **sycophancy** appears comparatively unstable.

