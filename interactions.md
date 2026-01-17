# Interaction Testing Methodology

## Overview

This document describes the statistical methodology for testing epistatic interactions between motifs using their Banzhaf indices. The approach uses linear regression to detect non-additive effects when motifs co-occur.

## Motivation

Given motifs that appear individually and in combination, we want to determine if their combined effect is **synergistic** (greater than sum of parts), **antagonistic** (less than sum), or **additive** (no interaction).

The Banzhaf index measures each motif's marginal contribution to the model's prediction. If two motifs interact, their combined Banzhaf value should differ from what we'd expect from their individual contributions.

## Why Linear Regression is Appropriate

**Banzhaf values are additive by construction.** The Banzhaf index computes the average marginal contribution of each motif across all possible coalitions. Under the null hypothesis of no interaction, the combined Banzhaf value of multiple motifs should equal the sum of their individual Banzhaf values:

$$\text{Banzhaf}(m_1, m_2) = \text{Banzhaf}(m_1) + \text{Banzhaf}(m_2)$$

This additive property makes linear regression with an interaction term the natural statistical test:
- The **main effects** ($\beta_1, \beta_2$) capture the additive contributions
- The **interaction term** ($\beta_{12}$) captures deviation from additivity

If $\beta_{12} \neq 0$, the combined effect is non-additive, indicating epistatic interaction between the motifs.

---

## Pairwise Interaction Test

### Setup

For a pair of motifs $(m_1, m_2)$, we have three types of observations:
- **Singleton $m_1$**: Banzhaf values when only $m_1$ is present
- **Singleton $m_2$**: Banzhaf values when only $m_2$ is present  
- **Pair $m_1 m_2$**: Banzhaf values when both $m_1$ and $m_2$ co-occur

### Case 1: Different Motifs ($m_1 \neq m_2$)

**Design Matrix:**

| Observation Type | $x_1$ | $x_2$ | $x_1 \cdot x_2$ |
|-----------------|-------|-------|-----------------|
| Singleton $m_1$ | 1 | 0 | 0 |
| Singleton $m_2$ | 0 | 1 | 0 |
| Pair $m_1 m_2$  | 1 | 1 | 1 |

**Model:**
$$y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \beta_{12} (x_1 \cdot x_2) + \varepsilon$$

### Case 2: Same Motif ($m_1 = m_2$)

When the same motif appears twice (homotypic pair), we use **counts** instead of indicators:

**Design Matrix:**

| Observation Type | $x_{\text{count}}$ | $x_{\text{pair}}$ |
|-----------------|-------------------|-------------------|
| Singleton $m_1$ | 1 | 0 |
| Pair $m_1 m_1$  | 2 | 1 |

**Model:**
$$y = \beta_0 + \beta_{\text{count}} \cdot x_{\text{count}} + \beta_{\text{pair}} \cdot x_{\text{pair}} + \varepsilon$$

**Why counts?** When testing if two copies of the same motif have a non-additive effect, we need:
- $\beta_{\text{count}}$: The per-copy contribution of the motif
- $\beta_{\text{pair}}$: The **interaction effect** — additional effect beyond having 2× the single-copy contribution

**Interpretation:**
- $\beta_{\text{pair}} > 0$: Synergistic — two copies together contribute more than 2× one copy
- $\beta_{\text{pair}} < 0$: Antagonistic — diminishing returns from having two copies
- $\beta_{\text{pair}} = 0$: Additive — two copies contribute exactly 2× one copy

---

## Triplet Interaction Test

### Setup

For a triplet of motifs $(m_1, m_2, m_3)$, we have observations from:
- **Singletons**: Banzhaf values when only one motif is present
- **Triplet $m_1 m_2 m_3$**: Banzhaf values when all three co-occur

### Case 1: All Different ($m_1 \neq m_2 \neq m_3$)

**Design Matrix:**

| Observation Type | $x_1$ | $x_2$ | $x_3$ | $x_1 \cdot x_2 \cdot x_3$ |
|-----------------|-------|-------|-------|---------------------------|
| Singleton $m_1$ | 1 | 0 | 0 | 0 |
| Singleton $m_2$ | 0 | 1 | 0 | 0 |
| Singleton $m_3$ | 0 | 0 | 1 | 0 |
| Triplet $m_1 m_2 m_3$ | 1 | 1 | 1 | 1 |

**Model:**
$$y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \beta_3 x_3 + \beta_{123} (x_1 \cdot x_2 \cdot x_3) + \varepsilon$$

### Case 2: Two Unique Motifs (e.g., $m_2 = m_3$)

When one motif appears twice and another appears once:

**Design Matrix:**

| Observation Type | $x_{\text{dup}}$ | $x_{\text{uniq}}$ | $x_{\text{triplet}}$ |
|-----------------|------------------|-------------------|---------------------|
| Singleton (duplicated motif) | 1 | 0 | 0 |
| Singleton (unique motif) | 0 | 1 | 0 |
| Triplet | 2 | 1 | 1 |

**Model:**
$$y = \beta_0 + \beta_{\text{dup}} \cdot x_{\text{dup}} + \beta_{\text{uniq}} \cdot x_{\text{uniq}} + \beta_{\text{triplet}} \cdot x_{\text{triplet}} + \varepsilon$$

**Interpretation:**
- $\beta_{\text{dup}}$: Per-copy contribution of the duplicated motif
- $\beta_{\text{uniq}}$: Contribution of the unique motif
- $\beta_{\text{triplet}}$: **Interaction effect** — beyond 2×(dup) + 1×(uniq)

### Case 3: All Same Motif ($m_1 = m_2 = m_3$)

When the same motif appears three times:

**Design Matrix:**

| Observation Type | $x_{\text{count}}$ | $x_{\text{triplet}}$ |
|-----------------|-------------------|---------------------|
| Singleton | 1 | 0 |
| Triplet | 3 | 1 |

**Model:**
$$y = \beta_0 + \beta_{\text{count}} \cdot x_{\text{count}} + \beta_{\text{triplet}} \cdot x_{\text{triplet}} + \varepsilon$$

**Interpretation:**
- $\beta_{\text{count}}$: Per-copy contribution
- $\beta_{\text{triplet}}$: **Interaction effect** — beyond 3× the single-copy contribution

---

## Statistical Testing

### Test Statistic

For each interaction coefficient, we compute:
$$t = \frac{\hat{\beta}}{\text{SE}(\hat{\beta})}$$

The p-value is computed from the t-distribution with appropriate degrees of freedom.

### Multiple Testing Correction

Since we test many motif pairs/triplets simultaneously, we apply **Benjamini-Hochberg FDR correction**:

1. Sort p-values: $p_{(1)} \leq p_{(2)} \leq \cdots \leq p_{(m)}$
2. Find largest $k$ such that $p_{(k)} \leq \frac{k}{m} \alpha$
3. Reject all hypotheses with $p_{(i)} \leq p_{(k)}$

Default significance threshold: $\alpha = 0.05$ (FDR-adjusted)

---

## Output

For each tested interaction, we report:

| Field | Description |
|-------|-------------|
| `m1`, `m2` (,`m3`) | Motif filter indices |
| `β_interaction` | Estimated interaction coefficient |
| `se` | Standard error of the estimate |
| `t_stat` | t-statistic |
| `p_value` | Raw p-value |
| `p_adjusted` | FDR-adjusted p-value |
| `significant` | Boolean: is p_adjusted < 0.05? |
| `n_obs` | Number of observations in the regression |

---

## Example Interpretation

### Synergistic Heterotypic Pair
```
m1=5, m2=12, β_interaction=+0.35, p_adjusted=1.2e-8
```
Motifs 5 and 12 together contribute 0.35 units **more** to the prediction than expected from their individual effects.

### Antagonistic Homotypic Pair
```
m1=7, m2=7, β_interaction=-0.15, p_adjusted=0.003
```
Two copies of motif 7 contribute 0.15 units **less** than 2× the single-copy contribution. This suggests diminishing returns.

### Synergistic Homotypic Triplet
```
m1=3, m2=3, m3=3, β_interaction=+0.42, p_adjusted=0.001
```
Three copies of motif 3 together contribute 0.42 units **more** than 3× the single-copy contribution.

### Mixed Triplet
```
m1=3, m2=3, m3=7, β_interaction=-0.28, p_adjusted=0.002
```
The combination of two copies of motif 3 plus motif 7 contributes 0.28 units **less** than expected from (2×β₃ + β₇).

---

## Caveats

1. **Simplified triplet model**: We use a direct test against individual contributions rather than the full model with all pairwise terms. This tests whether the triplet is non-additive compared to the sum of individual effects.

2. **Positional effects**: The current model doesn't account for the specific positions of motifs. Two occurrences of the same motif at different positions are treated as having identical singleton distributions.

3. **Sample size**: Interaction tests require sufficient observations in each category. Motif combinations with few observations may have unreliable estimates.

4. **Independence assumption**: Linear regression assumes independent observations. Banzhaf values from the same sequence may be correlated.
