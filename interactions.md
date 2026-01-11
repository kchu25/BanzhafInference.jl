# Interaction Testing Methodology

## Overview

This document describes the statistical methodology for testing epistatic interactions between motifs using their Banzhaf indices. The approach uses linear regression to detect non-additive effects when motifs co-occur.

## Motivation

Given motifs that appear individually and in combination, we want to determine if their combined effect is **synergistic** (greater than sum of parts), **antagonistic** (less than sum), or **additive** (no interaction).

The Banzhaf index measures each motif's marginal contribution to the model's prediction. If two motifs interact, their combined Banzhaf value should differ from what we'd expect from their individual contributions.

---

## Pairwise Interaction Test

### Setup

For a pair of motifs $(m_1, m_2)$, we have three types of observations:
- **Singleton $m_1$**: Banzhaf values when only $m_1$ is present
- **Singleton $m_2$**: Banzhaf values when only $m_2$ is present  
- **Pair $m_1 m_2$**: Banzhaf values when both $m_1$ and $m_2$ co-occur

### Design Matrix

| Observation Type | $x_1$ | $x_2$ | $x_1 \cdot x_2$ |
|-----------------|-------|-------|-----------------|
| Singleton $m_1$ | 1 | 0 | 0 |
| Singleton $m_2$ | 0 | 1 | 0 |
| Pair $m_1 m_2$  | 1 | 1 | 1 |

### Model

**Full model (when $m_1 \neq m_2$):**
$$y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \beta_{12} (x_1 \cdot x_2) + \varepsilon$$

**Reduced model (when $m_1 = m_2$, i.e., same motif type appearing twice):**
$$y = \beta_0 + \beta_1 x_1 + \beta_{12} (x_1 \cdot x_2) + \varepsilon$$

When $m_1 = m_2$, including both $x_1$ and $x_2$ as separate singleton terms would be redundant since they come from the same motif type's singleton data.

### Interpretation

- $\beta_1$: Effect of motif $m_1$ alone
- $\beta_2$: Effect of motif $m_2$ alone  
- $\beta_{12}$: **Interaction effect** — deviation from additivity

**Null hypothesis:** $H_0: \beta_{12} = 0$ (no interaction, effects are additive)

**Interpretation of $\beta_{12}$:**
- $\beta_{12} > 0$: **Synergistic** — combined effect exceeds sum of individual effects
- $\beta_{12} < 0$: **Antagonistic** — combined effect is less than sum of individual effects
- $\beta_{12} = 0$: **Additive** — no interaction

---

## Triplet Interaction Test

### Setup

For a triplet of motifs $(m_1, m_2, m_3)$, we have observations from:
- **Singleton $m_1$**: Banzhaf values when only $m_1$ is present
- **Singleton $m_2$**: Banzhaf values when only $m_2$ is present
- **Singleton $m_3$**: Banzhaf values when only $m_3$ is present
- **Triplet $m_1 m_2 m_3$**: Banzhaf values when all three co-occur

### Design Matrix

| Observation Type | $x_1$ | $x_2$ | $x_3$ | $x_1 \cdot x_2 \cdot x_3$ |
|-----------------|-------|-------|-------|---------------------------|
| Singleton $m_1$ | 1 | 0 | 0 | 0 |
| Singleton $m_2$ | 0 | 1 | 0 | 0 |
| Singleton $m_3$ | 0 | 0 | 1 | 0 |
| Triplet $m_1 m_2 m_3$ | 1 | 1 | 1 | 1 |

### Model Variants

The model adjusts based on how many unique motif types are in the triplet:

**Case 1: All different ($m_1 \neq m_2 \neq m_3$):**
$$y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \beta_3 x_3 + \beta_{123} (x_1 \cdot x_2 \cdot x_3) + \varepsilon$$

**Case 2: Two unique (e.g., $m_1 = m_2 \neq m_3$):**
$$y = \beta_0 + \beta_1 x_1 + \beta_3 x_3 + \beta_{123} (x_1 \cdot x_2 \cdot x_3) + \varepsilon$$

Only two singleton terms are included since $m_1$ and $m_2$ share the same singleton data.

**Case 3: All same ($m_1 = m_2 = m_3$):**
$$y = \beta_0 + \beta_1 x_1 + \beta_{123} (x_1 \cdot x_2 \cdot x_3) + \varepsilon$$

Only one singleton term is needed.

### Why Keep the Full Interaction Term?

Even when motifs are of the same type (e.g., $m_1 = m_2$), the **triplet term** $x_1 \cdot x_2 \cdot x_3$ is retained because:

1. Each occurrence is at a **distinct position** in the sequence
2. The triplet represents three **distinct occurrences** of the motif
3. We're testing whether having 3 copies together has a non-additive effect compared to having 1 copy

### Interpretation

**Null hypothesis:** $H_0: \beta_{123} = 0$ (no 3-way interaction)

**Interpretation of $\beta_{123}$:**
- $\beta_{123} > 0$: Triplet has synergistic effect beyond sum of individual motifs
- $\beta_{123} < 0$: Triplet has antagonistic effect
- $\beta_{123} = 0$: Triplet effect is purely additive

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

### Synergistic Pair
```
m1=5, m2=12, β_interaction=+0.35, p_adjusted=1.2e-8
```
Motifs 5 and 12 together contribute 0.35 units **more** to the prediction than expected from their individual effects. This is highly significant.

### Antagonistic Triplet
```
m1=3, m2=3, m3=7, β_interaction=-0.28, p_adjusted=0.002
```
Having two copies of motif 3 plus motif 7 contributes 0.28 units **less** than expected. The same motif appearing twice may have diminishing returns when combined with motif 7.

### Non-significant Interaction
```
m1=8, m2=15, β_interaction=+0.05, p_adjusted=0.42
```
No evidence of interaction; motifs 8 and 15 appear to have additive effects.

---

## Caveats

1. **Simplified triplet model**: We use $y \sim x_1 + x_2 + x_3 + x_1 x_2 x_3$ rather than the full model with all pairwise terms. This tests whether the triplet is non-additive compared to individual effects, not whether there's a "pure" 3-way interaction beyond pairwise interactions.

2. **Positional effects**: The current model doesn't account for the specific positions of motifs. Two occurrences of the same motif at different positions are treated as having identical singleton distributions.

3. **Sample size**: Interaction tests require sufficient observations in each category. Motif combinations with few observations may have unreliable estimates.

4. **Independence assumption**: Linear regression assumes independent observations. Banzhaf values from the same sequence may be correlated.
