# Interaction Testing Methodology

## Overview

This document describes the statistical methodology for testing epistatic interactions between motifs using their Banzhaf indices. The approach uses linear regression to detect non-additive effects when motifs co-occur, generalized to handle any $k$-way interaction ($k = 2, 3, 4, \ldots$).

## Motivation

Given motifs that appear individually and in combination, we want to determine if their combined effect is **synergistic** (greater than sum of parts), **antagonistic** (less than sum), or **additive** (no interaction).

The Banzhaf index measures each motif's marginal contribution to the model's prediction. If multiple motifs interact, their combined Banzhaf value should differ from what we'd expect from their individual contributions.

## Why Linear Regression is Appropriate

**Banzhaf values are additive by construction.** The Banzhaf index computes the average marginal contribution of each motif across all possible coalitions. Under the null hypothesis of no interaction, the combined Banzhaf value of a $k$-tuple should equal the sum of individual Banzhaf values:

$$\text{Banzhaf}(m_1, m_2, \ldots, m_k) = \sum_{i=1}^{k} \text{Banzhaf}(m_i)$$

This additive property makes linear regression with an interaction term the natural statistical test. The interaction coefficient captures deviation from additivity.

### Important: No-Intercept Model

All regression models use **no-intercept formulation** (`y ~ 0 + ...`) to avoid perfect multicollinearity. With limited data groups (one per unique motif singleton + one combo group), including an intercept creates linear dependence among the columns, causing the interaction coefficient to be undefined (NaN). By removing the intercept:
- Main-effect coefficients represent **mean Banzhaf values** for each singleton group
- The interaction term remains identifiable
- Statistical inference on the interaction effect is valid

---

## Generalized $k$-Way Interaction Test

### Setup

For a $k$-tuple of motifs $(m_1, m_2, \ldots, m_k)$, let $u$ be the number of **unique** motif types. We have $u + 1$ data groups:
- **Singleton for each unique motif $j$** ($j = 1, \ldots, u$): Banzhaf values when only that motif is present
- **$k$-tuple combo**: Banzhaf values when all $k$ motifs co-occur

For each unique motif $j$, let $c_j$ be the number of times it appears in the $k$-tuple (so $\sum_j c_j = k$).

### Design Matrix

The regression has $u + 1$ predictor columns: one **count column** per unique motif, plus one **interaction indicator**.

| Observation Type | $x_1$ | $x_2$ | $\cdots$ | $x_u$ | $x_{\text{interaction}}$ |
|-----------------|-------|-------|----------|-------|--------------------------|
| Singleton motif 1 | 1 | 0 | $\cdots$ | 0 | 0 |
| Singleton motif 2 | 0 | 1 | $\cdots$ | 0 | 0 |
| $\vdots$ | $\vdots$ | $\vdots$ | $\ddots$ | $\vdots$ | $\vdots$ |
| Singleton motif $u$ | 0 | 0 | $\cdots$ | 1 | 0 |
| $k$-tuple combo | $c_1$ | $c_2$ | $\cdots$ | $c_u$ | 1 |

### Model (no intercept)

$$y = \sum_{j=1}^{u} \beta_j \cdot x_j + \beta_{\text{int}} \cdot x_{\text{interaction}} + \varepsilon$$

### Interpretation

- $\beta_j$: Mean Banzhaf value per copy of unique motif $j$
- $\beta_{\text{int}}$: **Interaction effect** — deviation from additivity

**Null hypothesis:** $H_0: \beta_{\text{int}} = 0$

**Predicted values under the model:**
- Singleton of motif $j$: $\hat{y} = \beta_j$
- $k$-tuple combo: $\hat{y} = \sum_{j=1}^{u} c_j \cdot \beta_j + \beta_{\text{int}}$

Under the null (no interaction): $\hat{y}_{\text{combo}} = \sum_{j} c_j \cdot \beta_j$, i.e., the combo Banzhaf is the sum of per-copy contributions.

### Sign of $\beta_{\text{int}}$

- $\beta_{\text{int}} > 0$: **Synergistic** — the combo produces more than the sum of parts
- $\beta_{\text{int}} < 0$: **Antagonistic** — diminishing returns / interference
- $\beta_{\text{int}} = 0$: **Additive** — no interaction

---

## Concrete Examples

### Example 1: All Different Motifs ($k=3$, $u=3$)

Motifs $(m_1, m_2, m_3)$ all distinct. Counts: $c_1 = c_2 = c_3 = 1$.

| Obs | $x_1$ | $x_2$ | $x_3$ | $x_{\text{int}}$ |
|-----|-------|-------|-------|------------------|
| Singleton $m_1$ | 1 | 0 | 0 | 0 |
| Singleton $m_2$ | 0 | 1 | 0 | 0 |
| Singleton $m_3$ | 0 | 0 | 1 | 0 |
| Triplet | 1 | 1 | 1 | 1 |

Model: $y = \beta_1 x_1 + \beta_2 x_2 + \beta_3 x_3 + \beta_{\text{int}} \cdot x_{\text{int}}$

### Example 2: Homotypic Pair ($k=2$, $u=1$)

Same motif twice: $(m, m)$. Count: $c_1 = 2$.

| Obs | $x_1$ | $x_{\text{int}}$ |
|-----|-------|------------------|
| Singleton $m$ | 1 | 0 |
| Pair $(m,m)$ | 2 | 1 |

Model: $y = \beta_1 x_1 + \beta_{\text{int}} \cdot x_{\text{int}}$

Predicted pair value: $2\beta_1 + \beta_{\text{int}}$. Tests whether two copies deviate from $2\times$ one copy.

### Example 3: Mixed 4-tuple ($k=4$, $u=2$)

Motifs $(A, A, A, B)$. Counts: $c_A = 3, c_B = 1$.

| Obs | $x_A$ | $x_B$ | $x_{\text{int}}$ |
|-----|-------|-------|------------------|
| Singleton $A$ | 1 | 0 | 0 |
| Singleton $B$ | 0 | 1 | 0 |
| 4-tuple | 3 | 1 | 1 |

Model: $y = \beta_A x_A + \beta_B x_B + \beta_{\text{int}} \cdot x_{\text{int}}$

Predicted 4-tuple: $3\beta_A + \beta_B + \beta_{\text{int}}$

---

## Statistical Testing

### Test Statistic

For the interaction coefficient:
$$t = \frac{\hat{\beta}_{\text{int}}}{\text{SE}(\hat{\beta}_{\text{int}})}$$

The p-value is computed from the t-distribution with appropriate degrees of freedom.

### Multiple Testing Correction

Since we test many motif $k$-tuples simultaneously, we apply **Benjamini-Hochberg FDR correction**:

1. Sort p-values: $p_{(1)} \leq p_{(2)} \leq \cdots \leq p_{(m)}$
2. Find largest $i$ such that $p_{(i)} \leq \frac{i}{m} \alpha$
3. Reject all hypotheses with $p_{(j)} \leq p_{(i)}$

Default significance threshold: $\alpha = 10^{-10}$ (FDR-adjusted). Only significant results are retained.

---

## Output

For each tested $k$-way interaction, we report:

| Field | Description |
|-------|-------------|
| `m1`, `m2`, ..., `mk` | Motif filter indices in the $k$-tuple |
| `β_interaction` | Estimated interaction coefficient |
| `se` | Standard error of the estimate |
| `t_stat` | t-statistic |
| `p_value` | Raw p-value |
| `p_adjusted` | FDR-adjusted p-value |
| `n_obs` | Number of observations in the regression |

Results are returned as a vector of dictionaries, one per $k$ value:
- `summary_strs[1]` → pairs ($k=2$)
- `summary_strs[2]` → triplets ($k=3$)
- `summary_strs[i]` → $(i+1)$-tuples

---

## Example Interpretations

### Synergistic Heterotypic Pair
```
m1=5, m2=12, β_interaction=+0.35, p_adjusted=1.2e-8
```
Motifs 5 and 12 together contribute 0.35 units **more** than expected from their individual effects.

### Antagonistic Homotypic Pair
```
m1=7, m2=7, β_interaction=-0.15, p_adjusted=3e-3
```
Two copies of motif 7 contribute 0.15 units **less** than 2× the single-copy contribution.

### Synergistic 4-tuple
```
m1=3, m2=3, m3=3, m4=7, β_interaction=+0.52, p_adjusted=1e-12
```
Three copies of motif 3 plus motif 7 together contribute 0.52 units **more** than $(3\beta_3 + \beta_7)$.

---

## Caveats

1. **Direct interaction test**: We test whether the $k$-tuple is non-additive compared to the sum of individual motif effects. We do not decompose into lower-order interactions (pairwise within a triplet, etc.).

2. **Positional effects**: The current model doesn't account for the specific positions of motifs. Multiple occurrences of the same motif at different positions are treated as having identical singleton distributions.

3. **Sample size**: Interaction tests require sufficient observations in each category. Motif combinations with few observations may have unreliable estimates.

4. **Independence assumption**: Linear regression assumes independent observations. Banzhaf values from the same sequence may be correlated.
