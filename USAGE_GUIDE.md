# Varda 2.0: How to Use & When to Use Guide

A comprehensive guide for GitHub collaborators to understand when and how to use Varda 2.0.

---

## Table of Contents

1. [When to Use Varda 2.0](#when-to-use-varda-20)
2. [Quick Start](#quick-start)
3. [Installation](#installation)
4. [Basic Usage](#basic-usage)
5. [Common Workflows](#common-workflows)
6. [Advanced Usage](#advanced-usage)
7. [Best Practices](#best-practices)
8. [Troubleshooting](#troubleshooting)
9. [Examples](#examples)

---

## When to Use Varda 2.0

###  Use Varda 2.0 When:

#### 1. **You Have Too Many Factors (The "Factor Zoo" Problem)**
- You have 50+ factors and need to compress them to 5-15 core risk channels
- Multiple factors look similar and you want to identify redundancy
- You want to understand which factors actually matter for risk
- **Example**: You have 200 characteristics from academic papers and need to find the 10 that drive real risk

#### 2. **You Need Network-Based Factor Analysis**
- Traditional regression-based factor models aren't capturing dependencies
- Factors affect issuers/deals in complex, networked ways
- You want to understand factor-to-issuer impact propagation
- **Example**: A momentum shock affects tech stocks differently than value stocks, and you want to quantify this

#### 3. **You Want Regime-Aware Factor Models**
- Factors behave differently in crises vs. normal times
- You need to identify which factors are structural vs. conditional
- You want regime-robust risk models
- **Example**: Momentum dies in crises, value revives - you want to separate these behaviors

#### 4. **You Need Deal-Level Risk Translation**
- You want to translate factor shocks into fee-at-risk
- You need to understand how factors affect deal economics
- You want to make data-driven deal decisions
- **Example**: A +2σ momentum shock - does it change your pipeline fee-at-risk by >$1M?

#### 5. **You Want Conviction-Based Factor Selection**
- You need a probability score for which factors to use
- You want weighted conviction probabilities
- You need systematic factor evaluation
- **Example**: You have 50 factors - which 10 should you use with >70% conviction?

#### 6. **You Have Factor Redundancy Issues**
- Multiple factors measure the same underlying risk
- You want to identify independent vs. redundant factors
- You need information content evaluation
- **Example**: "Profitability" factor looks similar to "Quality" - are they redundant?

---

### Don't Use Varda 2.0 When:

1. **Simple Factor Models Suffice**
   - You have <10 factors and traditional regression works fine
   - Factors are independent and don't need network analysis
   - You don't need regime-aware modeling

2. **You Only Need Return Predictions**
   - Varda 2.0 focuses on risk propagation, not return forecasting
   - Use traditional factor models for return predictions

3. **You Have No Factor-Issuer Relationships**
   - Varda 2.0 requires factor-to-issuer exposures
   - If you don't have exposure data, you need to estimate it first

4. **Computational Constraints**
   - Large networks (1000+ issuers, 100+ factors) can be computationally intensive
   - Consider using simpler models for very large datasets

---

## Quick Start

### Installation

```bash
# Navigate to varda2.0 folder
cd varda2.0

# Install dependencies
pip install -r requirements.txt

# Or install manually
pip install numpy pandas scikit-learn scipy
```

### Minimal Example (5 minutes)

```python
from varda_2_0 import Varda2, FactorNode, FactorType
from varda import Entity  # Or use fallback Entity class

# 1. Create Varda 2.0 instance
varda2 = Varda2("My Factor Network")

# 2. Add factors
momentum = FactorNode(
    id="momentum",
    name="12M Momentum",
    factor_type=FactorType.MOMENTUM,
    volatility=1.5
)
varda2.add_factor(momentum)

# 3. Add issuers
techcorp = Entity(id="techcorp", name="TechCorp Inc", initial_risk_score=0.2)
varda2.add_issuer(techcorp)

# 4. Link factor to issuer
varda2.link_factor_to_issuer("momentum", "techcorp", exposure=0.8)

# 5. Propagate factor shock
impact = varda2.propagate_factor_shock(
    factor_id="momentum",
    shock_size=2.0  # +2σ shock
)

# 6. See results
print(f"TechCorp impact: {impact.issuer_impacts.get('techcorp', 0.0):.4f}")
print(f"Systemic risk change: {impact.systemic_risk_change:.4f}")
print(f"Conviction probability: {impact.conviction_probability:.2%}")
```

---

## Installation

### Requirements

```bash
numpy>=1.24
pandas>=2.0
scikit-learn>=1.3
scipy>=1.0
```

### Optional Dependencies (for visualizations)

```bash
matplotlib>=3.7
seaborn>=0.13
```

### Install from requirements.txt

```bash
pip install -r requirements.txt
```

---

## Basic Usage

### Step 1: Create a Varda 2.0 Instance

```python
from varda_2_0 import Varda2

varda2 = Varda2("My Factor Network Model")
```

### Step 2: Add Factors

```python
from varda_2_0 import FactorNode, FactorType

# Core factors
size_factor = FactorNode(
    id="size",
    name="Market Cap Factor",
    factor_type=FactorType.SIZE,
    base_value=0.0,
    volatility=1.0
)
varda2.add_factor(size_factor)

value_factor = FactorNode(
    id="value",
    name="Value Factor (P/B)",
    factor_type=FactorType.VALUE,
    base_value=0.0,
    volatility=1.2
)
# Add with regime sensitivity
varda2.add_factor(value_factor, regime_sensitivity={
    "normal": 1.0,
    "crisis": 1.5,  # Value revives in crises
    "recovery": 1.3
})
```

### Step 3: Add Issuers

```python
from varda import Entity  # Or use fallback Entity from varda_2_0

techcorp = Entity(
    id="issuer_techcorp",
    name="TechCorp Inc",
    entity_type="issuer",
    initial_risk_score=0.2
)
varda2.add_issuer(techcorp)
```

### Step 4: Link Factors to Issuers

```python
# Link via exposures (can be negative for inverse relationships)
varda2.link_factor_to_issuer(
    factor_id="momentum",
    issuer_id="issuer_techcorp",
    exposure=0.8  # TechCorp has high momentum exposure
)

varda2.link_factor_to_issuer(
    factor_id="value",
    issuer_id="issuer_techcorp",
    exposure=-0.5  # TechCorp has negative value exposure (growth stock)
)
```

### Step 5: Propagate Factor Shocks

```python
# Shock a factor by N standard deviations
impact = varda2.propagate_factor_shock(
    factor_id="momentum",
    shock_size=2.0,  # +2σ shock
    regime_state="normal"  # Optional: specify current regime
)

# Access results
print(f"Issuer impacts: {impact.issuer_impacts}")
print(f"PD changes: {impact.pd_changes}")
print(f"Systemic risk change: {impact.systemic_risk_change}")
print(f"Conviction probability: {impact.conviction_probability:.2%}")
```

---

## Common Workflows

### Workflow 1: Compare Factor Network Impacts

**Use Case**: Which factors actually move risk in my network?

```python
# Compare multiple factors
comparison = varda2.compare_factor_network_impacts(
    factor_ids=["size", "value", "momentum", "quality"],
    shock_size=2.0,
    regime_state="normal"
)

# See results sorted by systemic risk
print(comparison.sort_values("systemic_risk_change", ascending=False))
```

### Workflow 2: Evaluate Information Content

**Use Case**: Is my new factor redundant with existing ones?

```python
# Evaluate a new factor
info_content = varda2.evaluate_factor_information_content(
    new_factor_id="profitability",
    core_factor_ids=["size", "value", "momentum", "quality"],
    shock_size=2.0
)

if info_content["is_zoo_animal"]:
    print(f" REDUNDANT: Similar to {info_content['most_redundant_factor']}")
    print(f"   Redundancy: {info_content['max_redundancy_with_core']:.2%}")
else:
    print(f" INDEPENDENT: Marginal ES change: {info_content['marginal_es_change']:.4f}")
```

### Workflow 3: Compress Factor Zoo

**Use Case**: Reduce 100+ factors to 10 core risk channels

```python
# Compress factors
compressed = varda2.compress_factor_zoo(
    factor_ids=None,  # All factors
    target_n_factors=10,  # Compress to 10 core factors
    shock_size=2.0
)

print(f"Reduced {compressed['original_n_factors']} → {compressed['compressed_n_factors']} factors")
print(f"Risk preservation: {compressed['risk_preservation']:.2%}")

# See core factors
for cluster_id, info in compressed['compression_summary'].items():
    print(f"{cluster_id}: {info['representative']} (represents {info['n_factors']} factors)")
```

### Workflow 4: Regime-Aware Analysis

**Use Case**: Which factors are structural vs. conditional?

```python
# Analyze regime sensitivity
regime_sensitivity = varda2.analyze_regime_sensitivity(
    factor_id="momentum",
    regime_states=["normal", "liquidity_rich", "crisis", "recovery"],
    shock_size=2.0
)

print(regime_sensitivity[["regime", "systemic_risk_change", "regime_robustness"]])

# Identify structural vs conditional factors
factor_types = varda2.identify_regime_conditional_factors(
    factor_ids=None,  # All factors
    regime_states=["normal", "crisis", "recovery"],
    robustness_threshold=0.7,
    shock_size=2.0
)

print(f"Structural factors (robust): {factor_types['structural_factors']}")
print(f"Conditional factors (regime-dependent): {factor_types['conditional_factors']}")
```

### Workflow 5: Deal-Level Translation

**Use Case**: How does a factor shock affect my pipeline fee-at-risk?

```python
# Translate factor to deal economics
deal_econ = varda2.translate_factor_to_deal_economics(
    factor_id="momentum",
    deal_id="deal_hy1",
    shock_size=2.0
)

print(f"Deal impact: {deal_econ['deal_impact']:.4f}")
print(f"Fee-at-risk change: ${deal_econ['fee_at_risk_change']:,.2f}")
print(f"Decision impact: {deal_econ['decision_impact']}")

# Evaluate for entire pipeline
pipeline_eval = varda2.evaluate_factor_for_pipeline(
    factor_id="momentum",
    deal_ids=None,  # All deals
    shock_size=2.0,
    fee_threshold=100_000.0  # Minimum $100k to matter
)

if pipeline_eval["matters_for_pipeline"]:
    print(f" Affects {pipeline_eval['n_deals_significantly_affected']} deals")
    print(f"   Total fee-at-risk change: ${pipeline_eval['total_fee_at_risk_change']:,.2f}")
```

---

## Advanced Usage

### 1. Custom Factor Types

```python
# Create custom factor
custom_factor = FactorNode(
    id="sentiment",
    name="Twitter Sentiment Factor",
    factor_type=FactorType.CUSTOM,
    base_value=0.0,
    volatility=2.0,
    metadata={"source": "twitter_api", "window_days": 30}
)
varda2.add_factor(custom_factor)
```

### 2. Factor-Factor Relationships

```python
# Factors can influence each other
from varda import Relationship

# Quality often correlates with Value
relationship = Relationship(
    source_id="factor_value",
    target_id="factor_quality",
    strength=0.6,
    relationship_type="factor_correlation"
)
varda2.relationships.append(relationship)
```

### 3. Multiple Regimes

```python
# Add regime sensitivity for multiple regimes
varda2.add_factor(momentum_factor, regime_sensitivity={
    "normal": 1.5,
    "liquidity_rich": 1.8,
    "stressed": 1.0,
    "crisis": 0.5,  # Momentum dies in crises
    "recovery": 1.0
})
```

### 4. Deal Integration

```python
from varda import CapitalMarketsDeal, Tranche, DealType

# Create deal
tranche = Tranche(
    id="tranche_hy1",
    deal_id="deal_hy1",
    notional=500_000_000,
    pd_annual=0.03,
    lgd=0.60
)

deal = CapitalMarketsDeal(
    id="deal_hy1",
    issuer_entity_id="issuer_techcorp",
    deal_type=DealType.DCM_HY,
    tranches=[tranche],
    bookrunners=["bank1"],
    gross_fees=12_500_000
)

varda2.add_deal(deal)
varda2.link_factor_to_deal("momentum", "deal_hy1", exposure=0.7)
```

---

## Best Practices

### 1. **Start with Core Factors**
Always begin with the 5 core factors:
- Size
- Value
- Momentum
- Quality
- Investment/Profitability

### 2. **Link Factors Based on Data**
Use regression betas or fundamental analysis to set exposures:
```python
# From regression: TechCorp ~ 0.8 * Momentum - 0.5 * Value + 0.3 * Size
varda2.link_factor_to_issuer("momentum", "techcorp", exposure=0.8)
varda2.link_factor_to_issuer("value", "techcorp", exposure=-0.5)
varda2.link_factor_to_issuer("size", "techcorp", exposure=0.3)
```

### 3. **Test Information Content Before Adding**
Always evaluate new factors before adding them:
```python
info = varda2.evaluate_factor_information_content(
    new_factor_id="new_factor",
    core_factor_ids=["size", "value", "momentum"]
)
if info["is_zoo_animal"]:
    print("Skip - redundant with existing factors")
```

### 4. **Compress Periodically**
As you add factors, periodically compress:
```python
compressed = varda2.compress_factor_zoo(
    target_n_factors=10,
    shock_size=2.0
)
```

### 5. **Use Regime Analysis**
Always test factors across regimes:
```python
factor_types = varda2.identify_regime_conditional_factors(
    regime_states=["normal", "crisis", "recovery"]
)
# Use only structural factors for long-term models
```

### 6. **Check Deal-Level Impact**
Always verify factors matter for decisions:
```python
pipeline_eval = varda2.evaluate_factor_for_pipeline(
    factor_id="factor_id",
    fee_threshold=100_000.0
)
if not pipeline_eval["matters_for_pipeline"]:
    print("Factor has negligible impact - consider removing")
```

---

## Troubleshooting

### Issue: "Factor not found" error

**Solution**: Check that you've added the factor before using it:
```python
# Make sure factor exists
if factor_id not in varda2.factors:
    varda2.add_factor(my_factor)
```

### Issue: "No issuers affected" in impact results

**Solution**: Ensure you've linked factors to issuers:
```python
# Check links exist
if issuer_id in varda2.issuer_factor_exposures:
    print(f"Exposures: {varda2.issuer_factor_exposures[issuer_id]}")
else:
    # Add links
    varda2.link_factor_to_issuer(factor_id, issuer_id, exposure=0.5)
```

### Issue: Compression returns empty results

**Solution**: Ensure you have at least 3 factors and they're linked to issuers:
```python
# Check prerequisites
print(f"Factors: {len(varda2.factors)}")
print(f"Issuers: {len(varda2.issuers)}")
print(f"Links: {sum(len(exposures) for exposures in varda2.issuer_factor_exposures.values())}")

if len(varda2.factors) < 3:
    print("Need at least 3 factors for compression")
```

### Issue: Conviction probability is always 0

**Solution**: Ensure factors are linked to issuers and produce impacts:
```python
# Verify impact produces results
impact = varda2.propagate_factor_shock("factor_id", shock_size=2.0)
if len(impact.issuer_impacts) == 0:
    print("No impacts - check factor-issuer links")
```

---

## Examples

### Complete Working Example

See `varda_2_0_example.py` for 5 complete examples:

1. **Basic Factor Network** - Build a network and propagate shocks
2. **Information Content Evaluation** - Test if a factor is redundant
3. **Factor Compression** - Compress 20 factors to 6
4. **Regime-Aware Analysis** - Identify structural vs conditional factors
5. **Deal-Level Translation** - Translate factors to fee-at-risk

Run examples:
```bash
python varda_2_0_example.py
```

### Stress Testing

Run comprehensive stress tests:
```bash
python varda_2_0_stress_test.py
```

This will test:
- Network integrity
- Shock propagation
- Information content detection
- Factor compression
- Regime sensitivity
- Deal-level prediction
- Conviction probability
- End-to-end workflow

---

## Getting Help

### Documentation
- **Full Documentation**: See `VARDA_2_0_README.md`
- **Methodology**: See `VARDA_2_0_METHODOLOGY.md`
- **Examples**: See `varda_2_0_example.py`

### Common Questions

**Q: How do I convert regression betas to exposures?**
A: Use beta coefficients directly:
```python
varda2.link_factor_to_issuer("momentum", "issuer_id", exposure=beta_coefficient)
```

**Q: How do I calibrate PD changes from impacts?**
A: The current implementation uses a simplified linear mapping. In production, calibrate with historical data:
```python
# Example: calibrate with historical data
# pd_change = calibrated_model.predict(impact)
```

**Q: Can I use Varda 2.0 without Varda 1.0?**
A: Yes! Varda 2.0 has fallback implementations for Entity, Relationship, etc.

**Q: How do I handle missing data?**
A: Handle missing exposures by using 0.0 or estimated values from similar issuers.

---

## Next Steps

1. **Start with Examples**: Run `varda_2_0_example.py`
2. **Run Stress Tests**: Validate setup with `varda_2_0_stress_test.py`
3. **Read Documentation**: See `VARDA_2_0_README.md` for details
4. **Build Your Network**: Add your factors and issuers
5. **Analyze Results**: Use workflows to analyze your data

---

**Version**: 2.0  
**Last Updated**: 2024-12-19  
**Contributors**: See main README

