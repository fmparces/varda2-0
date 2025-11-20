# Varda 2.0: Network-Based Factor Model for Capital Markets

## Mission

**Instead of treating factors as a long flat list in a regression, Varda 2.0 models them as nodes in a capital-markets network.** Each factor connects to issuers, macro states, and deals through weighted relationships, and factor shocks are propagated through the graph like fluid. This lets Varda measure a factor's true contribution to default risk, spreads, and fee-at-risk, not just its t-stat in a backtest.

**By comparing the network impact of different factors, clustering redundant ones, and making the analysis regime-aware via Markov chains, Varda compresses the "factor zoo" into a small set of structural risk-propagation channels that actually move deal economics and systemic risk.**

---

## How Varda Reframes the "Factor Zoo"

Classic factor models treat each characteristic (size, value, momentum, quality, investment, etc.) as a separate line in a regression. The "factor zoo" problem is that hundreds of such characteristics all look statistically "special" in isolation, but many are noisy, overlapping, or regime-dependent.

**Varda 2.0 takes a different approach:**

### 1. Factors as Nodes in an Information Network

Instead of a flat list, Varda treats each factor as an entity in a network:

**Nodes:**
- **Issuers** (stocks, credit names)
- **Macro states** (inflation, growth, liquidity regimes)
- **Factors** (Size, Value, Momentum, Quality, Investment, plus custom signals)

**Edges:**
- **Sensitivity / exposure** (e.g., "TechCorp → Momentum: 0.8", "TechCorp → Value: −0.5")
- **Structural links** (supply chain, financing, sector, funding dependencies)

In code terms, factors become `FactorNode` objects connected by `Relationship` objects with weights. Varda's risk-propagation engine then treats factor shocks as fluid injected into the network and measures how that fluid spreads into issuer and deal risk.

**Interpretation:**
Instead of asking *"Is this factor significant in a regression?"*, Varda asks:

> **"How much network risk does this factor actually transmit into the names and deals I care about?"**

### 2. Information Content, Not Just t-Stats

Most new factors are just small twists on the old ones. **Varda's philosophy is:**

> **If a factor doesn't carry new information into the network, it belongs back in the zoo cage.**

You can implement this idea in three steps:

1. **Shock one factor at a time**
   - Create a `FactorScenario` where only one factor node is shocked (e.g., Momentum +2σ, others neutral).
   - Use `propagate_factor_shock(...)` to see how that shock changes entity risk and tranche loss distributions.

2. **Compare with existing core factors**
   - Repeat the same exercise for core factors (Size, Value, Momentum, Quality, Investment).
   - If the new factor's impact pattern is almost the same as a combination of existing ones, it's information-redundant.

3. **Score factors by marginal "risk information"**
   - Define a metric like "incremental Expected Shortfall" or "incremental systemic risk" when the factor is added to the network.
   - Factors with negligible marginal impact are flagged as zoo animals, not core risk drivers.

### 3. Compressing the Zoo into a Smaller Risk Basis

A big theme in the academic literature is that hundreds of factors can be compressed into 5–15 core dimensions. **Varda gives you a structural way to do this:**

1. Use Varda to run Monte Carlo scenarios across many factors simultaneously.
2. For each issuer/deal, collect simulated changes in:
   - default probability
   - spread
   - loss metrics (EL, VaR, ES)
3. Then:
   - **Cluster factors based on the similarity of their network impact** (not just return correlations).
   - For each cluster, keep a single "basis factor" – the one that explains most of the risk propagation in that cluster.

**Result:**

You don't just say *"we reduced 300 factors to 10 PCs."*

You say:

> **"We reduced 300 characteristics to ~10 risk-propagation channels that actually move default risk and deal economics in Varda's network."**

### 4. Regime-Aware Factors via Markov Chains

Many factors "work" only in certain regimes (e.g., momentum dies in crises, value revives after big drawdowns). Varda already has:

- Market regime Markov chains (Normal / Stressed / Crisis, etc.)
- Market constraints (`MarketConstraint`) that tilt those regime transitions
- Scenarios (`FactorScenario`) that adjust PDs and spreads based on regime

You can plug factors into this:

1. **Treat some factors as regime-sensitive nodes:**
   - Momentum nodes linked strongly to "Normal / Liquidity-rich" states
   - Value nodes linked to "Recovery / Dislocation" states

2. **Use Varda's regime simulation to see:**
   - Which factors only matter in specific regimes
   - Which factors are robust across regimes

This helps separate **structural factors** (true drivers of risk across regimes) from **conditional factors** (only useful in a narrow window).

### 5. From Factor Zoo to Deal-Level Decisions

Finally, Varda translates factor complexity into deal-level outputs:

1. **Simulate how factor shocks change:**
   - tranche loss distributions (`simulate_tranche_loss_distribution`)
   - expected underwriting P&L and fee-at-risk (`compute_pipeline_fee_at_risk`)

2. **Ask:**
   - *"Does this factor meaningfully change the fee-at-risk for our high-yield pipeline?"*
   - *"If I drop this factor entirely from my views, do my deal decisions change?"*

**If the answer is "no," that factor is noise for your use case.**

---

## Installation

```bash
pip install numpy pandas scikit-learn scipy
```

**Optional dependencies:**
- `financial_risk_lab` - For base `Entity`, `Relationship`, `MarketConstraint`, `MarkovChain` classes
  - If not available, Varda 2.0 uses minimal fallback implementations

---

## Quick Start

### 1. Create a Varda 2.0 Instance

```python
from varda_2_0 import Varda2, FactorNode, FactorType

# Initialize Varda 2.0
varda2 = Varda2("My Factor Network Model")
```

### 2. Add Factors to the Network

```python
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
    name="Value Factor (P/B, P/E)",
    factor_type=FactorType.VALUE,
    base_value=0.0,
    volatility=1.2
)
# Link to regimes: value works better in crises/recoveries
varda2.add_factor(value_factor, regime_sensitivity={
    "normal": 1.0,
    "crisis": 1.5,  # Value revives in crises
    "recovery": 1.3
})

momentum_factor = FactorNode(
    id="momentum",
    name="12M Momentum Factor",
    factor_type=FactorType.MOMENTUM,
    base_value=0.0,
    volatility=1.5
)
# Momentum dies in crises
varda2.add_factor(momentum_factor, regime_sensitivity={
    "normal": 1.5,
    "liquidity_rich": 1.8,
    "crisis": 0.5  # Momentum dies in crises
})
```

### 3. Add Issuers and Link Them to Factors

```python
from varda import Entity

# Add issuers
techcorp = Entity(
    id="issuer_techcorp",
    name="TechCorp Inc",
    entity_type="issuer",
    initial_risk_score=0.2
)
varda2.add_issuer(techcorp)

# Link factors to issuers via exposures
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

varda2.link_factor_to_issuer(
    factor_id="size",
    issuer_id="issuer_techcorp",
    exposure=0.3  # Medium size exposure
)
```

### 4. Propagate Factor Shocks Through the Network

```python
# Shock the Momentum factor by +2 standard deviations
impact = varda2.propagate_factor_shock(
    factor_id="momentum",
    shock_size=2.0,  # +2σ shock
    regime_state="normal"  # Current regime
)

# See how it affects issuers
print(f"TechCorp impact: {impact.issuer_impacts.get('issuer_techcorp', 0.0):.4f}")
print(f"TechCorp PD change: {impact.pd_changes.get('issuer_techcorp', 0.0):.4%}")
print(f"Systemic risk change: {impact.systemic_risk_change:.4f}")
```

### 5. Compare Factor Network Impacts

```python
# Compare how different factors move network risk
comparison = varda2.compare_factor_network_impacts(
    factor_ids=["size", "value", "momentum"],
    shock_size=2.0,
    regime_state="normal"
)

print(comparison[["factor_name", "systemic_risk_change", "expected_shortfall_change"]])
```

### 6. Evaluate Factor Information Content

```python
# Check if a new factor is redundant with existing ones
new_factor_info = varda2.evaluate_factor_information_content(
    new_factor_id="profitability",
    core_factor_ids=["size", "value", "momentum"],
    shock_size=2.0,
    regime_state="normal"
)

if new_factor_info["is_zoo_animal"]:
    print(f"❌ {new_factor_info['factor_name']} is redundant with {new_factor_info['most_redundant_factor']}")
    print(f"   Redundancy: {new_factor_info['max_redundancy_with_core']:.2%}")
else:
    print(f"✅ {new_factor_info['factor_name']} carries independent information")
    print(f"   Marginal ES change: {new_factor_info['marginal_es_change']:.4f}")
```

### 7. Compress the Factor Zoo

```python
# Cluster factors and find core risk-propagation channels
compressed = varda2.compress_factor_zoo(
    factor_ids=None,  # All factors
    target_n_factors=10,  # Compress to 10 core factors
    shock_size=2.0,
    regime_state="normal"
)

print(f"Reduced {compressed['original_n_factors']} factors to {compressed['compressed_n_factors']}")
print(f"Compression ratio: {compressed['compression_ratio']:.2%}")
print(f"Risk preservation: {compressed['risk_preservation']:.2%}")
print(f"\nCore factors:")
for cluster_id, info in compressed['compression_summary'].items():
    print(f"  {cluster_id}: {info['representative']} (represents {info['n_factors']} factors)")
```

### 8. Regime-Aware Factor Analysis

```python
# Analyze which factors are regime-robust vs regime-conditional
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

print(f"\nStructural factors (robust across regimes): {factor_types['structural_factors']}")
print(f"Conditional factors (regime-dependent): {factor_types['conditional_factors']}")
```

### 9. Translate Factors to Deal Economics

```python
# See how factor shocks affect deal-level fee-at-risk
deal_econ = varda2.translate_factor_to_deal_economics(
    factor_id="momentum",
    deal_id="deal_hy1",
    shock_size=2.0,
    regime_state="normal"
)

print(f"Deal impact: {deal_econ['deal_impact']:.4f}")
print(f"Fee-at-risk change: ${deal_econ['fee_at_risk_change']:,.2f}")
print(f"Decision impact: {deal_econ['decision_impact']}")

# Evaluate for entire pipeline
pipeline_eval = varda2.evaluate_factor_for_pipeline(
    factor_id="momentum",
    deal_ids=None,  # All deals
    shock_size=2.0,
    regime_state="normal",
    fee_threshold=100_000.0  # Minimum $100k fee-at-risk to matter
)

if pipeline_eval["matters_for_pipeline"]:
    print(f"✅ {pipeline_eval['factor_name']} matters for pipeline")
    print(f"   Affects {pipeline_eval['n_deals_significantly_affected']} deals")
    print(f"   Total fee-at-risk change: ${pipeline_eval['total_fee_at_risk_change']:,.2f}")
else:
    print(f"❌ {pipeline_eval['factor_name']} has negligible pipeline impact")
```

---

## Core Classes

### `Varda2`

Main class for network-based factor modeling.

**Key Methods:**
- `add_factor(factor, regime_sensitivity)` - Add factor to network
- `link_factor_to_issuer(factor_id, issuer_id, exposure)` - Link factor to issuer
- `propagate_factor_shock(factor_id, shock_size, ...)` - Propagate shock through network
- `compare_factor_network_impacts(factor_ids, ...)` - Compare factor impacts
- `evaluate_factor_information_content(new_factor_id, ...)` - Check if factor is redundant
- `cluster_factors_by_network_impact(...)` - Cluster factors by impact patterns
- `compress_factor_zoo(...)` - Compress factor zoo into core channels
- `analyze_regime_sensitivity(factor_id, regime_states, ...)` - Analyze regime dependence
- `identify_regime_conditional_factors(...)` - Separate structural from conditional factors
- `translate_factor_to_deal_economics(factor_id, deal_id, ...)` - Translate to deal-level
- `evaluate_factor_for_pipeline(factor_id, ...)` - Evaluate pipeline impact

### `FactorNode`

Represents a factor as a node in the network.

**Attributes:**
- `id` - Unique identifier
- `name` - Human-readable name
- `factor_type` - FactorType enum (SIZE, VALUE, MOMENTUM, etc.)
- `base_value` - Base factor value
- `volatility` - Factor volatility (for shock sizing)
- `regime_sensitivity` - Dict mapping regime names to sensitivity weights

### `FactorScenario`

Scenario for factor shock analysis.

**Attributes:**
- `name` - Scenario name
- `description` - Scenario description
- `factor_shocks` - Dict mapping factor_id to shock (in σ)
- `regime_state` - Current market regime
- `market_constraints` - List of MarketConstraint objects

### `NetworkImpactResult`

Results from propagating a factor shock.

**Attributes:**
- `factor_id` - Factor that was shocked
- `issuer_impacts` - Dict mapping issuer_id to risk change
- `deal_impacts` - Dict mapping deal_id to risk change
- `pd_changes` - Dict mapping issuer_id to PD change
- `spread_changes` - Dict mapping tranche_id to spread change (bps)
- `systemic_risk_change` - Overall systemic risk change
- `expected_shortfall_change` - Tail risk change

### `FactorCluster`

Cluster of factors with similar network impact patterns.

**Attributes:**
- `cluster_id` - Cluster identifier
- `factor_ids` - List of factor IDs in cluster
- `representative_factor_id` - Factor that best represents cluster
- `impact_pattern` - Representative impact vector
- `regime_robustness` - Dict mapping regime to robustness score

---

## Key Concepts

### Factor Network

Factors are nodes in a network, connected to:
- **Issuers** via exposure relationships
- **Deals** via deal exposure relationships
- **Regimes** via regime sensitivity links
- **Other factors** via factor-factor relationships

### Risk Propagation

Factor shocks propagate through the network like fluid:
1. Shock is injected at a factor node
2. Risk diffuses to connected issuers/deals
3. Regime sensitivity modulates propagation strength
4. Network topology determines contagion paths

### Information Content

A factor carries independent information if:
1. Its network impact pattern is distinct from existing factors
2. Marginal Expected Shortfall change is significant
3. Low redundancy with core factors (<85% similarity)

### Factor Clustering

Factors are clustered by:
- **Network impact similarity** (not return correlations)
- Similar issuer/deal impact patterns
- Similar systemic risk propagation channels

### Regime Robustness

- **Structural factors**: Robust across regimes (robustness > 0.7)
- **Conditional factors**: Only matter in specific regimes (robustness < 0.7)

---

## Workflow Examples

### Example 1: Adding a New Factor and Testing Information Content

```python
# Add a new "profitability" factor
profitability = FactorNode(
    id="profitability",
    name="Profitability Factor (ROE, ROA)",
    factor_type=FactorType.PROFITABILITY,
    base_value=0.0,
    volatility=1.1
)
varda2.add_factor(profitability)

# Link to issuers
varda2.link_factor_to_issuer("profitability", "issuer_techcorp", exposure=0.6)

# Evaluate information content
info_content = varda2.evaluate_factor_information_content(
    new_factor_id="profitability",
    core_factor_ids=["size", "value", "momentum"],
    shock_size=2.0
)

if not info_content["is_zoo_animal"]:
    print(f"✅ {info_content['factor_name']} adds value")
    print(f"   Marginal ES: {info_content['marginal_es_change']:.4f}")
else:
    print(f"❌ Redundant with {info_content['most_redundant_factor']}")
```

### Example 2: Compressing a Large Factor Set

```python
# Add many factors (example with 50 factors)
for i in range(50):
    factor = FactorNode(
        id=f"custom_factor_{i}",
        name=f"Custom Factor {i}",
        factor_type=FactorType.CUSTOM,
        base_value=0.0,
        volatility=1.0 + np.random.rand() * 0.5
    )
    varda2.add_factor(factor)
    # Link to random issuers
    for issuer_id in varda2.issuers.keys():
        if np.random.rand() < 0.3:  # 30% chance of link
            varda2.link_factor_to_issuer(factor.id, issuer_id, exposure=np.random.randn() * 0.5)

# Compress to 10 core factors
compressed = varda2.compress_factor_zoo(
    target_n_factors=10,
    shock_size=2.0
)

print(f"Compressed {compressed['original_n_factors']} → {compressed['compressed_n_factors']} factors")
print(f"Risk preservation: {compressed['risk_preservation']:.2%}")
```

### Example 3: Regime-Aware Factor Selection

```python
# Analyze all factors across regimes
regime_states = ["normal", "liquidity_rich", "crisis", "recovery"]

factor_types = varda2.identify_regime_conditional_factors(
    factor_ids=None,
    regime_states=regime_states,
    robustness_threshold=0.7,
    shock_size=2.0
)

# Use only structural factors for long-term risk models
structural_factors = factor_types["structural_factors"]
print(f"Structural factors (use in all regimes): {structural_factors}")

# Use conditional factors only for regime-specific analysis
conditional_factors = factor_types["conditional_factors"]
print(f"Conditional factors (regime-specific): {conditional_factors}")
```

---

## Integration with Varda 1.0

Varda 2.0 is designed to integrate with Varda 1.0's deal and tranche infrastructure:

```python
from varda import Varda, CapitalMarketsDeal, Tranche, DealType
from varda_2_0 import Varda2

# Create Varda 1.0 instance for deals
varda1 = Varda()

# Create Varda 2.0 instance for factor network
varda2 = Varda2()

# Add deal to Varda 1.0
deal = CapitalMarketsDeal(
    id="deal_hy1",
    issuer_entity_id="issuer_techcorp",
    deal_type=DealType.DCM_HY,
    tranches=[...],
    bookrunners=["bank1"],
    gross_fees=12_500_000
)
varda1.add_deal(deal)

# Add same deal to Varda 2.0
varda2.add_deal(deal)

# Link factors to deal
varda2.link_factor_to_deal("momentum", "deal_hy1", exposure=0.7)

# Propagate factor shock and see deal impact
impact = varda2.propagate_factor_shock("momentum", shock_size=2.0)
deal_impact = impact.deal_impacts.get("deal_hy1", 0.0)
print(f"Deal {deal.id} impact: {deal_impact:.4f}")

# Translate to fee-at-risk
deal_econ = varda2.translate_factor_to_deal_economics(
    factor_id="momentum",
    deal_id="deal_hy1",
    shock_size=2.0
)
print(f"Fee-at-risk change: ${deal_econ['fee_at_risk_change']:,.2f}")
```

---

## Advanced Features

### Custom Factor Types

```python
# Add custom factor types
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

### Factor-Factor Relationships

Factors can influence each other (e.g., value and quality are correlated):

```python
# Link factors to each other
from varda import Relationship

# Quality factor often correlates with Value
quality_factor = FactorNode(
    id="quality",
    name="Quality Factor (Debt/Equity, ROE)",
    factor_type=FactorType.QUALITY,
    base_value=0.0,
    volatility=1.1
)
varda2.add_factor(quality_factor)

# Create relationship: value → quality
relationship = Relationship(
    source_id="factor_value",
    target_id="factor_quality",
    strength=0.6,
    relationship_type="factor_correlation"
)
varda2.factor_relationships["value"].append(("factor_quality", 0.6))
varda2.relationships.append(relationship)
```

### Monte Carlo Factor Scenarios

```python
import numpy as np

# Run Monte Carlo with multiple factor shocks
n_simulations = 1000
systemic_risks = []

for sim in range(n_simulations):
    # Random factor shocks
    factor_shocks = {
        "size": np.random.randn() * 1.0,
        "value": np.random.randn() * 1.2,
        "momentum": np.random.randn() * 1.5
    }
    
    # Aggregate impact (simplified)
    total_impact = 0.0
    for factor_id, shock_size in factor_shocks.items():
        impact = varda2.propagate_factor_shock(
            factor_id=factor_id,
            shock_size=shock_size
        )
        total_impact += impact.systemic_risk_change
    
    systemic_risks.append(total_impact)

# Analyze distribution
systemic_risks = np.array(systemic_risks)
print(f"Mean systemic risk: {systemic_risks.mean():.4f}")
print(f"95th percentile: {np.percentile(systemic_risks, 95):.4f}")
print(f"Expected Shortfall (95%): {systemic_risks[systemic_risks >= np.percentile(systemic_risks, 95)].mean():.4f}")
```

---

## Best Practices

1. **Start with core factors**: Size, Value, Momentum, Quality, Investment
2. **Link factors to issuers based on data**: Use exposure betas from regressions or fundamental analysis
3. **Test information content before adding**: Use `evaluate_factor_information_content()` to avoid redundancy
4. **Compress periodically**: Run `compress_factor_zoo()` as you add factors to find core channels
5. **Analyze regime sensitivity**: Use `analyze_regime_sensitivity()` to understand when factors matter
6. **Translate to deal economics**: Always check `evaluate_factor_for_pipeline()` to see if factors matter for decisions

---

## FAQ

**Q: How do I convert regression betas to factor exposures?**

A: Use beta coefficients directly as exposures:
```python
# From regression: TechCorp ~ 0.8 * Momentum - 0.5 * Value + 0.3 * Size
varda2.link_factor_to_issuer("momentum", "issuer_techcorp", exposure=0.8)
varda2.link_factor_to_issuer("value", "issuer_techcorp", exposure=-0.5)
varda2.link_factor_to_issuer("size", "issuer_techcorp", exposure=0.3)
```

**Q: How do I calibrate PD changes from factor impacts?**

A: The current implementation uses a simplified linear mapping. In practice, you would:
1. Use historical data to estimate PD sensitivity to factor shocks
2. Calibrate a model (e.g., logit) mapping factor impacts to PD changes
3. Update the `propagate_factor_shock()` method to use calibrated models

**Q: Can I use Varda 2.0 without Varda 1.0?**

A: Yes! Varda 2.0 can work standalone, though it integrates best with Varda 1.0's deal infrastructure.

**Q: How do I handle regime transitions?**

A: Varda 2.0 supports regime-sensitive factors via `regime_sensitivity` in `add_factor()`. For dynamic regime transitions, integrate with Varda 1.0's Markov chain infrastructure.

---

## References

- **Factor Zoo Literature**: Harvey, C. R., Liu, Y., & Zhu, H. (2016). ... and the Cross-Section of Expected Returns. *Review of Financial Studies*, 29(1), 5-68.
- **Network Models**: Acemoglu, D., Carvalho, V. M., Ozdaglar, A., & Tahbaz-Salehi, A. (2012). The Network Origins of Aggregate Fluctuations. *Econometrica*, 80(5), 1977-2016.
- **Regime-Aware Factors**: Ang, A., & Bekaert, G. (2002). International Asset Allocation with Regime Shifts. *Review of Financial Studies*, 15(4), 1137-1187.

---

## License

[Add your license here]

## Author

Varda 2.0 - Network-Based Factor Model for Capital Markets

---

**Version**: 2.0  
**Last Updated**: 2025-11-19  
**Status**: Active Development

