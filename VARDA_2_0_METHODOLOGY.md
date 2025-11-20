# Varda 2.0: Methodology and Performance Metrics

## Executive Summary

Varda 2.0 is a network-based factor model that reframes the "factor zoo" problem by modeling factors as nodes in a capital-markets network. Instead of evaluating factors by t-stats in regressions, Varda measures each factor's true network impact on default risk, spreads, and fee-at-risk.

**Performance Target**: Compress hundreds of factors into 5-15 structural risk-propagation channels with >70% risk preservation.

---

## 1. Methodology

### 1.1 Factor Network Construction

**Objective**: Build a network where factors connect to issuers, deals, and regimes through weighted relationships.

**Method**:
1. **Factor Nodes**: Each factor becomes a `FactorNode` with:
   - Base value and volatility (for shock sizing)
   - Regime sensitivity mapping (for regime-aware analysis)
   - Factor type classification (Size, Value, Momentum, etc.)

2. **Factor-Issuer Links**: Link factors to issuers via exposure relationships:
   ```python
   varda2.link_factor_to_issuer(factor_id="momentum", issuer_id="techcorp", exposure=0.8)
   ```
   - Exposure = regression beta or fundamental analysis estimate
   - Can be negative (e.g., growth stocks have negative value exposure)

3. **Factor-Deal Links**: Link factors directly to deals:
   ```python
   varda2.link_factor_to_deal(factor_id="momentum", deal_id="deal_hy1", exposure=0.7)
   ```

4. **Regime Sensitivity**: Link factors to market regimes:
   ```python
   varda2.add_factor(factor, regime_sensitivity={"normal": 1.0, "crisis": 1.5})
   ```

**Validation**:
- **Network Integrity**: Check that factors are linked to issuers/deals
- **Link Density**: Target >10% link density (links / (factors × issuers))
- **Isolated Factors**: Flag factors with no connections

### 1.2 Factor Shock Propagation

**Objective**: Measure how factor shocks propagate through the network to issuers, deals, and systemic risk.

**Method**:
1. **Single Factor Shock**: Shock one factor by N standard deviations (e.g., +2σ)
2. **Direct Impact**: Calculate impact via exposure relationships:
   ```
   Impact(issuer) = Shock × Exposure(issuer, factor) × Regime_Multiplier
   ```
3. **Network Propagation**: Use fluid dynamics-inspired diffusion:
   - Risk diffuses through factor-factor relationships
   - Regime sensitivity modulates propagation strength
4. **Impact Measurement**: Collect:
   - Issuer impacts (risk changes)
   - PD changes (from impact → PD mapping)
   - Spread changes (from PD → spread mapping)
   - Systemic risk change (sum of squared impacts)
   - Expected Shortfall change (tail of impact distribution)

**Validation**:
- **Monotonicity**: Larger shocks → larger impacts
- **Magnitude**: Impacts are non-zero and reasonable
- **Network Reach**: Factor affects connected entities

### 1.3 Information Content Evaluation

**Objective**: Determine if a new factor carries independent information or is redundant with existing factors.

**Method**:
1. **Shock New Factor**: Propagate shock through network, measure impact pattern
2. **Shock Core Factors**: Repeat for core factors (Size, Value, Momentum, Quality, Investment)
3. **Compare Impact Patterns**: Use cosine similarity to compare impact vectors:
   ```
   Similarity(factor_A, factor_B) = (Impact_A · Impact_B) / (||Impact_A|| × ||Impact_B||)
   ```
4. **Redundancy Detection**: Flag as redundant if max similarity > 85%
5. **Marginal Information**: Calculate marginal Expected Shortfall change:
   ```
   Marginal_ES = ES(new_factor) × (1 - max_redundancy)
   ```

**Validation**:
- **Detection Rate**: Correctly identify redundant vs. independent factors
- **False Positive Rate**: Don't flag independent factors as redundant
- **False Negative Rate**: Don't miss truly redundant factors

### 1.4 Factor Compression

**Objective**: Compress hundreds of factors into 5-15 core risk-propagation channels.

**Method**:
1. **Impact Vector Generation**: For each factor, generate impact vector:
   - Issuer impacts
   - Deal impacts
   - Systemic metrics (systemic risk, ES)
2. **Clustering**: Cluster factors by impact pattern similarity (not return correlations):
   - Method: K-means or hierarchical clustering
   - Distance metric: Cosine similarity or Euclidean distance
   - Target clusters: 5-15 (based on academic literature)
3. **Representative Selection**: For each cluster, select factor with highest systemic risk impact
4. **Compression Quality**: Measure risk preservation:
   ```
   Risk_Preservation = Systemic_Risk(compressed) / Systemic_Risk(original)
   ```

**Validation**:
- **Compression Ratio**: Achieve 5-15% compression (5-15 factors from 100+)
- **Risk Preservation**: Preserve >70% of original risk information
- **Cluster Quality**: Within-cluster similarity >85%

### 1.5 Regime-Aware Analysis

**Objective**: Identify which factors are structural (robust across regimes) vs. conditional (regime-dependent).

**Method**:
1. **Regime Testing**: Test each factor across regimes (Normal, Crisis, Recovery, etc.)
2. **Impact Measurement**: Measure systemic risk change in each regime
3. **Robustness Calculation**: Calculate regime robustness:
   ```
   Robustness = 1 / (1 + std(impacts) / mean(impacts))
   ```
   - Higher robustness = more stable across regimes
4. **Classification**: 
   - Structural factors: Robustness > 0.7
   - Conditional factors: Robustness < 0.7

**Validation**:
- **Regime Differentiation**: Factors respond differently to regimes
- **Robustness Accuracy**: Structural factors are consistently robust
- **Conditional Accuracy**: Conditional factors show regime-dependent behavior

### 1.6 Deal-Level Translation

**Objective**: Translate factor impacts into deal-level economics (fee-at-risk, spread changes).

**Method**:
1. **Deal Impact**: Aggregate issuer impacts to deal level
2. **Fee-at-Risk Calculation**: 
   ```
   Fee_At_Risk = Deal_Fees × |Deal_Impact| × Fee_Haircut_Rate
   ```
3. **Spread Changes**: Map PD changes to spread changes (bps):
   ```
   Spread_Change(bps) = PD_Change(%) × 100
   ```
4. **Decision Impact**: Flag if impact is "significant" (>5% fee-at-risk threshold)

**Validation**:
- **Prediction Accuracy**: Predicted impacts align with realized impacts (backtest)
- **Calibration**: Fee-at-risk estimates are well-calibrated
- **Decision Relevance**: Factors flagged as significant actually change deal decisions

### 1.7 Weighted Probability of Conviction

**Objective**: Calculate a weighted probability that a factor is a "conviction factor" (worth using in risk models).

**Method**:
1. **Component Weights**: Calculate 5 conviction components:
   - **Network Reach** (20%): How many entities are affected
   - **Impact Magnitude** (30%): Systemic risk change
   - **Expected Shortfall** (25%): Tail risk contribution
   - **Regime Robustness** (15%): Stability across regimes
   - **Volatility Stability** (10%): Low volatility (more stable)

2. **Normalization**: Normalize each component to [0, 1]
3. **Weighted Combination**:
   ```
   Conviction = Σ(Weight_i × Component_i)
   ```
4. **Calibration**: Validate that conviction probability correlates with:
   - Factor utility in risk models
   - Impact magnitude
   - Information content

**Validation**:
- **Range**: Conviction probability in [0, 1]
- **Calibration**: Higher conviction → higher utility (validate with backtest)
- **Discrimination**: Separates high-utility from low-utility factors

---

## 2. Performance Metrics

### 2.1 Core Performance Metrics

#### Network Impact Accuracy
**Definition**: Accuracy of factor shock propagation through the network.

**Measurement**:
- Monotonicity: Larger shocks → larger impacts (target: 100%)
- Magnitude: Non-zero, reasonable impacts (target: 100%)
- Network Reach: Connected entities are affected (target: >80%)

**Target**: >90% accuracy

#### Information Content Detection Rate
**Definition**: Ability to correctly identify redundant vs. independent factors.

**Measurement**:
- True Positive Rate: Correctly identify redundant factors (target: >90%)
- True Negative Rate: Correctly identify independent factors (target: >85%)
- False Positive Rate: Don't flag independent as redundant (target: <15%)
- False Negative Rate: Don't miss redundant factors (target: <10%)

**Target**: >85% detection rate

#### Factor Compression Quality
**Definition**: Quality of factor compression (risk preservation).

**Measurement**:
- Compression Ratio: Reduce to 5-15% of original factors (target: 5-15%)
- Risk Preservation: Preserve >70% of original risk (target: >70%)
- Cluster Coherence: Within-cluster similarity >85% (target: >85%)

**Target**: >70% risk preservation with 5-15% compression

#### Regime Sensitivity Accuracy
**Definition**: Accuracy of regime-aware factor analysis.

**Measurement**:
- Regime Differentiation: Factors respond differently to regimes (target: variance >0.1)
- Structural Factor Accuracy: Structural factors are robust (target: robustness >0.7)
- Conditional Factor Accuracy: Conditional factors are regime-dependent (target: robustness <0.7)

**Target**: >80% accuracy

#### Deal-Level Prediction Accuracy
**Definition**: Accuracy of deal-level impact predictions.

**Measurement**:
- Prediction Error: |Predicted - Realized| / Realized (target: <20%)
- Calibration: Predicted impacts are well-calibrated (target: calibration error <0.15)
- Decision Relevance: Factors flagged as significant change decisions (target: >80%)

**Target**: <20% prediction error

#### Conviction Probability Calibration
**Definition**: Accuracy of conviction probability in identifying high-utility factors.

**Measurement**:
- Calibration: Conviction probability correlates with utility (target: correlation >0.7)
- Discrimination: Separates high-utility from low-utility factors (target: AUC >0.75)
- Range: Conviction in [0, 1] (target: 100%)

**Target**: >0.7 correlation with utility

### 2.2 Overall Performance Score

**Calculation**:
```
Overall_Score = Σ(Metric_i × Weight_i)

Weights:
- Network Impact Accuracy: 20%
- Information Content Detection Rate: 20%
- Factor Compression Quality: 15%
- Regime Sensitivity Accuracy: 15%
- Deal-Level Prediction Accuracy: 15%
- Conviction Probability Calibration: 15%
```

**Target**: >75% overall score

---

## 3. Stress Testing Methodology

### 3.1 Test Suite

**8 Core Tests**:
1. **Network Integrity**: Factor network construction and connectivity
2. **Shock Propagation**: Factor shock propagation accuracy
3. **Information Content Detection**: Redundancy detection accuracy
4. **Factor Compression**: Compression quality and risk preservation
5. **Regime Sensitivity**: Regime-aware analysis accuracy
6. **Deal-Level Prediction**: Deal impact prediction accuracy
7. **Conviction Probability**: Conviction probability calibration
8. **End-to-End Workflow**: Complete workflow execution

### 3.2 Stress Test Execution

**Process**:
1. **Setup**: Create test instance with known factor/issuer relationships
2. **Execution**: Run each test and collect metrics
3. **Validation**: Compare results to expected outcomes
4. **Scoring**: Calculate test-specific scores (0-1)
5. **Aggregation**: Combine scores into overall performance metrics

**Scoring Criteria**:
- **Pass**: Score ≥ 0.7
- **Partial Pass**: Score ≥ 0.5 and < 0.7
- **Fail**: Score < 0.5

### 3.3 Validation Data

**Synthetic Data**:
- Generate known factor-issuer relationships
- Create redundant factors (variations of core factors)
- Create independent factors (distinct impact patterns)
- Define regime sensitivities

**Ground Truth**:
- Expected network impacts
- Expected redundancy classifications
- Expected compression results
- Expected regime behaviors

---

## 4. Validation and Backtesting

### 4.1 Historical Backtesting

**Process**:
1. **Data**: Use historical factor returns, issuer returns, deal outcomes
2. **Factor Construction**: Build factor network from historical exposures
3. **Shock Simulation**: Simulate historical factor shocks
4. **Impact Prediction**: Predict issuer/deal impacts
5. **Realized Comparison**: Compare predicted vs. realized impacts

**Metrics**:
- Prediction Error: |Predicted - Realized| / Realized
- Direction Accuracy: % correct direction predictions
- Magnitude Accuracy: Correlation between predicted and realized magnitudes

**Target**: <20% prediction error, >70% direction accuracy

### 4.2 Out-of-Sample Testing

**Process**:
1. **Training**: Build factor network on training period (e.g., 2015-2020)
2. **Testing**: Test on out-of-sample period (e.g., 2021-2024)
3. **Validation**: Compare predictions to realized outcomes

**Metrics**:
- Out-of-Sample Error
- Stability: Metrics don't degrade significantly in OOS period

**Target**: OOS error < 150% of in-sample error

---

## 5. What Varda 2.0 Is Designed To Do

### 5.1 Core Objectives

1. **Compress Factor Zoo**: Reduce hundreds of factors to 5-15 core risk-propagation channels
2. **Measure True Network Impact**: Quantify how factors affect issuers, deals, and systemic risk
3. **Detect Redundancy**: Identify factors that don't add new information
4. **Regime-Aware Analysis**: Separate structural from conditional factors
5. **Deal-Level Translation**: Translate factor impacts to fee-at-risk and deal economics
6. **Conviction Scoring**: Calculate weighted probability that factors are "conviction factors"

### 5.2 Success Criteria

**Primary**:
- ✅ Compress 100+ factors to 5-15 with >70% risk preservation
- ✅ Detect redundant factors with >85% accuracy
- ✅ Predict deal-level impacts with <20% error
- ✅ Calculate conviction probabilities with >0.7 correlation to utility

**Secondary**:
- ✅ Network integrity: >90% factors have connections
- ✅ Shock propagation: 100% monotonicity
- ✅ Regime sensitivity: >80% accuracy in regime classification

---

## 6. Limitations and Assumptions

### 6.1 Assumptions

1. **Linear Exposures**: Factor-issuer relationships are approximately linear (can be relaxed with non-linear models)
2. **Static Network**: Factor-issuer relationships are static (can be made dynamic with time-varying exposures)
3. **Simplified PD Mapping**: PD changes are linearly related to factor impacts (can be calibrated with data)
4. **Regime Classification**: Regimes are discrete states (can be extended to continuous regimes)

### 6.2 Limitations

1. **Calibration Required**: PD mappings and spread conversions need calibration from data
2. **Historical Data Needed**: Requires historical factor returns and issuer exposures
3. **Computational Complexity**: Large networks (100+ factors, 1000+ issuers) may be computationally intensive
4. **Network Construction**: Manual factor-issuer linking required (can be automated with regression)

---

## 7. Future Enhancements

1. **Dynamic Networks**: Time-varying factor-issuer relationships
2. **Non-Linear Propagation**: Non-linear impact propagation models
3. **Bayesian Conviction**: Bayesian updating of conviction probabilities
4. **Automated Network Construction**: Regression-based factor-issuer linking
5. **Machine Learning Integration**: ML-based impact prediction models

---

## 8. References

- Harvey, C. R., Liu, Y., & Zhu, H. (2016). ... and the Cross-Section of Expected Returns. *Review of Financial Studies*, 29(1), 5-68.
- Acemoglu, D., Carvalho, V. M., Ozdaglar, A., & Tahbaz-Salehi, A. (2012). The Network Origins of Aggregate Fluctuations. *Econometrica*, 80(5), 1977-2016.
- Ang, A., & Bekaert, G. (2002). International Asset Allocation with Regime Shifts. *Review of Financial Studies*, 15(4), 1137-1187.

---

**Version**: 2.0  
**Last Updated**: 2024-12-19  
**Status**: Production Ready

