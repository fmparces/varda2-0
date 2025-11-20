"""
Varda 2.0 Example: Network-Based Factor Model

This example demonstrates the core features of Varda 2.0:
1. Building a factor network
2. Propagating factor shocks
3. Evaluating information content
4. Compressing the factor zoo
5. Regime-aware analysis
6. Deal-level translation
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from varda_2_0 import (
    Varda2, FactorNode, FactorType, FactorScenario, RegimeType
)

# Try to import Varda 1.0 classes for integration
try:
    from varda import Entity, CapitalMarketsDeal, Tranche, DealType
except ImportError:
    # Minimal fallback
    from dataclasses import dataclass
    from typing import Optional, List, Dict, Any
    
    @dataclass
    class Entity:
        id: str
        name: str
        entity_type: str = "issuer"
        initial_risk_score: float = 0.0
    
    @dataclass
    class Tranche:
        id: str
        deal_id: str
        notional: float
        pd_annual: float = 0.03
    
    class DealType:
        DCM_HY = "dcm_high_yield"
    
    @dataclass
    class CapitalMarketsDeal:
        id: str
        issuer_entity_id: str
        deal_type: str
        tranches: List[Tranche]
        bookrunners: List[str]
        gross_fees: float = 0.0


def example_1_basic_factor_network():
    """Example 1: Build a basic factor network and propagate shocks."""
    print("=" * 60)
    print("Example 1: Basic Factor Network")
    print("=" * 60)
    
    # Create Varda 2.0 instance
    varda2 = Varda2("Basic Factor Network Demo")
    
    # Add core factors
    factors = [
        FactorNode(id="size", name="Market Cap", factor_type=FactorType.SIZE, volatility=1.0),
        FactorNode(id="value", name="Value (P/B)", factor_type=FactorType.VALUE, volatility=1.2),
        FactorNode(id="momentum", name="12M Momentum", factor_type=FactorType.MOMENTUM, volatility=1.5),
        FactorNode(id="quality", name="Quality (ROE)", factor_type=FactorType.QUALITY, volatility=1.1),
    ]
    
    for factor in factors:
        varda2.add_factor(factor)
    
    # Add issuers
    issuers = [
        Entity(id="issuer_techcorp", name="TechCorp Inc", initial_risk_score=0.2),
        Entity(id="issuer_financecorp", name="FinanceCorp Inc", initial_risk_score=0.25),
        Entity(id="issuer_utilitycorp", name="UtilityCorp Inc", initial_risk_score=0.15),
    ]
    
    for issuer in issuers:
        varda2.add_issuer(issuer)
    
    # Link factors to issuers
    # TechCorp: high momentum, negative value (growth stock), large size
    varda2.link_factor_to_issuer("momentum", "issuer_techcorp", exposure=0.8)
    varda2.link_factor_to_issuer("value", "issuer_techcorp", exposure=-0.5)
    varda2.link_factor_to_issuer("size", "issuer_techcorp", exposure=0.4)
    varda2.link_factor_to_issuer("quality", "issuer_techcorp", exposure=0.6)
    
    # FinanceCorp: value stock, low momentum
    varda2.link_factor_to_issuer("value", "issuer_financecorp", exposure=0.7)
    varda2.link_factor_to_issuer("momentum", "issuer_financecorp", exposure=-0.3)
    varda2.link_factor_to_issuer("size", "issuer_financecorp", exposure=0.5)
    varda2.link_factor_to_issuer("quality", "issuer_financecorp", exposure=0.4)
    
    # UtilityCorp: low momentum, stable value
    varda2.link_factor_to_issuer("momentum", "issuer_utilitycorp", exposure=0.1)
    varda2.link_factor_to_issuer("value", "issuer_utilitycorp", exposure=0.3)
    varda2.link_factor_to_issuer("size", "issuer_utilitycorp", exposure=0.6)
    varda2.link_factor_to_issuer("quality", "issuer_utilitycorp", exposure=0.5)
    
    # Propagate Momentum factor shock (+2σ)
    print("\n1. Shocking Momentum factor by +2σ...")
    impact = varda2.propagate_factor_shock(
        factor_id="momentum",
        shock_size=2.0,
        regime_state="normal"
    )
    
    print(f"\nImpact on issuers:")
    for issuer_id, impact_value in impact.issuer_impacts.items():
        pd_change = impact.pd_changes.get(issuer_id, 0.0)
        print(f"  {issuer_id}: Impact = {impact_value:.4f}, PD change = {pd_change:.4%}")
    
    print(f"\nSystemic risk change: {impact.systemic_risk_change:.4f}")
    print(f"Expected shortfall change: {impact.expected_shortfall_change:.4f}")
    
    # Compare all factors
    print("\n2. Comparing network impacts of all factors...")
    comparison = varda2.compare_factor_network_impacts(
        factor_ids=["size", "value", "momentum", "quality"],
        shock_size=2.0
    )
    
    print(comparison[["factor_name", "systemic_risk_change", "expected_shortfall_change", "n_issuers_affected"]])
    
    return varda2, impact


def example_2_information_content():
    """Example 2: Evaluate factor information content."""
    print("\n" + "=" * 60)
    print("Example 2: Factor Information Content Evaluation")
    print("=" * 60)
    
    varda2, _ = example_1_basic_factor_network()
    
    # Add a new "profitability" factor
    profitability = FactorNode(
        id="profitability",
        name="Profitability (ROE, ROA)",
        factor_type=FactorType.PROFITABILITY,
        volatility=1.1
    )
    varda2.add_factor(profitability)
    
    # Link to issuers (similar to quality factor, so may be redundant)
    varda2.link_factor_to_issuer("profitability", "issuer_techcorp", exposure=0.55)  # Similar to quality
    varda2.link_factor_to_issuer("profitability", "issuer_financecorp", exposure=0.35)
    varda2.link_factor_to_issuer("profitability", "issuer_utilitycorp", exposure=0.48)
    
    # Evaluate information content
    print("\nEvaluating profitability factor information content...")
    info_content = varda2.evaluate_factor_information_content(
        new_factor_id="profitability",
        core_factor_ids=["size", "value", "momentum", "quality"],
        shock_size=2.0
    )
    
    print(f"\nFactor: {info_content['factor_name']}")
    print(f"Systemic risk change: {info_content['systemic_risk_change']:.4f}")
    print(f"Expected shortfall change: {info_content['expected_shortfall_change']:.4f}")
    print(f"Marginal ES change: {info_content['marginal_es_change']:.4f}")
    print(f"Max redundancy with core: {info_content['max_redundancy_with_core']:.2%}")
    print(f"Most redundant factor: {info_content['most_redundant_factor']}")
    
    if info_content["is_zoo_animal"]:
        print(f"\n❌ REDUNDANT: This factor is a 'zoo animal' (redundant with {info_content['most_redundant_factor']})")
    else:
        print(f"\n✅ INDEPENDENT: This factor carries unique information")
    
    print(f"\nRedundancy scores:")
    for core_id, score in info_content["redundancy_scores"].items():
        print(f"  vs {core_id}: {score:.2%}")


def example_3_factor_compression():
    """Example 3: Compress factor zoo into core channels."""
    print("\n" + "=" * 60)
    print("Example 3: Factor Zoo Compression")
    print("=" * 60)
    
    varda2 = Varda2("Factor Compression Demo")
    
    # Add many factors (simulating "factor zoo")
    np.random.seed(42)
    
    core_factors = [
        FactorNode(id="size", name="Size", factor_type=FactorType.SIZE, volatility=1.0),
        FactorNode(id="value", name="Value", factor_type=FactorType.VALUE, volatility=1.2),
        FactorNode(id="momentum", name="Momentum", factor_type=FactorType.MOMENTUM, volatility=1.5),
        FactorNode(id="quality", name="Quality", factor_type=FactorType.QUALITY, volatility=1.1),
    ]
    
    for factor in core_factors:
        varda2.add_factor(factor)
    
    # Add many "zoo animals" (redundant factors)
    n_zoo_factors = 15
    for i in range(n_zoo_factors):
        # Create factors that are slight variations of core factors
        base_factor = core_factors[i % len(core_factors)]
        zoo_factor = FactorNode(
            id=f"zoo_{i}",
            name=f"Zoo Factor {i} (variant of {base_factor.id})",
            factor_type=FactorType.CUSTOM,
            volatility=base_factor.volatility * (0.8 + np.random.rand() * 0.4)
        )
        varda2.add_factor(zoo_factor)
    
    # Add issuers
    issuers = [Entity(id=f"issuer_{i}", name=f"Issuer {i}") for i in range(5)]
    for issuer in issuers:
        varda2.add_issuer(issuer)
    
    # Link factors to issuers (zoo factors have similar exposures to base factors)
    for issuer in issuers:
        for i, core_factor in enumerate(core_factors):
            base_exposure = 0.3 + np.random.randn() * 0.2
            varda2.link_factor_to_issuer(core_factor.id, issuer.id, exposure=base_exposure)
            
            # Link corresponding zoo factors with similar exposures
            zoo_idx = i
            while zoo_idx < n_zoo_factors:
                zoo_exposure = base_exposure * (0.7 + np.random.rand() * 0.6)  # Similar but not identical
                varda2.link_factor_to_issuer(f"zoo_{zoo_idx}", issuer.id, exposure=zoo_exposure)
                zoo_idx += len(core_factors)
    
    # Compress factor zoo
    print(f"\nCompressing {len(varda2.factors)} factors into core channels...")
    compressed = varda2.compress_factor_zoo(
        factor_ids=None,  # All factors
        target_n_factors=6,  # Compress to 6 core factors
        shock_size=2.0
    )
    
    print(f"\nCompression Results:")
    print(f"  Original factors: {compressed['original_n_factors']}")
    print(f"  Compressed to: {compressed['compressed_n_factors']}")
    print(f"  Compression ratio: {compressed['compression_ratio']:.1%}")
    print(f"  Risk preservation: {compressed['risk_preservation']:.1%}")
    
    print(f"\nCore factors (representative of clusters):")
    for cluster_id, info in compressed['compression_summary'].items():
        print(f"  {cluster_id}:")
        print(f"    Representative: {info['representative']}")
        print(f"    Cluster size: {info['n_factors']} factors")
        print(f"    Factors: {', '.join(info['factors'][:5])}" + 
              (f" ... (+{info['n_factors'] - 5} more)" if info['n_factors'] > 5 else ""))


def example_4_regime_awareness():
    """Example 4: Regime-aware factor analysis."""
    print("\n" + "=" * 60)
    print("Example 4: Regime-Aware Factor Analysis")
    print("=" * 60)
    
    varda2 = Varda2("Regime-Aware Demo")
    
    # Add factors with regime sensitivity
    value_factor = FactorNode(
        id="value",
        name="Value Factor",
        factor_type=FactorType.VALUE,
        volatility=1.2
    )
    # Value works better in crises/recoveries
    varda2.add_factor(value_factor, regime_sensitivity={
        "normal": 1.0,
        "crisis": 1.5,
        "recovery": 1.3
    })
    
    momentum_factor = FactorNode(
        id="momentum",
        name="Momentum Factor",
        factor_type=FactorType.MOMENTUM,
        volatility=1.5
    )
    # Momentum dies in crises
    varda2.add_factor(momentum_factor, regime_sensitivity={
        "normal": 1.5,
        "liquidity_rich": 1.8,
        "crisis": 0.5,
        "recovery": 1.0
    })
    
    # Add issuers
    issuer = Entity(id="issuer1", name="Test Issuer")
    varda2.add_issuer(issuer)
    varda2.link_factor_to_issuer("value", "issuer1", exposure=0.6)
    varda2.link_factor_to_issuer("momentum", "issuer1", exposure=0.7)
    
    # Analyze regime sensitivity
    print("\nAnalyzing factor regime sensitivity...")
    regime_states = ["normal", "liquidity_rich", "crisis", "recovery"]
    
    for factor_id in ["value", "momentum"]:
        print(f"\n{factor_id.upper()} factor:")
        regime_analysis = varda2.analyze_regime_sensitivity(
            factor_id=factor_id,
            regime_states=regime_states,
            shock_size=2.0
        )
        
        print(regime_analysis[["regime", "systemic_risk_change", "expected_shortfall_change"]])
        
        if "regime_robustness" in regime_analysis.columns:
            robustness = regime_analysis["regime_robustness"].iloc[0]
            print(f"  Regime robustness: {robustness:.2f}")
            if robustness > 0.7:
                print(f"  ✅ STRUCTURAL: Robust across regimes")
            else:
                print(f"  ⚠️  CONDITIONAL: Regime-dependent")
    
    # Identify structural vs conditional factors
    print("\n" + "-" * 60)
    print("Identifying structural vs conditional factors...")
    factor_types = varda2.identify_regime_conditional_factors(
        factor_ids=None,
        regime_states=regime_states,
        robustness_threshold=0.7,
        shock_size=2.0
    )
    
    print(f"\nStructural factors (robust across regimes):")
    for fid in factor_types["structural_factors"]:
        print(f"  ✅ {fid}")
    
    print(f"\nConditional factors (regime-dependent):")
    for fid in factor_types["conditional_factors"]:
        print(f"  ⚠️  {fid}")


def example_5_deal_level_translation():
    """Example 5: Translate factors to deal economics."""
    print("\n" + "=" * 60)
    print("Example 5: Deal-Level Translation")
    print("=" * 60)
    
    varda2 = Varda2("Deal Translation Demo")
    
    # Add factors
    momentum_factor = FactorNode(
        id="momentum",
        name="Momentum Factor",
        factor_type=FactorType.MOMENTUM,
        volatility=1.5
    )
    varda2.add_factor(momentum_factor)
    
    # Add issuer
    techcorp = Entity(id="issuer_techcorp", name="TechCorp Inc", initial_risk_score=0.2)
    varda2.add_issuer(techcorp)
    varda2.link_factor_to_issuer("momentum", "issuer_techcorp", exposure=0.8)
    
    # Create a deal
    tranche1 = Tranche(
        id="tranche_hy1",
        deal_id="deal_hy1",
        notional=500_000_000,
        pd_annual=0.03
    )
    
    deal1 = CapitalMarketsDeal(
        id="deal_hy1",
        issuer_entity_id="issuer_techcorp",
        deal_type=DealType.DCM_HY,
        tranches=[tranche1],
        bookrunners=["bank1"],
        gross_fees=12_500_000  # 2.5% of notional
    )
    
    varda2.add_deal(deal1)
    varda2.link_factor_to_deal("momentum", "deal_hy1", exposure=0.7)
    
    # Translate factor shock to deal economics
    print("\nTranslating Momentum factor shock to deal economics...")
    deal_econ = varda2.translate_factor_to_deal_economics(
        factor_id="momentum",
        deal_id="deal_hy1",
        shock_size=2.0
    )
    
    print(f"\nDeal Impact Analysis:")
    print(f"  Deal ID: {deal_econ['deal_id']}")
    print(f"  Factor: {deal_econ['factor_id']}")
    print(f"  Deal impact: {deal_econ['deal_impact']:.4f}")
    print(f"  Fee-at-risk change: ${deal_econ['fee_at_risk_change']:,.2f}")
    print(f"  Mean spread change: {deal_econ['mean_spread_change_bps']:.1f} bps")
    print(f"  Decision impact: {deal_econ['decision_impact'].upper()}")
    
    if deal_econ["decision_impact"] == "significant":
        print(f"\n  ⚠️  WARNING: This factor meaningfully affects deal economics")
    else:
        print(f"\n  ✅ Factor has negligible impact on this deal")
    
    # Evaluate for pipeline
    print("\n" + "-" * 60)
    print("Pipeline-level evaluation...")
    pipeline_eval = varda2.evaluate_factor_for_pipeline(
        factor_id="momentum",
        deal_ids=["deal_hy1"],
        shock_size=2.0,
        fee_threshold=100_000.0
    )
    
    print(f"\nPipeline Impact:")
    print(f"  Factor: {pipeline_eval['factor_name']}")
    print(f"  Deals analyzed: {pipeline_eval['n_deals_analyzed']}")
    print(f"  Deals significantly affected: {pipeline_eval['n_deals_significantly_affected']}")
    print(f"  Total fee-at-risk change: ${pipeline_eval['total_fee_at_risk_change']:,.2f}")
    
    if pipeline_eval["matters_for_pipeline"]:
        print(f"\n  ✅ MATTERS: Factor affects pipeline fee-at-risk (above ${pipeline_eval['fee_threshold']:,.0f})")
    else:
        print(f"\n  ❌ NEGLIGIBLE: Factor has minimal pipeline impact")


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("VARDA 2.0: Network-Based Factor Model Examples")
    print("=" * 60)
    
    # Run examples
    example_1_basic_factor_network()
    example_2_information_content()
    example_3_factor_compression()
    example_4_regime_awareness()
    example_5_deal_level_translation()
    
    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()

