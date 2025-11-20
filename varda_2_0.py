"""
Varda 2.0: Network-Based Factor Model for Capital Markets

Instead of treating factors as a long flat list in a regression, Varda 2.0 models them 
as nodes in a capital-markets network. Each factor connects to issuers, macro states, 
and deals through weighted relationships, and factor shocks are propagated through the 
graph like fluid. This lets Varda measure a factor's true contribution to default risk, 
spreads, and fee-at-risk, not just its t-stat in a backtest.

MISSION: Compress the "factor zoo" into a small set of structural risk-propagation 
channels that actually move deal economics and systemic risk.
"""

from __future__ import annotations

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import warnings
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster


# Try to import base functionality, fallback to minimal implementations
try:
    from financial_risk_lab import (
        Entity, Relationship, MarketConstraint, MarketState, MarkovChain
    )
except ImportError:
    # Minimal fallback implementations
    @dataclass
    class Entity:
        id: str
        name: str
        entity_type: str
        initial_risk_score: float = 0.0
        metadata: Dict[str, Any] = field(default_factory=dict)
    
    @dataclass
    class Relationship:
        source_id: str
        target_id: str
        strength: float
        relationship_type: str = "generic"
        metadata: Dict[str, Any] = field(default_factory=dict)
    
    @dataclass
    class MarketConstraint:
        name: str
        constraint_type: str
        value: float
        impact_on_transitions: Dict[str, float] = field(default_factory=dict)
    
    @dataclass
    class MarketState:
        state_name: str
        description: str
        base_stability: float = 0.5
        economic_indicators: Dict[str, float] = field(default_factory=dict)
        metadata: Dict[str, Any] = field(default_factory=dict)
    
    class MarkovChain:
        def __init__(self, states: List[str], transition_matrix: np.ndarray):
            self.states = states
            self.transition_matrix = transition_matrix
            self.state_to_idx = {s: i for i, s in enumerate(states)}
        
        def stationary_distribution(self) -> np.ndarray:
            eigenvals, eigenvecs = np.linalg.eig(self.transition_matrix.T)
            idx = np.argmax(np.real(eigenvals))
            dist = np.real(eigenvecs[:, idx])
            return dist / dist.sum()
        
        def constrained_stationary_distribution(
            self, constraints: List[MarketConstraint], state_names: List[str]
        ) -> Tuple[np.ndarray, np.ndarray]:
            # Simplified: just return unconstrained for now
            return self.stationary_distribution(), self.transition_matrix


class FactorType(Enum):
    """Core factor categories."""
    SIZE = "size"
    VALUE = "value"
    MOMENTUM = "momentum"
    QUALITY = "quality"
    INVESTMENT = "investment"
    PROFITABILITY = "profitability"
    CUSTOM = "custom"


class RegimeType(Enum):
    """Market regime types."""
    NORMAL = "normal"
    LIQUIDITY_RICH = "liquidity_rich"
    STRESSED = "stressed"
    CRISIS = "crisis"
    RECOVERY = "recovery"
    DISLOCATION = "dislocation"


@dataclass
class FactorNode:
    """
    Represents a factor as a node in the capital markets network.
    
    Factors are entities that can be shocked and propagate risk through
    the network to issuers, deals, and macro states.
    """
    id: str
    name: str
    factor_type: FactorType
    base_value: float = 0.0
    volatility: float = 1.0
    regime_sensitivity: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FactorScenario:
    """
    Scenario for factor shock analysis.
    
    Specifies which factors are shocked and by how much (in standard deviations).
    """
    name: str
    description: str
    factor_shocks: Dict[str, float] = field(default_factory=dict)  # factor_id -> shock (in σ)
    regime_state: Optional[str] = None
    market_constraints: List[MarketConstraint] = field(default_factory=list)
    horizon_years: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NetworkImpactResult:
    """
    Results from propagating a factor shock through the network.
    """
    factor_id: str
    scenario_name: str
    issuer_impacts: Dict[str, float] = field(default_factory=dict)  # issuer_id -> risk change
    deal_impacts: Dict[str, float] = field(default_factory=dict)  # deal_id -> risk change
    pd_changes: Dict[str, float] = field(default_factory=dict)  # issuer_id -> PD change
    spread_changes: Dict[str, float] = field(default_factory=dict)  # tranche_id -> spread change (bps)
    systemic_risk_change: float = 0.0
    expected_shortfall_change: float = 0.0
    conviction_probability: float = 0.0  # Weighted probability of conviction (0-1)
    conviction_weights: Dict[str, float] = field(default_factory=dict)  # Component weights for conviction
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FactorCluster:
    """
    Represents a cluster of factors with similar network impact patterns.
    """
    cluster_id: int
    factor_ids: List[str]
    representative_factor_id: str  # Factor that explains most risk propagation
    impact_pattern: np.ndarray  # Representative impact vector
    regime_robustness: Dict[str, float] = field(default_factory=dict)


class Varda2:
    """
    Varda 2.0: Network-Based Factor Model
    
    Models factors as nodes in a capital-markets network. Factor shocks propagate
    through the graph to issuers, deals, and macro states, allowing measurement
    of true network impact rather than just t-stats.
    """
    
    def __init__(self, name: str = "Varda 2.0 Factor Network Model") -> None:
        """Initialize Varda 2.0."""
        self.name = name
        
        # Core network components (reusing Varda 1.0 structure)
        self.entities: Dict[str, Entity] = {}
        self.relationships: List[Relationship] = []
        self.markov_chains: Dict[str, MarkovChain] = {}
        self.entity_states: Dict[str, str] = {}
        self.market_states: Dict[str, MarketState] = {}
        self.market_constraints: List[MarketConstraint] = []
        
        # Varda 2.0 specific: Factor network
        self.factors: Dict[str, FactorNode] = {}
        self.factor_relationships: Dict[str, List[Tuple[str, float]]] = defaultdict(list)  # factor_id -> [(target_id, weight)]
        
        # Issuers, deals, tranches (from Varda 1.0)
        self.issuers: Dict[str, Entity] = {}
        self.deals: Dict[str, Any] = {}  # CapitalMarketsDeal
        self.tranches: Dict[str, Any] = {}  # Tranche
        
        # Factor exposure matrices: issuer_id -> factor_id -> exposure
        self.issuer_factor_exposures: Dict[str, Dict[str, float]] = defaultdict(dict)
        # deal_id -> factor_id -> exposure
        self.deal_factor_exposures: Dict[str, Dict[str, float]] = defaultdict(dict)
        
        # Regime-aware factor links
        self.regime_factor_links: Dict[str, Dict[str, float]] = defaultdict(dict)  # regime -> factor_id -> weight
        
        # Impact history for clustering
        self.impact_history: List[NetworkImpactResult] = []
        
    # -------------------------------------------------------------------------
    # Factor Network Construction
    # -------------------------------------------------------------------------
    
    def add_factor(
        self,
        factor: FactorNode,
        regime_sensitivity: Optional[Dict[str, float]] = None
    ) -> None:
        """
        Add a factor node to the network.
        
        Args:
            factor: Factor node to add
            regime_sensitivity: Optional dict mapping regime names to sensitivity weights
        """
        self.factors[factor.id] = factor
        
        # Create Entity for network compatibility
        entity = Entity(
            id=f"factor_{factor.id}",
            name=f"Factor: {factor.name}",
            entity_type="factor",
            initial_risk_score=abs(factor.base_value) * factor.volatility,
            metadata={"factor_id": factor.id, "factor_type": factor.factor_type.value}
        )
        self.entities[entity.id] = entity
        
        # Store regime sensitivity
        if regime_sensitivity:
            for regime, sensitivity in regime_sensitivity.items():
                if regime not in self.regime_factor_links:
                    self.regime_factor_links[regime] = {}
                self.regime_factor_links[regime][factor.id] = sensitivity
    
    def link_factor_to_issuer(
        self,
        factor_id: str,
        issuer_id: str,
        exposure: float,
        relationship_type: str = "factor_exposure"
    ) -> None:
        """
        Link a factor to an issuer with given exposure.
        
        Args:
            factor_id: ID of the factor
            issuer_id: ID of the issuer
            exposure: Exposure weight (can be negative)
            relationship_type: Type of relationship
        """
        if factor_id not in self.factors:
            raise ValueError(f"Factor {factor_id} not found")
        if issuer_id not in self.issuers:
            raise ValueError(f"Issuer {issuer_id} not found")
        
        self.issuer_factor_exposures[issuer_id][factor_id] = exposure
        
        # Create relationship in network
        relationship = Relationship(
            source_id=f"factor_{factor_id}",
            target_id=issuer_id,
            strength=abs(exposure),
            relationship_type=relationship_type,
            metadata={"exposure": exposure, "factor_id": factor_id}
        )
        self.relationships.append(relationship)
        self.factor_relationships[factor_id].append((issuer_id, exposure))
    
    def link_factor_to_deal(
        self,
        factor_id: str,
        deal_id: str,
        exposure: float
    ) -> None:
        """Link a factor to a deal with given exposure."""
        if factor_id not in self.factors:
            raise ValueError(f"Factor {factor_id} not found")
        if deal_id not in self.deals:
            raise ValueError(f"Deal {deal_id} not found")
        
        self.deal_factor_exposures[deal_id][factor_id] = exposure
    
    def add_issuer(self, issuer: Entity) -> None:
        """Add an issuer entity."""
        self.issuers[issuer.id] = issuer
        self.entities[issuer.id] = issuer
    
    def add_deal(self, deal: Any) -> None:
        """Add a deal (compatible with CapitalMarketsDeal from Varda 1.0)."""
        self.deals[deal.id] = deal
        # Register tranches if deal has them
        if hasattr(deal, 'tranches'):
            for tranche in deal.tranches:
                self.tranches[tranche.id] = tranche
    
    def link_factor_to_regime(
        self,
        factor_id: str,
        regime_name: str,
        sensitivity: float
    ) -> None:
        """
        Link a factor to a market regime with given sensitivity.
        
        Regime-sensitive factors have higher impact in specific regimes.
        """
        if factor_id not in self.factors:
            raise ValueError(f"Factor {factor_id} not found")
        
        if regime_name not in self.regime_factor_links:
            self.regime_factor_links[regime_name] = {}
        self.regime_factor_links[regime_name][factor_id] = sensitivity
    
    # -------------------------------------------------------------------------
    # Factor Shock Propagation
    # -------------------------------------------------------------------------
    
    def propagate_factor_shock(
        self,
        factor_id: str,
        shock_size: float,  # in standard deviations
        scenario: Optional[FactorScenario] = None,
        diffusion_rate: float = 0.1,
        iterations: int = 10,
        regime_state: Optional[str] = None
    ) -> NetworkImpactResult:
        """
        Propagate a factor shock through the network like fluid.
        
        This is the core method: shock one factor and measure how the risk
        flows through the network to issuers, deals, and systemic risk.
        
        Args:
            factor_id: ID of factor to shock
            shock_size: Shock magnitude in standard deviations (e.g., 2.0 = +2σ)
            scenario: Optional scenario with regime/constraints
            diffusion_rate: Rate of risk diffusion (0-1)
            iterations: Number of propagation steps
            regime_state: Current market regime (affects factor sensitivity)
            
        Returns:
            NetworkImpactResult with issuer/deal impacts and risk changes
        """
        if factor_id not in self.factors:
            raise ValueError(f"Factor {factor_id} not found")
        
        factor = self.factors[factor_id]
        
        # Apply regime adjustment if specified
        regime_multiplier = 1.0
        if regime_state and regime_state in self.regime_factor_links:
            regime_multiplier = self.regime_factor_links[regime_state].get(factor_id, 1.0)
        
        # Initial shock: convert σ to risk units
        initial_shock = shock_size * factor.volatility * regime_multiplier
        
        # Build initial shock dict for network propagation
        initial_shocks: Dict[str, float] = {}
        initial_shocks[f"factor_{factor_id}"] = initial_shock
        
        # Propagate through network using fluid dynamics
        impact_result = NetworkImpactResult(
            factor_id=factor_id,
            scenario_name=scenario.name if scenario else "single_factor_shock",
            issuer_impacts={},
            deal_impacts={},
            pd_changes={},
            spread_changes={},
            systemic_risk_change=0.0,
            expected_shortfall_change=0.0
        )
        
        # Direct factor-to-issuer impacts via exposure matrix
        for issuer_id, exposures in self.issuer_factor_exposures.items():
            if factor_id in exposures:
                exposure = exposures[factor_id]
                # Impact = shock * exposure * regime_multiplier
                impact = initial_shock * exposure * regime_multiplier
                impact_result.issuer_impacts[issuer_id] = impact
                
                # Convert to PD change (simplified: linear mapping)
                # In practice, this would use a calibrated model
                pd_change = impact * 0.01  # Example: 1 unit impact = 1% PD change
                impact_result.pd_changes[issuer_id] = pd_change
        
        # Factor-to-deal impacts
        for deal_id, exposures in self.deal_factor_exposures.items():
            if factor_id in exposures:
                exposure = exposures[factor_id]
                impact = initial_shock * exposure * regime_multiplier
                impact_result.deal_impacts[deal_id] = impact
        
        # Additional network propagation if factor is connected to other factors
        # (factors can influence each other)
        if factor_id in self.factor_relationships:
            for target_id, weight in self.factor_relationships[factor_id]:
                # Recursive impact through factor network
                if target_id in self.issuers:
                    cumulative_impact = initial_shock * weight
                    if target_id not in impact_result.issuer_impacts:
                        impact_result.issuer_impacts[target_id] = 0.0
                    impact_result.issuer_impacts[target_id] += cumulative_impact
        
        # Calculate systemic risk change: sum of squared impacts (simplified)
        systemic_risk = sum(imp ** 2 for imp in impact_result.issuer_impacts.values())
        impact_result.systemic_risk_change = systemic_risk
        
        # Expected shortfall change: tail of impact distribution
        if impact_result.issuer_impacts:
            impacts = list(impact_result.issuer_impacts.values())
            sorted_impacts = sorted(impacts, reverse=True)
            tail_size = max(1, len(sorted_impacts) // 10)  # Top 10%
            impact_result.expected_shortfall_change = np.mean(sorted_impacts[:tail_size])
        
        # Spread changes for tranches (simplified: proportional to PD changes)
        for issuer_id, pd_change in impact_result.pd_changes.items():
            # Find tranches for this issuer
            for deal_id, deal in self.deals.items():
                if hasattr(deal, 'issuer_entity_id') and deal.issuer_entity_id == issuer_id:
                    for tranche_id in self.tranches:
                        tranche = self.tranches[tranche_id]
                        if hasattr(tranche, 'deal_id') and tranche.deal_id == deal_id:
                            # Convert PD change to spread change (bps)
                            # Rough approximation: 1% PD ≈ 100 bps spread
                            spread_change_bps = pd_change * 100.0
                            impact_result.spread_changes[tranche_id] = spread_change_bps
        
        # Calculate weighted probability of conviction
        # Conviction = weighted combination of:
        # 1. Network reach (how many entities affected)
        # 2. Impact magnitude (systemic risk change)
        # 3. Expected shortfall (tail risk)
        # 4. Regime robustness (if regime specified)
        conviction_weights = {}
        
        # Component 1: Network reach (0-1 normalized)
        n_entities = len(impact_result.issuer_impacts) + len(impact_result.deal_impacts)
        max_possible_entities = len(self.issuers) + len(self.deals)
        network_reach = min(1.0, n_entities / max(max_possible_entities, 1))
        conviction_weights["network_reach"] = network_reach
        
        # Component 2: Impact magnitude (normalized systemic risk)
        max_systemic_risk = max(abs(impact_result.systemic_risk_change), 0.01)  # Avoid division by zero
        impact_magnitude = min(1.0, abs(impact_result.systemic_risk_change) / max(max_systemic_risk, 0.1))
        conviction_weights["impact_magnitude"] = impact_magnitude
        
        # Component 3: Expected shortfall (tail risk)
        max_es = max(abs(impact_result.expected_shortfall_change), 0.01)
        es_contribution = min(1.0, abs(impact_result.expected_shortfall_change) / max(max_es, 0.1))
        conviction_weights["expected_shortfall"] = es_contribution
        
        # Component 4: Regime robustness (if regime specified)
        regime_robustness = 1.0
        if regime_state:
            # Higher conviction if factor is regime-sensitive in this regime
            regime_sensitivity = self.regime_factor_links.get(regime_state, {}).get(factor_id, 1.0)
            regime_robustness = min(1.0, abs(regime_sensitivity))
        conviction_weights["regime_robustness"] = regime_robustness
        
        # Component 5: Factor stability (volatility-adjusted)
        factor = self.factors[factor_id]
        factor_volatility = factor.volatility
        volatility_stability = 1.0 / (1.0 + factor_volatility)  # Lower vol = higher conviction
        conviction_weights["volatility_stability"] = volatility_stability
        
        # Weighted combination (weights can be customized)
        weights = {
            "network_reach": 0.20,
            "impact_magnitude": 0.30,
            "expected_shortfall": 0.25,
            "regime_robustness": 0.15,
            "volatility_stability": 0.10
        }
        
        conviction_probability = sum(
            weights.get(key, 0.0) * conviction_weights.get(key, 0.0)
            for key in conviction_weights.keys()
        )
        
        # Normalize to [0, 1]
        conviction_probability = max(0.0, min(1.0, conviction_probability))
        
        impact_result.conviction_probability = conviction_probability
        impact_result.conviction_weights = conviction_weights
        
        # Store in history for clustering
        self.impact_history.append(impact_result)
        
        return impact_result
    
    def compare_factor_network_impacts(
        self,
        factor_ids: List[str],
        shock_size: float = 2.0,
        regime_state: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Compare network impacts of multiple factors.
        
        This answers: "Which factors actually move risk in the network?"
        Not just t-stats, but true network impact.
        
        Args:
            factor_ids: List of factor IDs to compare
            shock_size: Shock magnitude in σ
            regime_state: Optional regime state
            
        Returns:
            DataFrame comparing impacts across factors
        """
        results = []
        
        for factor_id in factor_ids:
            impact = self.propagate_factor_shock(
                factor_id=factor_id,
                shock_size=shock_size,
                regime_state=regime_state
            )
            
            # Summary metrics
            results.append({
                "factor_id": factor_id,
                "factor_name": self.factors[factor_id].name,
                "factor_type": self.factors[factor_id].factor_type.value,
                "n_issuers_affected": len(impact.issuer_impacts),
                "n_deals_affected": len(impact.deal_impacts),
                "mean_issuer_impact": np.mean(list(impact.issuer_impacts.values())) if impact.issuer_impacts else 0.0,
                "max_issuer_impact": max(impact.issuer_impacts.values()) if impact.issuer_impacts else 0.0,
                "systemic_risk_change": impact.systemic_risk_change,
                "expected_shortfall_change": impact.expected_shortfall_change,
                "conviction_probability": impact.conviction_probability,
                "mean_pd_change": np.mean(list(impact.pd_changes.values())) if impact.pd_changes else 0.0,
                "mean_spread_change_bps": np.mean(list(impact.spread_changes.values())) if impact.spread_changes else 0.0
            })
        
        return pd.DataFrame(results)
    
    # -------------------------------------------------------------------------
    # Information Content Analysis
    # -------------------------------------------------------------------------
    
    def evaluate_factor_information_content(
        self,
        new_factor_id: str,
        core_factor_ids: Optional[List[str]] = None,
        shock_size: float = 2.0,
        regime_state: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate whether a new factor carries independent information.
        
        Philosophy: If a factor doesn't carry new information into the network,
        it belongs back in the zoo cage.
        
        Method:
        1. Shock the new factor and measure network impact
        2. Shock core factors and measure their impacts
        3. Check if new factor's impact is redundant with core factors
        
        Args:
            new_factor_id: ID of factor to evaluate
            core_factor_ids: List of core factor IDs (default: all existing factors)
            shock_size: Shock magnitude in σ
            regime_state: Optional regime state
            
        Returns:
            Dict with information content metrics
        """
        if new_factor_id not in self.factors:
            raise ValueError(f"Factor {new_factor_id} not found")
        
        if core_factor_ids is None:
            core_factor_ids = [fid for fid in self.factors.keys() if fid != new_factor_id]
        
        # Get new factor impact
        new_impact = self.propagate_factor_shock(
            factor_id=new_factor_id,
            shock_size=shock_size,
            regime_state=regime_state
        )
        
        # Get core factor impacts
        core_impacts = {}
        for core_id in core_factor_ids:
            core_impacts[core_id] = self.propagate_factor_shock(
                factor_id=core_id,
                shock_size=shock_size,
                regime_state=regime_state
            )
        
        # Compare impact patterns
        new_impact_vector = self._impact_to_vector(new_impact)
        core_impact_vectors = {
            fid: self._impact_to_vector(imp) for fid, imp in core_impacts.items()
        }
        
        # Check redundancy: can we reconstruct new factor's impact from core factors?
        redundancy_scores = {}
        for core_id, core_vector in core_impact_vectors.items():
            # Cosine similarity as redundancy measure
            similarity = np.dot(new_impact_vector, core_vector) / (
                np.linalg.norm(new_impact_vector) * np.linalg.norm(core_vector) + 1e-10
            )
            redundancy_scores[core_id] = abs(similarity)
        
        max_redundancy = max(redundancy_scores.values()) if redundancy_scores else 0.0
        
        # Marginal information content
        # If redundant with existing factors, marginal content is low
        marginal_es_change = new_impact.expected_shortfall_change * (1.0 - max_redundancy)
        
        # Flag as zoo animal if highly redundant
        is_zoo_animal = max_redundancy > 0.85  # 85% similarity threshold
        
        return {
            "factor_id": new_factor_id,
            "factor_name": self.factors[new_factor_id].name,
            "systemic_risk_change": new_impact.systemic_risk_change,
            "expected_shortfall_change": new_impact.expected_shortfall_change,
            "marginal_es_change": marginal_es_change,
            "max_redundancy_with_core": max_redundancy,
            "most_redundant_factor": max(redundancy_scores, key=redundancy_scores.get) if redundancy_scores else None,
            "redundancy_scores": redundancy_scores,
            "is_zoo_animal": is_zoo_animal,
            "n_issuers_affected": len(new_impact.issuer_impacts),
            "n_deals_affected": len(new_impact.deal_impacts)
        }
    
    def _impact_to_vector(self, impact: NetworkImpactResult) -> np.ndarray:
        """Convert impact result to a vector for comparison."""
        # Combine issuer impacts, deal impacts, and systemic metrics
        all_keys = set(impact.issuer_impacts.keys()) | set(impact.deal_impacts.keys())
        
        vector = []
        for key in sorted(all_keys):
            issuer_val = impact.issuer_impacts.get(key, 0.0)
            deal_val = impact.deal_impacts.get(key, 0.0)
            vector.append(issuer_val + deal_val)
        
        # Add systemic metrics
        vector.append(impact.systemic_risk_change)
        vector.append(impact.expected_shortfall_change)
        
        return np.array(vector) if vector else np.array([0.0])
    
    # -------------------------------------------------------------------------
    # Factor Clustering and Compression
    # -------------------------------------------------------------------------
    
    def cluster_factors_by_network_impact(
        self,
        factor_ids: Optional[List[str]] = None,
        n_clusters: Optional[int] = None,
        shock_size: float = 2.0,
        regime_state: Optional[str] = None,
        method: str = "kmeans"  # "kmeans" or "hierarchical"
    ) -> List[FactorCluster]:
        """
        Cluster factors based on similarity of network impact patterns.
        
        This compresses the "factor zoo" into core risk-propagation channels.
        
        Args:
            factor_ids: List of factors to cluster (None = all factors)
            n_clusters: Number of clusters (None = auto-detect)
            shock_size: Shock magnitude for impact measurement
            regime_state: Optional regime state
            method: Clustering method ("kmeans" or "hierarchical")
            
        Returns:
            List of FactorCluster objects
        """
        if factor_ids is None:
            factor_ids = list(self.factors.keys())
        
        if len(factor_ids) < 2:
            return []
        
        # Generate impact vectors for all factors
        impact_vectors = {}
        for factor_id in factor_ids:
            impact = self.propagate_factor_shock(
                factor_id=factor_id,
                shock_size=shock_size,
                regime_state=regime_state
            )
            impact_vectors[factor_id] = self._impact_to_vector(impact)
        
        # Normalize vectors to [0, 1] for clustering
        all_vectors = np.array([impact_vectors[fid] for fid in factor_ids])
        if len(all_vectors) == 0 or all_vectors.shape[1] == 0:
            return []
        
        # Normalize
        min_vals = all_vectors.min(axis=0, keepdims=True)
        max_vals = all_vectors.max(axis=0, keepdims=True)
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1.0  # Avoid division by zero
        normalized_vectors = (all_vectors - min_vals) / range_vals
        
        # Determine number of clusters
        if n_clusters is None:
            # Auto-detect: use elbow method or default to sqrt(n/2)
            n_clusters = max(2, int(np.sqrt(len(factor_ids) / 2)))
            n_clusters = min(n_clusters, len(factor_ids))
        
        # Perform clustering
        if method == "kmeans":
            clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = clusterer.fit_predict(normalized_vectors)
        else:  # hierarchical
            distances = pdist(normalized_vectors, metric='euclidean')
            linkage_matrix = linkage(distances, method='ward')
            cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
            # Convert to 0-indexed
            cluster_labels = cluster_labels - 1
        
        # Build FactorCluster objects
        clusters: List[FactorCluster] = []
        unique_labels = np.unique(cluster_labels)
        
        for cluster_id in unique_labels:
            cluster_factor_ids = [factor_ids[i] for i in range(len(factor_ids)) if cluster_labels[i] == cluster_id]
            
            if not cluster_factor_ids:
                continue
            
            # Find representative factor: highest systemic risk impact
            representative_id = cluster_factor_ids[0]
            max_impact = impact_vectors[representative_id].sum()
            for fid in cluster_factor_ids[1:]:
                impact_sum = impact_vectors[fid].sum()
                if impact_sum > max_impact:
                    max_impact = impact_sum
                    representative_id = fid
            
            # Representative impact pattern (mean of cluster)
            cluster_vectors = np.array([impact_vectors[fid] for fid in cluster_factor_ids])
            representative_pattern = cluster_vectors.mean(axis=0)
            
            # Regime robustness (test across regimes if available)
            regime_robustness = {}
            if hasattr(self, 'regime_factor_links') and self.regime_factor_links:
                for regime in self.regime_factor_links.keys():
                    # Test representative factor in this regime
                    test_impact = self.propagate_factor_shock(
                        factor_id=representative_id,
                        shock_size=shock_size,
                        regime_state=regime
                    )
                    regime_robustness[regime] = test_impact.systemic_risk_change
            
            cluster = FactorCluster(
                cluster_id=int(cluster_id),
                factor_ids=cluster_factor_ids,
                representative_factor_id=representative_id,
                impact_pattern=representative_pattern,
                regime_robustness=regime_robustness
            )
            clusters.append(cluster)
        
        return clusters
    
    def compress_factor_zoo(
        self,
        factor_ids: Optional[List[str]] = None,
        target_n_factors: Optional[int] = None,
        shock_size: float = 2.0,
        regime_state: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Compress factor zoo into a smaller set of structural risk-propagation channels.
        
        This is the core "zoo compression" method. Instead of saying "we reduced
        300 factors to 10 PCs", you say "we reduced 300 characteristics to ~10
        risk-propagation channels that actually move default risk and deal economics."
        
        Args:
            factor_ids: List of factors to compress (None = all)
            target_n_factors: Target number of factors (None = auto)
            shock_size: Shock magnitude
            regime_state: Optional regime state
            
        Returns:
            Dict with compressed factor set and metadata
        """
        if factor_ids is None:
            factor_ids = list(self.factors.keys())
        
        if target_n_factors is None:
            # Auto: target ~5-15 factors based on literature
            target_n_factors = max(5, min(15, int(np.sqrt(len(factor_ids)))))
        
        # Cluster factors
        clusters = self.cluster_factors_by_network_impact(
            factor_ids=factor_ids,
            n_clusters=target_n_factors,
            shock_size=shock_size,
            regime_state=regime_state
        )
        
        # Extract representative factors from each cluster
        compressed_factors = [cluster.representative_factor_id for cluster in clusters]
        
        # Evaluate compression quality: how much risk information is preserved?
        original_risk = sum(
            self.propagate_factor_shock(fid, shock_size, regime_state=regime_state).systemic_risk_change
            for fid in factor_ids
        )
        
        compressed_risk = sum(
            self.propagate_factor_shock(fid, shock_size, regime_state=regime_state).systemic_risk_change
            for fid in compressed_factors
        )
        
        risk_preservation = compressed_risk / original_risk if original_risk > 0 else 1.0
        
        return {
            "original_n_factors": len(factor_ids),
            "compressed_n_factors": len(compressed_factors),
            "compression_ratio": len(compressed_factors) / len(factor_ids) if factor_ids else 0.0,
            "compressed_factor_ids": compressed_factors,
            "clusters": clusters,
            "risk_preservation": risk_preservation,
            "compression_summary": {
                f"Cluster {c.cluster_id}": {
                    "representative": c.representative_factor_id,
                    "n_factors": len(c.factor_ids),
                    "factors": c.factor_ids
                }
                for c in clusters
            }
        }
    
    # -------------------------------------------------------------------------
    # Regime-Aware Factor Analysis
    # -------------------------------------------------------------------------
    
    def analyze_regime_sensitivity(
        self,
        factor_id: str,
        regime_states: List[str],
        shock_size: float = 2.0
    ) -> pd.DataFrame:
        """
        Analyze how a factor's network impact varies across regimes.
        
        This helps separate structural factors (true drivers across regimes)
        from conditional factors (only useful in narrow windows).
        
        Args:
            factor_id: Factor to analyze
            regime_states: List of regime names to test
            shock_size: Shock magnitude
            
        Returns:
            DataFrame with impact metrics per regime
        """
        results = []
        
        for regime in regime_states:
            impact = self.propagate_factor_shock(
                factor_id=factor_id,
                shock_size=shock_size,
                regime_state=regime
            )
            
            results.append({
                "regime": regime,
                "systemic_risk_change": impact.systemic_risk_change,
                "expected_shortfall_change": impact.expected_shortfall_change,
                "n_issuers_affected": len(impact.issuer_impacts),
                "n_deals_affected": len(impact.deal_impacts),
                "mean_pd_change": np.mean(list(impact.pd_changes.values())) if impact.pd_changes else 0.0,
                "mean_spread_change_bps": np.mean(list(impact.spread_changes.values())) if impact.spread_changes else 0.0
            })
        
        df = pd.DataFrame(results)
        
        # Calculate regime robustness score
        if len(df) > 1:
            risk_std = df["systemic_risk_change"].std()
            risk_mean = df["systemic_risk_change"].mean()
            robustness = 1.0 / (1.0 + risk_std / (risk_mean + 1e-10))  # Higher = more robust
            df["regime_robustness"] = robustness
        
        return df
    
    def identify_regime_conditional_factors(
        self,
        factor_ids: Optional[List[str]] = None,
        regime_states: List[str] = None,
        robustness_threshold: float = 0.7,
        shock_size: float = 2.0
    ) -> Dict[str, Any]:
        """
        Identify which factors are regime-conditional vs. regime-robust.
        
        Structural factors: robust across regimes
        Conditional factors: only matter in specific regimes
        
        Args:
            factor_ids: Factors to analyze (None = all)
            regime_states: List of regime names
            robustness_threshold: Threshold for robustness (0-1)
            shock_size: Shock magnitude
            
        Returns:
            Dict with structural and conditional factor lists
        """
        if factor_ids is None:
            factor_ids = list(self.factors.keys())
        
        if regime_states is None:
            regime_states = list(self.regime_factor_links.keys()) if self.regime_factor_links else ["normal", "crisis"]
        
        structural_factors = []
        conditional_factors = []
        
        for factor_id in factor_ids:
            regime_analysis = self.analyze_regime_sensitivity(
                factor_id=factor_id,
                regime_states=regime_states,
                shock_size=shock_size
            )
            
            if "regime_robustness" in regime_analysis.columns:
                robustness = regime_analysis["regime_robustness"].iloc[0]
                if robustness >= robustness_threshold:
                    structural_factors.append(factor_id)
                else:
                    conditional_factors.append(factor_id)
            else:
                # If only one regime, treat as conditional
                conditional_factors.append(factor_id)
        
        return {
            "structural_factors": structural_factors,
            "conditional_factors": conditional_factors,
            "n_structural": len(structural_factors),
            "n_conditional": len(conditional_factors),
            "robustness_threshold": robustness_threshold
        }
    
    # -------------------------------------------------------------------------
    # Deal-Level Translation
    # -------------------------------------------------------------------------
    
    def translate_factor_to_deal_economics(
        self,
        factor_id: str,
        deal_id: str,
        shock_size: float = 2.0,
        regime_state: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Translate factor shock into deal-level economics.
        
        Answers: "Does this factor meaningfully change the fee-at-risk
        for our high-yield pipeline?"
        
        Args:
            factor_id: Factor to shock
            deal_id: Deal to analyze
            shock_size: Shock magnitude
            regime_state: Optional regime state
            
        Returns:
            Dict with deal-level impact metrics
        """
        if deal_id not in self.deals:
            raise ValueError(f"Deal {deal_id} not found")
        
        # Propagate factor shock
        impact = self.propagate_factor_shock(
            factor_id=factor_id,
            shock_size=shock_size,
            regime_state=regime_state
        )
        
        deal = self.deals[deal_id]
        deal_impact = impact.deal_impacts.get(deal_id, 0.0)
        
        # Calculate fee-at-risk change
        if hasattr(deal, 'gross_fees'):
            # Simplified: fee-at-risk proportional to deal risk impact
            fee_at_risk_change = deal.gross_fees * abs(deal_impact) * 0.1  # 10% of fees at risk per unit impact
        else:
            fee_at_risk_change = 0.0
        
        # Aggregate tranche impacts
        tranche_impacts = {}
        if hasattr(deal, 'tranches'):
            for tranche in deal.tranches:
                tranche_id = tranche.id if hasattr(tranche, 'id') else str(tranche)
                spread_change = impact.spread_changes.get(tranche_id, 0.0)
                tranche_impacts[tranche_id] = {
                    "spread_change_bps": spread_change,
                    "pd_change": impact.pd_changes.get(deal.issuer_entity_id if hasattr(deal, 'issuer_entity_id') else "", 0.0)
                }
        
        # Decision impact: does this change deal decisions?
        decision_impact = "significant" if abs(deal_impact) > 0.05 else "negligible"
        
        return {
            "factor_id": factor_id,
            "deal_id": deal_id,
            "deal_impact": deal_impact,
            "fee_at_risk_change": fee_at_risk_change,
            "tranche_impacts": tranche_impacts,
            "decision_impact": decision_impact,
            "mean_spread_change_bps": np.mean([t["spread_change_bps"] for t in tranche_impacts.values()]) if tranche_impacts else 0.0
        }
    
    def evaluate_factor_for_pipeline(
        self,
        factor_id: str,
        deal_ids: Optional[List[str]] = None,
        shock_size: float = 2.0,
        regime_state: Optional[str] = None,
        fee_threshold: float = 100_000.0  # Minimum fee-at-risk to matter
    ) -> Dict[str, Any]:
        """
        Evaluate whether a factor meaningfully affects pipeline fee-at-risk.
        
        Args:
            factor_id: Factor to evaluate
            deal_ids: Deals to analyze (None = all deals)
            shock_size: Shock magnitude
            regime_state: Optional regime state
            fee_threshold: Minimum fee-at-risk change to matter
            
        Returns:
            Dict with pipeline-level evaluation
        """
        if deal_ids is None:
            deal_ids = list(self.deals.keys())
        
        pipeline_impacts = []
        total_fee_at_risk_change = 0.0
        
        for deal_id in deal_ids:
            try:
                deal_econ = self.translate_factor_to_deal_economics(
                    factor_id=factor_id,
                    deal_id=deal_id,
                    shock_size=shock_size,
                    regime_state=regime_state
                )
                pipeline_impacts.append(deal_econ)
                total_fee_at_risk_change += deal_econ["fee_at_risk_change"]
            except Exception as e:
                warnings.warn(f"Error evaluating deal {deal_id}: {e}")
                continue
        
        # Decision: does this factor matter for pipeline?
        matters_for_pipeline = abs(total_fee_at_risk_change) >= fee_threshold
        
        n_deals_affected = sum(1 for imp in pipeline_impacts if imp["decision_impact"] == "significant")
        
        return {
            "factor_id": factor_id,
            "factor_name": self.factors[factor_id].name if factor_id in self.factors else factor_id,
            "n_deals_analyzed": len(deal_ids),
            "n_deals_significantly_affected": n_deals_affected,
            "total_fee_at_risk_change": total_fee_at_risk_change,
            "matters_for_pipeline": matters_for_pipeline,
            "fee_threshold": fee_threshold,
            "deal_impacts": pipeline_impacts
        }
    
    # -------------------------------------------------------------------------
    # Summary and Utilities
    # -------------------------------------------------------------------------
    
    def summary(self) -> str:
        """Generate summary of Varda 2.0 instance."""
        n_factors = len(self.factors)
        n_issuers = len(self.issuers)
        n_deals = len(self.deals)
        n_tranches = len(self.tranches)
        n_regimes = len(self.regime_factor_links)
        
        factor_types = {}
        for factor in self.factors.values():
            ft = factor.factor_type.value
            factor_types[ft] = factor_types.get(ft, 0) + 1
        
        summary = f"""
Varda 2.0: Network-Based Factor Model
=====================================
Name: {self.name}

Network Components:
  Factors: {n_factors}
    {', '.join(f'{k}: {v}' for k, v in factor_types.items())}
  Issuers: {n_issuers}
  Deals: {n_deals}
  Tranches: {n_tranches}
  Regimes: {n_regimes}

Factor-Issuer Links: {sum(len(exposures) for exposures in self.issuer_factor_exposures.values())}
Factor-Deal Links: {sum(len(exposures) for exposures in self.deal_factor_exposures.values())}
Impact History: {len(self.impact_history)} shocks analyzed
        """
        return summary.strip()


# Example usage
if __name__ == "__main__":
    # Create Varda 2.0 instance
    varda2 = Varda2("Factor Network Model Demo")
    
    # Add core factors
    from enum import Enum
    
    # Size factor
    size_factor = FactorNode(
        id="size",
        name="Market Cap Factor",
        factor_type=FactorType.SIZE,
        base_value=0.0,
        volatility=1.0
    )
    varda2.add_factor(size_factor)
    
    # Value factor
    value_factor = FactorNode(
        id="value",
        name="Value Factor",
        factor_type=FactorType.VALUE,
        base_value=0.0,
        volatility=1.2
    )
    varda2.add_factor(value_factor, regime_sensitivity={"normal": 1.0, "crisis": 1.5})
    
    # Momentum factor
    momentum_factor = FactorNode(
        id="momentum",
        name="Momentum Factor",
        factor_type=FactorType.MOMENTUM,
        base_value=0.0,
        volatility=1.5
    )
    varda2.add_factor(momentum_factor, regime_sensitivity={"normal": 1.5, "crisis": 0.5})  # Dies in crises
    
    print("Varda 2.0 created with core factors!")
    print(varda2.summary())

