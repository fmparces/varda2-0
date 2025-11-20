# Varda 2.0: Quick Start Guide

Get started with Varda 2.0 in 5 minutes!

---

## Installation

```bash
cd varda2.0
pip install -r requirements.txt
```

---

## 5-Minute Example

```python
from varda_2_0 import Varda2, FactorNode, FactorType
from varda import Entity  # Or use fallback from varda_2_0

# 1. Create instance
varda2 = Varda2("My Factor Network")

# 2. Add factors
momentum = FactorNode(id="momentum", name="12M Momentum", 
                     factor_type=FactorType.MOMENTUM, volatility=1.5)
varda2.add_factor(momentum)

# 3. Add issuers
techcorp = Entity(id="techcorp", name="TechCorp Inc", initial_risk_score=0.2)
varda2.add_issuer(techcorp)

# 4. Link factor to issuer
varda2.link_factor_to_issuer("momentum", "techcorp", exposure=0.8)

# 5. Propagate shock
impact = varda2.propagate_factor_shock("momentum", shock_size=2.0)

# 6. See results
print(f"Impact: {impact.issuer_impacts.get('techcorp', 0.0):.4f}")
print(f"Conviction: {impact.conviction_probability:.2%}")
```

---

## Run Examples

```bash
python varda_2_0_example.py
```

## Run Tests

```bash
python varda_2_0_stress_test.py
```

---

## Documentation

- **📖 Full Guide**: See `USAGE_GUIDE.md` for complete usage instructions
- **📚 API Reference**: See `VARDA_2_0_README.md` for API documentation
- **🔬 Methodology**: See `VARDA_2_0_METHODOLOGY.md` for methodology details

---

## When to Use Varda 2.0

✅ **Use When:**
- You have 50+ factors and need to compress to 5-15 core channels
- Traditional regression models aren't capturing factor dependencies
- You need regime-aware factor analysis
- You want to translate factors to deal-level fee-at-risk
- You need conviction-based factor selection

❌ **Don't Use When:**
- You have <10 factors and simple regression works fine
- You only need return predictions (not risk propagation)
- You have no factor-issuer exposure data

---

## Next Steps

1. ✅ Install dependencies: `pip install -r requirements.txt`
2. ✅ Run examples: `python varda_2_0_example.py`
3. ✅ Read full guide: `USAGE_GUIDE.md`
4. ✅ Build your network: Follow examples in `varda_2_0_example.py`

---

**Need Help?** See `USAGE_GUIDE.md` for detailed instructions and troubleshooting.

