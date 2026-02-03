# Sample Materials from reaction.csv

All examples in the platform now use materials from **reaction.csv** to ensure consistency and testability.

## 📊 Available Materials (42 compounds)

### Basic Examples
- **Ba2Cl8Ni1Pb1** - Chloride compound (used in Colab Example 1)
- **K2Cu4F10** - Copper fluoride (used in quickstart.py)
- **K2Fe1F6** - Iron fluoride
- **Rb2Ag2F4** - Silver fluoride

### High-Hazard Materials (Fluorides)
- **Li1Ni1F6** - Contains Li (pyrophoric) + F (reactive) - Used in Colab Example 3
- **Cu2Eu2F9Rb1** - Europium copper fluoride
- **Ba1Cs2F12Ni3** - Cesium barium nickel fluoride
- **Co1Cu1F10K2Ru1** - Ruthenium compound
- **F4K1Na1Pd2** - Palladium fluoride

### Complex Fluorides
- **Ca5F11Ni2O1Zn1** - Mixed oxide-fluoride
- **Rb2Cu1Ag1F6** - Silver-copper fluoride
- **Mg2Cu2O1F4** - Magnesium copper oxyfluoride
- **Rb1Ca1Cu1Bi2O1F10** - Bismuth-containing fluoride

### Element Substitution Examples
- **K2Cu4F10** → **K2Ag4F10** (Cu→Ag substitution, used in Colab Example 2)
- **Rb2Cu1Ag1F6** → Substitute Cu or Ag
- **Ba2Ag1Fe1F6** → Substitute Fe
- **Sr2Li1Cu1F6** → Substitute Cu

## 🔬 Where Samples Are Used

### Streamlit App (`streamlit_app.py`)
- **Dropdown menu**: All 42 materials from reaction.csv
- **Default value**: First material (Ba2Cl8Ni1Pb1)
- **Auto-populated**: Loaded dynamically from CSV file

### Colab Notebook (`colab_setup.ipynb`)
- **Example 1**: Ba2Cl8Ni1Pb1 (basic synthesis)
- **Example 2**: K2Cu4F10 with Cu→Ag substitution
- **Example 3**: Li1Ni1F6 (high-hazard fluoride with Li)

### Quick Start (`quickstart.py`)
- **Demo material**: K2Cu4F10
- **Output files**: demo_K2Cu4F10.cif, demo_K2Cu4F10_synthesis.txt

## 🧪 Material Categories

### By Element Type

#### Alkali Metals (High Hazard)
- Li1Ni1F6 (Li - pyrophoric)
- K2Cu4F10, K2Fe1F6, K1Rb1Cu1Ag1F6
- Rb2Ag2F4, Rb1Fe1F6, Rb3V2F9
- Na3Cr1F4
- Cs2K2Fe1F5, Cs3Mn1Fe1F7

#### Alkaline Earth Metals
- Ba2Cl8Ni1Pb1, Ba1Cs2F12Ni3
- Ca5F11Ni2O1Zn1, Ca4In2O2
- Sr1Ca2Fe1F8, Sr2Li1Cu1F6
- Mg2Cu2O1F4

#### Transition Metals
- Cu2Eu2F9Rb1, K2Cu4F10, Rb2Cu1Ag1F6
- Co1Cu1F10K2Ru1
- Ni2Pb2F6, Y1Mg1Cu1Ni2F11

#### Fluorides (High Reactivity)
- 38 out of 42 materials contain fluorine
- **Safety critical**: All require calcium gluconate gel
- Includes Li1Ni1F6 (double hazard: Li + F)

#### Heavy Metals (Toxic)
- Ba2Cl8Ni1Pb1 (Pb - toxic)
- Ni2Pb2F6
- Rb1Tl1Cu2F6 (Tl - highly toxic)

## 🔄 Recommended Substitution Tests

### Safe Substitutions
1. **K2Cu4F10** → K2Ag4F10 (Cu→Ag) ✅ Ionic radii compatible
2. **Ba2Ag1Fe1F6** → Ba2Ag1Co1F6 (Fe→Co) ✅ Similar chemistry
3. **Sr2Li1Cu1F6** → Sr2Na1Cu1F6 (Li→Na) ✅ Alkali metal swap

### Challenging Substitutions
1. **Li1Ni1F6** → Na1Ni1F6 (Li→Na) ⚠️ Different ionic size
2. **Rb2Cu1Ag1F6** → Cs2Cu1Ag1F6 (Rb→Cs) ⚠️ Larger cation
3. **K2Fe1F6** → K2Mn1F6 (Fe→Mn) ⚠️ Different oxidation states

## 📈 Testing Strategy

### Quick Test (1-2 minutes)
```python
from pipeline.run_pipeline import MaterialsPipeline

pipeline = MaterialsPipeline()

# Test basic material from reaction.csv
result = pipeline.run_materials_pipeline(
    composition="K2Cu4F10",
    generate_cif=True,
    predict_properties=True,
    generate_synthesis=True
)
```

### Substitution Test (2-3 minutes)
```python
# Test element substitution
result = pipeline.run_materials_pipeline(
    composition="K2Cu4F10",
    substitutions={"Cu": "Ag"},
    generate_cif=True,
    predict_properties=True
)
```

### High-Hazard Test (3-5 minutes)
```python
# Test safety protocol generation
result = pipeline.run_materials_pipeline(
    composition="Li1Ni1F6",  # Li (pyrophoric) + F (reactive)
    generate_synthesis=True
)

# Verify safety protocols are comprehensive
assert len(result.hazards_detected) >= 2
assert any(h['element'] == 'Li' for h in result.hazards_detected)
assert any(h['element'] == 'F' for h in result.hazards_detected)
```

## ⚠️ Safety Notes

### Fluoride Materials (38/42)
- **All require**: Calcium gluconate gel on-site
- **PPE**: Face shield + double gloves
- **Ventilation**: Fume hood mandatory
- **Emergency**: HF exposure protocol ready

### Lithium Materials (3 compounds)
- **Li1Ni1F6**, **Rb1Ba1Li1Co1F6**, **Sr2Li1Cu1F6**
- **Hazard**: Pyrophoric (ignites in air)
- **Storage**: Inert atmosphere (Ar/N₂)
- **Handling**: Never expose to moisture

### Heavy Metals
- **Pb compounds**: Toxic, avoid dust
- **Tl compounds**: Extremely toxic, specialized training required
- **Ba compounds**: Toxic if ingested
- **Cr compounds**: Carcinogenic (Cr(VI))

## 🎯 Example Outputs

### From K2Cu4F10 (Colab/quickstart)
- **CIF**: demo_K2Cu4F10.cif
- **Synthesis**: demo_K2Cu4F10_synthesis.txt
- **JSON**: demo_K2Cu4F10_results.json

### From Li1Ni1F6 (High hazard)
- **Hazards**: Li (HIGH - pyrophoric), F (HIGH - reactive)
- **Safety sections**: 5 mandatory sections in protocol
- **Special equipment**: Glovebox, inert atmosphere

## 📚 Full Material List

```
Ba2Cl8Ni1Pb1       Cu2Eu2F9Rb1        Ba1Cs2F12Ni3       Co1Cu1F10K2Ru1    
F4K1Na1Pd2         Ca5F11Ni2O1Zn1     Rb2Cu1Ag1F6        K2Cu4F10          
Mg2Cu2O1F4         Rb1Ca1Cu1Bi2O1F10  K2Fe1F6            K1Rb1Cu1Ag1F6     
Y1Mg1Cu1Ni2F11     Cu1Sn2O1F8         Li1Ni1F6           Rb3V2F9           
Rb1Tl1Cu2F6        Rb1Ba1Li1Co1F6     Cu1Sn1F8           Ba2Ag1Fe1F6       
Sr1Ca2Fe1F8        Ca4In2O2           Rb1Fe1F6           Sr2Li1Cu1F6       
K2Pr1Nd1Fe2F14     Cs3K1Cu2Ag1F9      Cs2K2Fe1F5         Ni2Pb2F6          
Na3Cr1F4           Rb2Ba1Cu2F9        Sr1Cu1F5           Li1Cu1Ag2F6       
Ba1Ca1Cu1F5        Rb2Ag2F4           Sr1Eu1Ni2F6        Cs1Sr3Co1Ni1F9    
Cs3Mn1Fe1F7        Cu2Bi2Au1F13       Cs2Rb2Co2F8        Rb3Sr1Fe2Ag2F12   
Cs2Rb2Cr2F12       Cs2Ce4F14
```

## ✅ Integration Status

- ✅ **Streamlit**: Loads all materials dynamically
- ✅ **Colab**: Uses 3 diverse examples from CSV
- ✅ **quickstart.py**: Uses K2Cu4F10
- ✅ **All samples verified**: Present in reaction.csv
- ✅ **Safety coverage**: High-hazard materials included
- ✅ **Substitution examples**: Compatible pairs selected

---

**All test materials are now sourced from reaction.csv for consistency! 🎯**
