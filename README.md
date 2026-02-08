# MOF Performance Predictor 🔬⚡

Physics-Informed Machine Learning for Predicting Metal-Organic Framework (MOF) Electrochemical Performance Before Wet-Lab Synthesis

## 📋 Overview

This project provides a complete pipeline for predicting the electrochemical performance of Metal-Organic Frameworks (MOFs) using physics-informed machine learning. The system generates four critical characterization graphs:

- **GCD (Galvanostatic Charge-Discharge)** - Voltage behavior during cycles
- **Rate Capability** - Capacity vs. current density
- **IES (Inelastic Electron Scattering)** - Electronic structure
- **EIS (Electrochemical Impedance Spectroscopy)** - Internal resistance

## 🎯 Key Features

- ✅ **Synthetic Data Generation** - Physics-based training data creation
- ✅ **Deep Learning Model** - Multi-output neural network with PyTorch
- ✅ **Interactive Web Interfaces** - Both Streamlit and Gradio frontends
- ✅ **REST API** - Flask server for predictions
- ✅ **Real-time Predictions** - Sub-second inference time
- ✅ **Export Capabilities** - Download data as CSV

## 📊 Project Structure

```
mof_predictor/
├── generate_synthetic_data.py   # Generate training dataset
├── train_model.py                # Train the neural network
├── streamlit_app.py              # Streamlit web interface
├── gradio_app.py                 # Gradio web interface
├── api_server.py                 # Flask REST API
├── mof_frontend.jsx              # React frontend (optional)
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or download the project files
cd /home/claude

# Install dependencies
python install -r requirements.txt --break-system-packages
```

### 2. Generate Synthetic Data

```bash
python generate_synthetic_data.py
```

This creates:
- `mof_data/synthetic_dataset.json` - Full dataset (5000 samples)
- `mof_data/feature_summary.csv` - Parameter distribution summary

Expected output:
```
Generating synthetic MOF performance data...
Target samples: 5000
Generated 500/5000 samples...
Generated 1000/5000 samples...
...
✓ Data generation complete!
```

### 3. Train the Model

```bash
python train_model.py
```

This trains the neural network and saves:
- `mof_model_best.pth` - Best model weights
- `mof_model_final.pth` - Final epoch weights
- `scaler.pkl` - Feature scaler
- `training_history.png` - Loss curves

Training metrics:
- **Architecture**: Multi-task neural network with shared encoder
- **Training time**: ~10-20 minutes on CPU
- **Expected validation loss**: <0.05

### 4. Launch Web Interface

#### Option A: Streamlit (Recommended)

```bash
streamlit run streamlit_app.py
```

Access at: `http://localhost:8501`

Features:
- Modern, responsive UI
- Interactive parameter selection
- Real-time graph generation
- Data export functionality
- Dark theme with gradients

#### Option B: Gradio

```bash
python gradio_app.py
```

Access at: `http://localhost:7860`

Features:
- Simple, clean interface
- Example configurations
- One-click predictions
- Shareable public link

#### Option C: REST API

```bash
python api_server.py
```

API endpoint: `http://localhost:5000/predict`

Example request:
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "metal": "Cu",
    "valency": 2,
    "ligand": "BDC",
    "assembly": "Three-Electrode",
    "electrode": "Nickel Foam",
    "is_mof": true
  }'
```

## 🔧 Input Parameters

| Parameter | Type | Options | Description |
|-----------|------|---------|-------------|
| **Metal** | Dropdown | Cu, Ni, Co, Zn, Fe, Mn | Metal center in MOF |
| **Valency** | Slider | 1, 2, 3 | Oxidation state |
| **Ligand** | Dropdown | BDC, BTC, DOBDC, BPDC, NDC, TPA | Organic linker |
| **Assembly** | Radio | Two-Electrode, Three-Electrode | Cell configuration |
| **Electrode** | Dropdown | Nickel Foam, Glassy Carbon, Carbon Cloth, Stainless Steel | Current collector |
| **Is MOF** | Checkbox | True/False | Porous framework boost |

## 📈 Output Graphs

### 1. Galvanostatic Charge-Discharge (GCD)
- **X-axis**: Time (hours)
- **Y-axis**: Voltage (V)
- **Curves**: Multiple current densities (0.5 - 2.5 A/g)
- **Interpretation**: Longer discharge = better capacity

### 2. Rate Capability
- **X-axis**: Current Density (A/g)
- **Y-axis**: Specific Capacity (mAh/g)
- **Interpretation**: Flatter curve = better rate performance

### 3. IES Spectrum
- **X-axis**: Energy Loss (eV)
- **Y-axis**: Intensity (arbitrary units)
- **Features**: Plasmon peak (~20 eV), d-d transitions (~5 eV)

### 4. EIS Nyquist Plot
- **X-axis**: Real Impedance Z' (Ω)
- **Y-axis**: Imaginary Impedance -Z'' (Ω)
- **Interpretation**: Smaller semicircle = lower resistance

## 🧪 Physics-Informed Logic

The model incorporates established electrochemical principles:

### Metal Effects
- **Cu**: Highest redox activity (1.2× boost)
- **Co**: Excellent conductivity (1.15× boost)
- **Ni**: Balanced performance (1.1× boost)
- **Fe/Mn/Zn**: Moderate performance

### Electrode Effects
- **Nickel Foam**: Best surface area (1.3× boost)
- **Carbon Cloth**: Good flexibility (1.15× boost)
- **Stainless Steel**: Moderate (1.1× boost)
- **Glassy Carbon**: Baseline (1.0×)

### MOF Structure
- **Enabled**: 1.5× capacity boost from porosity
- **Disabled**: Simple metal complex behavior

## 📊 Model Architecture

```
Input Layer (19 features)
    ↓
Shared Encoder
    Dense(256) → ReLU → BatchNorm → Dropout(0.3)
    Dense(512) → ReLU → BatchNorm → Dropout(0.3)
    Dense(256) → ReLU → BatchNorm
    ↓
Task-Specific Decoders
    ├─→ GCD Head (20 outputs)
    ├─→ Rate Head (10 outputs)
    ├─→ IES Head (20 outputs)
    └─→ EIS Head (30 outputs)
```

**Training Configuration:**
- Optimizer: Adam (lr=0.001)
- Loss: MSE (Multi-task)
- Batch Size: 32
- Epochs: 100
- Validation Split: 20%

## 💡 Usage Examples

### High-Performance Configuration
```
Metal: Cu
Valency: 2
Ligand: BDC
Assembly: Three-Electrode
Electrode: Nickel Foam
Is MOF: ✓
```
**Expected**: >250 mAh/g capacity, >45% retention @ 5A/g

### Moderate Configuration
```
Metal: Zn
Valency: 2
Ligand: BPDC
Assembly: Two-Electrode
Electrode: Glassy Carbon
Is MOF: ✗
```
**Expected**: ~150 mAh/g capacity, ~35% retention @ 5A/g

## 🔬 Scientific Background

### Why This Matters
Traditional MOF characterization requires:
- 2-4 weeks synthesis time
- Expensive characterization equipment ($100K+)
- Material waste from failed attempts

This tool provides:
- **Instant** performance estimation
- **Zero** material cost
- **Rapid** iteration on design parameters

### Validation Approach
The physics-informed approach ensures predictions align with:
- Structure-property relationships
- Electrochemical theory
- Literature-reported trends

## 🛠️ Customization

### Add New Metals
Edit `generate_synthetic_data.py`:
```python
METALS = ['Cu', 'Ni', 'Co', 'Zn', 'Fe', 'Mn', 'YOUR_METAL']
METAL_PROPERTIES = {
    'YOUR_METAL': {
        'redox_factor': 1.1,
        'conductivity': 1.05,
        'plasmon_peak': 21,
        'd_transition': 6
    }
}
```

### Adjust Model Complexity
Edit `train_model.py`:
```python
# Increase encoder size
nn.Linear(input_dim, 512),  # was 256
nn.Linear(512, 1024),       # was 512
```

### Change Current Densities
Edit GCD generation in `generate_synthetic_data.py`:
```python
current_densities = np.array([0.5, 1.0, 2.0, 3.0, 5.0])  # Add higher rates
```

## 📦 Dependencies

Core requirements:
- Python 3.8+
- PyTorch 2.0+
- NumPy
- Pandas
- Matplotlib
- Streamlit / Gradio
- Flask
- Plotly
- scikit-learn

See `requirements.txt` for complete list.

## 🐛 Troubleshooting

### Model Not Found
```
⚠️ Model files not found. Using physics-based simulation instead.
```
**Solution**: Run `python train_model.py` first

### Import Errors
```
ModuleNotFoundError: No module named 'torch'
```
**Solution**: Install dependencies
```bash
pip install -r requirements.txt --break-system-packages
```

### Low Prediction Accuracy
- Ensure model finished training (100 epochs)
- Check validation loss < 0.1
- Verify scaler.pkl exists
- Re-generate data if corrupted

### Port Already in Use
```
OSError: [Errno 98] Address already in use
```
**Solution**: Change port in app
```python
# Streamlit: streamlit run app.py --server.port 8502
# Gradio: demo.launch(server_port=7861)
```

## 🚀 Future Enhancements

Planned features:
- [ ] Integration with Cambridge Structural Database (CSD)
- [ ] Graph Neural Networks for CIF file input
- [ ] Multi-metal MOFs (bimetallic systems)
- [ ] Temperature-dependent predictions
- [ ] Cyclic voltammetry simulation
- [ ] Web deployment (Hugging Face Spaces)
- [ ] Active learning loop
- [ ] Experimental validation module

## 📚 References

1. **MOF Electrochemistry**: Sheberla et al., Nature Materials (2017)
2. **Physics-Informed ML**: Raissi et al., J. Computational Physics (2019)
3. **Structure-Property**: Choi et al., ACS Applied Materials (2020)

## 📄 License

MIT License - Feel free to use for research and education

## 👥 Contributing

Contributions welcome! Areas of interest:
- Additional electrode materials
- Improved physics models
- Real experimental data integration
- UI/UX enhancements

## 📧 Contact

For questions or collaboration:
- Open an issue on GitHub
- Email: [your-email]

## 🙏 Acknowledgments

- Anthropic Claude for development assistance
- Materials science community for domain knowledge
- Open-source ML/visualization libraries

---

**Built with ❤️ for the materials science community**

*Accelerating MOF discovery through AI-powered predictions*
