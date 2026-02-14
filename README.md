# XRD Geometry Simulation (Python)

Python implementation of the X-ray diffraction (XRD) geometry simulator for a 4-circle diffractometer setup with pump-probe geometry.

## Features
- Diffraction and pump-probe geometry visualization
- Computation of omega, two-theta, chi, phi alignment angles
- Optical incidence/polarization angle outputs
- Interactive GUI with 3D rotation test controls (`Rx`, `Ry`, `Rz`)

## Requirements
- Python 3.10+
- `numpy`
- `matplotlib`

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run (default parameters)

```bash
python run_python_simulation.py
```

## Run with custom parameters

```bash
python run_python_simulation.py --h 2 --k 1 --l 1 --alpha-deg 3 --dev-angle-deg 10 --surface 1 0 0
```

## Save output figures

```bash
python run_python_simulation.py --save-prefix output/xrd
```

Generated files:
- `output/xrd_lab_frame.png`
- `output/xrd_sample_views.png`

## Launch interactive viewer

```bash
python run_python_simulation.py --interactive
```

Interactive controls:
- Input sliders: `h`, `k`, `l`, `alpha`, `dev`
- Display rotation test sliders: `Rx`, `Ry`, `Rz`
- Mouse rotation and zoom
