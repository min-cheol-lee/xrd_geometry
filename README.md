# XRD_geometry_simulation
Simulation for X-ray diffraction (XRD) geometric conditions of 4-circle diffractometer system with a fixed grazing angle condition, and pump beam geometry for pump-probe experiments. You can find the following results from this simulation.

- visualization of diffraction + pump-probe geometry
- omega, two-theta, chi, phi angles 4C alignment for X-ray beam
- incidence grazing angle, polarization angle for optical beam

![image](https://user-images.githubusercontent.com/75286302/113194679-9f0c7500-921e-11eb-9265-8095a240602f.png)![image](https://user-images.githubusercontent.com/75286302/113194490-6ff60380-921e-11eb-9e03-a5ba6bce64aa.png)

example of simulation output for geometric conditions of optical-pump and XRD-probe on YBa2Cu3O6.7 with grazing incidence angle 3 deg.

## MATLAB entry point
Run:

```matlab
XRD_geometry_simulation
```

## Python migration (new)
A Python version is included to make further expansion (GUI, additional materials, and richer workflows) easier.

### Requirements
- Python 3.10+
- `numpy`
- `matplotlib`

Install dependencies:

```bash
pip install numpy matplotlib
```

### Run with default YBCO-style parameters

```bash
python run_python_simulation.py
```

### Run with custom input parameters

```bash
python run_python_simulation.py --h 2 --k 1 --l 1 --alpha-deg 3 --dev-angle-deg 10 --surface 1 0 0
```

### Save figures to files

```bash
python run_python_simulation.py --save-prefix output/xrd
```

This writes:
- `output/xrd_lab_frame.png`
- `output/xrd_sample_views.png`

### Launch an interactive viewer (starter GUI)

```bash
python run_python_simulation.py --interactive
```

The interactive view supports:
- slider inputs for `h, k, l`, grazing incidence angle, and pump deviation angle
- display rotation test sliders `Rx, Ry, Rz` to rotate the full 3D scene
- mouse rotation and zoom to inspect geometry details
- recomputation of output alignment angles on parameter updates

### Launch a modern web GUI (Dash + Plotly)

Install additional web dependencies:

```bash
pip install dash plotly
```

Run:

```bash
python run_dash_app.py
```

Then open:
- `http://127.0.0.1:8050`

The web GUI includes:
- styled control panel for `h, k, l, alpha, dev`
- interactive 3D Plotly scene
- display rotation test sliders `Rx, Ry, Rz`
- live angle table updates
