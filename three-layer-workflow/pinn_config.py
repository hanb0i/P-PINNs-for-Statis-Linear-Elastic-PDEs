import os
import torch
import numpy as np

# --- Geometry Dimensions ---
Lx = 1.0
Ly = 1.0
H = 0.1  # Total height (baseline thickness)
# Three layers (homogeneous per layer); z goes from 0 to H.
NUM_LAYERS = 3
LAYER_THICKNESSES = [H / 3.0, H / 3.0, H / 3.0]
Layer_Interfaces = [0.0, LAYER_THICKNESSES[0], LAYER_THICKNESSES[0] + LAYER_THICKNESSES[1], H]

# --- Material Properties ---
# Young's Modulus (E) and Poisson's Ratio (nu)
# Baseline single-material values to match FEM.
E_vals = [1.0] # Normalized
nu_vals = [0.3]
# Parameterized PINN settings (do not alter baseline values)
E_RANGE = [1.0, 10.0]
T1_RANGE = [0.02, 0.10]
T2_RANGE = [0.02, 0.10]
T3_RANGE = [0.02, 0.10]
RESTITUTION_RANGE = [0.5, 0.5]
FRICTION_RANGE = [0.3, 0.3]
IMPACT_VELOCITY_RANGE = [1.0, 1.0]
# Params: [E1, t1, E2, t2, E3, t3, r, mu, v0]
PARAM_DIM = 9

# Optional: explicit E sweep values for `verify_parametric_pinn.py`.
# If not set, it uses `np.linspace(E_RANGE[0], E_RANGE[1], PINN_VERIFY_E_STEPS)`.
# VERIFY_E_SWEEP_VALUES = np.linspace(E_RANGE[0], E_RANGE[1], 10).tolist()
# Optional: explicit restitution/friction sweep values for verification.
# VERIFY_RESTITUTION_SWEEP_VALUES = np.linspace(RESTITUTION_RANGE[0], RESTITUTION_RANGE[1], 7).tolist()
# VERIFY_FRICTION_SWEEP_VALUES = np.linspace(FRICTION_RANGE[0], FRICTION_RANGE[1], 7).tolist()
# VERIFY_IMPACT_VELOCITY_SWEEP_VALUES = np.linspace(IMPACT_VELOCITY_RANGE[0], IMPACT_VELOCITY_RANGE[1], 7).tolist()

# Reference parameter values for parity with baseline FEA (which has no restitution/friction).
RESTITUTION_REF = 0.5
FRICTION_REF = 0.3
IMPACT_VELOCITY_REF = 1.0

# Inference-time compliance correction for E/t:
# Use u = v / E^p instead of v / E (p=1.0). This can help slightly reduce
# high-E under/over-shoot without retraining.
E_COMPLIANCE_POWER = 0.95
# Global compliance calibration applied at evaluation/inference time.
DISPLACEMENT_COMPLIANCE_SCALE = 1.0

# --- Parametric compliance scaling ---
# Many plate-like problems scale strongly with thickness (often ~ 1/t^3).
# We apply a simple thickness-aware scaling in the physics layer:
#   u = (v / E) * (H / t)^alpha
# where H is the baseline thickness (config.H) and t is the sampled thickness.
# Set alpha=0.0 to disable.
THICKNESS_COMPLIANCE_ALPHA = 3.0

def get_lame_params(E, nu):
    lm = (E * nu) / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    return lm, mu

Lame_Params = [get_lame_params(e, n) for e, n in zip(E_vals, nu_vals)]

# --- Loading ---
p0 = 1.0 # Load magnitude

# --- FEA supervision mesh ---
# Default to the refined benchmark mesh so supervised training and reported
# FEM comparisons use the same load integration and comparable mesh quality.
FEM_NE_X = 16
FEM_NE_Y = 16
FEM_NE_Z = 8

# --- Unit-consistent loss scaling ---
# div(sigma) has units of stress/length; scale by a characteristic length.
PDE_LENGTH_SCALE = H

# --- Boundary condition handling ---
# Use hard mask early for shape, then switch to soft BCs for magnitude.
USE_HARD_SIDE_BC = True
HARD_BC_EPOCHS = 1000

# Load patch boundaries (normalized coordinates)
LOAD_PATCH_X = [Lx/3, 2*Lx/3]  # [0.333, 0.667]
LOAD_PATCH_Y = [Ly/3, 2*Ly/3]  # [0.333, 0.667]

# Per-edge side-face BCs: x0 (x=0), x1 (x=Lx), y0 (y=0), y1 (y=Ly).
# Types: "clamped" | "simply_supported" | "free". Empty dict = all clamped (Case A).
# Env override example: PINN_EDGE_BCS="x1=simply_supported,y1=simply_supported"
EDGE_BCS = {}

# --- Network Architecture ---
LAYERS = 4
NEURONS = 64
INTERFACE_FEATURE_BETA = 20.0

# --- Training Hyperparameters ---
LEARNING_RATE = 1e-3
EPOCHS_ADAM = 2000
EPOCHS_LBFGS = 100
# SOAP optimizer
SOAP_PRECONDITION_FREQUENCY = 10 # Lower = more frequent curvature updates; higher = cheaper but less responsive
#Plot Physical Residuals Every N Epochs every 100 epochs. 
WEIGHTS = {
    # Tuned for 3-layer parametric sweep (<5% worst MAE in compare_three_layer_pinn_fem.py)
    'pde': 10.0,
    'bc': 0.7,      # Slightly softer sides so load can gather more budget
    'load': 5.0, # Optimal load weight
    'energy': 0.63, # Per user request
    'impact_invariance': 0.0,  # Set >0 only for neutral-parameter mode
    'impact_contact': 0.0002,   # Reduced to preserve FEA parity in no-supervision mode
    'friction_coulomb': 0.001,  # Reduced to preserve FEA parity in no-supervision mode
    'friction_stick': 0.0005,   # Reduced to preserve FEA parity in no-supervision mode
    'interface_u': 300.0,
    'data': 400.0
}

def _env_flag(name: str, default: bool = False) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "y", "on"}

def _env_int(name: str, default: int) -> int:
    val = os.getenv(name)
    if val is None:
        return default
    try:
        return int(val)
    except ValueError:
        return default

def _env_float(name: str, default: float) -> float:
    val = os.getenv(name)
    if val is None:
        return default
    try:
        return float(val)
    except ValueError:
        return default

def _env_float_list(name: str, default):
    val = os.getenv(name)
    if val is None:
        return default
    out = []
    for part in val.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(float(part))
        except ValueError:
            return default
    return out if out else default

# --- Env overrides (tuning without edits) ---
DISPLACEMENT_COMPLIANCE_SCALE = _env_float("PINN_DISPLACEMENT_COMPLIANCE_SCALE", DISPLACEMENT_COMPLIANCE_SCALE)
THICKNESS_COMPLIANCE_ALPHA = _env_float("PINN_THICKNESS_COMPLIANCE_ALPHA", THICKNESS_COMPLIANCE_ALPHA)
E_COMPLIANCE_POWER = _env_float("PINN_E_COMPLIANCE_POWER", E_COMPLIANCE_POWER)
NEURONS = _env_int("PINN_NEURONS", NEURONS)
LAYERS = _env_int("PINN_LAYERS", LAYERS)

for _k, _env in [
    ("pde", "PINN_W_PDE"),
    ("interface_u", "PINN_W_INTERFACE_U"),
    ("load", "PINN_W_LOAD"),
    ("bc", "PINN_W_BC"),
    ("data", "PINN_W_DATA"),
]:
    if _env in os.environ:
        WEIGHTS[_k] = _env_float(_env, float(WEIGHTS.get(_k, 0.0)))

# Loss weights for simply-supported edges (relative to the other side-BC terms).
#   S_LINE_WEIGHT:     uz=0 on the supported bottom line, averaged over its OWN points
#                      (not diluted by clamped-edge samples).
#   S_TRACTION_WEIGHT: multiplier on the traction-free loss for S/free faces.
S_LINE_WEIGHT = _env_float("PINN_S_LINE_WEIGHT", 1.0)
S_TRACTION_WEIGHT = _env_float("PINN_S_TRACTION_WEIGHT", 1.0)
# Share of each S edge's side samples placed on the supported line (z=0).
S_LINE_FRACTION = _env_float("PINN_S_LINE_FRACTION", 0.25)
# Hard-enforce the simply-supported line (uz=0 on the bottom edge line) with a mask on uz:
#   m = 1 - (1 - z/T) * exp(-d/ell),  d = distance to the S edge, T = total thickness.
# m=0 on the supported line, ->1 away from it. Only active while the hard mask is on.
S_HARD_MASK = _env_flag("PINN_S_HARD_MASK", False)
S_HARD_LENGTH = _env_float("PINN_S_HARD_LENGTH", 0.15)

_edge_env = os.getenv("PINN_EDGE_BCS")
if _edge_env:
    EDGE_BCS = dict(item.strip().split("=", 1) for item in _edge_env.split(",") if item.strip())

# Case geometry/loading overrides (Case B etc.) without editing this file.
#   PINN_LX / PINN_LY: plate dimensions
#   PINN_PATCH_X_FRAC / PINN_PATCH_Y_FRAC: "lo,hi" patch bounds as fractions of Lx / Ly
Lx = _env_float("PINN_LX", Lx)
Ly = _env_float("PINN_LY", Ly)
_px = _env_float_list("PINN_PATCH_X_FRAC", None)
_py = _env_float_list("PINN_PATCH_Y_FRAC", None)
if _px is not None and len(_px) == 2:
    LOAD_PATCH_X = [_px[0] * Lx, _px[1] * Lx]
elif "PINN_LX" in os.environ:
    LOAD_PATCH_X = [Lx / 3, 2 * Lx / 3]
if _py is not None and len(_py) == 2:
    LOAD_PATCH_Y = [_py[0] * Ly, _py[1] * Ly]
elif "PINN_LY" in os.environ:
    LOAD_PATCH_Y = [Ly / 3, 2 * Ly / 3]

# In-plane domain: "rect" (default) or "L" (rectangle minus its top-right corner
# x > cut*Lx and y > cut*Ly; all outer faces clamped). See domain.py.
DOMAIN = os.getenv("PINN_DOMAIN", "rect").strip()
if DOMAIN not in ("rect", "L"):
    raise ValueError(f"PINN_DOMAIN must be 'rect' or 'L', got {DOMAIN!r}")
L_CUT_FRACTION = _env_float("PINN_L_CUT_FRACTION", 0.5)

# PDE decomposition toggle (balances stiff/soft layers during training).
PDE_DECOMPOSE_BY_LAYER = _env_flag("PINN_PDE_DECOMPOSE_BY_LAYER", True)

# Loss weight ramp: load-first to raise displacement while preserving shape.
WEIGHT_RAMP_EPOCHS = 0
LOAD_WEIGHT_START = WEIGHTS['load']
PDE_WEIGHT_START = WEIGHTS['pde']
ENERGY_WEIGHT_START = WEIGHTS['energy']
# Force soft side boundary conditions from the beginning.
FORCE_SOFT_SIDE_BC_FROM_START = True
# Env overrides: train WITH the hard mask (PINN_FORCE_SOFT_SIDE_BC=0) for HARD_BC_EPOCHS epochs.
FORCE_SOFT_SIDE_BC_FROM_START = _env_flag("PINN_FORCE_SOFT_SIDE_BC", FORCE_SOFT_SIDE_BC_FROM_START)
HARD_BC_EPOCHS = _env_int("PINN_HARD_BC_EPOCHS", HARD_BC_EPOCHS)
# Keep the hard mask on during the L-BFGS stage (default off = original behaviour).
LBFGS_KEEP_HARD_BC = _env_flag("PINN_LBFGS_KEEP_HARD_BC", False)
SOFT_MODE_PDE_WEIGHT_SCALE = 3.0
SOFT_MODE_LOAD_WEIGHT_SCALE = 1.0
# Sampling
N_INTERIOR = _env_int("PINN_N_INTERIOR", 15000) # Per layer
N_SIDES = _env_int("PINN_N_SIDES", 2000)  # Clamped side faces
N_TOP_LOAD = _env_int("PINN_N_TOP_LOAD", 6000)  # Load patch (more points to boost displacement)
N_TOP_FREE = _env_int("PINN_N_TOP_FREE", 2000)  # Top free surface
N_BOTTOM = _env_int("PINN_N_BOTTOM", 2000)  # Bottom free surface
N_INTERFACE = _env_int("PINN_N_INTERFACE", 16000)  # Exact points on the layer interface
UNDER_PATCH_FRACTION = 0.95 # More interior points focus under the load patch
INTERFACE_SAMPLE_FRACTION = _env_float("PINN_INTERFACE_SAMPLE_FRACTION", 0.75)
INTERFACE_BAND = 0.05 * H
# Bias a portion of patch samples toward the center.
PATCH_CENTER_BIAS_FRACTION = 0.8
PATCH_CENTER_BIAS_SHAPE = 3.0

#Resampling/perturbation control
SAMPLING_NOISE_SCALE = 0.08  # Larger perturbations widen coverage while still sampling residual-rich zones.

# Auxiliary load-patch average displacement penalty
LOAD_PATCH_UZ_TARGET = -0.05  # Encourage the mean vertical deflection on the load patch
LOAD_PATCH_UZ_WEIGHT = 0.02   # Keep the auxiliary penalty small so shape stays intact

# Fourier Features
FOURIER_DIM = 0 # Number of Fourier frequencies
FOURIER_SCALE = 1.0 # Standard deviation for frequency sampling

# Hybrid / Parametric Training Data
N_DATA_POINTS = _env_int("PINN_N_DATA_POINTS", 36000)
ADAPTIVE_RESAMPLE_EVERY = _env_int("PINN_ADAPTIVE_RESAMPLE_EVERY", 500)
DATA_E_VALUES = [1.0, 10.0]
DATA_T1_VALUES = [0.02, 0.10]
DATA_T2_VALUES = [0.02, 0.10]
DATA_T3_VALUES = [0.02, 0.10]
# Smaller evaluation grid for quick sweeps.
EVAL_E_VALUES = [1.0, 10.0]
EVAL_T1_VALUES = [0.02, 0.10]
EVAL_T2_VALUES = [0.02, 0.10]
EVAL_T3_VALUES = [0.02, 0.10]
USE_SUPERVISION_DATA = _env_flag("PINN_USE_SUPERVISION_DATA", True)

# Default supervision allocation bias toward thin stacks (overridable via env).
SUPERVISION_THICKNESS_POWER = 3.0

DATA_E_VALUES = _env_float_list("PINN_DATA_E_VALUES", DATA_E_VALUES)
DATA_T1_VALUES = _env_float_list("PINN_DATA_T1_VALUES", DATA_T1_VALUES)
DATA_T2_VALUES = _env_float_list("PINN_DATA_T2_VALUES", DATA_T2_VALUES)
DATA_T3_VALUES = _env_float_list("PINN_DATA_T3_VALUES", DATA_T3_VALUES)

EVAL_E_VALUES = _env_float_list("PINN_EVAL_E_VALUES", EVAL_E_VALUES)
EVAL_T1_VALUES = _env_float_list("PINN_EVAL_T1_VALUES", EVAL_T1_VALUES)
EVAL_T2_VALUES = _env_float_list("PINN_EVAL_T2_VALUES", EVAL_T2_VALUES)
EVAL_T3_VALUES = _env_float_list("PINN_EVAL_T3_VALUES", EVAL_T3_VALUES)

# FEM mesh resolution for supervision generation (lower for faster runs).
FEM_NE_X = _env_int("PINN_FEM_NE_X", FEM_NE_X)
FEM_NE_Y = _env_int("PINN_FEM_NE_Y", FEM_NE_Y)
FEM_NE_Z = _env_int("PINN_FEM_NE_Z", FEM_NE_Z)

# --- Explicit impact/friction physics controls ---
# When enabled, restitution/friction influence boundary losses directly.
USE_EXPLICIT_IMPACT_PHYSICS = False
# If True, keeps restitution/friction neutral (used before explicit physics).
ENFORCE_IMPACT_INVARIANCE = False
# Restitution-coupled load amplification gain.
IMPACT_RESTITUTION_GAIN = 0.03
# Impact-velocity gain for dynamic traction amplification.
IMPACT_VELOCITY_GAIN = 0.03
