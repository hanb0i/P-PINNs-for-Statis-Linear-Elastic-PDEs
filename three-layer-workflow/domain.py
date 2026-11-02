"""In-plane domain geometry for the PINN: rectangle (default) or an L-shape.

The L-shape is the Lx x Ly rectangle minus its top-right corner
x > cut*Lx and y > cut*Ly, with all outer faces (including the two re-entrant
faces) clamped. Everything geometry-specific lives here so the samplers, the
hard BC mask and the loss all agree on the domain.

For the default rectangle, sample_xy draws x then y with torch.rand, exactly as
the original samplers did, so the random stream (and Case A) is unchanged.
"""
from __future__ import annotations

import math

import torch

import pinn_config as config


def is_l_shape() -> bool:
    return getattr(config, "DOMAIN", "rect") == "L"


def _cut():
    return float(getattr(config, "L_CUT_FRACTION", 0.5))


def area() -> float:
    full = float(config.Lx) * float(config.Ly)
    return full * (1.0 - (1.0 - _cut()) ** 2) if is_l_shape() else full


def inside(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Boolean mask of points inside the domain (inclusive of its boundary)."""
    ok = torch.ones_like(x, dtype=torch.bool)
    if is_l_shape():
        ok = ~((x > _cut() * float(config.Lx)) & (y > _cut() * float(config.Ly)))
    return ok


def sample_xy(n: int):
    """n interior in-plane points, shapes (n,1) each."""
    if not is_l_shape():
        x = torch.rand(n, 1) * config.Lx
        y = torch.rand(n, 1) * config.Ly
        return x, y
    if n <= 0:
        return torch.rand(0, 1), torch.rand(0, 1)
    xs, ys, count = [], [], 0
    while count < n:
        x = torch.rand(2 * n, 1) * config.Lx
        y = torch.rand(2 * n, 1) * config.Ly
        keep = inside(x, y).squeeze(1)
        xs.append(x[keep])
        ys.append(y[keep])
        count += int(keep.sum())
    return torch.cat(xs)[:n], torch.cat(ys)[:n]


def _l_edges():
    """The six outer edges of the L as (x0, y0, x1, y1) segments."""
    Lx, Ly, c = float(config.Lx), float(config.Ly), _cut()
    cx, cy = c * Lx, c * Ly
    return [
        (0.0, 0.0, Lx, 0.0),    # bottom
        (Lx, 0.0, Lx, cy),      # right (lower leg end)
        (Lx, cy, cx, cy),       # re-entrant: top of lower leg
        (cx, cy, cx, Ly),       # re-entrant: right of upper leg
        (cx, Ly, 0.0, Ly),      # top (upper leg end)
        (0.0, Ly, 0.0, 0.0),    # left
    ]


def sample_boundary_xy(n: int):
    """n in-plane points on the outer boundary, spread by edge length. Shapes (n,1)."""
    segs = _l_edges()
    lengths = torch.tensor([math.hypot(x1 - x0, y1 - y0) for x0, y0, x1, y1 in segs])
    which = torch.multinomial(lengths / lengths.sum(), n, replacement=True)
    t = torch.rand(n)
    seg_t = torch.tensor(segs)
    x = seg_t[which, 0] + t * (seg_t[which, 2] - seg_t[which, 0])
    y = seg_t[which, 1] + t * (seg_t[which, 3] - seg_t[which, 1])
    return x.unsqueeze(1), y.unsqueeze(1)


def hard_mask(x_coord: torch.Tensor, y_coord: torch.Tensor) -> torch.Tensor:
    """Smooth multiplier that is 0 on the whole boundary of the L and >0 inside.

    R-function union of the two legs. phi_i is positive inside leg i and negative
    outside it, so  phi = phi1 + phi2 + sqrt(phi1^2 + phi2^2)  is positive in the
    union and exactly zero on its boundary (including the re-entrant faces).
    Scaled so its maximum inside is about 1.
    """
    Lx, Ly, c = float(config.Lx), float(config.Ly), _cut()
    u, v = x_coord / Lx, y_coord / Ly
    phi1 = u * (1.0 - u) * v * (c - v)          # lower leg:  0<u<1, 0<v<c
    phi2 = u * (c - u) * v * (1.0 - v)          # left leg:   0<u<c, 0<v<1
    phi = phi1 + phi2 + torch.sqrt(phi1 ** 2 + phi2 ** 2 + 1e-24)
    return phi * _SCALE.get(round(c, 6), _scale_for(c))


_SCALE: dict = {}


def _scale_for(c: float) -> float:
    g = torch.linspace(0, 1, 401)
    uu, vv = torch.meshgrid(g, g, indexing="ij")
    p1 = uu * (1 - uu) * vv * (c - vv)
    p2 = uu * (c - uu) * vv * (1 - vv)
    phi = p1 + p2 + torch.sqrt(p1 ** 2 + p2 ** 2)
    inside_l = ~((uu > c) & (vv > c))
    _SCALE[round(c, 6)] = float(1.0 / phi[inside_l].max())
    return _SCALE[round(c, 6)]
