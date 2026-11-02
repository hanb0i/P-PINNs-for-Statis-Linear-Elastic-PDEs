
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

EDGE_NAMES = ("x0", "x1", "y0", "y1")
_BC_FIXED_COMPONENTS = {
    "clamped": (0, 1, 2),
    # Classical-analogue simple support for a 3D solid: uz=0 along the bottom
    # edge line only (z=0); the rest of the face is traction-free, so the cross
    # section can rotate (w=0, M_n~0 in plate terms).
    "simply_supported": (2,),
    # uz=0 over the whole side face (in-plane motion free); stiffer than a hinge.
    "roller_face": (2,),
    "free": (),
    # Symmetry planes for reduced models: normal displacement fixed, rest free.
    "sym_ux": (0,),
    "sym_uy": (1,),
}
# Edge types that constrain only the bottom edge line (k == 0).
_BOTTOM_LINE_ONLY = {"simply_supported"}


def resolve_edge_bcs(cfg):
    """Return {edge: bc_type} with clamped as the default for every edge."""
    given = cfg.get("boundary", {}) or {}
    unknown = set(given) - set(EDGE_NAMES)
    if unknown:
        raise ValueError(f"Unknown boundary edges: {sorted(unknown)}")
    out = {}
    for edge in EDGE_NAMES:
        kind = given.get(edge, "clamped")
        if kind not in _BC_FIXED_COMPONENTS:
            raise ValueError(f"Unknown BC type {kind!r} on edge {edge}")
        out[edge] = kind
    return out


def active_element_mask(cfg):
    """Boolean (ne_y, ne_x) mask of elements kept in the model.

    cfg["domain"] = {"type": "L", "cut_x": 0.5, "cut_y": 0.5} removes the corner
    region x > cut_x*Lx and y > cut_y*Ly (an L-shaped plate). Without "domain"
    every element is active (rectangle). Cut lines must lie on mesh lines.
    """
    g = cfg["geometry"]
    ne_x, ne_y = int(g.get("ne_x", 30)), int(g.get("ne_y", 30))
    dom = cfg.get("domain")
    active = np.ones((ne_y, ne_x), dtype=bool)
    if not dom:
        return active
    if dom.get("type") != "L":
        raise ValueError(f"Unknown domain type {dom.get('type')!r}")
    for frac, n in ((dom["cut_x"], ne_x), (dom["cut_y"], ne_y)):
        if abs(frac * n - round(frac * n)) > 1e-9:
            raise ValueError("L-domain cut must lie on a mesh line (cut * ne must be an integer)")
    cx, cy = int(round(dom["cut_x"] * ne_x)), int(round(dom["cut_y"] * ne_y))
    active[cy:, cx:] = False
    return active


def active_node_mask(cfg):
    """Boolean (nx, ny) mask of nodes that belong to at least one active element."""
    act = active_element_mask(cfg)
    ne_y, ne_x = act.shape
    nodes = np.zeros((ne_x + 1, ne_y + 1), dtype=bool)
    for dj in (0, 1):
        for di in (0, 1):
            nodes[di:di + ne_x, dj:dj + ne_y] |= act.T
    return nodes


def _domain_boundary_columns(cfg):
    """Set of (i, j) node columns lying on the boundary of the active domain."""
    act = active_element_mask(cfg)
    ne_y, ne_x = act.shape
    cols = set()
    for j in range(ne_y):
        for i in range(ne_x):
            if not act[j, i]:
                continue
            for di, dj, nodes in (
                (-1, 0, ((i, j), (i, j + 1))),
                (1, 0, ((i + 1, j), (i + 1, j + 1))),
                (0, -1, ((i, j), (i + 1, j))),
                (0, 1, ((i, j + 1), (i + 1, j + 1))),
            ):
                ni, nj = i + di, j + dj
                if ni < 0 or nj < 0 or ni >= ne_x or nj >= ne_y or not act[nj, ni]:
                    cols.update(nodes)
    return cols


def _layer_ids_from_z_centers(z_centers, t_layers):
    """Assign each element to the layer containing its centroid.

    Layers are ordered bottom-to-top. Adjacent layers share mesh nodes at their
    interface, so displacement continuity is enforced by the global DOFs. The
    assembled equilibrium equations transmit equal-and-opposite interface
    tractions; only material stiffness changes by layer.
    """
    interfaces = np.cumsum(np.array(t_layers, dtype=float))
    layer_ids = np.searchsorted(interfaces, z_centers, side="right")
    return np.clip(layer_ids, 0, len(t_layers) - 1)


def _hex8_stiffness(dx, dy, dz, E, nu):
    lam = (E * nu) / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))

    gp = [-1 / np.sqrt(3), 1 / np.sqrt(3)]
    ke = np.zeros((24, 24))
    c_diag = [lam + 2 * mu, lam + 2 * mu, lam + 2 * mu, mu, mu, mu]
    c_mat = np.zeros((6, 6))
    c_mat[0:3, 0:3] = lam
    np.fill_diagonal(c_mat, c_diag)

    for r_val in gp:
        for s_val in gp:
            for t_val in gp:
                inv_j = np.diag([2 / dx, 2 / dy, 2 / dz])
                det_j = dx * dy * dz / 8.0
                b_mat = np.zeros((6, 24))
                node_signs = [
                    [-1, -1, -1],
                    [1, -1, -1],
                    [1, 1, -1],
                    [-1, 1, -1],
                    [-1, -1, 1],
                    [1, -1, 1],
                    [1, 1, 1],
                    [-1, 1, 1],
                ]
                for node_idx, (xi, eta, zeta) in enumerate(node_signs):
                    dN_dxi = 0.125 * xi * (1 + eta * s_val) * (1 + zeta * t_val)
                    dN_deta = 0.125 * eta * (1 + xi * r_val) * (1 + zeta * t_val)
                    dN_dzeta = 0.125 * zeta * (1 + xi * r_val) * (1 + eta * s_val)
                    d_global = inv_j @ np.array([dN_dxi, dN_deta, dN_dzeta])
                    nx_val, ny_val, nz_val = d_global
                    col = 3 * node_idx
                    b_mat[0, col] = nx_val
                    b_mat[1, col + 1] = ny_val
                    b_mat[2, col + 2] = nz_val
                    b_mat[3, col + 1] = nz_val
                    b_mat[3, col + 2] = ny_val
                    b_mat[4, col] = nz_val
                    b_mat[4, col + 2] = nx_val
                    b_mat[5, col] = ny_val
                    b_mat[5, col + 1] = nx_val
                with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                    ke += b_mat.T @ c_mat @ b_mat * det_j
    return ke


def _assemble_and_solve(cfg, ke_per_element):
    Lx, Ly, H = cfg["geometry"]["Lx"], cfg["geometry"]["Ly"], cfg["geometry"]["H"]
    ne_x = int(cfg["geometry"].get("ne_x", 30))
    ne_y = int(cfg["geometry"].get("ne_y", 30))
    ne_z = int(cfg["geometry"].get("ne_z", 10))

    dx, dy, dz = Lx / ne_x, Ly / ne_y, H / ne_z
    nx, ny, nz = ne_x + 1, ne_y + 1, ne_z + 1
    n_dof = nx * ny * nz * 3

    print("Assembling...")
    el_indices = np.arange(ne_x * ne_y * ne_z)
    ek, ej, ei = np.unravel_index(el_indices, (ne_z, ne_y, ne_x))
    active_el = active_element_mask(cfg)
    keep = active_el[ej, ei]
    if not keep.all():
        ek, ej, ei = ek[keep], ej[keep], ei[keep]
        ke_per_element = ke_per_element[keep]

    n0 = (ei) + (ej) * nx + (ek) * nx * ny
    n1 = (ei + 1) + (ej) * nx + (ek) * nx * ny
    n2 = (ei + 1) + (ej + 1) * nx + (ek) * nx * ny
    n3 = (ei) + (ej + 1) * nx + (ek) * nx * ny
    n4 = (ei) + (ej) * nx + (ek + 1) * nx * ny
    n5 = (ei + 1) + (ej) * nx + (ek + 1) * nx * ny
    n6 = (ei + 1) + (ej + 1) * nx + (ek + 1) * nx * ny
    n7 = (ei) + (ej + 1) * nx + (ek + 1) * nx * ny

    conn = np.stack([n0, n1, n2, n3, n4, n5, n6, n7], axis=1)

    dof_indices = np.zeros((conn.shape[0], 24), dtype=int)
    for i in range(8):
        dof_indices[:, i * 3 : i * 3 + 3] = conn[:, i : i + 1] * 3 + np.array([0, 1, 2])

    dof_rows = np.repeat(dof_indices, 24, axis=1).ravel()
    dof_cols = np.tile(dof_indices, (1, 24)).ravel()

    vals = ke_per_element.reshape(-1)
    K = sp.coo_matrix((vals, (dof_rows, dof_cols)), shape=(n_dof, n_dof)).tocsr()

    F = np.zeros(n_dof)
    p0 = cfg["load_patch"]["pressure"]

    k = nz - 1
    x_nodes = np.linspace(0, Lx, nx)
    y_nodes = np.linspace(0, Ly, ny)

    patch_x_min = cfg["load_patch"]["x_start"] * Lx
    patch_x_max = cfg["load_patch"]["x_end"] * Lx
    patch_y_min = cfg["load_patch"]["y_start"] * Ly
    patch_y_max = cfg["load_patch"]["y_end"] * Ly

    def load_mask(x, y):
        if x < patch_x_min or x > patch_x_max or y < patch_y_min or y > patch_y_max:
            return 0.0

        x_norm = (x - patch_x_min) / (patch_x_max - patch_x_min)
        y_norm = (y - patch_y_min) / (patch_y_max - patch_y_min)
        return 16.0 * x_norm * (1.0 - x_norm) * y_norm * (1.0 - y_norm)

    # Top traction load integrated over element faces with 2x2 Gauss quadrature.
    # This is more mesh-consistent than assigning dx*dy loads directly to nodes
    # and uses the same smooth patch profile as the PINN load residual.
    gp = [-1 / np.sqrt(3), 1 / np.sqrt(3)]
    face_det_j = dx * dy / 4.0
    top_k = nz - 1
    for elem_j in range(ne_y):
        for elem_i in range(ne_x):
            if not active_el[elem_j, elem_i]:
                continue
            face_nodes = [
                elem_i + elem_j * nx + top_k * nx * ny,
                (elem_i + 1) + elem_j * nx + top_k * nx * ny,
                (elem_i + 1) + (elem_j + 1) * nx + top_k * nx * ny,
                elem_i + (elem_j + 1) * nx + top_k * nx * ny,
            ]
            x0 = elem_i * dx
            y0 = elem_j * dy
            for r_val in gp:
                for s_val in gp:
                    shape = np.array(
                        [
                            0.25 * (1.0 - r_val) * (1.0 - s_val),
                            0.25 * (1.0 + r_val) * (1.0 - s_val),
                            0.25 * (1.0 + r_val) * (1.0 + s_val),
                            0.25 * (1.0 - r_val) * (1.0 + s_val),
                        ]
                    )
                    x_q = x0 + 0.5 * (r_val + 1.0) * dx
                    y_q = y0 + 0.5 * (s_val + 1.0) * dy
                    traction_z = -p0 * load_mask(x_q, y_q)
                    for node_id, n_val in zip(face_nodes, shape):
                        F[3 * node_id + 2] += traction_z * n_val * face_det_j

    # Per-edge side-face boundary conditions. cfg["boundary"] maps edge name to
    # a type; missing keys (or a missing "boundary" entry) default to clamped,
    # which reproduces the original all-clamped benchmark (Case A).
    #   x0: x=0, x1: x=Lx, y0: y=0, y1: y=Ly
    #   clamped          -> ux=uy=uz=0 on the whole side face
    #   simply_supported -> uz=0 on the bottom edge line only (classical analogue)
    #   roller_face      -> uz=0 on the whole side face (in-plane motion free)
    #   free             -> no constraint (natural traction-free)
    bc = resolve_edge_bcs(cfg)
    if cfg.get("domain"):
        # Non-rectangular domain: clamp every node column on the outer boundary
        # (including re-entrant faces) and fix nodes not in any active element.
        if cfg.get("boundary"):
            raise ValueError("per-edge 'boundary' is not supported together with 'domain' (all outer faces are clamped)")
        bc = {e: "free" for e in EDGE_NAMES}
    edge_nodes = {
        "x0": [(0 + j * nx + k * nx * ny, k) for j in range(ny) for k in range(nz)],
        "x1": [((nx - 1) + j * nx + k * nx * ny, k) for j in range(ny) for k in range(nz)],
        "y0": [(i + 0 * nx + k * nx * ny, k) for i in range(nx) for k in range(nz)],
        "y1": [(i + (ny - 1) * nx + k * nx * ny, k) for i in range(nx) for k in range(nz)],
    }
    fixed_dofs = []
    for edge, nodes in edge_nodes.items():
        kind = bc[edge]
        comps = _BC_FIXED_COMPONENTS[kind]
        for n_id, k_level in nodes:
            if kind in _BOTTOM_LINE_ONLY and k_level != 0:
                continue
            fixed_dofs.extend(3 * n_id + c for c in comps)

    if cfg.get("domain"):
        node_active = active_node_mask(cfg)
        boundary_cols = _domain_boundary_columns(cfg)
        for j in range(ny):
            for i in range(nx):
                if (not node_active[i, j]) or ((i, j) in boundary_cols):
                    for k in range(nz):
                        n_id = i + j * nx + k * nx * ny
                        fixed_dofs.extend([3 * n_id, 3 * n_id + 1, 3 * n_id + 2])

    if len(fixed_dofs) == 0:
        raise ValueError("All edges are free: rigid-body modes are unconstrained.")

    fixed_dofs = np.unique(fixed_dofs)

    penalty = 1e12
    K = K + sp.coo_matrix(
        (np.ones(len(fixed_dofs)) * penalty, (fixed_dofs, fixed_dofs)),
        shape=(n_dof, n_dof),
    ).tocsr()

    print("Solving...")
    if cfg.get("diagonal_symmetry"):
        u = _solve_diagonal_reduced(K, F, nx, ny, nz)
    else:
        u = spla.spsolve(K, F)

    u_grid = np.zeros((nx, ny, nz, 3))
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                idx = i + j * nx + k * nx * ny
                u_grid[i, j, k] = u[3 * idx : 3 * idx + 3]

    return x_nodes, y_nodes, np.linspace(0, H, nz), u_grid


def _solve_diagonal_reduced(K, F, nx, ny, nz):
    """Solve exploiting diagonal (x<->y) symmetry of a square quarter model.

    The solution is invariant under swapping x and y: u_x(i,j,k)=u_y(j,i,k),
    u_z(i,j,k)=u_z(j,i,k). With P mapping reduced DOFs (nodes i>=j) to the full
    vector, solve (P^T K P) u_r = P^T F, so the linear system has about half the
    unknowns of the quarter model (about 1/8 of the full plate).
    """
    if nx != ny:
        raise ValueError("diagonal_symmetry requires a square (nx == ny) model")
    node = lambda i, j, k: i + j * nx + k * nx * ny
    red_index = {}
    for k in range(nz):
        for j in range(ny):
            for i in range(j, nx):
                red_index[(i, j, k)] = len(red_index)
    n_full = nx * ny * nz * 3
    rows, cols = [], []
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                n = node(i, j, k)
                if i >= j:
                    r = red_index[(i, j, k)]
                    comps = ((0, 0), (1, 1), (2, 2))
                else:
                    r = red_index[(j, i, k)]
                    comps = ((0, 1), (1, 0), (2, 2))  # (full component, reduced component)
                for fc, rc in comps:
                    rows.append(3 * n + fc)
                    cols.append(3 * r + rc)
    P = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n_full, 3 * len(red_index))).tocsr()
    Kr = (P.T @ K @ P).tocsr()
    Fr = P.T @ F
    return P @ spla.spsolve(Kr, Fr)


def solve_fem(cfg):
    print("Initializing FEA Solver...")
    Lx, Ly, H = cfg["geometry"]["Lx"], cfg["geometry"]["Ly"], cfg["geometry"]["H"]
    ne_x = int(cfg["geometry"].get("ne_x", 30))
    ne_y = int(cfg["geometry"].get("ne_y", 30))
    ne_z = int(cfg["geometry"].get("ne_z", 10))
    dx, dy, dz = Lx / ne_x, Ly / ne_y, H / ne_z

    material = cfg["material"]
    ke = _hex8_stiffness(dx, dy, dz, material["E"], material["nu"])
    ke_per_element = np.tile(ke.ravel(), ne_x * ne_y * ne_z).reshape(-1, 24 * 24)
    return _assemble_and_solve(cfg, ke_per_element)


def solve_two_layer_fem(cfg):
    print("Initializing Two-Layer FEA Solver...")
    Lx, Ly, H = cfg["geometry"]["Lx"], cfg["geometry"]["Ly"], cfg["geometry"]["H"]
    ne_x = int(cfg["geometry"].get("ne_x", 30))
    ne_y = int(cfg["geometry"].get("ne_y", 30))
    ne_z = int(cfg["geometry"].get("ne_z", 10))
    dx, dy, dz = Lx / ne_x, Ly / ne_y, H / ne_z

    material = cfg["material"]
    nu = material["nu"]
    e_layers = material["E_layers"]
    t_layers = material.get("t_layers", [0.5 * H, 0.5 * H])
    if len(e_layers) != 2 or len(t_layers) != 2:
        raise ValueError("solve_two_layer_fem expects exactly two layers.")

    ke_bottom = _hex8_stiffness(dx, dy, dz, float(e_layers[0]), nu)
    ke_top = _hex8_stiffness(dx, dy, dz, float(e_layers[1]), nu)

    ek, _, _ = np.unravel_index(np.arange(ne_x * ne_y * ne_z), (ne_z, ne_y, ne_x))
    z_centers = (ek + 0.5) * dz
    # Element material lookup: bottom layer is index 0, top layer is index 1.
    # Interface continuity is not imposed with duplicate constraints; it follows
    # from shared nodes and the single global displacement vector.
    layer_ids = _layer_ids_from_z_centers(z_centers, t_layers)
    ke_flat_bottom = ke_bottom.ravel()
    ke_flat_top = ke_top.ravel()
    ke_per_element = np.where(layer_ids[:, None] == 0, ke_flat_bottom[None, :], ke_flat_top[None, :])

    return _assemble_and_solve(cfg, ke_per_element)


def solve_three_layer_fem(cfg):
    print("Initializing Three-Layer FEA Solver...")
    Lx, Ly, H = cfg["geometry"]["Lx"], cfg["geometry"]["Ly"], cfg["geometry"]["H"]
    ne_x = int(cfg["geometry"].get("ne_x", 30))
    ne_y = int(cfg["geometry"].get("ne_y", 30))
    ne_z = int(cfg["geometry"].get("ne_z", 10))
    dx, dy, dz = Lx / ne_x, Ly / ne_y, H / ne_z

    material = cfg["material"]
    nu = material["nu"]
    e_layers = material["E_layers"]
    t_layers = material.get("t_layers", [H / 3.0, H / 3.0, H / 3.0])
    if len(e_layers) != 3 or len(t_layers) != 3:
        raise ValueError("solve_three_layer_fem expects exactly three layers.")

    ke_by_layer = np.stack(
        [_hex8_stiffness(dx, dy, dz, float(e_layers[i]), nu).ravel() for i in range(3)],
        axis=0,
    )

    ek, _, _ = np.unravel_index(np.arange(ne_x * ne_y * ne_z), (ne_z, ne_y, ne_x))
    z_centers = (ek + 0.5) * dz
    # Element material lookup: layers are ordered bottom-to-top as
    # [E1, E2, E3] with interfaces at cumulative [t1, t1+t2].
    # Shared interface nodes carry one displacement value, and the assembled
    # stiffness matrix enforces traction balance weakly across the interface.
    layer_ids = _layer_ids_from_z_centers(z_centers, t_layers)
    ke_per_element = ke_by_layer[layer_ids]

    return _assemble_and_solve(cfg, ke_per_element)
