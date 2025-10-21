# Finite strain ellipsoid: plane strain / constriction / flattening
# with optional simple shear in xy, xz, yz
#
# Python translation of the provided MATLAB script using numpy + pyvista
#
# Based on the MATLAB script in supplementary material of:
# Spitz, R., Schmalholz, S.M., Kaus, B.J.P. and Popov, A.A. (2020) 
# Quantification and visualization of finite strain in 3D viscous numerical models 
# of folding and overthrusting. 
# Journal of Structural Geology, 131, 103945. doi: 10.1016/j.jsg.2019.103945.


import numpy as np
import pyvista as pv

# -----------------------------
# Input
# -----------------------------
plane_strain = 0
constriction = 1
flattening   = 0

# Pure shear rate
ps_rate = 1.0

# Simple shear rates (will change over time below)
xy_rate = 0.0
xz_rate = 0.0
yz_rate = 0.0

# Time controls
dt   = 0.01     # time step
nt   = 70       # number of time steps
tsxy = 20       # activate xy shear after this step
tsxz = 50       # switch to xz shear after this step

# -----------------------------
# Pre-processing
# -----------------------------
# Sphere parametric grid (matches MATLAB)
r = np.pi * (np.arange(-24, 24.0 + 0.5, 0.5) / 24.0)
s = np.pi * (np.arange(0,  24.0 + 0.5, 0.5) / 24.0)
theta, phi = np.meshgrid(r, s, indexing="xy")

X = np.sin(phi) * np.cos(theta)
Y = np.sin(phi) * np.sin(theta)
Z = np.cos(phi)

# Save initial coordinates as 3xN
XYZ_ini = np.vstack([X.ravel(), Y.ravel(), Z.ravel()])

# Initial deformation gradient tensor (3x3 identity)
F = np.eye(3)

# Diagnostics
nadai_strain = 0.0
lodes_ratio  = 0.0
time         = 0.0

# -----------------------------
# PyVista scene setup
# -----------------------------
pv.set_plot_theme("document")

# StructuredGrid from the parametric surface
grid = pv.StructuredGrid()
# Note: pyvista expects coordinates of shape (ny, nx) for 2D structured grid
grid.points = np.c_[X.ravel(), Y.ravel(), Z.ravel()]
grid.dimensions = X.shape[1], X.shape[0], 1  # (nx, ny, nz=1)

pl = pv.Plotter(window_size=(1100, 800), notebook=False)
pl.open_gif("finite_strain_3D.gif")
mesh_actor = pl.add_mesh(grid, color=(0, 204/255, 153/255), opacity=0.3, show_edges=True)

pl.add_axes(line_width=2)
pl.show_grid()
txt_actor = pl.add_text("", position='upper_right', font_size=12)

pl.write_frame()

# Hold arrow actors so we can update them each frame
arrow_actors = []

# Helper to (re)draw principal strain axes and velocity vectors
def update_overlays(FS, VXr_tot, VYr_tot, VZr_tot, defmode, shearmode, nadai, lode, time, ps_rate, xy_rate, xz_rate, yz_rate):
    global arrow_actors
    # Remove previous arrows
    for act in arrow_actors:
        try:
            pl.remove_actor(act)
        except Exception:
            pass
    arrow_actors = []

    # Sort columns of FS by max |component| to keep colors consistent
    col_strength = np.max(np.abs(FS), axis=0)
    idx = np.argsort(col_strength)  # ascending
    FSsorted = FS[:, idx]
    psa_minor = FSsorted[:, 0]
    psa_imed  = FSsorted[:, 1]
    psa_major = FSsorted[:, 2]

    # Build arrows for principal axes (± directions)
    def add_axis(vec, color):
        a1 = pv.Arrow(start=(0, 0, 0), direction=vec, tip_length=0.15, tip_radius=0.02, shaft_radius=0.008)
        a2 = pv.Arrow(start=(0, 0, 0), direction=-vec, tip_length=0.15, tip_radius=0.02, shaft_radius=0.008)
        arrow_actors.append(pl.add_mesh(a1, color=color))
        arrow_actors.append(pl.add_mesh(a2, color=color))

    add_axis(psa_major, "red")
    add_axis(psa_minor, "blue")
    add_axis(psa_imed,  "green")

    # Velocity field arrows: sample every ~4th grid point
    step = 4
    Xs = X[::step, ::step]
    Ys = Y[::step, ::step]
    Zs = Z[::step, ::step]
    VXs = VXr_tot[::step, ::step]
    VYs = VYr_tot[::step, ::step]
    VZs = VZr_tot[::step, ::step]

    pts = np.c_[Xs.ravel(), Ys.ravel(), Zs.ravel()]
    vec = np.c_[VXs.ravel(), VYs.ravel(), VZs.ravel()]

    if pts.size > 0:
        arr_actor = pl.add_arrows(pts, vec, mag=0.1)
        arrow_actors.append(arr_actor)

    # Update text HUD
    txt_actor.SetText(
        3,
        f"Time: {time:.2f} units\n"
        f"eii rate = {ps_rate:.2f}, exy rate = {xy_rate:.2f}\n"
        f"exz rate = {xz_rate:.2f}, eyz rate = {yz_rate:.2f}\n"
        f"Deformation: {defmode}\n"
        f"Shearing: {shearmode}\n"
        f"Nadai strain es = {nadai:.2f}\n"
        fr"Lodes ratio nu = {lode:.3f}"
    )


# -----------------------------
# Time loop
# -----------------------------
for it in range(1, nt + 1):
    # Increment time
    time += dt

    # Activate shear at given steps (match MATLAB logic)
    if it > tsxy:
        xy_rate = 3.0
    if it > tsxz:
        xz_rate = -1.0
        xy_rate = 0.0

    shearmode = "active" if np.sum(np.abs([xy_rate, xz_rate, yz_rate])) != 0 else "none"

    # Deformation mode and principal pure-shear rates
    if plane_strain == 1:
        ps_rateX, ps_rateY, ps_rateZ = ps_rate, ps_rate, 0.0
        defmode = "plane strain"
    elif constriction == 1:
        ps_rateX, ps_rateY, ps_rateZ = -0.5 * ps_rate, 0.5 * ps_rate, ps_rate
        defmode = "constriction"
    elif flattening == 1:
        ps_rateX, ps_rateY, ps_rateZ = 0.5 * ps_rate, ps_rate, 0.5 * ps_rate
        defmode = "flattening"
    else:
        raise RuntimeError("Please define a deformation mode.")

    # Velocity fields (pure shear + simple shear)
    VXr_ps =  X * ps_rateX
    VYr_ps = -Y * ps_rateY
    VZr_ps =  Z * ps_rateZ

    VXr_xy =  Y * xy_rate
    VXr_xz =  Z * xz_rate

    VYr_xy = np.zeros_like(VXr_xy)
    VYr_yz =  Z * yz_rate

    VZr_xz = np.zeros_like(VXr_xy)
    VZr_yz = np.zeros_like(VXr_xy)

    VXr_tot = VXr_ps + VXr_xy + VXr_xz
    VYr_tot = VYr_xy + VYr_ps + VYr_yz
    VZr_tot = VZr_xz + VZr_yz + VZr_ps

    # Update surface nodes
    X = X + VXr_tot * dt
    Y = Y + VYr_tot * dt
    Z = Z + VZr_tot * dt

    # Incremental transformation and update deformation gradient
    D = np.array([
        [1 + dt * ps_rateX,  dt * xy_rate,       dt * xz_rate],
        [0.0,                1 - dt * ps_rateY,  dt * yz_rate],
        [0.0,                0.0,                1 + dt * ps_rateZ]
    ])
    F = D @ F

    # Apply deformation gradient to initial sphere (safety check)
    XYZ_strain_el = F @ XYZ_ini
    XFS = XYZ_strain_el[0, :].reshape(X.shape[0], X.shape[1])
    YFS = XYZ_strain_el[1, :].reshape(X.shape[0], X.shape[1])
    ZFS = XYZ_strain_el[2, :].reshape(X.shape[0], X.shape[1])

    errVF = np.max(np.abs(np.concatenate([
        (X.ravel() - XFS.ravel()),
        (Y.ravel() - YFS.ravel()),
        (Z.ravel() - ZFS.ravel())
    ])))
    # (errVF can be logged if you want to compare velocity vs tensor update)

    # Polar decomposition via Left Cauchy–Green B = F F^T
    B = F @ F.T
    # Symmetric -> eigh
    evals, evecs = np.linalg.eigh(B)  # ascending eigenvalues
    # Principal stretches are sqrt of evals (ensure nonnegative)
    evals = np.clip(evals, 0.0, None)
    stretches = np.sqrt(evals)
    VE = np.diag(stretches)
    FS = evecs @ VE  # scale eigenvectors by principal stretches

    # Nadai strain and Lode's ratio (using principal stretches)
    e = np.sort(stretches)[::-1]  # e1 >= e2 >= e3
    # guard against log(0)
    e = np.clip(e, 1e-12, None)
    nadai_strain = (1 / np.sqrt(3)) * np.sqrt(
        (np.log(e[0]) - np.log(e[1]))**2 +
        (np.log(e[0]) - np.log(e[2]))**2 +
        (np.log(e[2]) - np.log(e[0]))**2
    )
    lodes_ratio = (2 * np.log(e[1]) - np.log(e[0]) - np.log(e[2])) / (np.log(e[0]) - np.log(e[2]) + 1e-12)

    # Update mesh geometry
    points = np.c_[X.ravel(), Y.ravel(), Z.ravel()]
    #pl.update_coordinates(grid.points, render=False)
    grid_update = pv.StructuredGrid()
    grid_update.points = points
    grid_update.dimensions = X.shape[1], X.shape[0], 1  # (nx, ny, nz=1)
    grid.shallow_copy(grid_update)
   
    # Redraw overlays (principal axes + velocity field) and HUD text
    update_overlays(FS, VXr_tot, VYr_tot, VZr_tot, defmode, shearmode, nadai_strain, lodes_ratio, time, ps_rate, xy_rate, xz_rate, yz_rate)

    # Write and Render this frame
    pl.write_frame()

pl.close()