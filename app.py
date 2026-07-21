# =============================================================================
# Facility Location: a Streamlit tutorial app.
#
# Uncapacitated Facility Location (UFL): given a set of candidate facility
# sites and a set of customers, decide which facilities to open and how to
# route customer demand so that total cost is minimized:
#
#     min  sum_i  f_i * y_i        (fixed cost of opening facility i)
#        + sum_ij c_ij * d_j * x_ij (transport: distance * demand * fraction)
#
#     s.t. sum_i  x_ij = 1           (each customer fully served)
#          x_ij  <= y_i              (only ship from open facilities)
#          y_i   in {0, 1}, x_ij >= 0
#
# The "uncapacitated" part means open facilities have no upper limit on
# how much demand they can serve. The classic Knapsack-style twist for
# this app: the user proposes their own set of opened facilities (toggle
# buttons) and the app shows their total cost alongside the optimum.
#
# Library roadmap:
#   - streamlit : UI framework. Each interaction reruns this script
#                  top-to-bottom; persistent state lives in `st.session_state`.
#   - pyomo     : algebraic modeling: sets, params, vars, objective,
#                  constraints. Continuous + binary variables.
#   - Gurobi    : the MILP solver, called via Pyomo's appsi_gurobi interface.
#                  Ships as a pip wheel (`gurobipy`); needs a Gurobi license
#                  (WLS via Fly secrets in production, or a local license file
#                  pointed to by GRB_LICENSE_FILE).
#   - pandas    : DataFrames for the editable site/customer tables.
#   - altair    : the 2D map plot (sites + customers + assignment lines).
#
# File roadmap:
#   1. Solver      : model definition, Gurobi log capture, top-level solve.
#   2. State       : session_state init / reset.
#   3. Utilities   : random-scenario generation, distance, user cost.
#   4. LaTeX       : General + Instance formulation rendering.
#   5. CSS         : small style tweaks for the toggle button grid.
#   6. Tabs        : render_optimizer / render_data / render_formulation /
#                     render_logs.
#   7. Main        : page config, sidebar, tab assembly.
# =============================================================================

import base64
import math
import os
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import pyomo.environ as pyo
import streamlit as st
from pyomo.common.errors import ApplicationError
from pyomo.common.tee import capture_output


def _materialize_gurobi_license():
    """Production license shim. Fly secrets surface as environment
    variables, but gurobipy wants a license FILE: so if the three WLS
    values are present and no license file is configured, write one to
    the home directory and point GRB_LICENSE_FILE at it. Local dev is
    untouched: there GRB_LICENSE_FILE already points at a file on disk.
    The values never enter the repo or image: only Fly's secret store
    and the container's private filesystem."""
    if os.environ.get("GRB_LICENSE_FILE"):
        return
    access = os.environ.get("GRB_WLSACCESSID")
    secret = os.environ.get("GRB_WLSSECRET")
    license_id = os.environ.get("GRB_LICENSEID")
    if not (access and secret and license_id):
        return
    lic_path = Path.home() / "gurobi.lic"
    if not lic_path.exists():
        lic_path.write_text(
            f"WLSACCESSID={access}\n"
            f"WLSSECRET={secret}\n"
            f"LICENSEID={license_id}\n",
            encoding="utf-8",
        )
    os.environ["GRB_LICENSE_FILE"] = str(lic_path)


_materialize_gurobi_license()


# ---------- Constants ----------

MAX_SITES = 25
MAX_CUSTOMERS = 60

# Color palette: red for the user's solution, green for the optimal,
# purple where they overlap (facility opened in both).
COLOR_USER = "#dc2626"
COLOR_OPT = "#16a34a"
COLOR_BOTH = "#7c3aed"
COLOR_CANDIDATE = "#9ca3af"  # gray X for closed candidate sites
COLOR_CUSTOMER = "#1f2937"   # near-black dot for customers


# ---------- Solver ----------

def build_model(sites_df, customers_df, fixed_mult=1.0, transport_mult=1.0):
    """Build the Pyomo UFL model. sites_df has columns x,y,fixed_cost; customers_df
    has columns x,y,demand. Returns the Pyomo ConcreteModel."""
    n_sites = len(sites_df)
    n_customers = len(customers_df)

    m = pyo.ConcreteModel()
    m.I = pyo.RangeSet(0, n_sites - 1)
    m.J = pyo.RangeSet(0, n_customers - 1)

    # Parameters
    m.f = pyo.Param(m.I, initialize={
        i: float(sites_df.iloc[i]["fixed_cost"]) * fixed_mult for i in range(n_sites)
    })
    m.d = pyo.Param(m.J, initialize={
        j: float(customers_df.iloc[j]["demand"]) for j in range(n_customers)
    })
    # c_ij = Euclidean distance from site i to customer j, scaled by transport_mult
    def _cost(i, j):
        dx = sites_df.iloc[i]["x"] - customers_df.iloc[j]["x"]
        dy = sites_df.iloc[i]["y"] - customers_df.iloc[j]["y"]
        return float(math.hypot(dx, dy)) * transport_mult
    m.c = pyo.Param(m.I, m.J, initialize={
        (i, j): _cost(i, j) for i in range(n_sites) for j in range(n_customers)
    })

    # Variables
    m.y = pyo.Var(m.I, within=pyo.Binary)
    m.x = pyo.Var(m.I, m.J, bounds=(0.0, 1.0))

    # Objective: fixed cost + transport cost (multi-source UFL)
    m.obj = pyo.Objective(
        expr=sum(m.f[i] * m.y[i] for i in m.I)
        + sum(m.c[i, j] * m.d[j] * m.x[i, j] for i in m.I for j in m.J),
        sense=pyo.minimize,
    )

    # Each customer's demand is fully assigned across (one or more) facilities
    m.assign = pyo.Constraint(m.J, rule=lambda m, j: sum(m.x[i, j] for i in m.I) == 1)

    # Can only ship from open facilities
    m.linking = pyo.Constraint(m.I, m.J, rule=lambda m, i, j: m.x[i, j] <= m.y[i])

    return m


def _solve_capturing(m):
    """Run Gurobi and return (results, log_text)."""
    log_text = ""
    try:
        with capture_output(capture_fd=True) as buf:
            solver = pyo.SolverFactory("appsi_gurobi")
            results = solver.solve(m, tee=True)
        log_text = buf.getvalue()
    except TypeError:
        with capture_output() as buf:
            solver = pyo.SolverFactory("appsi_gurobi")
            results = solver.solve(m, tee=True)
        log_text = buf.getvalue()
    # Scrub license-identifying lines from the captured log before it
    # reaches the public Logs tab: Gurobi's WLS banner prints the
    # license ID and registrant ("WLS license NNNNNNN - registered to
    # ..."). Substring match keeps this robust to wording shifts across
    # Gurobi versions.
    log_text = "\n".join(
        ln for ln in log_text.splitlines()
        if not any(
            marker in ln.lower()
            for marker in ("wls", "registered to", "academic license")
        )
    )
    return results, log_text


def solve(sites_df, customers_df, fixed_mult=1.0, transport_mult=1.0):
    """Top-level solve. Returns a dict with optimal y_i, x_ij, total cost, and log."""
    if len(sites_df) == 0 or len(customers_df) == 0:
        return {"status": "no_data", "y": {}, "x": {}, "cost": None, "log": ""}

    m = build_model(sites_df, customers_df, fixed_mult, transport_mult)

    try:
        results, log = _solve_capturing(m)
    except ApplicationError as e:
        return {
            "status": "solver_missing",
            "message": (
                f"Gurobi solver not available. Run `pip install gurobipy` "
                f"and make sure a license is configured. ({e})"
            ),
            "y": {}, "x": {}, "cost": None, "log": "",
        }

    tc = str(results.solver.termination_condition)
    if tc != "optimal":
        return {"status": tc, "y": {}, "x": {}, "cost": None, "log": log}

    y = {i: int(round(pyo.value(m.y[i]))) for i in m.I}
    x = {
        (i, j): float(pyo.value(m.x[i, j]))
        for i in m.I for j in m.J
        if pyo.value(m.x[i, j]) > 1e-6
    }
    cost = float(pyo.value(m.obj))

    return {"status": "optimal", "y": y, "x": x, "cost": cost, "log": log}


# ---------- Random scenario generation ----------

def make_scenario(n_sites, n_customers, seed):
    """Generate a random scenario: sites and customers placed uniformly on a
    2x1 rectangle (matching the map's aspect ratio), fixed costs vary
    modestly, demands uniform."""
    rng = np.random.default_rng(seed)
    sites = pd.DataFrame({
        "x": rng.uniform(0.05, 1.95, n_sites).round(3),
        "y": rng.uniform(0.05, 0.95, n_sites).round(3),
        # Fixed cost varies in [0.4, 1.2]: meaningful spread, doesn't dominate
        "fixed_cost": rng.uniform(0.4, 1.2, n_sites).round(3),
    })
    customers = pd.DataFrame({
        "x": rng.uniform(0.0, 2.0, n_customers).round(3),
        "y": rng.uniform(0.0, 1.0, n_customers).round(3),
        # Demand in [0.5, 1.5]: modest spread, no zero demand
        "demand": rng.uniform(0.5, 1.5, n_customers).round(3),
    })
    return sites, customers


def compute_user_cost(sites_df, customers_df, opened, fixed_mult=1.0, transport_mult=1.0):
    """Given a set of opened facility indices, compute the user's cost by
    auto-assigning each customer to its nearest open facility (single-source
    greedy). Returns (cost, assignment dict {customer_j: facility_i})."""
    if not opened:
        return None, {}
    fixed = sum(
        float(sites_df.iloc[i]["fixed_cost"]) * fixed_mult for i in opened
    )
    transport = 0.0
    assign = {}
    for j in range(len(customers_df)):
        cx, cy = customers_df.iloc[j]["x"], customers_df.iloc[j]["y"]
        d = customers_df.iloc[j]["demand"]
        best_i, best_dist = None, float("inf")
        for i in opened:
            sx, sy = sites_df.iloc[i]["x"], sites_df.iloc[i]["y"]
            dist = math.hypot(sx - cx, sy - cy)
            if dist < best_dist:
                best_dist = dist
                best_i = i
        assign[j] = best_i
        transport += best_dist * float(d) * transport_mult
    return fixed + transport, assign


# ---------- State ----------

DEFAULT_N_SITES = 12
DEFAULT_N_CUSTOMERS = 40
DEFAULT_SEED = 0
DEFAULT_FIXED_MULT = 1.0
DEFAULT_TRANSPORT_MULT = 1.0


def init_state():
    if "scenario_initialized" in st.session_state:
        return
    sites, customers = make_scenario(DEFAULT_N_SITES, DEFAULT_N_CUSTOMERS, DEFAULT_SEED)
    st.session_state.sites = sites
    st.session_state.customers = customers
    st.session_state.user_opened = set()      # empty: all closed
    st.session_state.optimal = None           # filled after Solve click
    st.session_state.map_ver = 0              # bumps per scenario: fresh chart widget
    st.session_state.scenario_initialized = True


def reset_scenario(n_sites, n_customers, seed):
    """Re-roll scenario; reset user candidate to all-closed."""
    sites, customers = make_scenario(n_sites, n_customers, seed)
    st.session_state.sites = sites
    st.session_state.customers = customers
    st.session_state.user_opened = set()
    st.session_state.optimal = None
    # Drop the map-selection tracker and version the chart's widget key: the
    # browser-side chart retains its click-selection across scenario changes
    # under a fixed key, and that stale selection would replay old toggles
    # into the fresh scenario on a later rebuild (e.g. after Solve).
    st.session_state.pop("map_sel_prev", None)
    st.session_state.map_ver = st.session_state.get("map_ver", 0) + 1


def set_at_optimum():
    """Copy the optimal opened-facility set into the user candidate."""
    opt = st.session_state.optimal
    if opt and opt.get("status") == "optimal":
        st.session_state.user_opened = {i for i, yi in opt["y"].items() if yi == 1}


def toggle_facility(i):
    """Toggle facility i's open/closed state in the user candidate."""
    if i in st.session_state.user_opened:
        st.session_state.user_opened.remove(i)
    else:
        st.session_state.user_opened.add(i)


# ---------- LaTeX ----------

def render_general():
    st.markdown("**Sets**")
    st.latex(r"""
        \begin{aligned}
            \mathcal{I} &= \{1, \dots, |\mathcal{I}|\}\quad \text{candidate sites} \\
            \mathcal{J} &= \{1, \dots, |\mathcal{J}|\}\quad \text{customers}
        \end{aligned}
    """)
    st.markdown("**Parameters**")
    st.latex(r"""
        \begin{aligned}
            f_i &\quad\text{fixed cost to open facility } i \\
            d_j &\quad\text{demand at customer } j \\
            c_{ij} &\quad\text{unit transport cost between } i \text{ and } j
        \end{aligned}
    """)
    st.markdown("**Variables**")
    st.latex(r"""
        \begin{aligned}
            y_i \in \{0, 1\} &\quad\text{open facility } i \\
            x_{ij} \in [0, 1] &\quad\text{fraction of } j\text{'s demand from } i
        \end{aligned}
    """)
    st.markdown("**Objective and constraints**")
    st.latex(r"""
        \begin{aligned}
            \min \quad & \sum_{i\in\mathcal{I}} f_i\, y_i
                       + \sum_{i\in\mathcal{I}} \sum_{j\in\mathcal{J}} c_{ij}\, d_j\, x_{ij} \\
            \text{s.t.} \quad & \sum_{i\in\mathcal{I}} x_{ij} = 1
                                \quad \forall j \in \mathcal{J} \\
                              & x_{ij} \leq y_i
                                \quad \forall i \in \mathcal{I},\, j \in \mathcal{J} \\
                              & y_i \in \{0, 1\},\ x_{ij} \in [0, 1]
        \end{aligned}
    """)
    st.caption(
        "Note: this is the **multi-source** UFL: a customer could split its "
        "demand across facilities, but with no capacity limits a split is "
        "never beneficial, so optimal assignments are single-source (up to "
        "exact-distance ties)."
    )

    st.markdown("**References**")
    st.markdown(
        "[1] A. A. Kuehn and M. J. Hamburger, \"A Heuristic Program for "
        "Locating Warehouses,\" *Management Science*, vol. 9, no. 4, "
        "pp. 643–666, 1963. "
        "[INFORMS](https://pubsonline.informs.org/doi/10.1287/mnsc.9.4.643)\n\n"
        "[2] D. Erlenkotter, \"A Dual-Based Procedure for Uncapacitated "
        "Facility Location,\" *Operations Research*, vol. 26, no. 6, "
        "pp. 992–1009, 1978. "
        "[INFORMS](https://pubsonline.informs.org/doi/10.1287/opre.26.6.992)\n\n"
        "[3] G. Cornuéjols, G. L. Nemhauser, and L. A. Wolsey, \"The "
        "Uncapacitated Facility Location Problem,\" in *Discrete Location "
        "Theory*, P. B. Mirchandani and R. L. Francis, Eds. New York: "
        "Wiley, 1990, pp. 119–171.\n\n"
        "[4] M. L. Bynum, G. A. Hackebeil, W. E. Hart, C. D. Laird, "
        "B. L. Nicholson, J. D. Siirola, J.-P. Watson, and D. L. Woodruff, "
        "*Pyomo: Optimization Modeling in Python*, 3rd ed. "
        "Cham: Springer, 2021. "
        "[Springer](https://link.springer.com/book/10.1007/978-3-030-68928-5)"
    )


def render_instance():
    """Substitute the current instance's numbers into the formulation. The
    transport term has |I|*|J| coefficients, so the objective's fixed-cost
    part is expanded in full while the transport part and the repeated
    constraint families are shown once with a representative customer."""
    sites = st.session_state.sites
    customers = st.session_state.customers
    fixed_mult = st.session_state.get("fixed_mult", 1.0)
    transport_mult = st.session_state.get("transport_mult", 1.0)
    n_sites = len(sites)
    n_customers = len(customers)
    st.markdown("**This instance**")
    st.markdown(
        f"- $|\\mathcal{{I}}| = {n_sites}$ candidate sites\n"
        f"- $|\\mathcal{{J}}| = {n_customers}$ customers\n"
        f"- Decision variables: {n_sites} binary $y_i$ + "
        f"{n_sites * n_customers} continuous $x_{{ij}}$\n"
        f"- Constraints: {n_customers} demand + "
        f"{n_sites * n_customers} linking\n"
        f"- Cost multipliers applied: fixed × {fixed_mult:g}, "
        f"transport × {transport_mult:g}"
    )

    # Objective: fixed-cost terms expanded numerically; transport term kept
    # as the double sum (its coefficients are the c_ij * d_j products).
    terms = [
        f"{float(sites.iloc[i]['fixed_cost']) * fixed_mult:.2f}\\,y_{{{i + 1}}}"
        for i in range(n_sites)
    ]
    # Break the expanded fixed-cost sum across aligned rows so wide
    # instances don't run off screen.
    _PER_ROW = 8
    chunks = [terms[k:k + _PER_ROW] for k in range(0, len(terms), _PER_ROW)]
    rows = []
    for r, chunk in enumerate(chunks):
        lead = r"\min \; & " if r == 0 else r"& {} + "
        rows.append(lead + " + ".join(chunk) + r" \\")
    rows.append(
        r"& {} + \sum_{i=1}^{" + str(n_sites) + r"} \sum_{j=1}^{"
        + str(n_customers) + r"} c_{ij} d_j \, x_{ij}"
    )
    st.markdown("**Objective**")
    st.latex(r"\small \begin{aligned}" + "\n".join(rows) + r"\end{aligned}")

    # One representative customer: its demand constraint and its numeric
    # transport coefficients across all sites.
    coeffs = " & ".join(
        f"{float(math.hypot(sites.iloc[i]['x'] - customers.iloc[0]['x'], sites.iloc[i]['y'] - customers.iloc[0]['y'])) * transport_mult * float(customers.iloc[0]['demand']):.2f}"
        for i in range(n_sites)
    )
    st.markdown(f"**Customer 1 of {n_customers}** (each customer contributes "
                "one demand constraint and one column of transport "
                "coefficients like these)")
    st.latex(
        r"\small x_{1,1} + x_{2,1} + \dots + x_{" + str(n_sites)
        + r",1} = 1"
    )
    st.latex(
        r"\small (c_{i,1} d_1)_{i=1}^{" + str(n_sites) + r"} = ("
        + coeffs.replace(" & ", r",\; ") + r")"
    )

    st.markdown("**Linking and domains**")
    st.latex(
        r"\small x_{ij} \le y_i \quad \forall i,j, \qquad "
        r"y_i \in \{0,1\}, \quad x_{ij} \ge 0"
    )


# ---------- CSS ----------

CSS = """
<style>
/* Top padding shared across the template family: clears the sticky
   header without clipping the title. See griffith-pse-app-template. */
.block-container,
[data-testid="stMainBlockContainer"] {
    padding-top: 2.5rem !important;
}
/* Pull the map chart up toward its legend caption. */
[class*="st-key-map_legend"] {
    margin-bottom: -1rem;
}
/* Home-link logo at the very top of the sidebar, in normal document flow
   so it scrolls with the sidebar content (not pinned to the viewport). */
.home-logo-corner {
    display: inline-block;   /* shrink to the icon so only the G is clickable */
    margin: 0 0 0.75rem;
}
.home-logo-corner img {
    width: 32px; height: 32px; border-radius: 4px; display: block;
}
/* Hide Streamlit's sticky sidebar header (which hosts the «« collapse
   arrow) so the home-logo sits at the very top of the sidebar with no
   chrome above it. Trade-off: the user can no longer collapse the sidebar
   via the button. The sidebar is the app's control panel and is meant
   to stay visible, so this is fine for this app. */
[data-testid="stSidebarHeader"] {
    display: none !important;
}
[data-testid="stSidebarUserContent"] {
    padding-top: 0.5rem !important;
}
</style>
"""


# ---------- Plot ----------

def build_map(sites, customers, user_opened, optimal, user_assign):
    """Build the ghosted-overlay 2D map showing both solutions."""
    n_sites = len(sites)
    n_customers = len(customers)

    # Site states: which color to use for each facility
    opt_open = set()
    if optimal and optimal.get("status") == "optimal":
        opt_open = {i for i, yi in optimal["y"].items() if yi == 1}

    site_records = []
    for i in range(n_sites):
        in_user = i in user_opened
        in_opt = i in opt_open
        if in_user and in_opt:
            kind, color = "Open in both", COLOR_BOTH
        elif in_user:
            kind, color = "Open in your solution", COLOR_USER
        elif in_opt:
            kind, color = "Open in optimal", COLOR_OPT
        else:
            kind, color = "Closed (candidate)", COLOR_CANDIDATE
        site_records.append({
            "i": i,
            "site": i + 1,  # 1-based display label; "i" stays the internal id
            "x": float(sites.iloc[i]["x"]),
            "y": float(sites.iloc[i]["y"]),
            "kind": kind,
            "color": color,
            "fixed_cost": float(sites.iloc[i]["fixed_cost"]),
        })
    sites_plot = pd.DataFrame(site_records)

    customers_plot = pd.DataFrame({
        "j": list(range(n_customers)),
        "customer": [j + 1 for j in range(n_customers)],
        "x": customers["x"].astype(float).values,
        "y": customers["y"].astype(float).values,
        "demand": customers["demand"].astype(float).values,
    })

    # Routes present in BOTH solutions draw once in purple (matching the
    # "open in both" factory color) instead of translucent red-over-green,
    # which blended into an unintentional brown.
    opt_pairs = set()
    if optimal and optimal.get("status") == "optimal":
        opt_pairs = {(i, j) for (i, j), frac in optimal["x"].items()
                     if frac > 1e-6}
    shared_pairs = {(i, j) for j, i in user_assign.items()
                    if i is not None and (i, j) in opt_pairs}

    # User assignment lines (red, single-source from each customer to its
    # nearest user-open facility). Only drawn if user has any open facilities.
    user_lines_records = []
    for j, i in user_assign.items():
        if i is None or (i, j) in shared_pairs:
            continue
        user_lines_records.append({
            "j": j, "i": i,
            "x": float(customers.iloc[j]["x"]),
            "y": float(customers.iloc[j]["y"]),
            "x2": float(sites.iloc[i]["x"]),
            "y2": float(sites.iloc[i]["y"]),
        })
    user_lines_df = pd.DataFrame(user_lines_records) if user_lines_records else \
        pd.DataFrame(columns=["j", "i", "x", "y", "x2", "y2"])

    # Optimal assignment lines (green, opacity = x_ij)
    opt_lines_records = []
    shared_lines_records = []
    if optimal and optimal.get("status") == "optimal":
        for (i, j), frac in optimal["x"].items():
            if frac < 1e-6:
                continue
            rec = {
                "j": j, "i": i, "frac": frac,
                "x": float(customers.iloc[j]["x"]),
                "y": float(customers.iloc[j]["y"]),
                "x2": float(sites.iloc[i]["x"]),
                "y2": float(sites.iloc[i]["y"]),
            }
            (shared_lines_records if (i, j) in shared_pairs
             else opt_lines_records).append(rec)
    opt_lines_df = pd.DataFrame(opt_lines_records) if opt_lines_records else \
        pd.DataFrame(columns=["j", "i", "frac", "x", "y", "x2", "y2"])
    shared_lines_df = pd.DataFrame(shared_lines_records) if shared_lines_records \
        else pd.DataFrame(columns=["j", "i", "frac", "x", "y", "x2", "y2"])

    # ---- Layers ----
    base_axis = alt.Axis(grid=False, labels=False, ticks=False, domain=False, title=None)
    # Padded domains with the x padding doubled, so the x span (2.2) is
    # exactly twice the y span (1.1): matching the 1200x600 canvas keeps
    # pixels-per-unit identical on both axes.
    x_dom = [-0.1, 2.1]
    y_dom = [-0.05, 1.05]

    # Stylized "service territory" backdrop: a soft sage region with a white
    # street-like grid. Purely decorative: the coordinates are synthetic, so
    # this deliberately avoids looking like real geography.
    bg_rect = (
        alt.Chart(pd.DataFrame({"x": [0.0], "y": [0.0],
                                "x2": [2.0], "y2": [1.0]}))
        .mark_rect(color="#eceef0", stroke="#d3d7db", strokeWidth=1.5,
                   cornerRadius=6)
        .encode(
            x=alt.X("x:Q", scale=alt.Scale(domain=x_dom), axis=base_axis),
            y=alt.Y("y:Q", scale=alt.Scale(domain=y_dom), axis=base_axis),
            x2="x2:Q", y2="y2:Q",
        )
    )
    _grid_step = 0.125
    _v = [round(_grid_step * k, 3) for k in range(1, int(2.0 / _grid_step))]
    _h = [round(_grid_step * k, 3) for k in range(1, int(1.0 / _grid_step))]
    grid_df = pd.DataFrame(
        [{"x": gx, "y": 0.0, "x2": gx, "y2": 1.0} for gx in _v]
        + [{"x": 0.0, "y": gy, "x2": 2.0, "y2": gy} for gy in _h]
    )
    bg_grid = (
        alt.Chart(grid_df)
        .mark_rule(color="#ffffff", strokeWidth=1.2, opacity=0.8)
        .encode(
            x=alt.X("x:Q", scale=alt.Scale(domain=x_dom), axis=base_axis),
            y=alt.Y("y:Q", scale=alt.Scale(domain=y_dom), axis=base_axis),
            x2="x2:Q", y2="y2:Q",
        )
    )

    # Optimal lines: very thin, semi-transparent green
    if not opt_lines_df.empty:
        opt_lines = (
            alt.Chart(opt_lines_df)
            .mark_rule(color=COLOR_OPT, opacity=0.4, strokeWidth=1)
            .encode(
                x=alt.X("x:Q", scale=alt.Scale(domain=x_dom), axis=base_axis),
                y=alt.Y("y:Q", scale=alt.Scale(domain=y_dom), axis=base_axis),
                x2="x2:Q", y2="y2:Q",
            )
        )
    else:
        opt_lines = None

    # User lines: thin, semi-transparent red
    if not user_lines_df.empty:
        user_lines = (
            alt.Chart(user_lines_df)
            .mark_rule(color=COLOR_USER, opacity=0.4, strokeWidth=1.2)
            .encode(
                x=alt.X("x:Q", scale=alt.Scale(domain=x_dom), axis=base_axis),
                y=alt.Y("y:Q", scale=alt.Scale(domain=y_dom), axis=base_axis),
                x2="x2:Q", y2="y2:Q",
            )
        )
    else:
        user_lines = None

    # Shared routes: in both solutions, drawn once in the "both" purple
    if not shared_lines_df.empty:
        shared_lines = (
            alt.Chart(shared_lines_df)
            .mark_rule(color=COLOR_BOTH, opacity=0.5, strokeWidth=1.2)
            .encode(
                x=alt.X("x:Q", scale=alt.Scale(domain=x_dom), axis=base_axis),
                y=alt.Y("y:Q", scale=alt.Scale(domain=y_dom), axis=base_axis),
                x2="x2:Q", y2="y2:Q",
            )
        )
    else:
        shared_lines = None

    # Customers: dark dots, sized proportionally to demand. Range chosen so
    # the smallest demand is still visible and the largest doesn't dwarf
    # the facility markers.
    customers_layer = (
        alt.Chart(customers_plot)
        .mark_circle(color=COLOR_CUSTOMER, opacity=0.7)
        .encode(
            x=alt.X("x:Q", scale=alt.Scale(domain=x_dom), axis=base_axis),
            y=alt.Y("y:Q", scale=alt.Scale(domain=y_dom), axis=base_axis),
            size=alt.Size("demand:Q",
                          scale=alt.Scale(range=[20, 180]),
                          legend=None),
            tooltip=[alt.Tooltip("customer:O", title="customer"),
                     alt.Tooltip("demand:Q", format=".2f"),
                     alt.Tooltip("x:Q", format=".3f"),
                     alt.Tooltip("y:Q", format=".3f")],
        )
    )

    # Sites: filled square if open (in either or both), gray X if closed.
    # Selection param turns site marks into clickable toggles. Initialized
    # with the current user_opened set so the chart stays in sync after
    # button-grid clicks. Read back via st.altair_chart on_select="rerun".
    site_select = alt.selection_point(
        name="site_select",
        on="click",
        fields=["i"],
        toggle="true",
        value=[{"i": int(i)} for i in sorted(user_opened)],
    )

    closed_sites = sites_plot[sites_plot["kind"] == "Closed (candidate)"]
    open_sites = sites_plot[sites_plot["kind"] != "Closed (candidate)"]

    # Marker size encodes the site's fixed open cost. One shared domain over
    # all sites so open squares and closed crosses scale consistently; the
    # square range sits higher so open sites stay visually dominant.
    fc = sites_plot["fixed_cost"]
    fc_domain = [float(fc.min()), float(fc.max())]

    closed_layer = (
        alt.Chart(closed_sites)
        .mark_point(color=COLOR_CANDIDATE, shape="cross", opacity=0.6,
                    strokeWidth=2, cursor="pointer")
        .encode(
            x=alt.X("x:Q", scale=alt.Scale(domain=x_dom), axis=base_axis),
            y=alt.Y("y:Q", scale=alt.Scale(domain=y_dom), axis=base_axis),
            size=alt.Size("fixed_cost:Q",
                          scale=alt.Scale(domain=fc_domain, range=[60, 260]),
                          legend=None),
            tooltip=[alt.Tooltip("site:O", title="site"),
                     alt.Tooltip("fixed_cost:Q", format=".2f"),
                     alt.Tooltip("x:Q", format=".3f"),
                     alt.Tooltip("y:Q", format=".3f"),
                     alt.Tooltip("kind:N")],
        )
        .add_params(site_select)
    )

    # Open sites draw as a factory silhouette (custom SVG path: sawtooth roof
    # + chimney, y-down coordinates centered on the site), colored by kind
    # like the squares it replaces.
    factory_shape = (
        "M -1.9 1.9 L -1.9 -0.3 L -1.0 -1.0 L -1.0 -0.3 L -0.1 -1.0 "
        "L -0.1 -0.3 L 0.8 -1.0 L 0.8 -0.3 L 1.9 -0.3 L 1.9 1.9 Z "
        "M 1.1 -0.3 L 1.16 -1.9 L 1.64 -1.9 L 1.7 -0.3 Z"
    )
    open_layer = (
        alt.Chart(open_sites)
        .mark_point(shape=factory_shape, filled=True, opacity=0.92,
                    stroke="white", strokeWidth=1, cursor="pointer")
        .encode(
            x=alt.X("x:Q", scale=alt.Scale(domain=x_dom), axis=base_axis),
            y=alt.Y("y:Q", scale=alt.Scale(domain=y_dom), axis=base_axis),
            size=alt.Size("fixed_cost:Q",
                          scale=alt.Scale(domain=fc_domain, range=[500, 1600]),
                          legend=None),
            color=alt.Color("color:N", scale=None, legend=None),
            tooltip=[alt.Tooltip("site:O", title="site"),
                     alt.Tooltip("fixed_cost:Q", format=".2f"),
                     alt.Tooltip("x:Q", format=".3f"),
                     alt.Tooltip("y:Q", format=".3f"),
                     alt.Tooltip("kind:N")],
        )
        .add_params(site_select)
    )

    # Draw order: lines at the bottom, then customers, then sites (closed
    # crosses and open factories) so a site overlapping a customer stays on
    # top and clickable.
    layers = [l for l in [bg_rect, bg_grid, opt_lines, user_lines,
                          shared_lines, customers_layer, closed_layer,
                          open_layer]
              if l is not None]
    chart = alt.layer(*layers).properties(width=1200, height=600).configure_view(strokeOpacity=0)
    return chart


# ---------- Tabs ----------

def render_optimizer():
    sites = st.session_state.sites
    customers = st.session_state.customers
    user_opened = st.session_state.user_opened
    optimal = st.session_state.optimal

    fixed_mult = st.session_state.fixed_mult
    transport_mult = st.session_state.transport_mult

    # User cost (live, computed every rerun based on currently-open facilities)
    user_cost, user_assign = compute_user_cost(
        sites, customers, user_opened, fixed_mult, transport_mult
    )

    # Headline row: the two cost metrics side by side on the left, and the
    # getting-started hint in a fixed column on the right so the content
    # below doesn't shift when the hint appears or disappears.
    metric_l, metric_r, hint_col, _spacer = st.columns(
        [1, 1, 4, 2], vertical_alignment="center"
    )

    def _colored_metric(col, label, value, color):
        col.markdown(
            f"<div style='font-size: 0.85rem; color: #6b7280;'>{label}</div>"
            f"<div style='font-size: 1.9rem; font-weight: 600; "
            f"color: {color}; line-height: 1.2;'>{value}</div>",
            unsafe_allow_html=True,
        )

    solved = bool(optimal and optimal.get("status") == "optimal")
    # Your cost: neutral until an optimum exists to compare against, then
    # green on a match and red otherwise.
    if user_cost is None:
        _colored_metric(metric_l, "Your cost", "-", "inherit")
    elif not solved:
        _colored_metric(metric_l, "Your cost", f"{user_cost:.2f}", "inherit")
    else:
        matches = abs(user_cost - optimal["cost"]) <= 1e-6 * max(1.0, optimal["cost"])
        _colored_metric(metric_l, "Your cost", f"{user_cost:.2f}",
                        COLOR_OPT if matches else COLOR_USER)
    if solved:
        _colored_metric(metric_r, "Optimal cost", f"{optimal['cost']:.2f}",
                        COLOR_OPT)
    else:
        _colored_metric(metric_r, "Optimal cost", "-", "inherit")

    # Solve-state badge: a small fixed pill instead of an info box that
    # appears and disappears (which shifted the layout).
    if solved:
        hint_col.markdown(
            "<span style='display: inline-block; padding: 0.25rem 0.75rem; "
            f"border-radius: 999px; background: {COLOR_OPT}1a; "
            f"color: {COLOR_OPT}; font-weight: 600; font-size: 0.9rem;'>"
            "Solved</span>",
            unsafe_allow_html=True,
        )
    else:
        hint_col.markdown(
            "<span style='display: inline-block; padding: 0.25rem 0.75rem; "
            "border-radius: 999px; background: #6b72801a; color: #6b7280; "
            "font-weight: 600; font-size: 0.9rem;' "
            "title='Toggle sites on the map, then click Solve Optimization "
            "in the sidebar.'>Not solved</span>",
            unsafe_allow_html=True,
        )

    # Map. Click any site marker (× or ■) to toggle it open/closed -
    # clicks propagate back via on_select="rerun". Marker size encodes the
    # site's fixed open cost.
    # Keyed container so CSS can pull the chart below closer to the legend.
    # Factory glyphs are inline SVGs of the same path the map uses, filled
    # with the matching solution color.
    def _factory_svg(color):
        return (
            f"<svg viewBox='-2.1 -2.1 4.2 4.2' width='14' height='14' "
            f"style='vertical-align: -0.12em;'><path fill='{color}' "
            f"d='M -1.9 1.9 L -1.9 -0.3 L -1.0 -1.0 L -1.0 -0.3 L -0.1 -1.0 "
            f"L -0.1 -0.3 L 0.8 -1.0 L 0.8 -0.3 L 1.9 -0.3 L 1.9 1.9 Z "
            f"M 1.1 -0.3 L 1.16 -1.9 L 1.64 -1.9 L 1.7 -0.3 Z'/></svg>"
        )

    st.container(key="map_legend").caption(
        f"{_factory_svg(COLOR_USER)} = open in your solution &nbsp;&nbsp; "
        f"{_factory_svg(COLOR_OPT)} = open in optimal &nbsp;&nbsp; "
        f"{_factory_svg(COLOR_BOTH)} = open in both &nbsp;&nbsp; "
        f"<svg viewBox='-6.5 -6.5 13 13' width='13' height='13' "
        f"style='vertical-align: -0.1em;'>"
        f"<path stroke='{COLOR_CANDIDATE}' stroke-width='1.6' fill='none' "
        f"d='M -1.7 -5 H 1.7 V -1.7 H 5 V 1.7 H 1.7 V 5 H -1.7 V 1.7 "
        f"H -5 V -1.7 H -1.7 Z'/></svg> "
        f"= closed candidate (size = cost) &nbsp;&nbsp; "
        f"<span style='color: {COLOR_CUSTOMER};'>●</span> "
        f"= customer (size = demand)",
        unsafe_allow_html=True,
    )
    chart = build_map(sites, customers, user_opened, optimal, user_assign)
    chart_state = st.altair_chart(
        chart, use_container_width=False, on_select="rerun",
        key=f"map_chart_{st.session_state.get('map_ver', 0)}"
    )
    if chart_state and isinstance(chart_state, dict):
        sel = chart_state.get("selection", {}).get("site_select", []) or []
        selected_ids = {int(p["i"]) for p in sel if isinstance(p, dict) and "i" in p}
        # The map's selection state persists across reruns and can go stale
        # (e.g. Set at Optimum writes user_opened directly), so it can't be
        # treated as the source of truth. Instead, diff it against its own
        # previous value and apply only the sites the user actually clicked.
        prev = st.session_state.get("map_sel_prev")
        if prev is None:
            prev = selected_ids
        # Guard against stale ids (e.g. a selection remembered from a larger
        # scenario): only ids that exist as sites can toggle.
        clicked = {i for i in selected_ids ^ prev if 0 <= i < len(sites)}
        st.session_state.map_sel_prev = selected_ids
        if clicked:
            for i in clicked:
                toggle_facility(i)
            st.rerun()

def render_data():
    st.subheader("Sites (candidate facilities)")
    st.caption(
        f"Edit positions and fixed costs. Cap: {MAX_SITES} sites. "
        "Edits take effect on the next Solve."
    )
    # Leading read-only "site" column with 1-based numbers matching the map
    # labels and set notation (a dynamic-row editor hides the dataframe
    # index, so the numbers need a real column); internal storage stays
    # 0-based.
    display_sites = st.session_state.sites.copy()
    display_sites.insert(0, "site", range(1, len(display_sites) + 1))
    edited_sites = st.data_editor(
        display_sites,
        num_rows="dynamic",
        width="stretch",
        height=min(35 * (len(st.session_state.sites) + 1) + 3, 400),
        column_config={
            "site": st.column_config.NumberColumn("site", disabled=True, alignment="left"),
            "x": st.column_config.NumberColumn("x", format="%.3f", min_value=0.0, max_value=2.0, alignment="left"),
            "y": st.column_config.NumberColumn("y", format="%.3f", min_value=0.0, max_value=1.0, alignment="left"),
            "fixed_cost": st.column_config.NumberColumn(
                "fixed_cost", format="%.3f", min_value=0.0,
                alignment="left"
            ),
        },
        key="sites_editor",
    )
    edited_sites = edited_sites.drop(columns=["site"]).reset_index(drop=True)
    if not edited_sites.equals(st.session_state.sites):
        if len(edited_sites) > MAX_SITES:
            st.warning(f"Capped at {MAX_SITES} sites; ignoring excess rows.")
            edited_sites = edited_sites.iloc[:MAX_SITES]
        st.session_state.sites = edited_sites.reset_index(drop=True)
        st.session_state.user_opened = set()
        st.session_state.optimal = None
        st.rerun()

    st.subheader("Customers (demand points)")
    st.caption(
        f"Edit positions and demands. Cap: {MAX_CUSTOMERS} customers. "
        "Edits take effect on the next Solve."
    )
    display_customers = st.session_state.customers.copy()
    display_customers.insert(0, "customer", range(1, len(display_customers) + 1))
    edited_customers = st.data_editor(
        display_customers,
        num_rows="dynamic",
        width="stretch",
        height=min(35 * (len(st.session_state.customers) + 1) + 3, 400),
        column_config={
            "customer": st.column_config.NumberColumn("customer", disabled=True, alignment="left"),
            "x": st.column_config.NumberColumn("x", format="%.3f", min_value=0.0, max_value=2.0, alignment="left"),
            "y": st.column_config.NumberColumn("y", format="%.3f", min_value=0.0, max_value=1.0, alignment="left"),
            "demand": st.column_config.NumberColumn("demand", format="%.3f", min_value=0.0, alignment="left"),
        },
        key="customers_editor",
    )
    edited_customers = edited_customers.drop(columns=["customer"]).reset_index(drop=True)
    if not edited_customers.equals(st.session_state.customers):
        if len(edited_customers) > MAX_CUSTOMERS:
            st.warning(f"Capped at {MAX_CUSTOMERS} customers; ignoring excess rows.")
            edited_customers = edited_customers.iloc[:MAX_CUSTOMERS]
        st.session_state.customers = edited_customers.reset_index(drop=True)
        st.session_state.user_opened = set()
        st.session_state.optimal = None
        st.rerun()


def render_formulation():
    sub_general, sub_instance = st.tabs(["General", "Instance"])
    with sub_general:
        render_general()
    with sub_instance:
        render_instance()


def render_logs():
    optimal = st.session_state.optimal
    if not optimal:
        st.info("Run the optimizer to see solver logs.")
        return
    log = optimal.get("log", "") or ""
    if not log.strip():
        if optimal.get("status") == "solver_missing":
            st.warning(optimal.get("message", "Gurobi solver not available."))
        else:
            st.info("No solver output captured for the last run.")
        return
    st.code(log, language="text")


# ---------- Main ----------

st.set_page_config(
    page_title="Facility Location",
    page_icon="favicon.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(CSS, unsafe_allow_html=True)

# Home-link logo (sidebar variant): clicking the Griffith PSE blackletter G
# returns to the portfolio site. Image embedded as a base64 data URL so
# loading the page makes no third-party network call.
_FAVICON_DATA_URL = "data:image/png;base64," + base64.b64encode(
    (Path(__file__).parent / "favicon.png").read_bytes()
).decode()
st.sidebar.markdown(
    f'<a class="home-logo-corner" href="https://griffith-pse.com" target="_self">'
    f'<img src="{_FAVICON_DATA_URL}" alt="Griffith PSE: home" width="32" height="32" style="width:32px;height:32px;border-radius:4px;display:block" />'
    f'</a>',
    unsafe_allow_html=True,
)

# Initialize session state on first run
init_state()

# ---- Sidebar: scenario controls + cost multipliers + Solve button ----

st.sidebar.header("Scenario")
n_sites = st.sidebar.slider("Number of candidate sites", 5, MAX_SITES,
                            DEFAULT_N_SITES, key="n_sites_slider")
n_customers = st.sidebar.slider("Number of customers", 10, MAX_CUSTOMERS,
                                DEFAULT_N_CUSTOMERS, key="n_customers_slider")
seed = st.sidebar.number_input("Random seed", 0, 9999, DEFAULT_SEED, step=1,
                               key="seed_input")
if st.sidebar.button("Reset scenario", use_container_width=True):
    reset_scenario(n_sites, n_customers, seed)
    st.rerun()
# If the user changed N or seed without clicking re-randomize, do it for them
# (small UX nicety: slider drag should feel live)
if (len(st.session_state.sites) != n_sites
    or len(st.session_state.customers) != n_customers
    or st.session_state.get("_last_seed") != seed):
    reset_scenario(n_sites, n_customers, seed)
    st.session_state._last_seed = seed
    st.rerun()

st.sidebar.header("Costs")
fixed_mult = st.sidebar.slider("Fixed-cost multiplier", 0.1, 5.0, DEFAULT_FIXED_MULT,
                               step=0.1, format="%.1f", key="fixed_mult",
                               help="Scales every facility's fixed-open cost.")
transport_mult = st.sidebar.slider("Transport-cost multiplier", 0.1, 5.0,
                                   DEFAULT_TRANSPORT_MULT, step=0.1,
                                   format="%.1f", key="transport_mult",
                                   help="Scales transport cost per unit distance.")

# When cost multipliers change, the previous optimum is stale. Clear it so
# the user sees a "Run Solve again" cue rather than misleading numbers.
last_costs = st.session_state.get("_last_costs")
this_costs = (fixed_mult, transport_mult)
if last_costs is not None and last_costs != this_costs:
    st.session_state.optimal = None
st.session_state._last_costs = this_costs

solve_btn = st.sidebar.button("Solve Optimization", type="primary",
                              use_container_width=True)
# Placeholder for Set at Optimum: filled AFTER the solve handler below, so
# the button appears on the same run a solve completes (the sidebar renders
# before the solve runs).
_setopt_slot = st.sidebar.empty()

# ---- Title ----
st.markdown(
    "<h2 style='margin: 0 0 0.25rem 0; padding: 0; font-size: 1.5rem; font-weight: 700;'>"
    "Facility Location MIP Optimizer "
    "<a href='https://github.com/devin-griff/facility-location' target='_blank' "
    "title='View source on GitHub' "
    "style='display: inline-block; vertical-align: 0.02em; margin: 0 0.35rem 0 0.1rem; "
    "color: inherit;'>"
    "<svg viewBox='0 0 16 16' width='20' height='20' fill='currentColor' "
    "aria-label='GitHub'>"
    "<path d='M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17."
    "55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-"
    ".82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 "
    "2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59."
    "82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27"
    ".68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51"
    ".56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1."
    "07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-"
    "8-8-8z'/></svg></a>"
    "<span style='font-size: 1.15rem; font-weight: 400; color: #6b7280;'>"
    "powered by "
    "<a href='https://github.com/Pyomo/pyomo' target='_blank' "
    "style='color: #6b7280; text-decoration: underline;'>Pyomo</a>"
    " + "
    "<a href='https://www.gurobi.com' target='_blank' "
    "style='color: #6b7280; text-decoration: underline;'>Gurobi</a>"
    "</span>"
    "</h2>",
    unsafe_allow_html=True,
)
_caption_col, _ = st.columns([6, 3])
with _caption_col:
    st.markdown(
        "Choose which candidate sites to open so that every customer's demand "
        "is served at minimum total cost. Opening a site incurs its fixed "
        "cost, and serving each customer costs its demand times the distance "
        "to its nearest open facility (there are no capacity limits). "
        "Toggle sites open on the map and watch your cost, then "
        "click **Solve Optimization** in the sidebar to compare against the "
        "true optimum."
    )

# ---- Tabs ----
tab_opt, tab_data, tab_formulation, tab_logs = st.tabs(
    ["▶ Optimizer", "📊 Data", "📐 Formulation", "📋 Logs"]
)

# ---- Solve handler ----
if solve_btn:
    with st.spinner("Solving UFL via Gurobi..."):
        result = solve(st.session_state.sites, st.session_state.customers,
                       fixed_mult, transport_mult)
    st.session_state.optimal = result

# Fill the sidebar's Set at Optimum slot now that any fresh solve result is
# stored. No rerun: a rerun here would skip rendering the map this run, which
# destroys its selection-widget state and replays stale toggles.
if st.session_state.optimal and st.session_state.optimal.get("status") == "optimal":
    _setopt_slot.button("Set at Optimum", on_click=set_at_optimum,
                        use_container_width=True)

with tab_opt:
    render_optimizer()
with tab_data:
    render_data()
with tab_formulation:
    render_formulation()
with tab_logs:
    render_logs()
