# =============================================================================
# Facility Location — a Streamlit tutorial app.
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
#   - streamlit  — UI framework. Each interaction reruns this script
#                  top-to-bottom; persistent state lives in `st.session_state`.
#   - pyomo      — algebraic modeling: sets, params, vars, objective,
#                  constraints. Continuous + binary variables.
#   - HiGHS      — the MILP solver, called via Pyomo's appsi_highs interface.
#                  Ships as a pip wheel (`highspy`).
#   - pandas     — DataFrames for the editable site/customer tables.
#   - altair     — the 2D map plot (sites + customers + assignment lines).
#
# File roadmap:
#   1. Solver       — model definition, HiGHS log capture, top-level solve.
#   2. State        — session_state init / reset.
#   3. Utilities    — random-scenario generation, distance, user cost.
#   4. LaTeX        — General + Instance formulation rendering.
#   5. CSS          — small style tweaks for the toggle button grid.
#   6. Tabs         — render_optimizer / render_data / render_formulation /
#                     render_logs.
#   7. Main         — page config, sidebar, tab assembly.
# =============================================================================

import base64
import math
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import pyomo.environ as pyo
import streamlit as st
from pyomo.common.errors import ApplicationError
from pyomo.common.tee import capture_output


# ---------- Constants ----------

MAX_SITES = 25
MAX_CUSTOMERS = 60

# Color palette — red for the user's solution, green for the optimal,
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
    """Run HiGHS and return (results, log_text)."""
    log_text = ""
    try:
        with capture_output(capture_fd=True) as buf:
            solver = pyo.SolverFactory("appsi_highs")
            results = solver.solve(m, tee=True)
        log_text = buf.getvalue()
    except TypeError:
        with capture_output() as buf:
            solver = pyo.SolverFactory("appsi_highs")
            results = solver.solve(m, tee=True)
        log_text = buf.getvalue()
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
                f"HiGHS solver not available. Run `pip install highspy` "
                f"in your environment. ({e})"
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
    unit square, fixed costs vary modestly, demands uniform."""
    rng = np.random.default_rng(seed)
    sites = pd.DataFrame({
        "x": rng.uniform(0.05, 0.95, n_sites).round(3),
        "y": rng.uniform(0.05, 0.95, n_sites).round(3),
        # Fixed cost varies in [0.4, 1.2] — meaningful spread, doesn't dominate
        "fixed_cost": rng.uniform(0.4, 1.2, n_sites).round(3),
    })
    customers = pd.DataFrame({
        "x": rng.uniform(0.0, 1.0, n_customers).round(3),
        "y": rng.uniform(0.0, 1.0, n_customers).round(3),
        # Demand in [0.5, 1.5] — modest spread, no zero demand
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
    st.session_state.scenario_initialized = True


def reset_scenario(n_sites, n_customers, seed):
    """Re-roll scenario; reset user candidate to all-closed."""
    sites, customers = make_scenario(n_sites, n_customers, seed)
    st.session_state.sites = sites
    st.session_state.customers = customers
    st.session_state.user_opened = set()
    st.session_state.optimal = None


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
        "Note: this is the **multi-source** UFL — a customer can in principle be "
        "served fractionally by multiple facilities. The LP relaxation usually "
        "gives integer optima, so the visualization typically looks single-source."
    )


def render_instance():
    sites = st.session_state.sites
    customers = st.session_state.customers
    n_sites = len(sites)
    n_customers = len(customers)
    st.markdown("**This instance**")
    st.markdown(
        f"- $|\\mathcal{{I}}| = {n_sites}$ candidate sites\n"
        f"- $|\\mathcal{{J}}| = {n_customers}$ customers\n"
        f"- Decision variables: {n_sites} binary $y_i$ + "
        f"{n_sites * n_customers} continuous $x_{{ij}}$\n"
        f"- Constraints: {n_customers} demand + "
        f"{n_sites * n_customers} linking"
    )
    st.markdown("**Fixed costs $f_i$**")
    st.dataframe(
        sites.reset_index().rename(columns={"index": "i"})[["i", "x", "y", "fixed_cost"]],
        width="stretch", height=min(35 * (n_sites + 1) + 3, 350),
    )
    st.markdown("**Demands $d_j$**")
    st.dataframe(
        customers.reset_index().rename(columns={"index": "j"})[["j", "x", "y", "demand"]],
        width="stretch", height=min(35 * (n_customers + 1) + 3, 350),
    )


# ---------- CSS ----------

CSS = """
<style>
.block-container,
[data-testid="stMainBlockContainer"] {
    padding-top: 4rem !important;
}
/* Toggle buttons for facility selection. Smaller, denser than default. */
.stButton > button {
    white-space: pre-line;
    font-size: 0.85rem;
    padding: 0.4rem 0.5rem;
    line-height: 1.3;
    min-height: 60px;
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
            "x": float(sites.iloc[i]["x"]),
            "y": float(sites.iloc[i]["y"]),
            "kind": kind,
            "color": color,
            "fixed_cost": float(sites.iloc[i]["fixed_cost"]),
        })
    sites_plot = pd.DataFrame(site_records)

    customers_plot = pd.DataFrame({
        "j": list(range(n_customers)),
        "x": customers["x"].astype(float).values,
        "y": customers["y"].astype(float).values,
        "demand": customers["demand"].astype(float).values,
    })

    # User assignment lines (red, single-source from each customer to its
    # nearest user-open facility). Only drawn if user has any open facilities.
    user_lines_records = []
    for j, i in user_assign.items():
        if i is None:
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
    if optimal and optimal.get("status") == "optimal":
        for (i, j), frac in optimal["x"].items():
            if frac < 1e-6:
                continue
            opt_lines_records.append({
                "j": j, "i": i, "frac": frac,
                "x": float(customers.iloc[j]["x"]),
                "y": float(customers.iloc[j]["y"]),
                "x2": float(sites.iloc[i]["x"]),
                "y2": float(sites.iloc[i]["y"]),
            })
    opt_lines_df = pd.DataFrame(opt_lines_records) if opt_lines_records else \
        pd.DataFrame(columns=["j", "i", "frac", "x", "y", "x2", "y2"])

    # ---- Layers ----
    base_axis = alt.Axis(grid=False, labels=False, ticks=False, domain=False, title=None)
    domain = [-0.05, 1.05]

    # Optimal lines: very thin, semi-transparent green
    if not opt_lines_df.empty:
        opt_lines = (
            alt.Chart(opt_lines_df)
            .mark_rule(color=COLOR_OPT, opacity=0.4, strokeWidth=1)
            .encode(
                x=alt.X("x:Q", scale=alt.Scale(domain=domain), axis=base_axis),
                y=alt.Y("y:Q", scale=alt.Scale(domain=domain), axis=base_axis),
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
                x=alt.X("x:Q", scale=alt.Scale(domain=domain), axis=base_axis),
                y=alt.Y("y:Q", scale=alt.Scale(domain=domain), axis=base_axis),
                x2="x2:Q", y2="y2:Q",
            )
        )
    else:
        user_lines = None

    # Customers: dark dots, sized proportionally to demand. Range chosen so
    # the smallest demand is still visible and the largest doesn't dwarf
    # the facility markers.
    customers_layer = (
        alt.Chart(customers_plot)
        .mark_circle(color=COLOR_CUSTOMER, opacity=0.7)
        .encode(
            x=alt.X("x:Q", scale=alt.Scale(domain=domain), axis=base_axis),
            y=alt.Y("y:Q", scale=alt.Scale(domain=domain), axis=base_axis),
            size=alt.Size("demand:Q",
                          scale=alt.Scale(range=[20, 180]),
                          legend=None),
            tooltip=[alt.Tooltip("j:O", title="customer"),
                     alt.Tooltip("demand:Q", format=".2f")],
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

    closed_layer = (
        alt.Chart(closed_sites)
        .mark_point(color=COLOR_CANDIDATE, shape="cross", size=110, opacity=0.6,
                    strokeWidth=2)
        .encode(
            x=alt.X("x:Q", scale=alt.Scale(domain=domain), axis=base_axis),
            y=alt.Y("y:Q", scale=alt.Scale(domain=domain), axis=base_axis),
            tooltip=[alt.Tooltip("i:O", title="site"),
                     alt.Tooltip("fixed_cost:Q", format=".2f"),
                     alt.Tooltip("kind:N")],
        )
        .add_params(site_select)
    )

    open_layer = (
        alt.Chart(open_sites)
        .mark_square(size=240, opacity=0.92, stroke="white", strokeWidth=1.5)
        .encode(
            x=alt.X("x:Q", scale=alt.Scale(domain=domain), axis=base_axis),
            y=alt.Y("y:Q", scale=alt.Scale(domain=domain), axis=base_axis),
            color=alt.Color("color:N", scale=None, legend=None),
            tooltip=[alt.Tooltip("i:O", title="site"),
                     alt.Tooltip("fixed_cost:Q", format=".2f"),
                     alt.Tooltip("kind:N")],
        )
        .add_params(site_select)
    )

    # Site index labels — only on the open ones (keeps the closed cloud clean)
    open_labels = (
        alt.Chart(open_sites)
        .mark_text(color="white", fontWeight="bold", fontSize=10)
        .encode(
            x=alt.X("x:Q", scale=alt.Scale(domain=domain), axis=base_axis),
            y=alt.Y("y:Q", scale=alt.Scale(domain=domain), axis=base_axis),
            text="i:O",
        )
    )

    layers = [l for l in [opt_lines, user_lines, closed_layer, customers_layer,
                          open_layer, open_labels] if l is not None]
    chart = alt.layer(*layers).properties(width=600, height=600).configure_view(strokeOpacity=0)
    return chart


# ---------- Tabs ----------

def _grid_rows(n, cols=4):
    """Chunk indices 0..n-1 into rows of `cols`, padding with None."""
    rows = []
    for r in range(0, n, cols):
        rows.append([i if i < n else None for i in range(r, r + cols)])
    return rows


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

    # Headline metrics row
    cols = st.columns(4)
    if user_cost is None:
        cols[0].metric("Your cost", "—")
    else:
        cols[0].metric("Your cost", f"{user_cost:.2f}")

    if optimal and optimal.get("status") == "optimal":
        opt_cost = optimal["cost"]
        cols[1].metric("Optimal cost", f"{opt_cost:.2f}")
        if user_cost is not None and opt_cost > 0:
            gap = 100.0 * (user_cost - opt_cost) / opt_cost
            cols[2].metric("Gap", f"{gap:+.1f}%",
                           help="Your cost minus optimal, as a % of optimal.")
            cols[3].metric("Match",
                           "✓" if gap < 0.01 else f"+{gap:.1f}%")
        else:
            cols[2].metric("Gap", "—")
            cols[3].metric("Match", "—")
    else:
        cols[1].metric("Optimal cost", "—")
        cols[2].metric("Gap", "—")
        cols[3].metric("Match", "—")

    if not user_opened:
        st.info(
            "Click the facility buttons below to toggle them open. "
            "Then hit **Solve Optimization** in the sidebar to see the optimum."
        )

    st.markdown("**Toggle facilities open / closed**")
    st.caption(
        "Each button shows the candidate site index and its fixed open cost. "
        "Click to open or close. Customer assignment auto-routes to the nearest "
        "open facility."
    )

    # Button grid: 4 columns, ceil(n_sites/4) rows
    for row in _grid_rows(len(sites), cols=4):
        button_cols = st.columns(len(row))
        for col, idx in zip(button_cols, row):
            if idx is None:
                col.empty()
                continue
            site = sites.iloc[idx]
            opened = idx in user_opened
            label = f"{idx}\nf={site['fixed_cost']:.2f}"
            if col.button(
                label,
                key=f"toggle_{idx}",
                type="primary" if opened else "secondary",
                use_container_width=True,
            ):
                toggle_facility(idx)
                st.rerun()

    # Map. Click any site marker (× or ■) to toggle it open/closed —
    # selection state is initialized from user_opened so toggles via the
    # button grid above are reflected on the map, and clicks on the map
    # propagate back via on_select="rerun".
    st.markdown("**Network map** — click site markers to toggle open / closed")
    chart = build_map(sites, customers, user_opened, optimal, user_assign)
    chart_state = st.altair_chart(
        chart, use_container_width=False, on_select="rerun", key="map_chart"
    )
    if chart_state and isinstance(chart_state, dict):
        sel = chart_state.get("selection", {}).get("site_select", []) or []
        selected_ids = {int(p["i"]) for p in sel if isinstance(p, dict) and "i" in p}
        if selected_ids != user_opened:
            st.session_state.user_opened = selected_ids
            st.rerun()

    # Legend
    st.caption(
        f":red[**■ Red**] = open in your solution &nbsp;&nbsp; "
        f":green[**■ Green**] = open in optimal &nbsp;&nbsp; "
        f":violet[**■ Purple**] = open in both &nbsp;&nbsp; "
        f"× gray = closed candidate &nbsp;&nbsp; • dark dot = customer (size = demand)"
    )


def render_data():
    st.subheader("Sites (candidate facilities)")
    st.caption(
        f"Edit positions and fixed costs. Cap: {MAX_SITES} sites. "
        "Edits take effect on the next Solve."
    )
    edited_sites = st.data_editor(
        st.session_state.sites,
        num_rows="dynamic",
        width="stretch",
        height=min(35 * (len(st.session_state.sites) + 1) + 3, 400),
        column_config={
            "x": st.column_config.NumberColumn("x", format="%.3f", min_value=0.0, max_value=1.0),
            "y": st.column_config.NumberColumn("y", format="%.3f", min_value=0.0, max_value=1.0),
            "fixed_cost": st.column_config.NumberColumn(
                "fixed_cost", format="%.3f", min_value=0.0
            ),
        },
        key="sites_editor",
    )
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
    edited_customers = st.data_editor(
        st.session_state.customers,
        num_rows="dynamic",
        width="stretch",
        height=min(35 * (len(st.session_state.customers) + 1) + 3, 400),
        column_config={
            "x": st.column_config.NumberColumn("x", format="%.3f", min_value=0.0, max_value=1.0),
            "y": st.column_config.NumberColumn("y", format="%.3f", min_value=0.0, max_value=1.0),
            "demand": st.column_config.NumberColumn("demand", format="%.3f", min_value=0.0),
        },
        key="customers_editor",
    )
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
            st.warning(optimal.get("message", "HiGHS solver not available."))
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
st.markdown("""
<style>
.home-logo-corner {
    position: fixed; top: 0.5rem; left: 0.75rem; z-index: 999999;
}
.home-logo-corner img {
    width: 32px; height: 32px; border-radius: 4px; display: block;
}
</style>
""", unsafe_allow_html=True)
st.sidebar.markdown(
    f'<a class="home-logo-corner" href="https://griffith-pse.com" target="_self">'
    f'<img src="{_FAVICON_DATA_URL}" alt="Griffith PSE — home" />'
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
if st.sidebar.button("Re-randomize scenario", use_container_width=True,
                     help="Regenerate sites/customers from the chosen seed."):
    reset_scenario(n_sites, n_customers, seed)
    st.rerun()
# If the user changed N or seed without clicking re-randomize, do it for them
# (small UX nicety — slider drag should feel live)
if (len(st.session_state.sites) != n_sites
    or len(st.session_state.customers) != n_customers
    or st.session_state.get("_last_seed") != seed):
    reset_scenario(n_sites, n_customers, seed)
    st.session_state._last_seed = seed
    st.rerun()

st.sidebar.header("Costs")
fixed_mult = st.sidebar.slider("Fixed-cost multiplier", 0.1, 5.0, DEFAULT_FIXED_MULT,
                               step=0.1, key="fixed_mult",
                               help="Scales every facility's fixed-open cost.")
transport_mult = st.sidebar.slider("Transport-cost multiplier", 0.1, 5.0,
                                   DEFAULT_TRANSPORT_MULT, step=0.1,
                                   key="transport_mult",
                                   help="Scales transport cost per unit distance.")

# When cost multipliers change, the previous optimum is stale. Clear it so
# the user sees a "Run Solve again" cue rather than misleading numbers.
last_costs = st.session_state.get("_last_costs")
this_costs = (fixed_mult, transport_mult)
if last_costs is not None and last_costs != this_costs:
    st.session_state.optimal = None
st.session_state._last_costs = this_costs

st.sidebar.divider()
solve_btn = st.sidebar.button("Solve Optimization", type="primary",
                              use_container_width=True)
if st.session_state.optimal and st.session_state.optimal.get("status") == "optimal":
    st.sidebar.button("Set at Optimum", on_click=set_at_optimum,
                      use_container_width=True,
                      help="Copy the optimal opened-facility set into your candidate.")

# ---- Title ----
st.title("Facility Location")
st.caption("Where to open warehouses — uncapacitated facility location")
st.markdown(
    "Toggle which candidate sites to open in the **Optimizer** tab and watch "
    "your cost. Click **Solve Optimization** in the sidebar to compute the "
    "true optimum, then compare side-by-side on the same map."
)

# ---- Tabs ----
tab_opt, tab_data, tab_formulation, tab_logs = st.tabs(
    ["▶ Optimizer", "📊 Data", "📐 Formulation", "📋 Logs"]
)

# ---- Solve handler ----
if solve_btn:
    with st.spinner("Solving UFL via HiGHS..."):
        result = solve(st.session_state.sites, st.session_state.customers,
                       fixed_mult, transport_mult)
    st.session_state.optimal = result

with tab_opt:
    render_optimizer()
with tab_data:
    render_data()
with tab_formulation:
    render_formulation()
with tab_logs:
    render_logs()
