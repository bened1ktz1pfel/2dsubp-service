"""
2D Batch Scheduling with Layered Multi-element Setups (2D-BS-LMS)
Gurobi MILP implementation of the formulation (3)-(23) from the paper.

Notation (matching the paper):
  Sets:
    I : items           (indices 0..n-1)
    B : potential batches (we use |B| = n as an upper bound)
    E : elements
    L : layers
    E_il subset of E : elements required by item i in layer l

  Variables:
    x[i,b]   in {0,1}  - item i assigned to batch b
    z[b]     in {0,1}  - batch b used
    y[e,l,b] in {0,1}  - element e required in layer l of batch b
    d[e,l,b] in {0,1}  - element e used in both layer l-1 and l (l>=1)
    f[l,b]   in {0,1}  - some element shared between layers l-1 and l (l>=1)
    r[l,b]   in {0,1}  - some element shared between layers l and l+1
    u[e,b]   in {0,1}  - element e gets a setup between batches b-1 and b
    u0[e]    in {0,1}  - element e setup for the very first used batch
    P[b]     >= 0      - processing time of batch b
    X[i], Y[i] >= 0    - placement coordinates of item i
    a[i,j], b_lr[i,j] in {0,1} - relative position (left / below) for no-overlap

Objective: minimize sum_b P[b] + sum_b sum_e s_e * u[e,b] + sum_e s_e * u0[e]
"""

import gurobipy as gp
from gurobipy import GRB
from dataclasses import dataclass
from typing import Dict, List, Set


@dataclass
class Instance:
    n_items: int
    n_elements: int
    n_layers: int
    w: List[float]
    h: List[float]
    W: float
    H: float
    p: List[float]
    s: List[float]
    c: List[float]
    Q: float
    E_il: List[List[Set[int]]]
    n_batches: int = None

    def __post_init__(self):
        if self.n_batches is None:
            self.n_batches = self.n_items


def build_gurobi_model(inst: Instance, time_limit: float = 300.0,
                       verbose: bool = True) -> gp.Model:
    I = range(inst.n_items)
    B = range(inst.n_batches)
    E = range(inst.n_elements)
    L = range(inst.n_layers)

    m = gp.Model("2D-BS-LMS")
    m.Params.OutputFlag = 1 if verbose else 0
    m.Params.TimeLimit = time_limit

    M_geo = max(inst.W, inst.H) + max(max(inst.w), max(inst.h))

    # Variables
    x  = m.addVars(I, B, vtype=GRB.BINARY, name="x")
    z  = m.addVars(B,    vtype=GRB.BINARY, name="z")
    y  = m.addVars(E, L, B, vtype=GRB.BINARY, name="y")
    v  = m.addVars(E, L, B, vtype=GRB.BINARY, name="d")
    u  = m.addVars(E, B, vtype=GRB.BINARY, name="u")
    u0 = m.addVars(E,    vtype=GRB.BINARY, name="u0")
    P  = m.addVars(B,    lb=0.0, name="P")

    X  = m.addVars(I,    lb=0.0, name="X")
    Y  = m.addVars(I,    lb=0.0, name="Y")
    a_lr = m.addVars(I, I, vtype=GRB.BINARY, name="a")
    b_bl = m.addVars(I, I, vtype=GRB.BINARY, name="b")

    # (2) every item in exactly one batch
    m.addConstrs((gp.quicksum(x[i, b] for b in B) == 1 for i in I), name="assign")

    # (3) item only in used batch
    m.addConstrs((x[i, b] <= z[b] for i in I for b in B), name="used_batch")

    # (4) empty batch must not be "used"
    m.addConstrs((z[b] <= gp.quicksum(x[i, b] for i in I) for b in B),
                 name="nonempty_if_used")

    # (5) element required if item assigned
    for i in I:
        for l in L:
            for e in inst.E_il[i][l]:
                for b in B:
                    m.addConstr(x[i, b] <= y[e, l, b], name=f"req_{i}_{e}_{l}_{b}")

    # (6) weight capacity per layer
    m.addConstrs(
        (gp.quicksum(inst.c[e] * y[e, l, b] for e in E) <= inst.Q * z[b]
         for b in B for l in L),
        name="capacity",
    )

     # (7), (8): inter-batch setup
    for e in E:
        for b in range(1, inst.n_batches):
            for l in L:
                m.addConstr(u[e, b] >= y[e, l, b]
                            - gp.quicksum(y[e, lp, b - 1] for lp in L)
                            - (1 - z[b]))
                m.addConstr(u[e, b] >= y[e, l, b - 1]
                            - gp.quicksum(y[e, lp, b] for lp in L)
                            - (1 - z[b]))

    # (9): initial setup
    for e in E:
        for l in L:
            m.addConstr(u0[e] >= y[e, l, 0])

    # (10): connecting element must be in l-1
    for e in E:
        for l in range(1, inst.n_layers):
            for b in B:
                m.addConstr(v[e, l, b] <= y[e, l - 1, b])

    # (11): connecting element must be in l
    for e in E:
        for l in range(1, inst.n_layers):
            for b in B:
                m.addConstr(v[e, l, b] <= y[e, l, b])
    
    # (12): There is only one connecting element between layers l-1 and l
    for l in range(1, inst.n_layers):
        for b in B:
            m.addConstr(gp.quicksum(v[e, l, b] for e in E) <= 1)
    
    # (13): if element is connecting in l-1 and l, then there can not be another element in l
    for l in range(1, inst.n_layers - 1):
        for b in B:
            for e in E:
                for ep in E:
                    if ep != e:
                        m.addConstr(v[e, l, b] + v[e, l + 1, b] + y[ep, l, b] <= 1)

    # (14) processing time
    for b in B:
        m.addConstr(
            P[b] == gp.quicksum(inst.p[i] * x[i, b] for i in I)
                  + 2 * gp.quicksum(y[e, l, b] for e in E for l in L)
                  - 2 * gp.quicksum(v[e, l, b] for e in E for l in range(1, inst.n_layers)),
            name=f"P_{b}"
        )
  

    # (15), (16): plate bounds (every item is placed by (4))
    for i in I:
        m.addConstr(X[i] + inst.w[i] <= inst.W)
        m.addConstr(Y[i] + inst.h[i] <= inst.H)

    # (17)-(19): no overlap inside the same batch
    for i in I:
        for j in I:
            if i == j:
                continue
            m.addConstr(X[i] + inst.w[i] <= X[j] + (1 - a_lr[i, j]) * M_geo)
            m.addConstr(Y[i] + inst.h[i] <= Y[j] + (1 - b_bl[i, j]) * M_geo)

    for b in B:
        for i in I:
            for j in I:
                if i >= j:
                    continue
                m.addConstr(
                    a_lr[i, j] + a_lr[j, i] + b_bl[i, j] + b_bl[j, i]
                    >= x[i, b] + x[j, b] - 1,
                    name=f"sep_{i}_{j}_{b}",
                )

    # symmetry breaking: use batches in order
    m.addConstrs((z[b] >= z[b + 1] for b in range(inst.n_batches - 1)),
                 name="batch_order")
    
    setup_cost = (gp.quicksum(inst.s[e] * u0[e] for e in E)
                  + gp.quicksum(inst.s[e] * u[e, b]
                                for e in E for b in range(1, inst.n_batches)))
    m.setObjective(gp.quicksum(P[b] for b in B) + setup_cost, GRB.MINIMIZE)

    m._vars = dict(x=x, z=z, y=y, P=P, u=u, u0=u0, X=X, Y=Y)
    return m


def extract_solution(m: gp.Model, inst: Instance) -> Dict:
    x = m._vars["x"]; z = m._vars["z"]; P = m._vars["P"]
    X = m._vars["X"]; Y = m._vars["Y"]
    batches = []
    for b in range(inst.n_batches):
        if z[b].X < 0.5:
            continue
        items = [i for i in range(inst.n_items) if x[i, b].X > 0.5]
        batches.append({
            "batch": b,
            "items": items,
            "processing_time": P[b].X,
            "placement": {i: (X[i].X, Y[i].X) for i in items},
        })
    return {"objective": m.ObjVal, "batches": batches,
            "gap": m.MIPGap, "runtime": m.Runtime}


def demo_instance() -> Instance:
    return Instance(
        n_items=15,
        n_elements=4,           # e.g. 4 different filament colors / materials
        n_layers=10,             # 10 print layers
        # geometry: build plate 10 x 10
        W=10, H=10,
        w=[4, 3, 5, 2, 6, 3, 4, 2, 5, 3, 4, 3, 5, 2, 6],
        h=[3, 4, 4, 2, 5, 3, 3, 2, 4, 5, 3, 4, 2, 3, 5],
        # processing time per item (volume-dependent)
        p=[5, 6, 8, 3, 10, 5, 6, 3, 7, 6, 5, 4, 8, 3, 9],
        # setup time per element
        s=[2, 3, 2, 4],
        # weight per element (capacity Q = 6 -> at most 3 heavier elements together)
        c=[1, 1, 1, 1],
        Q=3,
        # Element requirements per (item, layer).
        # Pattern: some items use the same element across all layers (good for
        # layer-sharing), others switch between layers (no sharing bonus).
        E_il=[
            # item 0: uses {0} in every layer -> strong self-sharing
            [{0},     {0, 1},     {0},     {0},     {0},    {0},     {0},     {0},     {0},     {0}],
            # item 1: uses {1} then {1,2} then {2} -> partial sharing
            [{1},     {1, 2},  {2},     {2},     {2},   {2},     {2},     {2},     {2},     {2}],
            # item 2: uses {0,2} in layers 0 and 1, then {2} -> sharing on 2
            [{0, 2},  {0, 2},  {2},     {2},     {2},   {2},     {2},     {2},     {2},     {2}],
            # item 3: small, uses only {3}
            [{3},     {3},     {3},     {3},     {3},   {3},     {3},     {3},     {3},     {3}],
            # item 4: large, uses {0,1} in layer 0, {1} later
            [{0, 1},  {1},     {1, 2},     {1},     {1},   {1},     {1},     {1},     {1},     {1}],
            # item 5: uses {2,3} then {3} then {3}
            [{2, 3},  {3},     {3},     {3},     {3}],
            # item 6: uses {0} then {0,3} then {3}
            [{0},     {0, 3},  {3},     {3, 1},     {3}],
            # item 7: small filler, uses {1}
            [{1},     {1, 3},     {1},     {1},     {3}],
            # item 8: uses {2} consistently
            [{2},     {2},     {2, 3},     {2, 3},     {2, 3}],
            # item 9: uses {3} then {2,3} then {2}
            [{3},     {2, 3},  {2},     {0, 2, 3},     {2}],
            # item 10: uses {0,1} then {1} then {1}
            [{0, 1},  {1},     {1},     {1},     {1}],
            # item 11: uses {1,2} then {2} then {2}
            [{1, 2},  {2},     {2},     {2},     {2}],
            # item 12: uses {0,3} then {3} then {3}
            [{0, 3},  {3},     {3},     {3},     {3}],
            # item 13: uses {1} then {1,2} then {2}
            [{1},     {1, 2},  {2},     {2},     {2}],
            # item 14: uses {2,3} then {3} then {3}
            [{2, 3},  {3},     {3},     {3},     {0, 2,3}],
        ],
        # upper bound on number of batches; 4 is generous for 10 items on a 10x10 plate
        n_batches=5,
    )


if __name__ == "__main__":
    inst = demo_instance()
    m = build_gurobi_model(inst, time_limit=60)
    m.Params.Threads = 1
    m.Params.TimeLimit = 300
    m.optimize()
    if m.SolCount > 0:
        sol = extract_solution(m, inst)
        print("\n=== Solution ===")
        print(f"Objective: {sol['objective']:.2f}  (gap {sol['gap']*100:.1f}%)")
        for bat in sol["batches"]:
            print(f" Batch {bat['batch']}: items={bat['items']} "
                  f"P={bat['processing_time']:.2f}")
            for i, (xi, yi) in bat["placement"].items():
                print(f"   item {i} at ({xi:.2f}, {yi:.2f})")