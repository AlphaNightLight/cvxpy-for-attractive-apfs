import marimo

__generated_with = "0.24.2"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # CVXPY for attractive APFs
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Library imports
    """)
    return


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt

    import cvxpy as cp
    import numpy as np

    return cp, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Given Parameters

    $$
    \begin{array}{rl}
    N \in \mathbb{N}:& \text{Number of waypoints}\\
    M \in \mathbb{N}:& \text{Number of obstacles}\\
    d \in \mathbb{N}:& \text{Vector space dimension}\\
    \gamma \in \mathbb{R}:& \text{Cost weighting factor}\\
    \delta \in \mathbb{R}:& \text{Maximum distance between waypoint}\\
    \end{array}
    $$

    $$
    \begin{array}{rl}
    a,b \in \mathbb{R}^d:& \text{Start and goal points}\\
    c_m \in \mathbb{R}^d:& \text{Position of obstacle m}\\
    \end{array}
    $$
    """)
    return


@app.cell
def _():
    N = 50
    M = 5
    d = 2
    gamma = 1.
    delta = 0.5

    a = (0., 0.)
    b = (10., 10.)
    #c = ((2., 2.),(5., 5.),(6., 8.),(7., 6.),(8., 9.))
    c = (
        (2., 2.),
        (3., 6.),
        (4., 8.),
        (6., 8.),
        (8., 9.),
    )

    assert len(a) == d, f"Start is not in R^{d}"
    assert len(b) == d, f"Goal is not in R^{d}"
    assert len(c) == M, f"M={M} but c contains {len(c)} values"
    for mm in range(M):
        assert len(c[mm]) == d, f"Obstacle {mm} is not in R^{d}"

    print("All parameters are fine")
    return M, N, a, b, c, d, delta, gamma


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Plot map for $\mathbb{R}^2$ case
    """)
    return


@app.cell
def _(c, d, np, plt):
    assert d==2, "Plot is just for R^2"

    c_np = np.array(c)

    plt.scatter(c_np[:,0], c_np[:,1], color="b", marker=".", s=500)
    #plt.scatter(a[0], a[1], color="r", marker="*")
    #plt.scatter(b[0], b[1], color="r", marker="*")

    plt.axis((0, 10, 0, 10))
    plt.grid()

    plt.show()
    return (c_np,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Variables

    $$
    \begin{array}{rl}
    x_n \in \mathbb{R}^d:& \text{Position of waypoint n}\\
    L \in \mathbb{R}:& \text{Upper bound of path length}\\
    J_{n,m} \in \mathbb{R}:& \text{Upper bound of obstacle m cost for point n}\\
    \end{array}
    $$
    """)
    return


@app.cell
def _(M, N, cp, d):
    x = cp.Variable((N+1, d))
    L = cp.Variable()
    J = cp.Variable((N+1,M))
    return J, L, x


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Problem Formulation

    $$
    \begin{array}{lll}
    \text{minimize} & L + \gamma\sum_{n=0}^N \sum_{m=0}^{M-1} J_{n,m} &\\
    \text{subject to} & x_0 = a,\ x_N = b &\\
    & ||x_{n+1} - x_{n}||_2 \le L/N & \text{for}\ n \in 0..N-1 \\
    & ||x_{n+1} - x_{n}||_2 \le \delta & \text{for}\ n \in 0..N-1 \\
    & V(||x_n - c_m||_2) \le J_{n,m}/N & \text{for}\ n \in 0..N, m \in 0..M-1 \\
    %& V(||x_n - c_m||_2) \le J_{n,m} & \text{for}\ n \in 0..N, m \in 0..M-1 \\
    %& ||x_n - c_m||_2 \ge V^{-1}(J_{n,m}) & \text{for}\ n \in 0..N, m \in 0..M-1 \\
    \end{array}
    $$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Common forms of V

    - The function $V(v):\mathbb{R}^+\rightarrow\mathbb{R}^+$ is scalar and convex

    - For repulsive charges it's strictly decreasing in $\mathbb{R}^+$, and $V(\infty)\rightarrow0$.
    Common repulsive forms of V are:
    $$
    V(v) = \frac{1}{v^2},\ V(v) = e^{-v},\ V(v) = \frac{1}{v}
    $$

    - For attractive charges it's strictly increasing in $\mathbb{R}^+$, and $V(0)=0$.
    Common attractive forms of V are:
    $$
    V(v) = v^2,\ V(v) = e^{v}-1,\ V(v) = v
    $$

    - For attractive charges, the problem is convex! I will use $V(v) = v$
    """)
    return


@app.cell
def _(J, L, M, N, a, b, c, cp, delta, gamma, x):
    obj = cp.Minimize(L + gamma * cp.sum(J))

    constr = [x[0] == a, x[N] == b]

    for n in range(N):
        constr += [cp.norm(x[n+1] - x[n]) <= L/N]
        constr += [cp.norm(x[n+1] - x[n]) <= delta]
        for m in range(M):
            constr += [cp.norm(x[n] - c[m]) <= J[n,m]/N]

    for m in range(M):
        constr += [cp.norm(x[N] - c[m]) <= J[N,m]/N]

    prob = cp.Problem(obj, constr)
    return constr, obj, prob


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Solve the problem
    """)
    return


@app.cell
def _(prob):
    prob.solve()
    #prob.solve(method="dccp")
    #prob.solve(nlp=True, solver=cp.IPOPT, verbose=False, print_level=0)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Printing Results
    """)
    return


@app.cell
def _(J, L, np, obj):
    print(f"Objective: {obj.value}")
    print(f"L: {L.value}")
    print(f"sum(J): {np.sum(J.value)}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Plot solution for $\mathbb{R}^2$ case
    """)
    return


@app.cell
def _(c_np, d, plt, x):
    assert d==2, "Plot is just for R^2"

    # c_np = np.array(c) # uncomment if not defined before
    x_np = x.value

    plt.scatter(c_np[:,0], c_np[:,1], color="b", marker=".", s=500)
    plt.scatter(x_np[:,0], x_np[:,1], color="r", marker="x")

    plt.axis((0, 10, 0, 10))
    plt.grid()

    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Multiplot solution for $\mathbb{R}^2$ case
    """)
    return


@app.cell
def _(J, L, c_np, constr, cp, d, plt, x):
    assert d==2, "Plot is just for R^2"

    multi_gamma = [1e-5, 1e-4, 1e-3, 1]

    for g in multi_gamma:
        obj2 = cp.Minimize(L + g * cp.sum(J))
        prob2 = cp.Problem(obj2, constr)
        prob2.solve()

        x_np2 = x.value
        plt.scatter(x_np2[:,0], x_np2[:,1], marker="x")

    plt.scatter(c_np[:,0], c_np[:,1], color="b", marker=".", s=500)
    plt.axis((0, 10, 0, 10))
    plt.grid()
    plt.show()
    return


if __name__ == "__main__":
    app.run()
