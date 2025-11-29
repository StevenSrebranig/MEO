"""
example_meo.py

Toy demonstration of the MEO / OO continuous update.

We define a few simple channels with smooth value and cost functions,
initialize allocations, and let MEO iterate toward equilibrium.
"""

from math import sqrt
from meo import Channel, MEO


def make_channels():
    # Simple smooth value / cost functions with diminishing returns.

    # Channel A: strong early returns, then saturating.
    def V_a(q: float) -> float:
        return 4.0 * sqrt(max(q, 0.0))

    def C_a(q: float) -> float:
        return 0.5 * q  # linear cost

    # Channel B: more gradual but steady.
    def V_b(q: float) -> float:
        return 3.0 * (1.0 - 2.0 ** (-max(q, 0.0)))  # saturating exponential

    def C_b(q: float) -> float:
        return 0.3 * q

    # Channel C: weaker early, stronger mid-range.
    def V_c(q: float) -> float:
        return 2.5 * sqrt(max(q, 0.0) + 0.5) - 1.0

    def C_c(q: float) -> float:
        return 0.2 * q + 0.05 * q * q  # slightly convex cost

    ch_a = Channel(V_a, C_a, q0=1.0, name="A")
    ch_b = Channel(V_b, C_b, q0=1.0, name="B")
    ch_c = Channel(V_c, C_c, q0=1.0, name="C")

    return [ch_a, ch_b, ch_c]


def main():
    channels = make_channels()

    # Total resource of 10 units, alpha controls step size.
    meo = MEO(
        channels,
        alpha=0.05,
        total_resource=10.0,
        tol=1e-4,
        max_iters=5000,
    )

    print("Initial allocations:", meo.allocations())
    iters = meo.run()
    print(f"Converged (or stopped) in {iters} iterations.")
    print("Final allocations:", meo.allocations())
    print("Total profit:", meo.total_profit())

    for ch in channels:
        print(ch)


if __name__ == "__main__":
    main()
