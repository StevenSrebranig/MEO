"""
Microeconomic Equilibrium Optimizer (MEO) / Opportunity Optimization (OO)
Option A: continuous updates q_k <- q_k + alpha * MP_k

Minimal, didactic implementation following the MEO / OO manifesto (v1.3.2):

- Each channel k has a value function V_k(q_k) and cost function C_k(q_k).
- We maintain a vector of allocations Q = (q_1, ..., q_K).
- At each iteration, we compute for each channel:

    MU_k  = dV_k/dq_k           (marginal utility / revenue)
    UC_k  = dC_k/dq_k           (marginal unit cost)
    PU_k  = MU_k / UC_k         (profit-usefulness ratio)

- The baseline channel b is the one with the highest PU_k.

- For non-baseline channels, we compute MRTS-based opportunity cost:

    MRTS_bk = PU_b / PU_k
    P_k     = MU_k
    OC_k    = MRTS_bk * P_k - UC_k
    MP_k    = MU_k - UC_k - OC_k

  For the baseline channel, OC_b = 0 and:

    MP_b    = MU_b - UC_b

- Allocations are updated continuously:

    q_k <- q_k + alpha * MP_k

  This allows q_k to move up when MP_k > 0 and down when MP_k < 0.

- If total_resource is provided, allocations are normalized after each step
  so that sum(q_k) = total_resource with q_k >= 0.

- The optimizer stops when max(|MP_k|) < tol or when max_iters is reached.

This implementation is intended as a faithful conceptual translation, not as a
production solver. It uses simple central differences to approximate
derivatives.
"""

from typing import Callable, List, Optional
import math


ValueFunc = Callable[[float], float]
CostFunc = Callable[[float], float]


class Channel:
    """
    A single channel with value and cost functions.

    value(q): total value / revenue at allocation q
    cost(q): total cost at allocation q

    Derivatives (MU, UC) are approximated via central finite differences.
    """

    def __init__(
        self,
        value: ValueFunc,
        cost: CostFunc,
        q0: float = 0.0,
        dq: float = 1e-3,
        name: str = "",
    ) -> None:
        self.value = value
        self.cost = cost
        self.q = float(q0)
        self.dq = float(dq)
        self.name = name or "channel"

    def _central_diff(self, f: Callable[[float], float]) -> float:
        """Central finite difference for derivative at q."""
        h = self.dq
        q_minus = max(self.q - h, 0.0)
        q_plus = self.q + h
        f_minus = f(q_minus)
        f_plus = f(q_plus)
        return (f_plus - f_minus) / (q_plus - q_minus)

    def marginal_utility(self) -> float:
        """Approximate dV/dq via central finite difference."""
        return self._central_diff(self.value)

    def marginal_cost(self) -> float:
        """Approximate dC/dq via central finite difference."""
        return self._central_diff(self.cost)

    def profit(self) -> float:
        """Total profit at current allocation."""
        return self.value(self.q) - self.cost(self.q)

    def __repr__(self) -> str:
        return f"Channel(name={self.name!r}, q={self.q:.4f}, profit={self.profit():.4f})"


class MEO:
    """
    Microeconomic Equilibrium Optimizer (continuous-update version).

    At each iteration:

        1. Compute MU_k, UC_k, PU_k for all channels.
        2. Choose baseline channel b = argmax PU_k.
        3. For each channel:

            P_k = MU_k
            if k == b:
                OC_k = 0
                MP_k = MU_k - UC_k
            else:
                MRTS_bk = PU_b / PU_k
                OC_k    = MRTS_bk * P_k - UC_k
                MP_k    = MU_k - UC_k - OC_k

        4. Update allocations:

            q_k <- q_k + alpha * MP_k

        5. Project onto feasible set (q_k >= 0) and optionally rescale
           allocations to keep sum(q_k) = total_resource.

    Stop when max(|MP_k|) < tol or max_iters is reached.
    """

    def __init__(
        self,
        channels: List[Channel],
        alpha: float = 0.01,
        total_resource: Optional[float] = None,
        tol: float = 1e-3,
        max_iters: int = 10_000,
    ) -> None:
        if not channels:
            raise ValueError("MEO requires at least one channel.")
        self.channels = channels
        self.alpha = float(alpha)
        self.total_resource = total_resource
        self.tol = float(tol)
        self.max_iters = int(max_iters)

        self.iterations = 0
        self.last_marginal_profits: List[float] = []

    def _compute_marginals(self):
        """Compute MU_k, UC_k, PU_k for each channel."""
        MU: List[float] = []
        UC: List[float] = []
        PU: List[float] = []

        for ch in self.channels:
            mu = ch.marginal_utility()
            uc = ch.marginal_cost()

            # Guard against degenerate UC.
            if uc <= 0.0:
                pu = 0.0
            else:
                pu = mu / uc

            MU.append(mu)
            UC.append(uc)
            PU.append(pu)

        return MU, UC, PU

    def _normalize_allocations(self):
        """Normalize q so that sum(q_k) = total_resource and q_k >= 0."""
        if self.total_resource is None:
            # Just clamp to non-negative if no fixed pool is enforced.
            for ch in self.channels:
                ch.q = max(ch.q, 0.0)
            return

        # Enforce non-negativity first.
        for ch in self.channels:
            ch.q = max(ch.q, 0.0)

        total = sum(ch.q for ch in self.channels)
        if total <= 0.0:
            # If everything collapsed, spread evenly.
            equal = self.total_resource / len(self.channels)
            for ch in self.channels:
                ch.q = equal
            return

        scale = self.total_resource / total
        for ch in self.channels:
            ch.q *= scale

    def step(self) -> float:
        """
        Perform a single MEO update step.

        Returns:
            max_abs_MP: max(|MP_k|) across channels after this step.
        """
        MU, UC, PU = self._compute_marginals()

        # Baseline: channel with highest PU_k.
        b = max(range(len(self.channels)), key=lambda k: PU[k])
        PU_b = PU[b]

        MP: List[float] = []

        for k, ch in enumerate(self.channels):
            mu_k = MU[k]
            uc_k = UC[k]
            P_k = mu_k  # treat marginal utility as local "price"

            if k == b:
                OC_k = 0.0
                MP_k = mu_k - uc_k
            else:
                # MRTS_bk = PU_b / PU_k (with guard for PU_k ~ 0)
                pu_k = PU[k]
                if abs(pu_k) < 1e-12:
                    MRTS_bk = 0.0
                else:
                    MRTS_bk = PU_b / pu_k
                OC_k = MRTS_bk * P_k - uc_k
                MP_k = mu_k - uc_k - OC_k

            MP.append(MP_k)

        # Continuous update: q_k <- q_k + alpha * MP_k
        for k, ch in enumerate(self.channels):
            ch.q += self.alpha * MP[k]

        # Project to feasible set and optionally enforce fixed pool.
        self._normalize_allocations()

        self.last_marginal_profits = MP
        max_abs_MP = max(abs(mp) for mp in MP)
        self.iterations += 1
        return max_abs_MP

    def run(self) -> int:
        """
        Run MEO until equilibrium or max_iters.

        Returns:
            Number of iterations performed.
        """
        for _ in range(self.max_iters):
            max_abs_MP = self.step()
            if max_abs_MP < self.tol:
                break
        return self.iterations

    def total_profit(self) -> float:
        """Total profit across all channels at current allocations."""
        return sum(ch.profit() for ch in self.channels)

    def allocations(self) -> List[float]:
        """Return current allocation vector Q."""
        return [ch.q for ch in self.channels]
