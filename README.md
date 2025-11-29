Microeconomic Equilibrium Optimizer (MEO) / Opportunity Optimization (OO)

This repository contains a minimal, manifesto-faithful, continuous-update reference implementation of the MEO / OO framework (v1.3.2).

MEO is an equilibrium-seeking allocation method grounded in marginal comparisons and dynamic opportunity cost.

Key features of this implementation:

• Each channel k has smooth value and cost functions V_k(q_k) and C_k(q_k).
• MEO computes marginal utility MU_k, marginal cost UC_k, and the profit-usefulness ratio PU_k = MU_k / UC_k.
• The baseline channel is the one with the highest PU_k.
• For each non-baseline channel k, an opportunity cost OC_k is computed using the MRTS definition MRTS_bk = PU_b / PU_k, and marginal profit is MP_k = MU_k – UC_k – OC_k.
• Allocations are updated continuously using q_k = q_k + alpha * MP_k, with optional normalization to enforce a fixed total resource.
• The system converges when all marginal profits are near zero, indicating equilibrium.

This implementation is didactic, deterministic, and finite-difference-based. It is intended to illustrate the structure and dynamics of MEO / OO, not to serve as an industrial or production-grade optimizer.
