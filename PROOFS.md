# Plain-English Intuition & One-Page Mathematical Proofs

This document explains the entire theoretical core of `jm-online` using one intuitive physical picture followed by **three self-contained elementary proofs** (five lines each) that can be verified by hand with pen and paper in ten minutes.

---

## Part 1: Plain-English Intuition ("The Needle in a $[-\lambda, +\lambda]$ Box")

Suppose an automated system monitors noisy sensors to switch between two operating regimes—**State `0` (Normal / Inactive)** and **State `1` (Alert / Active)**—such as room HVAC ventilation, urban air-pollution warnings, or market volatility risk alerts.
- Without a transition penalty ($\lambda = 0$), single-step sensor noise causes rapid on/off flickering, wearing out equipment and causing alarm fatigue.
- A **Statistical Jump Model (SJM)** discourages switching by charging a fixed cost $\lambda > 0$ each time the regime changes.

### The Core Picture
At each time step $t$, incoming observation $x_t$ produces a pointwise loss difference $g_t = \ell_t(0) - \ell_t(1)$:
- $g_t > 0$ means current data favors State `1`;
- $g_t < 0$ means current data favors State `0`.

**Theorem 1** proves that two-state SJM dynamic programming is mathematically identical to moving a **needle inside a bounded box $[-\lambda, +\lambda]$**:
- Each step shifts the needle by $g_t$, stopping at the right wall $+\lambda$ or the left wall $-\lambda$ whenever it hits either boundary.
- The emitted online decision simply checks which side of $0$ the needle sits on: **output `1` if needle $> 0$, and output `0` if needle $\le 0$**.

From this single picture, two practical engineering questions resolve immediately:

1. **Should real-time systems carry the needle forward (`carried` / `predict_online`) or re-solve a sliding window of $W$ steps (`rolling-W`)?**
   - `carried` keeps the needle moving continuously from $t = 0$ in $O(1)$ memory. `rolling-W` lifts the needle back to $0$ at the start of every length-$W$ window and replays the last $W$ steps.
   - **Theorem 2** shows that if the last $W$ observations push the needle against either wall $-\lambda$ or $+\lambda$ even once, all older history is erased and **both methods output the exact same state (`0.00%` disagreement)**. Conversely, if $W$ is short and $\lambda$ is wide so the needle never reaches the walls, repeatedly resetting to $0$ turns the rule into an unanchored moving sum that can **double false switching** when $\lambda$ increases.
2. **How wide should $\lambda$ be without guessing or overfitting backtests?**
   - When settled in State `0`, the needle rests against the left wall $-\lambda$. If a genuine shift to State `1` occurs (pushing the needle right by $\Delta > 0$ per step), the needle must travel a distance of $\lambda$ to cross $0$.
   - **Theorem 3** proves that this takes $\lfloor \lambda / \Delta \rfloor + 1$ steps. Specifying a physical **delay budget** $B$ (e.g., $\le 5$ minutes or $\le 4$ hours) immediately imposes the exact ceiling **$\lambda < B \cdot \Delta$**, allowing automatic, label-free penalty calibration.

---

## Part 2: Complete Human-Readable Proofs

Fix two distinct centroids $\mu_0, \mu_1 \in \mathbb{R}^p$ and transition penalty $\lambda > 0$. For observations $x_t \in \mathbb{R}^p$, let $\ell_t(k) = \frac{1}{2}\|x_t - \mu_k\|_2^2$ for $k \in \{0, 1\}$, and let $g_t = \ell_t(0) - \ell_t(1)$. The path objective over $s_{1:t} \in \{0,1\}^t$ is:
$$J_\lambda(s_{1:t}) = \sum_{i=1}^{t} \ell_i(s_i) + \lambda \sum_{i=2}^{t} \mathbf{1}\{s_i \ne s_{i-1}\}.$$
Let $V_t(k) = \min_{s_{1:t}:\, s_t = k} J_\lambda(s_{1:t})$ be the minimum path cost ending in state $k$, with terminal cost difference $D_t = V_t(0) - V_t(1)$. Under standard `argmin` tie-breaking, the causal label at step $t$ is $\hat{s}_t = 1$ if $V_t(1) < V_t(0)$ and $\hat{s}_t = 0$ if $V_t(0) \le V_t(1)$. Define $\operatorname{clip}_\lambda(z) = \max(-\lambda, \min(\lambda, z))$.

---

### Theorem 1 (Exact Bounded-Score Reduction)
Define $y_0 = 0$ and $y_t = \operatorname{clip}_\lambda(y_{t-1} + g_t)$ for $t \ge 1$. Then for every $t \ge 1$, $y_t = \operatorname{clip}_\lambda(D_t)$ and the causal terminal label is $\hat{s}_t = \mathbf{1}\{y_t > 0\}$.

**Proof:**
1. At $t = 1$, there is no transition charge so $V_1(k) = \ell_1(k)$, giving $D_1 = g_1$ and $y_1 = \operatorname{clip}_\lambda(g_1) = \operatorname{clip}_\lambda(D_1)$.
2. For $t \ge 2$, dynamic programming gives $V_t(0) = \ell_t(0) + \min\{V_{t-1}(0),\, V_{t-1}(1) + \lambda\}$ and $V_t(1) = \ell_t(1) + \min\{V_{t-1}(1),\, V_{t-1}(0) + \lambda\}$.
3. Subtracting the two equations and factoring out $V_{t-1}(1)$ yields:
   $$D_t = g_t + \min(D_{t-1}, \lambda) - \min(0, D_{t-1} + \lambda).$$
4. Direct inspection on the three intervals $D_{t-1} < -\lambda$, $|D_{t-1}| \le \lambda$, and $D_{t-1} > \lambda$ verifies that $\min(D_{t-1}, \lambda) - \min(0, D_{t-1} + \lambda) = \operatorname{clip}_\lambda(D_{t-1})$.
5. By the induction hypothesis $\operatorname{clip}_\lambda(D_{t-1}) = y_{t-1}$, hence $D_t = g_t + y_{t-1}$ and $\operatorname{clip}_\lambda(D_t) = \operatorname{clip}_\lambda(y_{t-1} + g_t) = y_t$. Since $\lambda > 0$, clipping at $\pm\lambda$ preserves strict positivity: $V_t(1) < V_t(0) \iff D_t > 0 \iff y_t > 0$. $\blacksquare$

---

### Theorem 2 (Memory Coalescence vs. Short-Window Switching Inflation)
Fix $\lambda > 0$ and $t \ge W \ge 1$. Let $u_j$ and $v_j$ denote the `carried` and `rolling-W` trajectories across $j \in \{t-W, \dots, t\}$, initialized at $u_{t-W} = y_{t-W}^{\mathrm{car}} \in [-\lambda, \lambda]$ and $v_{t-W} = 0$.
1. **(Monotone Contraction & Exact Coalescence):** $|u_j - v_j| \le |u_{j-1} - v_{j-1}| \le \lambda$ for all $j$. Moreover, if $u_{t-W} \ge 0$ and at some step $j$ inside the window either $v_{j-1} + g_j \ge \lambda$ or $u_{j-1} + g_j \le -\lambda$ (symmetrically for $u_{t-W} \le 0$), then $u_j = v_j$ and therefore **$y_t^{\mathrm{car}} = y_t^{(W)}$ identically**.
2. **(Short-Window Doubling Counterexample):** On the $8$-step periodic sequence repeating `[-1, -1, -1, +1, -1, +1, +1, +1]` with $W = 7$, increasing $\lambda$ from $1.5$ to $2.5$ reduces internal jumps inside every fitted $7$-step window to $0$, yet **doubles** `rolling-7` emitted switches from $2/8$ ($25\%$) to $4/8$ ($50\%$), whereas `carried` switches only $2/8$ ($25\%$).

**Proof:**
- *Part 1:* For any fixed $g$, $f(x) = \operatorname{clip}_\lambda(x + g)$ is nondecreasing and $1$-Lipschitz ($|f(a) - f(b)| \le |a - b|$), so $|u_j - v_j|$ never increases and starts at $|u_{t-W} - 0| \le \lambda$. When $u_{t-W} \ge 0 = v_{t-W}$, monotonicity maintains $u_{j-1} \ge v_{j-1}$. As soon as $v_{j-1} + g_j \ge \lambda$, we also have $u_{j-1} + g_j \ge \lambda$, giving $u_j = v_j = +\lambda$; as soon as $u_{j-1} + g_j \le -\lambda$, we also have $v_{j-1} + g_j \le -\lambda$, giving $u_j = v_j = -\lambda$. Equal scores at step $j$ remain equal through step $t$.
- *Part 2:* Every $7$-element slice of `[-1, -1, -1, +1, -1, +1, +1, +1]` contains four signs of one polarity and three of the other, so the constant-path cost is $3$, whereas any path with an internal jump at $\lambda = 2.5$ costs at least $1 + 2.5 = 3.5 > 3$. Thus every `rolling-7` fit at $\lambda = 2.5$ is flat ($0$ internal jumps) and outputs the $7$-element majority sign. Across the $8$ phases, the $7$-step sums are `[-1, +1, +1, +1, -1, +1, -1, -1]`, yielding labels `(0, 1, 1, 1, 0, 1, 0, 0)` (**4 transitions per 8 steps**). Direct evaluation of $y_7$ at $\lambda = 1.5$ gives `(0, 0, 1, 1, 1, 0, 0, 0)` (**2 transitions**), and steady-state `carried` decoding at $\lambda = 2.5$ crosses $0$ once upward and once downward per cycle (**2 transitions**). $\blacksquare$

---

### Theorem 3 (Exact Step-Response Delay & Delay-Budget Ceiling)
Let $\Delta = \frac{1}{2}\|\mu_1 - \mu_0\|_2^2 > 0$ denote the per-step loss separation between centroids.
1. Starting saturated in State `0` ($y_{t_0} = -\lambda$), under clean State-`1` observations $x_t = \mu_1$ ($g_t = +\Delta$) for $t > t_0$, the number of steps until $\hat{s}_{t_0+k} = 1$ is:
   $$d_{0 \to 1}(\lambda) = \left\lfloor \frac{\lambda}{\Delta} \right\rfloor + 1.$$
2. Starting saturated in State `1` ($y_{t_0} = +\lambda$), under $x_t = \mu_0$ ($g_t = -\Delta$) for $t > t_0$, the number of steps until $\hat{s}_{t_0+k} = 0$ is:
   $$d_{1 \to 0}(\lambda) = \max\!\left(1,\, \left\lceil \frac{\lambda}{\Delta} \right\rceil\right).$$
3. For any integer delay budget $B \ge 1$, the worst-case step-response delay satisfies $\max\{d_{0 \to 1}(\lambda),\, d_{1 \to 0}(\lambda)\} \le B$ if and only if:
   $$\lambda < B \cdot \Delta.$$

**Proof:**
1. Starting at $y_{t_0} = -\lambda$ with $g_t = +\Delta > 0$, after $k \ge 1$ steps $y_{t_0+k} = \min(\lambda,\, -\lambda + k\Delta)$. Since $\lambda > 0$, $y_{t_0+k} > 0 \iff -\lambda + k\Delta > 0 \iff k > \lambda / \Delta$, whose smallest positive integer solution is $\lfloor \lambda / \Delta \rfloor + 1$.
2. Starting at $y_{t_0} = +\lambda$ with $g_t = -\Delta$, $y_{t_0+k} = \max(-\lambda,\, \lambda - k\Delta)$, which satisfies $y_{t_0+k} \le 0 \iff \lambda - k\Delta \le 0 \iff k \ge \lambda / \Delta$, giving $\max(1, \lceil \lambda / \Delta \rceil)$.
3. Because $\lceil r \rceil \le \lfloor r \rfloor + 1$ for every $r = \lambda / \Delta > 0$, worst-case delay is $d_{0 \to 1}(\lambda) = \lfloor \lambda / \Delta \rfloor + 1$. Therefore $\max(d_{0 \to 1}, d_{1 \to 0}) \le B \iff \lfloor \lambda / \Delta \rfloor \le B - 1 \iff \lambda / \Delta < B \iff \lambda < B\Delta$. $\blacksquare$
