import torch
from typing import Optional, Sequence
import numpy as np


def get_butcher_tableau(method):
    # 3-tuples of (a,b,c)
    if method == 'euler': return (
        torch.tensor([[0]], dtype=torch.float32),
        torch.tensor([1], dtype=torch.float32),
        torch.tensor([0], dtype=torch.float32),
    )
    elif method == 'midpoint': return (
        torch.tensor([[0,    0],
                      [1/2,  0]], dtype=torch.float32),
        torch.tensor([0, 1], dtype=torch.float32),
        torch.tensor([0, 1/2], dtype=torch.float32),
    )
    # Composition: "NXmethod" means running `method` `N` times in sequence, represented as a single composite Butcher tableau.
    # (e.g. "5xeuler", "4xmidpoint", but allows any method defined herein)
    # NOTE: not 100% sure this is correct for all methods -- tread carefully
    elif 'x' in method:
        parts = method.split('x', 1)
        try:
            N = int(parts[0])
        except ValueError:
            raise ValueError(f"Invalid composition prefix: {parts[0]!r}. Expected integer like '4x'.")
        base_method = parts[1]
        assert N >= 1

        A_base_t, b_base_t, c_base_t = get_butcher_tableau(base_method)
        A_base = A_base_t.numpy().astype(np.float32)
        b_base = b_base_t.numpy().astype(np.float32)
        c_base = c_base_t.numpy().astype(np.float32)

        s = b_base.size
        h = 1.0 / N
        S = N * s

        # Allocate and fill composite tableau
        A = np.zeros((S, S), dtype=np.float32)
        b = np.zeros(S, dtype=np.float32)
        c = np.zeros(S, dtype=np.float32)
        for n in range(N):
            for i in range(s):
                idx = n*s+i
                # global c: (n + c_i) * h
                c[idx] = (n + c_base[i]) * h
                # intra-substep coupling: scale base A by h and place on block (n,n)
                for j in range(s):
                    A[idx, n*s+j] = h * A_base[i, j]
                # coupling from all previous completed substeps m < n:
                # each previous substep's stages j contribute with weight h * b_base[j]
                if n > 0:
                    for m in range(n):
                        A[idx, m*s:(m+1)*s] = h * b_base  # broadcasts across the block
        for n in range(N):
            b[n*s:(n+1)*s] = h * b_base
        return (torch.tensor(A, dtype=torch.float32),
                torch.tensor(b, dtype=torch.float32),
                torch.tensor(c, dtype=torch.float32))
    elif method == 'ralston2': return (
        torch.tensor([[0,    0],
                      [2/3,  0]], dtype=torch.float32),
        torch.tensor([1/4, 3/4], dtype=torch.float32),
        torch.tensor([0, 2/3], dtype=torch.float32),
    )
    elif method.startswith('generic2-'):
        alpha = float(method.split('-')[1])
        assert 0 < alpha <= 1
        return (
            torch.tensor([[0,     0],
                          [alpha, 0]], dtype=torch.float32),
            torch.tensor([1-1/(2*alpha), 1/(2*alpha)], dtype=torch.float32),
            torch.tensor([0, alpha], dtype=torch.float32),
        )
    elif method == 'heun3': return (
        torch.tensor([[0,     0,  0],
                      [1/3,   0,  0],
                      [0,   2/3,  0]], dtype=torch.float32),
        torch.tensor([1/4, 0, 3/4], dtype=torch.float32),
        torch.tensor([0, 1/3, 2/3], dtype=torch.float32),
    )
    elif method == 'ralston3': return (
        torch.tensor([[0,     0,  0],
                      [1/2,   0,  0],
                      [0,   3/4,  0]], dtype=torch.float32),
        torch.tensor([2/9, 1/3, 4/9], dtype=torch.float32),
        torch.tensor([0, 1/2, 3/4], dtype=torch.float32),
    )
    elif method == 'ssprk3': return (
        torch.tensor([[0,     0,  0],
                      [1,     0,  0],
                      [1/4, 1/4,  0]], dtype=torch.float32),
        torch.tensor([1/6, 1/6, 2/3], dtype=torch.float32),
        torch.tensor([0, 1, 1/2], dtype=torch.float32),
    )
    elif method == 'rk4_3over8': return (
        torch.tensor([[0,      0,   0, 0],
                      [1/3,    0,   0, 0],
                      [-1/3,   1,   0, 0],
                      [1,     -1,   1, 0]], dtype=torch.float32),
        torch.tensor([1/8, 3/8, 3/8, 1/8], dtype=torch.float32),
        torch.tensor([0, 1/3, 2/3, 1], dtype=torch.float32),
    )
    elif method == 'rk4': return (
        torch.tensor([[0,    0,    0,   0],
                      [1/2,  0,    0,   0],
                      [0,    1/2,  0,   0],
                      [0,    0,    1,   0]], dtype=torch.float32),
        torch.tensor([1/6, 1/3, 1/3, 1/6], dtype=torch.float32),
        torch.tensor([0, 1/2, 1/2, 1], dtype=torch.float32),
    )
    # this one is a nonstandard stage-4 method that avoids c_i = 1 for all i
    # it should have order 3 and better stability than e.g. heun3
    elif method == 'rk4_order3_custom': return (
        torch.tensor([[0,     0,    0,   0],
                      [1/3,   0,    0,   0],
                      [1/3, 1/3,    0,   0],
                      [0,     0,  2/3,   0]], dtype=torch.float32),
        torch.tensor([1/4, 0, 1/2, 1/4], dtype=torch.float32),
        torch.tensor([0, 1/3, 2/3, 2/3], dtype=torch.float32),
    )
    elif method.endswith('xeuler'):
        # N times Euler steps with step size 1/N, represented as a Butcher tableau
        N = int(method[:-6])
        assert N >= 1
        c_ = np.linspace(0, 1, N, endpoint=False, dtype=np.float32)
        A_ = np.tril(1/N * np.ones((N, N), dtype=np.float32), k=-1)
        b_ = np.array([1/N]*N, dtype=np.float32)
        return (torch.tensor(A_, dtype=torch.float32),
                torch.tensor(b_, dtype=torch.float32),
                torch.tensor(c_, dtype=torch.float32))
    elif method == 'rk4p1_3over8_euler':
        a = 0.8   # scale factor for the RK4 3/8 method, we evaluate only at max. t+a*dt
        b = 0.85  # time point until which we let the RK4 scheme run and from which we run one Euler step
        g = b / a
        return (
            torch.tensor([[0,     0,   0, 0, 0],
                          [a/3,   0,   0, 0, 0],
                          [-a/3,  a,   0, 0, 0],
                          [a,    -a,   a, 0, 0],
                          # the next line gets the RK4 3/8 result to feed into the Euler step
                          [g*a/8, g*a*3/8, g*a*3/8, g*a/8, 0]], dtype=torch.float32),
            torch.tensor([g*a/8, g*a*3/8, g*a*3/8, g*a/8, 1-b], dtype=torch.float32),
            torch.tensor([0, a*1/3, a*2/3, a, b], dtype=torch.float32),
        )
    elif method == 'rk5_ralston3_ralston2':
        a = 0.55
        return (
            torch.tensor([[0,       0, 0, 0, 0],
                          [a/2,     0, 0, 0, 0],
                          [0,   a*3/4, 0, 0, 0],
                          # assemble the Ralston3 output for the subsequent Ralston2 stage
                          [a*2/9, a*1/3, a*4/9, 0, 0],
                          # get the second-to-last output to feed into the final Ralston2 step
                          [a*2/9, a*1/3, a*4/9, (1-a)*2/3, 0]], dtype=torch.float32),
            torch.tensor([a*2/9, a*1/3, a*4/9, (1-a)/4, 3*(1-a)/4], dtype=torch.float32),
            torch.tensor([0, a/2, a*3/4, a, 2/3 + a/3], dtype=torch.float32),
        )
    # Custom task/data/DNN-tuned pseudo-RK solvers
    elif method == "tuned_storm_wham_speechbert_logspec0.001":
        return (
            np.array(
                [[0.0,  0.0, 0.0, 0.0],
                [0.27450722,  0.0, 0.0, 0.0],
                [-0.7793591, 1.5663809, 0.0, 0.0],
                [2.2804186, -1.2022496, -0.07816912, 0.0]]
            ),
            np.array([0.56421536, 0.12971826, 0.16723365, 0.13883278]),
            np.array([0.0, 0.27450722, 0.78702176, 1.0])
        )
    elif method == "tuned_storm_wham_speechbert_logspec_rk12_sep08":
        return (
            torch.tensor(
                [[0.0,  0.0, 0.0, 0.0],
                [0.420, 0.0, 0.0, 0.0],
                [-0.591, 1.326, 0.0, 0.0],
                [3.151, -2.936, 0.634, 0.0]]
            ),
            torch.tensor([0.230, 0.530, 0.141, 0.099]),
            torch.tensor([0.0, 0.420, 0.735, 0.850])
        )

    raise ValueError(f"Unknown Runge-Kutta method {method}.")


def _solver_step_euler(X_t, t, v_theta, dt):
    return X_t + dt * v_theta(X_t, t)


def _solver_step_rk(X_t, t, v_theta, dt, a, b, c):
    """
    Implements a generic arbitrary-stage explicit Runge-Kutta method
    Assumes that `a` is lower-triangular. Any elements above and on the diagonal are ignored.
    """
    q = a.shape[0]
    k_list = []
    for i in range(q):
        if i == 0:
            stage_input = X_t
        else:
            k_prev = torch.stack(k_list[:i], dim=0)  # (q, B, C, F, T)
            weighted_sum = torch.sum(a[i, :i][:, None, None, None, None] * k_prev, dim=0)
            stage_input = X_t + dt * weighted_sum
        t_i = t + c[i:i+1] * dt
        k_i = v_theta(stage_input, t_i)
        k_list.append(k_i)
    k_all = torch.stack(k_list, dim=0)  # (q, B, C, F, T)
    k_total = torch.sum(b[:, None, None, None, None] * k_all, dim=0)
    out = X_t + dt * k_total
    return out


def solve_ode(X_0, v_theta, solver, N, solver_args: Optional[dict] = None, grad_for_steps: Sequence[int] = ()):
    euler_last = solver_args.get('euler_last', False) if solver_args is not None else False

    if solver == 'rk':
        assert solver_args is not None and all(k in solver_args for k in ('a', 'b', 'c')), \
            "For 'rk' solver, solver_args must contain 'a', 'b', and 'c' keys."
        a = solver_args['a']
        b = solver_args['b']
        c = solver_args['c']
        a, b, c = [torch.tensor(v, dtype=torch.float32, device=X_0.device) if isinstance(v, np.ndarray) else v for v in (a, b, c)]
    elif solver != 'euler':
        # get butcher tableau for other RK methods
        a, b, c = get_butcher_tableau(solver)
        a, b, c = [v.to(X_0.device) for v in (a, b, c)]

    dt = 1/N
    t = torch.zeros(1, dtype=torch.float32, device=X_0.device)
    X_t = X_0

    euler_step = lambda X_t, t: _solver_step_euler(X_t, t, v_theta, dt)
    if solver == 'euler':
        solver_step = euler_step
    else:
        solver_step = lambda X_t, t: _solver_step_rk(X_t, t, v_theta, dt, a, b, c)

    if len(grad_for_steps):
        for i in range(N):
            with torch.set_grad_enabled(i in grad_for_steps):
                if euler_last and i == N-1:
                    X_t = euler_step(X_t, t)
                else:
                    X_t = solver_step(X_t, t)
                t += dt
    else:
        with torch.inference_mode():
            for i in range(N):
                if euler_last and i == N-1:
                    X_t = euler_step(X_t, t)
                else:
                    X_t = solver_step(X_t, t)
                t += dt

    return X_t
