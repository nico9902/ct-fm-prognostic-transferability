"""Full-feature L2 Cox regression with Efron ties and an L-BFGS optimizer.

The optimizer uses gradients of the partial likelihood, avoiding the dense
p-by-p Hessian that makes Newton fitting slow when embedding dimension > n.
The Breslow baseline cumulative hazard is used for survival probabilities,
matching the convention used by lifelines after Efron coefficient fitting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import torch


class RidgeCoxLBFGS:
    def __init__(self, penalizer: float, maxiter: int = 1000, device: str = "cpu"):
        if penalizer <= 0:
            raise ValueError("Ridge Cox requires a positive penalizer")
        self.penalizer = float(penalizer)
        self.maxiter = maxiter
        self.device = device

    def fit(self, features, durations, events):
        x = np.asarray(features, dtype=np.float64)
        time = np.asarray(durations, dtype=np.float64)
        event = np.asarray(events, dtype=np.int64)
        if x.ndim != 2 or len(x) != len(time) or len(x) != len(event):
            raise ValueError("Features, durations and events have inconsistent shapes")
        if not np.isfinite(x).all() or not np.isfinite(time).all() or (time <= 0).any():
            raise ValueError("Cox input must be finite with positive durations")
        if not np.isin(event, (0, 1)).all() or not event.any():
            raise ValueError("Cox model requires binary events and at least one death")
        order = np.argsort(-time, kind="mergesort")
        x = x[order]
        time = time[order]
        event = event[order]
        ends = np.flatnonzero(np.r_[time[:-1] != time[1:], True])
        starts = np.r_[0, ends[:-1] + 1]
        deaths = np.add.reduceat(event, starts)
        step_group = np.repeat(np.arange(len(ends)), deaths)
        fractions = np.concatenate([np.arange(count) / count for count in deaths if count])
        step_ends = ends[step_group]
        n_deaths = int(event.sum())
        observed_x = x[event == 1].sum(axis=0)

        def objective(beta):
            eta = x @ beta
            shift = float(eta.max())
            weights = np.exp(eta - shift)
            weighted_x = weights[:, None] * x
            sum_weights = np.cumsum(weights)
            sum_weighted_x = np.cumsum(weighted_x, axis=0)
            event_weights = np.add.reduceat(weights * event, starts)
            event_weighted_x = np.add.reduceat(weighted_x * event[:, None], starts, axis=0)
            denominators = sum_weights[step_ends] - fractions * event_weights[step_group]
            numerators = (sum_weighted_x[step_ends]
                          - fractions[:, None] * event_weighted_x[step_group])
            loss = ((np.log(denominators).sum() + n_deaths * shift - eta[event == 1].sum())
                    / n_deaths + 0.5 * self.penalizer * np.dot(beta, beta))
            gradient = ((numerators / denominators[:, None]).sum(axis=0) - observed_x) / n_deaths
            gradient += self.penalizer * beta
            return loss, gradient

        if self.device != "cpu":
            device = torch.device(self.device)
            tx = torch.as_tensor(x, dtype=torch.float64, device=device)
            tevent = torch.as_tensor(event, dtype=torch.bool, device=device)
            tends = torch.as_tensor(ends, dtype=torch.long, device=device)
            tstarts = torch.as_tensor(starts, dtype=torch.long, device=device)
            tstep_group = torch.as_tensor(step_group, dtype=torch.long, device=device)
            tstep_ends = torch.as_tensor(step_ends, dtype=torch.long, device=device)
            tfractions = torch.as_tensor(fractions, dtype=torch.float64, device=device)
            tobserved_x = tx[tevent].sum(dim=0)

            def objective(beta):
                with torch.no_grad():
                    tbeta = torch.as_tensor(beta, dtype=torch.float64, device=device)
                    eta = tx @ tbeta
                    shift = eta.max()
                    weights = torch.exp(eta - shift)
                    weighted_x = weights[:, None] * tx
                    sum_weights = torch.cumsum(weights, dim=0)
                    sum_weighted_x = torch.cumsum(weighted_x, dim=0)
                    event_sum_weights = torch.cumsum(weights * tevent, dim=0)
                    event_sum_weighted_x = torch.cumsum(weighted_x * tevent[:, None], dim=0)
                    event_weights = event_sum_weights[tends]
                    event_weighted_x = event_sum_weighted_x[tends]
                    nonfirst = tstarts > 0
                    event_weights[nonfirst] -= event_sum_weights[tstarts[nonfirst] - 1]
                    event_weighted_x[nonfirst] -= event_sum_weighted_x[tstarts[nonfirst] - 1]
                    denominators = (sum_weights[tstep_ends]
                                    - tfractions * event_weights[tstep_group])
                    numerators = (sum_weighted_x[tstep_ends]
                                  - tfractions[:, None] * event_weighted_x[tstep_group])
                    loss = ((torch.log(denominators).sum() + n_deaths * shift - eta[tevent].sum())
                            / n_deaths + 0.5 * self.penalizer * torch.dot(tbeta, tbeta))
                    gradient = ((numerators / denominators[:, None]).sum(dim=0)
                                - tobserved_x) / n_deaths + self.penalizer * tbeta
                    return float(loss.item()), gradient.cpu().numpy()

        result = minimize(objective, np.zeros(x.shape[1], dtype=np.float64), jac=True,
                          method="L-BFGS-B", options={"maxiter": self.maxiter,
                                                       "maxls": 50, "ftol": 1e-8,
                                                       "gtol": 1e-5})
        if not result.success:
            raise RuntimeError(f"Ridge Cox did not converge: {result.message}")
        self.coef_ = result.x
        self.n_iter_ = result.nit
        eta = x @ self.coef_
        self.shift_ = float(eta.max())
        weights = np.exp(eta - self.shift_)
        risk_sums = np.cumsum(weights)[ends]
        positive = deaths > 0
        event_times_desc = time[ends[positive]]
        hazard_desc = deaths[positive] / risk_sums[positive]
        self.event_times_ = event_times_desc[::-1]
        self.cumulative_baseline_hazard_ = np.cumsum(hazard_desc[::-1])
        return self

    def predict_log_partial_hazard(self, features):
        return pd.Series(np.asarray(features, dtype=np.float64) @ self.coef_)

    def predict_survival_function(self, features, times):
        query = np.asarray(times, dtype=np.float64)
        indices = np.searchsorted(self.event_times_, query, side="right") - 1
        baseline = np.where(indices < 0, 0.0,
                            self.cumulative_baseline_hazard_[np.maximum(indices, 0)])
        eta = np.asarray(features, dtype=np.float64) @ self.coef_
        relative_risk = np.exp(np.clip(eta - self.shift_, -700, 700))
        survival = np.exp(-relative_risk[:, None] * baseline[None, :])
        return pd.DataFrame(survival.T, index=query)
