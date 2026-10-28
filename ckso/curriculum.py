"""Four-stage training curriculum (Section III-E, Appendix D).

Stage A (15 ep): L_rec, L_var                                   (autoencoding)
Stage B (25 ep): + L_lin, L_freq, L_pred with S = 2, L_perp     (linearity)
Stage C (30 ep): S annealed 2 -> S_max geometrically            (horizon extension)
Stage D (30 ep): + L_vol, L_ef, L_pos; beta annealed 2 -> 40    (clinical readout)

The manuscript does not say in which stage L_perp enters; we add it with L_lin in stage B.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class StageState:
    name: str
    index: int
    epoch_in_stage: int
    weights: dict
    horizon: int
    beta: float
    clinical: bool = field(default=False)


def _geom(a: float, b: float, p: float) -> float:
    if a <= 0 or b <= 0:
        return a + (b - a) * p
    return a * (b / a) ** p


class Curriculum:
    def __init__(self, cfg: dict):
        """cfg = {'weights': {...final weights...}, 'stages': [ {name, epochs, losses, horizon, beta}, ...]}

        ``horizon`` and ``beta`` are either a number or a [start, end] pair annealed
        geometrically over the stage's epochs.
        """
        self.weights = dict(cfg["weights"])
        self.stages = list(cfg["stages"])
        if not self.stages:
            raise ValueError("curriculum needs at least one stage")
        self.total_epochs = sum(int(s["epochs"]) for s in self.stages)

    def state(self, epoch: int) -> StageState:
        e = epoch
        for i, st in enumerate(self.stages):
            n = int(st["epochs"])
            if e < n or i == len(self.stages) - 1:
                e = min(e, n - 1)
                p = 0.0 if n <= 1 else e / (n - 1)
                hz = st.get("horizon", 1)
                if isinstance(hz, (list, tuple)):
                    hz = _geom(float(hz[0]), float(hz[1]), p)
                bt = st.get("beta", 1.0)
                if isinstance(bt, (list, tuple)):
                    bt = _geom(float(bt[0]), float(bt[1]), p)
                losses = st["losses"]
                weights = {k: float(self.weights[k]) for k in losses}
                clinical = any(k in weights for k in ("vol", "ef", "pos"))
                return StageState(st["name"], i, e, weights, max(1, int(round(hz))), float(bt), clinical)
            e -= n
        raise RuntimeError("unreachable")  # pragma: no cover
