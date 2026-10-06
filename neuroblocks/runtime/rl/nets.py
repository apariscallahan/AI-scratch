"""Policy / value networks built from the user's layer blocks, plus fast forward paths.

The layer blocks inside 'create neural network' become the *body* of every network an
algorithm needs (actor, critic, Q-network); the output layer is sized automatically.
Small RL networks are called thousands of times per second, so plain MLPs (dense layers
and activations) get a forward pass without nn.Module overhead, and whole populations
(neuro-evolution) are evaluated in one batched matrix multiply per layer.
"""
from __future__ import annotations

import contextlib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..errors import NBError
from ..layers import BuildCtx, Dense, Output, _Seq, build_layers

_ACTS = {
    nn.Tanh: torch.tanh, nn.ReLU: torch.relu, nn.Sigmoid: torch.sigmoid, nn.GELU: F.gelu, nn.SiLU: F.silu,
    nn.ELU: F.elu, nn.Softplus: F.softplus,
}


def default_specs():
    return [Dense(64, "tanh"), Dense(64, "tanh"), Output()]


def body_specs(model):
    specs = list(getattr(model, "specs", None) or [])
    return specs if specs else default_specs()


def build_net(model, in_size: int, out_size: int, out_gain: float | None = None):
    """Build a network from the model's layer blocks for ``in_size`` inputs and ``out_size`` outputs."""
    specs = body_specs(model)
    bid = getattr(model, "bid", None)
    ctx = BuildCtx(task="regression", output_size=int(out_size), input_is_tokens=False)
    try:
        mods, rows, shape, _ = build_layers(specs, ctx, (int(in_size),), False)
    except NBError as e:
        if e.nb_block is None:
            e.nb_block = bid
        if "token" in e.message.lower() or "image" in e.message.lower() or "sequence" in e.message.lower():
            e.hint = e.hint or "A network that plays in a world reads a short list of numbers (its senses): use " \
                               "dense layers."
        raise
    if tuple(shape) != (int(out_size),):
        last = specs[-1] if specs else None
        if isinstance(last, Output):
            raise NBError(f"The network ends with shape {tuple(shape)} but the world needs {out_size} outputs.",
                          block_id=getattr(last, "bid", None) or bid)
        extra, extra_rows, shape, _ = build_layers([Output()], ctx, tuple(shape), False)
        for r in extra_rows:
            r["auto"] = True
        mods += extra
        rows += extra_rows
        if tuple(shape) != (int(out_size),):
            raise NBError(f"The network ends with shape {tuple(shape)} but the world needs {out_size} outputs.",
                          block_id=bid)
    net = _Seq(mods)
    if out_gain is not None:
        lin = last_linear(net)
        if lin is not None:
            with torch.no_grad():
                lin.weight.mul_(out_gain)
                if lin.bias is not None:
                    lin.bias.zero_()
    net.eval()
    return net, rows


def last_linear(net):
    found = None
    for m in net.modules():
        if isinstance(m, nn.Linear):
            found = m
    return found


def reinit(net, gain_last: float | None = None):
    """Fresh random weights (default PyTorch initialisation)."""
    for m in net.modules():
        if m is not net and hasattr(m, "reset_parameters"):
            m.reset_parameters()
    if gain_last is not None:
        lin = last_linear(net)
        if lin is not None:
            with torch.no_grad():
                lin.weight.mul_(gain_last)
                if lin.bias is not None:
                    lin.bias.zero_()


# ---------------------------------------------------------------------------
# Fast paths
# ---------------------------------------------------------------------------


def _mlp_plan(net):
    """[(kind, payload)] for nets made only of Linear layers and simple activations, else None."""
    plan = []

    def walk(m):
        if isinstance(m, (nn.Sequential, nn.ModuleList, _Seq)):
            children = m.layers if isinstance(m, _Seq) else m
            for c in children:
                if not walk(c):
                    return False
            return True
        if isinstance(m, nn.Linear):
            plan.append(("linear", m))
            return True
        if isinstance(m, (nn.Dropout, nn.Identity)):
            return True
        if type(m) in _ACTS:
            plan.append(("act", _ACTS[type(m)]))
            return True
        if isinstance(m, nn.LeakyReLU):
            slope = m.negative_slope
            plan.append(("act", lambda x, s=slope: F.leaky_relu(x, s)))
            return True
        if isinstance(m, nn.Softmax):
            plan.append(("act", lambda x: torch.softmax(x, -1)))
            return True
        return False

    return plan if walk(net) else None


class FastForward:
    """``f(np_array) -> np_array`` through a network, for acting.

    It runs exactly the same computation as :meth:`Population.forward` (one copy of the weights per
    row), so the outputs match bit for bit: a champion found by neuro-evolution then replays
    identically when it is watched or scored (physics is chaotic — tiny rounding differences grow).
    """

    def __init__(self, net):
        self.net = net
        self.pop = Population(net)
        self._theta = None
        self._version = None

    def _flat(self):
        version = tuple(p._version for p in self.net.parameters())
        if self._theta is None or version != self._version:
            self._theta = self.pop.flat()[None]
            self._version = version
        return self._theta

    @torch.no_grad()
    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        theta = self._flat()
        return self.pop.forward(theta if x.shape[0] == 1 else theta.expand(x.shape[0], -1), x)


class Population:
    """Many copies of one network with different weights (rows of a matrix), evaluated together."""

    def __init__(self, net):
        self.net = net
        self.names = [n for n, _ in net.named_parameters()]
        self.shapes = [tuple(p.shape) for _, p in net.named_parameters()]
        self.sizes = [p.numel() for _, p in net.named_parameters()]
        self.dim = int(sum(self.sizes))
        self.slices = {}
        o = 0
        for n, k, s in zip(self.names, self.sizes, self.shapes):
            self.slices[n] = (o, k, s)
            o += k
        plan = _mlp_plan(net)
        self.plan = None
        if plan is not None:
            ids = {id(p): n for n, p in net.named_parameters()}
            steps = []
            for kind, p in plan:
                if kind == "linear":
                    wn = ids[id(p.weight)]
                    bn = ids.get(id(p.bias)) if p.bias is not None else None
                    steps.append(("linear", wn, bn))
                else:
                    steps.append(("act", p))
            self.plan = steps
        self._vmap_ok = None

    def flat(self, net=None) -> torch.Tensor:
        return torch.cat([p.detach().reshape(-1).float() for p in (net or self.net).parameters()])

    @torch.no_grad()
    def load(self, theta: torch.Tensor, net=None):
        """Copy one row of weights into the network (in place, so cached copies notice the change)."""
        o = 0
        for p in (net or self.net).parameters():
            k = p.numel()
            p.copy_(theta[o:o + k].view_as(p))
            o += k

    def _view(self, theta, name):
        o, k, shape = self.slices[name]
        return theta[:, o:o + k].reshape(theta.shape[0], *shape)

    def _params(self, theta):
        return {n: self._view(theta, n) for n in self.names}

    @torch.no_grad()
    def forward(self, theta: torch.Tensor, x: np.ndarray) -> np.ndarray:
        """theta: (P, D) weights, x: (P, inputs) -> (P, outputs)."""
        xt = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
        if self.plan is not None:
            h = xt.unsqueeze(1)
            for step in self.plan:
                if step[0] == "linear":
                    w = self._view(theta, step[1])
                    if step[2] is not None:
                        b = self._view(theta, step[2]).unsqueeze(1)
                        h = torch.baddbmm(b, h, w.transpose(1, 2))
                    else:
                        h = torch.bmm(h, w.transpose(1, 2))
                else:
                    h = step[1](h)
            return h.squeeze(1).numpy()
        from torch.func import functional_call, vmap
        params = self._params(theta)
        if self._vmap_ok is not False:
            try:
                out = vmap(lambda p, xi: functional_call(self.net, p, (xi.unsqueeze(0),)).squeeze(0))(params, xt)
                self._vmap_ok = True
                return out.float().numpy()
            except Exception:  # noqa: BLE001 - some layers (e.g. LSTM) can't be vmapped
                self._vmap_ok = False
        outs = [functional_call(self.net, {n: v[i] for n, v in params.items()}, (xt[i:i + 1],))[0]
                for i in range(theta.shape[0])]
        return torch.stack(outs).float().numpy()


@contextlib.contextmanager
def few_threads(n: int = 1):
    """Tiny networks run fastest on one thread (thread start-up costs more than the maths)."""
    old = torch.get_num_threads()
    try:
        if old != n:
            torch.set_num_threads(n)
        yield
    finally:
        if torch.get_num_threads() != old:
            torch.set_num_threads(old)
