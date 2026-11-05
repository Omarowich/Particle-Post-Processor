"""3D order of a particle configuration: Steinhardt bond order and layer-resolved psi6.
- q6 (local, Steinhardt 1983) and the averaged q6-bar (Lechner & Dellago 2008) over neighbours within rc.
- crystal-like particles (ten Wolde, Ruiz-Montero & Frenkel 1996): bond i-j is 'coherent' if the normalised
  q6(i).q6(j)* > 0.7; a particle is crystal-like with >= NC coherent bonds. NC = 4 so that a hexagonal monolayer
  (6 in-plane neighbours) and close-packed 3D crystals (12) both count; random dense liquids rarely reach it.
- layer-resolved psi6: particles sorted into height layers (bottom: z < 1.6a); psi6 with in-layer neighbours only,
  so a crystalline bottom layer under a disordered second layer is still seen as ordered.
Works for 2D input too (z = 0): then every particle is in the bottom layer."""
import numpy as np
from scipy.special import sph_harm_y

RC, NC, A = 2.9, 4, 1.0

def _neigh(P, rc):
    d = P[:, None] - P[None]; r = np.sqrt((d ** 2).sum(-1)); np.fill_diagonal(r, np.inf)
    return d, r < rc

def q6_vectors(P, rc=RC):
    d, adj = _neigh(P, rc); n = len(P); q = np.zeros((n, 13), complex); nb = adj.sum(1)
    i, j = np.nonzero(adj); v = -d[i, j]; rr = np.linalg.norm(v, axis=1)
    th = np.arccos(np.clip(v[:, 2] / rr, -1, 1)); ph = np.arctan2(v[:, 1], v[:, 0])
    for k, m in enumerate(range(-6, 7)):
        np.add.at(q[:, k], i, sph_harm_y(6, m, th, ph))
    q[nb > 0] /= nb[nb > 0, None]
    return q, adj, nb

def analyse(P, rc=RC, nc=NC):
    q, adj, nb = q6_vectors(P, rc)
    norm = np.sqrt((np.abs(q) ** 2).sum(1)); q6 = np.sqrt(4 * np.pi / 13) * norm
    qbar = np.array([q[np.r_[k, np.nonzero(adj[k])[0]]].mean(0) for k in range(len(P))])
    q6bar = np.sqrt(4 * np.pi / 13) * np.sqrt((np.abs(qbar) ** 2).sum(1))
    i, j = np.nonzero(np.triu(adj, 1))
    ok = (norm[i] > 0) & (norm[j] > 0)
    dot = np.zeros(len(i)); dot[ok] = np.real((q[i[ok]] * np.conj(q[j[ok]])).sum(1)) / (norm[i[ok]] * norm[j[ok]])
    coh = dot > 0.7; nco = np.bincount(np.r_[i[coh], j[coh]], minlength=len(P))
    xtal = nco >= nc
    # layer-resolved psi6 (bottom layer and everything above)
    z = P[:, 2]; z0 = z.min() if np.ptp(z) > 0 else 0.0
    bottom = (z - z0) < 1.0 * A if np.ptp(z) > 0 else np.ones(len(P), bool)
    def psi6(mask):
        Q = P[mask]; dd = Q[:, None, :2] - Q[None, :, :2]; r = np.hypot(dd[..., 0], dd[..., 1]); np.fill_diagonal(r, np.inf)
        a = r < rc; zz = a.sum(1); ang = np.arctan2(dd[..., 1], dd[..., 0])
        s = np.where(a, np.exp(6j * ang), 0).sum(1); ok = zz >= 2
        return float(np.mean(np.abs(s[ok] / zz[ok]))) if ok.sum() > 5 else np.nan
    return dict(q6=float(q6[nb >= 2].mean()), q6bar=float(q6bar[nb >= 2].mean()), f_xtal=float(xtal.mean()),
                f_bottom=float(bottom.mean()), psi6_bottom=psi6(bottom), psi6_upper=psi6(~bottom),
                f_xtal_bottom=float(xtal[bottom].mean()), f_xtal_upper=float(xtal[~bottom].mean()) if (~bottom).any() else np.nan)

if __name__ == "__main__":
    # sanity: perfect hexagonal monolayer and fcc block
    a = 2.05; xs = [(i + 0.5 * (j % 2)) * a for i in range(12) for j in range(12)]; ys = [j * a * np.sqrt(3) / 2 for i in range(12) for j in range(12)]
    print("hex", analyse(np.c_[xs, ys, np.zeros(len(xs))]))
    rng = np.random.default_rng(1); print("random gas", analyse(rng.uniform(0, 30, (300, 3))))
