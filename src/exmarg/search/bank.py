"""
the template bank and the matched filter, on the CPU (numpy) or the GPU (cupy).

  z_I(t) = FFT_t[ conj(d_I) / sig2_I * a e^{i psi0} ]   (zero padded to n_time = 8 nu_max)

the template factor a e^{i psi0} depends on the template only, so the
bank is built once; a segment gives (n_template, n_time) arrays of
ln|z_I| and arg z_I at every arrival time of the segment.
"""

import numpy as np

from ..profiles import profile
from .segments import D0, read_segments


def array_module(device):
    """numpy for device 'cpu', cupy for device 'gpu'"""
    if device == "cpu":
        return np
    if device == "gpu":
        import cupy
        return cupy
    raise ValueError("device must be 'cpu' or 'gpu'")


class MassBank:
    """templates for one frequency grid and the noise curves of the two detectors.

    M, q, chi  (n_template,) detector-frame total mass, mass ratio
               m2/m1 <= 1 and common aligned spin
    u_H, u_L   (n_template,) template power u_I = sum a^2 / sig2_I at D0 (E_0|z_I|^2, the data's normalization)
    gain       {'H': kappa_H, 'L': kappa_L} the amplitude a signal carries in the data of each detector relative to the
               true strain (strain.signal_gain: the conditioning with estimated spectra), 1.0 for both by default.  The
               model of detector I is the template kappa_I a, of power kappa_I^2 u_I: its matched filter is kappa_I z_I
               (z_I of trajectories, which does not depend on the gain) and E_0|kappa_I z_I|^2 = kappa_I^2 u_I.  The
               data's statistics |z_I|^2 / u_I are those of the template a and do not change; pe reads the tables of
               the model with the data scaled by kappa_I (pe._plan)
    u          (n_template,) the mean model power ubar = (kappa_H^2 u_H + kappa_L^2 u_L) / 2 ((u_H + u_L) / 2 with
               gains 1, u_H with equal curves, to the last bit)
    asym       (n_template,) w of the model powers, (kappa_H^2 u_H - kappa_L^2 u_L) / (2 ubar), so kappa_H^2 u_H =
               u (1 + w), kappa_L^2 u_L = u (1 - w)
    W          (n_template, n_freq) complex64 a e^{i psi0}, on the device"""

    def __init__(self, M, q, chi, seg, device="cpu", gain=None):
        self.M, self.q, self.chi = M, q, chi
        self.device, self.xp = device, array_module(device)
        self.T, self.fmin, self.fmax = seg["T"], seg["fmin"], seg["fmax"]
        self.nu = seg["nu"]
        if not np.all(np.diff(self.nu) == 1):
            raise ValueError("frequency bins must be contiguous")
        self.lo, self.hi = int(self.nu[0]), int(self.nu[-1]) + 1
        self.n_time = 8 * int(self.nu.max())
        W = np.empty((M.size, self.nu.size), np.complex64)
        self.u_H, self.u_L = np.empty(M.size), np.empty(M.size)
        for i in range(M.size):
            m1 = M[i] / (1.0 + q[i]); m2 = M[i] * q[i] / (1.0 + q[i])
            nu, _, a1, p0 = profile(m1, m2, chi[i], chi[i], self.T, self.fmin, self.fmax, 1.0)
            j = np.searchsorted(self.nu, nu)
            a = np.zeros(self.nu.size); a[j] = a1 / D0
            psi0 = np.zeros(self.nu.size); psi0[j] = p0
            W[i] = a * np.exp(1j * psi0)
            self.u_H[i] = float((a ** 2 / seg["sig2_H"]).sum())
            self.u_L[i] = float((a ** 2 / seg["sig2_L"]).sum())
        g = {I: 1.0 for I in "HL"} if gain is None else {I: float(gain[I]) for I in "HL"}
        if not all(np.isfinite(v) and v > 0 for v in g.values()):
            raise ValueError("the gains must be finite and positive, not %s" % g)
        self.gain = g
        mH, mL = self.u_H, self.u_L
        if g["H"] != 1.0 or g["L"] != 1.0:                   # gains 1: the same operations as before, bit for bit
            mH, mL = g["H"] ** 2 * self.u_H, g["L"] ** 2 * self.u_L
        self.u = (mH + mL) / 2.0
        self.asym = (mH - mL) / (mH + mL)
        self.W = self.xp.asarray(W)

    def trajectories(self, d, sig2, chunk=1024):
        """matched filter of one segment against every template.

        d, sig2  dicts {'H': (n_freq,), 'L': (n_freq,)} of data and noise
        returns dict of float32 (n_template, n_time) arrays on the device:
        lqH, phH, lqL, phL = ln|z_I| and arg z_I."""
        xp = self.xp
        nm = self.W.shape[0]
        out = {}
        for I in "HL":
            w = xp.asarray(np.ascontiguousarray(np.conj(d[I]) / sig2[I], np.complex64))
            lq = xp.empty((nm, self.n_time), np.float32)
            ph = xp.empty((nm, self.n_time), np.float32)
            for a in range(0, nm, chunk):
                b = min(a + chunk, nm)
                pad = xp.zeros((b - a, self.n_time), np.complex64)
                pad[:, self.lo:self.hi] = self.W[a:b] * w[None, :]
                z = xp.fft.fft(pad, axis=-1)
                lq[a:b] = 0.5 * xp.log(xp.maximum(z.real ** 2 + z.imag ** 2, 1e-38))
                ph[a:b] = xp.arctan2(z.imag, z.real)
                del pad, z
            out["lq" + I], out["ph" + I] = lq, ph
        return out


def mass_bank(bank, segments, device="cpu", gain=None):
    """the template bank.

    bank      .npy path or array: (n,) detector-frame total masses (equal
              mass, zero spin) or (n, 3) rows (M_det, q, chi)
    segments  anything read_segments takes; its frequency grid, T, band
              and sig2_H, sig2_L define the templates
    device    'cpu' or 'gpu'
    gain      {'H': kappa_H, 'L': kappa_L}, the signal gain of the data
              (strain.signal_gain; MassBank), default 1"""
    B = np.load(bank) if isinstance(bank, (str, bytes)) or hasattr(bank, "__fspath__") else np.asarray(bank)
    B = np.asarray(B, float)
    if B.ndim == 1:
        B = np.stack([B, np.ones_like(B), np.zeros_like(B)], axis=1)
    seg = read_segments(segments, rows=[0])
    return MassBank(*[np.ascontiguousarray(B[:, i]) for i in range(3)], seg, device, gain)
