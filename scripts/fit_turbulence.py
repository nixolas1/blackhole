"""Fit the spatial spectrum, coherence time and emission distribution of the moving gas."""
import base64
import gzip
import json
from pathlib import Path

import numpy as np
from scipy.special import ndtr, ndtri
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]


def read_model(name):
    return json.loads((ROOT / f'data/{name}.js').read_text().split('=', 1)[1].strip().rstrip(';'))


def main():
    disk, source = read_model('disk'), read_model('accretion')
    raw = np.frombuffer(gzip.decompress(base64.b64decode(source['packed'])), dtype='B', offset=768)
    all_heat = raw.reshape(-1, source['radii'], source['angles'], 3)[:disk['openingFrames'], :, :, 0]
    heat = all_heat[:, ::2, ::2]
    frames, radii, angles = heat.shape
    ranks = rankdata(heat, axis=0, method='average')
    normal = ndtri((ranks - 0.5) / frames)
    z_limit = 2.6
    levels = 17
    quantiles = np.quantile(heat, ndtr(np.linspace(-z_limit, z_limit, levels)), axis=0)
    spectrum = np.fft.rfft(normal, axis=2) / angles
    harmonics, components = 32, 8
    # The radial covariance of each angular harmonic describes an eddy's width
    # and tilt across neighboring orbits. Keep the leading complex eigenvectors.
    basis = np.zeros((harmonics, components, radii), dtype='complex128')
    for m in range(harmonics):
        _, singular, vectors = np.linalg.svd(spectrum[:, :, m], full_matrices=False)
        basis[m] = vectors[:components] * (singular[:components, None] / np.sqrt(frames))
    # A real Fourier series counts non-zero frequencies twice. Normalize after
    # truncation so a generated field has unit variance at every sampled radius.
    variance = np.sum(np.abs(basis[0])**2, axis=0) + 2*np.sum(np.abs(basis[1:])**2, axis=(0,1))
    basis /= np.sqrt(np.maximum(variance, 0.01))
    # Measure temporal coherence after undoing the observed angular displacement.
    r_px, omega = np.asarray(disk['patternSpeeds']).T
    sample = spectrum[:, (r_px/4).astype(int), :]
    m = np.arange(sample.shape[-1])
    lags = np.arange(1, 13)
    corr = []
    for lag in lags:
        shift = np.exp(1j * omega[:,None]*m*lag/source['fps'])
        numerator = (sample[lag:]*sample[:-lag].conj()*shift).real.sum()
        denom = np.sqrt((abs(sample[lag:])**2).sum()*(abs(sample[:-lag])**2).sum())
        corr.append(numerator/denom)
    half_time = np.interp(0.5, corr[::-1], lags[::-1]) / source['fps']
    # For overlapping quarter-sine windows, the field correlation at one launch
    # interval is 1/pi. Match the measured half-correlation at ~0.79 intervals.
    interval = float(half_time / 0.79)
    # Carry the texture of the last measured state across the boundary.
    tail = np.rint(128 + 40*np.clip(normal[-1], -3.15, 3.15)).astype('B')
    coeff = np.stack([basis.real, basis.imag], axis=-1).astype('<f4')
    packed = np.rint(quantiles).astype('B').tobytes() + tail.tobytes() + coeff.tobytes()
    spec = dict(angles=angles,radii=radii,harmonics=harmonics,components=components,levels=levels,
                limit=z_limit,interval=round(interval, 4),packed=base64.b64encode(packed).decode())
    (ROOT/'data/turbulence.js').write_text('const turbulenceModel = '+json.dumps(spec,separators=(',',':'))+';\n')
    print(f'{frames} training states, co-moving half-correlation {half_time:.3f}s, renewal interval {interval:.3f}s')


if __name__ == '__main__':
    main()
