"""Measure the reference and fit the initial and steady-state disk emission."""
import argparse
import base64
import csv
import gzip
import json
from pathlib import Path

import cv2
import numpy as np

# The square crop is the black plotting area in the 1000 px reference.
CX, CY, EXTENT = 512, 495, 576
LEFT, TOP, SIDE = 128, 121, 768
ANGLES, RADII = 192, 193
TRACK_RADII = np.array([60, 70, 80, 90, 100, 120, 150])
ROOT = Path(__file__).resolve().parents[1]


def polar_maps():
    theta = np.arange(ANGLES) * (2 * np.pi / ANGLES)
    index = np.arange(RADII)
    radii = 2 * np.minimum(index, 128) + 5 * np.maximum(index - 128, 0)
    return tuple(a.astype('f') for a in (
        CX - LEFT + radii[:, None] * np.cos(theta),
        CY - TOP + radii[:, None] * np.sin(theta),
    ))


def pattern_speeds(tracks, fps):
    """Estimate signed pattern speeds in image coordinates where +y is down."""
    tracks = np.asarray(tracks)
    if len(tracks) < 3:
        return []
    # Removing the stationary crescent isolates the moving hot patches.
    modes = np.fft.rfft(tracks - tracks.mean(axis=0), axis=-1)
    corr = np.fft.irfft((modes[1:] * modes[:-1].conj()).sum(axis=0), axis=-1)
    speeds = []
    for radius, ring in zip(TRACK_RADII, corr):
        peak = int(np.argmax(ring))
        left, middle, right = ring[peak-1], ring[peak], ring[(peak+1) % ANGLES]
        curvature = 2 * (2*middle - left - right)
        if curvature <= 0:
            continue
        shift = peak + (right-left)/curvature
        if shift > ANGLES/2:
            shift -= ANGLES
        speeds.append([int(radius), round(float(shift * 2*np.pi*fps/ANGLES), 3)])
    return speeds


def main(path, rebuild_opening=False):
    capture = cv2.VideoCapture(str(path))
    fps = capture.get(cv2.CAP_PROP_FPS)
    if not capture.isOpened() or (capture.get(3), capture.get(4)) != (1000, 1000):
        raise ValueError('Expected the 1000 × 1000 reference clip')
    map_x, map_y = polar_maps()
    y, x = np.indices((SIDE, SIDE))
    palette_sum = np.zeros((256, 3))
    palette_count = np.zeros(256)
    initial = None
    total_heat = np.zeros((RADII, ANGLES))
    total_heat_squared = np.zeros_like(total_heat)
    active_tracks = []
    opening = []
    metrics = []
    motion = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    motion.setFinestScale(0)
    last = None
    last_gas = None
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        rgb = frame[TOP:TOP+SIDE, LEFT:LEFT+SIDE, ::-1]
        rgb = cv2.GaussianBlur(rgb.astype('f'), (0, 0), .9)
        gas = rgb.mean(axis=2)
        # Square-root light coordinates preserve the dim red gas in an 8-bit field.
        temp = np.sqrt(gas / 255) * 255
        bins = np.rint(temp).astype(np.uint8).ravel()
        palette_count += np.bincount(bins, minlength=256)
        for c in range(3):
            palette_sum[:, c] += np.bincount(bins, weights=rgb[:, :, c].ravel(), minlength=256)
        heat = cv2.remap(temp, map_x, map_y, cv2.INTER_LINEAR)
        harmonics = np.fft.rfft(heat, axis=1)
        harmonics[:, 65:] = 0
        heat = np.clip(np.fft.irfft(harmonics, n=ANGLES, axis=1), 0, 255)
        if initial is None:
            initial = heat.copy()
        index = len(metrics)
        # The GRMHD sequence ends at 5 s; later frames morph to an observational still.
        if index / fps <= 5:
            total_heat += heat
            total_heat_squared += heat * heat
            active_tracks.append(heat[TRACK_RADII//2])
        gray = np.rint(temp).astype('B')
        dt = velocity = 0.
        if last is not None:
            flow = motion.calc(last, gray, None)
            flow[np.maximum(last, gray) < 8] = 0
            dt = np.abs(gas-last_gas)[np.maximum(gas,last_gas)>8].mean()
            velocity = np.median(np.linalg.norm(flow, axis=2)[last_gas>12])
            if rebuild_opening and index - 1 < len(opening):
                polar_flow = cv2.remap(flow, map_x, map_y, cv2.INTER_LINEAR)
                opening[-1][:, :, 1:] = np.clip(np.rint(polar_flow * 4 + 128), 0, 255)
        if rebuild_opening and index / fps <= 5:
            layer = np.full((RADII, ANGLES, 3), 128, dtype='B')
            layer[:, :, 0] = np.rint(heat)
            opening.append(layer)
        smoothed = cv2.GaussianBlur(gas, (0, 0), 3)
        peak_y, peak_x = np.unravel_index(smoothed.argmax(), gas.shape)
        total = gas.sum()
        metrics.append([index, index/fps, LEFT+(gas*x).sum()/total,
                        TOP+(gas*y).sum()/total, LEFT+peak_x, TOP+peak_y,
                        *rgb[peak_y, peak_x], dt, velocity])
        last, last_gas = gray, gas
    capture.release()
    if initial is None:
        raise ValueError('No frames decoded')
    count = len(metrics)
    opening_frames = len(active_tracks)
    mean = total_heat / opening_frames
    sigma = np.sqrt(np.maximum(0, total_heat_squared/opening_frames - mean**2))
    ring_mean = np.broadcast_to(mean.mean(axis=1, keepdims=True), mean.shape)
    field = np.stack((initial, mean, 2*sigma, ring_mean), axis=-1)
    field = np.clip(np.rint(field), 0, 255).astype('B')
    valid = palette_count > 0
    curve = np.stack([np.interp(np.arange(256), np.flatnonzero(valid),
                      palette_sum[valid, c]/palette_count[valid]) for c in range(3)], axis=1)
    curve[0] = 0
    palette = np.rint(curve).astype('B').tobytes()
    raw = palette + field.tobytes()
    speeds = pattern_speeds(active_tracks, fps)
    spec = dict(angles=ANGLES, radii=RADII, extent=EXTENT, center=[CX,CY],
                fieldCenter=[512.5,505], fieldSize=770, openingFrames=opening_frames, patternSpeeds=speeds,
                packed=base64.b64encode(raw).decode())
    (ROOT/'data/disk.js').write_text('const diskModel = '+json.dumps(spec, separators=(',',':'))+';\n')
    if rebuild_opening:
        raw = palette + np.stack(opening).tobytes()
        packed = base64.b64encode(gzip.compress(raw, compresslevel=9, mtime=0)).decode()
        sequence = dict(frames=len(opening), fps=fps, angles=ANGLES, radii=RADII, extent=EXTENT,
                        center=[CX,CY], fieldCenter=[512.5,505], fieldSize=770, packed=packed)
        (ROOT/'data/accretion.js').write_text(
            'const accretionField = '+json.dumps(sequence, separators=(',',':'))+';\n')
    with (ROOT/'data/frames.csv').open('w') as output:
        writer = csv.writer(output)
        writer.writerow(['frame','seconds','light_center_x','light_center_y','peak_x','peak_y',
                         'peak_R','peak_G','peak_B','active_mean_change','median_flow_px_per_frame'])
        writer.writerows([[round(float(v),3) if j not in (0,4,5) else int(v)
                           for j,v in enumerate(row)] for row in metrics])
    print(f'Fitted static disk from the first {opening_frames} of {count} frames at {fps:g} fps. '
          f'Angular pattern speeds [radius in px, radians/s]: {speeds}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('clip', type=Path)
    parser.add_argument('--opening', action='store_true', help='Also rebuild the reference opening sequence')
    args = parser.parse_args()
    main(args.clip, args.opening)
