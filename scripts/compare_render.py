"""Compare the exported close view to the physical opening at matching time and scale."""
import argparse
import csv
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def compare(clip, movie, report):
    source = cv2.VideoCapture(str(clip))
    if not source.isOpened():
        raise ValueError(f'Cannot open {clip}')
    config = (ROOT / "data/disk.js").read_text().split("=", 1)[1].strip().rstrip(";")
    opening_frames = json.loads(config)["openingFrames"]
    size = 420
    y, x = np.indices((size, size), dtype='f')
    # Pixel-center mapping of the 420-unit close view to the original panel.
    source_x, source_y = x + 303, y + 295.5
    command = ['ffmpeg', '-v', 'error', '-i', str(movie), '-vf',
               r'select=not(mod(n\,3)),scale=420:420:flags=area', '-fps_mode', 'passthrough',
               '-frames:v', str(opening_frames), '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-']
    rows = []
    with subprocess.Popen(command, stdout=subprocess.PIPE) as decoder:
        for _ in range(opening_frames):
            ok, image = source.read()
            block = decoder.stdout.read(size*size*3)
            if not ok or len(block) != size*size*3:
                raise ValueError('Source and rendered frame counts differ')
            reference = cv2.remap(image[:, :, ::-1], source_x, source_y, cv2.INTER_LINEAR).astype('f')
            render = np.frombuffer(block, 'B').reshape(size, size, 3).astype('f')
            active = reference.max(axis=2) > 24
            error = np.abs(reference-render)[active].mean()
            centers = []
            for rgb in (reference, render):
                light = rgb.mean(axis=2)
                centers.append(np.array([(light*x).sum(), (light*y).sum()]) / light.sum())
            rows.append((len(rows), float(error), float(np.linalg.norm(centers[0]-centers[1]))))
        if decoder.wait():
            raise RuntimeError('FFmpeg did not finish decoding')
    source.release()
    if not rows:
        raise ValueError('No matching frames')
    with report.open('w') as out:
        writer = csv.writer(out)
        writer.writerow(['frame', 'active_rgb_mean_absolute_error', 'light_centroid_error_px'])
        writer.writerows((i, round(mae, 4), round(center, 4)) for i, mae, center in rows)
    print(f'{len(rows)} matched frames; mean active RGB error {np.mean([r[1] for r in rows]):.2f}/255')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('clip', type=Path)
    parser.add_argument('--movie', type=Path, default=ROOT/'renders/black-hole-2160p60.mp4')
    parser.add_argument('--report', type=Path, default=ROOT/'data/render-comparison.csv')
    args = parser.parse_args()
    compare(args.clip, args.movie, args.report)
