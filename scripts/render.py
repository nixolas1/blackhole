"""Export a finite preview of the live particle simulation to an H.264 movie."""
import argparse
import base64
import math
import subprocess
import tempfile
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
ENCODE = """async ({size, fps, field, duration}) => {
  const canvas = document.createElement('canvas');
  const simulation = await createAccretionRenderer(canvas, diskModel);
  simulation.reset();
  const total = Math.max(1, Math.round(duration * fps));
  const stepsPerSecond = 60;
  let completedSteps = 0;
  const config = { codec:'avc1.640034', width:size, height:size, framerate:fps,
    bitrate:32000000, latencyMode:'realtime', avc:{format:'annexb'} };
  if (!(await VideoEncoder.isConfigSupported(config)).supported)
    throw new Error('This browser does not support the requested H.264 encode.');
  let failure;
  let chunks = [];
  const encoder = new VideoEncoder({
    output(chunk) {
      const data = new Uint8Array(chunk.byteLength);
      chunk.copyTo(data);
      chunks.push(data);
    },
    error(error) { failure = error; }
  });
  encoder.configure(config);
  for (let i = 0; i < total; i++) {
    const targetStep = Math.floor(i * stepsPerSecond / fps);
    while (completedSteps < targetStep) {
      simulation.advance(1 / stepsPerSecond);
      completedSteps++;
    }
    simulation.draw(field, size, size);
    const frame = new VideoFrame(canvas, {timestamp:Math.round(i*1e6/fps),
      duration: Math.round((i+1)*1e6/fps)-Math.round(i*1e6/fps)});
    encoder.encode(frame, {keyFrame:i%fps===0});
    frame.close();
    if (i%12===11 || i===total-1) {
      await encoder.flush();
      if (failure) throw failure;
      const blob = new Blob(chunks);
      chunks = [];
      const uri = await new Promise((resolve,reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result);
        reader.onerror = reject;
        reader.readAsDataURL(blob);
      });
      await writePackets(uri.split(',')[1]);
    }
  }
  encoder.close();
  return total;
}"""


def render(path, size, fps, field, duration=24):
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError('Duration must be a positive number of seconds.')
    if fps <= 0 or size <= 0 or size % 2 or not math.isfinite(field) or field <= 0:
        raise ValueError('Choose a positive fps, a positive even pixel size, and a positive field width.')
    path.parent.mkdir(exist_ok=True, parents=True)
    with tempfile.TemporaryDirectory() as folder, sync_playwright() as automation:
        raw = Path(folder) / 'frames.h264'
        browser = automation.chromium.launch(channel='chrome', args=['--use-angle=metal'])
        page = browser.new_page(reduced_motion='reduce')
        page.goto((ROOT / 'index.html').as_uri())
        with raw.open('wb') as stream:
            page.expose_function('writePackets', lambda data: stream.write(base64.b64decode(data)))
            count = page.evaluate(ENCODE, {'size': size, 'fps': fps, 'field': field, 'duration': duration})
        browser.close()
        subprocess.run(['ffmpeg','-v','error','-y','-r',str(fps),'-f','h264','-i',str(raw),
                        '-c:v','copy','-movflags','+faststart',str(path)], check=True)
    print(f'{count} frames rendered to {path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'renders/black-hole-2160p60.mp4')
    parser.add_argument('--size', type=int, default=2160)
    parser.add_argument('--fps', type=int, default=60)
    parser.add_argument('--duration', type=float, default=24, help='Length of the exported preview in seconds')
    parser.add_argument('--field', type=float, default=420, help='Field width in reference pixels (770 for the original)')
    args = parser.parse_args()
    render(args.output, args.size, args.fps, args.field, args.duration)
