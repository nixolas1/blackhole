(async function start() {
  const $ = id => document.getElementById(id);
  const canvas = $('sky');
  const notice = $('notice');

  try {
    const population = $('particles');
    for (const n of particleCounts) population.add(new Option(n.toLocaleString(), String(n)));
    let requestedCount = defaultParticleCount;
    try {
      const saved = Number(localStorage.getItem('ekko.particles'));
      if (particleCounts.includes(saved)) requestedCount = saved;
    } catch { /* A file preview can disable storage. */ }
    population.value = String(requestedCount);
    const simulation = await createAccretionRenderer(canvas, diskModel, requestedCount);
    const controls = $('controls');
    const framing = $('framing');
    const speed = $('speed');
    const play = $('play');
    const fullscreen = $('fullscreen');
    const time = $('time');
    const step = 1 / 60;
    let steps = 0;
    let remainder = 0;
    let last = null;
    let playing = !matchMedia('(prefers-reduced-motion: reduce)').matches;
    let dirty = true;
    let quietTimer;
    let scheduled = null;
    let contextLost = false;

    function schedule() {
      if (scheduled === null && !contextLost && !document.hidden) {
        scheduled = requestAnimationFrame(tick);
      }
    }

    function redraw() {
      dirty = true;
      schedule();
    }

    function updatePlay() {
      const label = playing ? 'Pause' : 'Play';
      $('play-label').textContent = label;
      $('play-icon').setAttribute('d', playing ? 'M5 3h5v18H5zm9 0h5v18h-5z' : 'M6 3v18l15-9z');
      play.title = label + ' (Space)';
    }

    function formatTime(seconds) {
      if (seconds < 60) return seconds.toFixed(2) + ' s';
      const minutes = Math.floor(seconds / 60);
      const pad = n => String(n).padStart(2, '0');
      if (minutes < 60) return minutes + ':' + pad(Math.floor(seconds % 60));
      return Math.floor(minutes / 60) + 'h ' + pad(minutes % 60) + 'm';
    }

    function reveal() {
      document.body.classList.remove('quiet');
      clearTimeout(quietTimer);
      if (playing && document.fullscreenElement) {
        quietTimer = setTimeout(() => document.body.classList.add('quiet'), 2200);
      }
    }

    function toggle() {
      playing = !playing;
      last = null;
      schedule();
      updatePlay();
      reveal();
    }

    play.onclick = toggle;
    canvas.onclick = toggle;
    $('reset').onclick = () => {
      simulation.reset();
      steps = 0;
      remainder = 0;
      last = null;
      redraw();
      reveal();
    };
    speed.onchange = () => { last = null; };
    framing.onchange = redraw;
    population.onchange = () => {
      simulation.setParticleCount(Number(population.value));
      try { localStorage.setItem('ekko.particles', population.value); } catch {}
      last = null;
      redraw();
      reveal();
    };
    fullscreen.hidden = !document.fullscreenEnabled;
    fullscreen.onclick = async () => {
      try {
        if (document.fullscreenElement) await document.exitFullscreen();
        else await document.documentElement.requestFullscreen();
      } catch { reveal(); }
    };
    document.addEventListener('fullscreenchange', () => {
      const label = document.fullscreenElement ? 'Exit fullscreen' : 'Enter fullscreen';
      fullscreen.title = label;
      fullscreen.setAttribute('aria-label', label);
      redraw();
      reveal();
    });
    document.addEventListener('keydown', event => {
      if (event.code === 'Space' && [document.body, canvas].includes(event.target)) {
        event.preventDefault();
        if (!event.repeat) toggle();
      }
      reveal();
    });
    document.addEventListener('pointermove', reveal);
    document.addEventListener('pointerdown', reveal);
    document.addEventListener('visibilitychange', () => {
      last = null;
      if (!document.hidden) redraw();
    });
    window.addEventListener('resize', redraw);

    function tick(now) {
      scheduled = null;
      if (document.hidden) {
        last = null;
      } else {
        if (playing && last !== null) {
          // Discard long browser stalls. The fixed physics step is identical at every speed.
          const elapsed = Math.min(0.1, Math.max(0, now - last) / 1000);
          remainder += elapsed * Number(speed.value);
          const pending = Math.floor(remainder / step);
          for (let i = 0; i < pending; i++) simulation.advance(step);
          remainder -= pending * step;
          steps += pending;
          if (pending) dirty = true;
        }
        last = now;
        if (dirty) {
          const dpr = Math.min(devicePixelRatio, 4096 / Math.max(innerWidth, innerHeight));
          const width = Math.max(1, Math.round(innerWidth * dpr));
          const height = Math.max(1, Math.round(innerHeight * dpr));
          simulation.draw(Number(framing.value), width, height);
          time.value = formatTime(steps * step);
          dirty = false;
        }
      }
      if (playing) schedule();
    }

    canvas.addEventListener('webglcontextlost', event => {
      event.preventDefault();
      playing = false;
      contextLost = true;
      if (scheduled !== null) cancelAnimationFrame(scheduled);
      dirty = false;
      controls.hidden = true;
      notice.hidden = false;
      notice.textContent = 'Graphics context interrupted. Reload to restart the simulation.';
    });
    updatePlay();
    notice.hidden = true;
    controls.hidden = false;
    schedule();
  } catch (error) {
    notice.textContent = error.message + ' ';
    const link = document.createElement('a');
    link.href = 'renders/black-hole-2160p60.mp4';
    link.textContent = 'Watch the rendered simulation';
    notice.append(link);
  }
})();
