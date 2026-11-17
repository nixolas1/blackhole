// A reduced induction/dissipation model in disk-plane coordinates (source pixels, simulation seconds).
function createMagneticField(gravity, horizon) {
  const count = 24;
  const uniforms = new Float32Array(count*8);
  let cells = [], seed = 1;
  function random() {
    seed = (Math.imul(1664525, seed) + 1013904223) >>> 0;
    return seed / 4294967296;
  }
  function cell(initial) {
    const r = 57 + 105*random()**1.6;
    const omega = Math.sqrt(gravity/r)/(r-horizon);
    const threshold = 1.5 + 2.6*random();
    return {r, omega, theta:random()*2*Math.PI,
      shear:omega*(0.5 + r/(r-horizon)), radial:0.36+0.67*random(),
      toroidal:initial ? threshold*random() : 0, threshold, reconnecting:false,
      width:7+14*random(), length:16+25*random(), decay:0.13+0.25*random()};
  }
  function reset() {
    seed = 274981;
    cells = Array.from({length:count}, () => cell(true));
    uniforms.fill(0);
  }
  function advance(dt) {
    for (let i=0; i<count; i++) {
      const c = cells[i];
      c.theta = (c.theta + c.omega*dt) % (2*Math.PI);
      let power = 0;
      if (c.reconnecting) {
        const before = c.toroidal*c.toroidal;
        c.toroidal *= Math.exp(-dt/c.decay);
        // Magnetic energy density is proportional to B²/2. Lost energy heats the gas.
        power = (before-c.toroidal*c.toroidal)/(2*dt);
      } else {
        // Local induction dB_phi/dt = r dOmega/dr B_r (field magnitudes).
        c.toroidal += c.shear*c.radial*dt;
        c.reconnecting = c.toroidal >= c.threshold;
      }
      const cos = Math.cos(c.theta), sin = Math.sin(c.theta);
      uniforms.set([c.r*cos, c.r*sin, cos, sin, c.width, c.length, power, c.toroidal/c.threshold], i*8);
      if (c.reconnecting && c.toroidal < 0.045*c.threshold) cells[i] = cell(false);
    }
    return uniforms;
  }
  reset();
  return {count, reset, advance};
}
