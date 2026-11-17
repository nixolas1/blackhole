// Sample new density fluctuations from the opening's azimuthal spectrum.
// Each harmonic has correlated amplitude and phase across disk radii. The
// GPU advects the resulting fields and mixes overlapping generations.
function createTurbulence(gl, model, startTime) {
  const {angles, radii, harmonics, components, levels, interval} = model;
  const bytes = Uint8Array.from(atob(model.packed), c => c.charCodeAt(0));
  const plane = radii * angles;
  const endpoint = bytes.slice(levels*plane, (levels+1)*plane);
  const basis = new Float32Array(bytes.slice((levels+1)*plane).buffer);
  const fields = [];
  for (let unit = 5; unit <= 7; unit++) {
    const target = unit === 7 ? gl.TEXTURE_3D : gl.TEXTURE_2D;
    gl.activeTexture(gl.TEXTURE0+unit);
    const tex = gl.createTexture();
    gl.bindTexture(target, tex);
    gl.texParameteri(target, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(target, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(target, gl.TEXTURE_WRAP_S, gl.REPEAT);
    gl.texParameteri(target, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    if (unit === 7) {
      gl.texParameteri(target, gl.TEXTURE_WRAP_R, gl.CLAMP_TO_EDGE);
      gl.texImage3D(target, 0, gl.R8, angles, radii, levels, 0, gl.RED, gl.UNSIGNED_BYTE, bytes.subarray(0, levels*plane));
    } else {
      fields.push(tex);
      gl.texImage2D(target, 0, gl.R8, angles, radii, 0, gl.RED, gl.UNSIGNED_BYTE, null);
    }
  }
  const cosines = new Float32Array(angles*harmonics);
  const sines = new Float32Array(angles*harmonics);
  for (let a = 0; a < angles; a++) for (let m = 0; m < harmonics; m++) {
    const phi = 2*Math.PI*a*m/angles;
    cosines[a*harmonics+m] = Math.cos(phi);
    sines[a*harmonics+m] = Math.sin(phi);
  }
  let seed, previous, launches;
  const real = new Float32Array(radii*harmonics);
  const imag = new Float32Array(radii*harmonics);
  const encoded = new Uint8Array(plane);
  function random() {
    seed = (Math.imul(1664525, seed) + 1013904223) >>> 0;
    return (seed + 0.5)/4294967296;
  }
  function generate() {
    real.fill(0);
    imag.fill(0);
    for (let m = 0; m < harmonics; m++) for (let k = 0; k < components; k++) {
      const phi = 2*Math.PI*random();
      const magnitude = Math.sqrt(-2*Math.log(random()))*(m ? Math.SQRT2 : 1);
      const x = magnitude*Math.cos(phi), y = m ? magnitude*Math.sin(phi) : 0;
      for (let r = 0; r < radii; r++) {
        const i = 2*((m*components+k)*radii+r);
        real[r*harmonics+m] += x*basis[i] - y*basis[i+1];
        imag[r*harmonics+m] += x*basis[i+1] + y*basis[i];
      }
    }
    for (let r = 0; r < radii; r++) for (let a = 0; a < angles; a++) {
      let value = 0;
      for (let m = 0; m < harmonics; m++) {
        value += real[r*harmonics+m]*cosines[a*harmonics+m] - imag[r*harmonics+m]*sines[a*harmonics+m];
      }
      encoded[r*angles+a] = Math.round(Math.max(0,Math.min(255,128+40*value)));
    }
    return encoded;
  }
  function upload(index, values) {
    gl.activeTexture(gl.TEXTURE5 + index);
    gl.bindTexture(gl.TEXTURE_2D, fields[index]);
    gl.texSubImage2D(gl.TEXTURE_2D,0,0,0,angles,radii,gl.RED,gl.UNSIGNED_BYTE,values);
  }
  function reset() {
    seed = 731201;
    previous = 0;
    launches = 0;
    upload(0, endpoint);
    upload(1, generate());
  }
  function setTime(seconds, uniform) {
    const elapsed = Math.max(0, seconds-startTime);
    const launch = Math.floor(elapsed/interval);
    while (launches < launch) {
      previous = 1-previous;
      upload(1-previous, generate());
      launches++;
    }
    const age = elapsed - launch*interval;
    gl.uniform4f(uniform, age+(launch ? interval : 0), age, age/interval, previous);
  }
  return {reset,setTime};
}
