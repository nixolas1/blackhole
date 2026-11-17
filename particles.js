// Persistent disk-plane positions, velocities, thermal emission and parcel masses.
const particleCounts = [125000, 250000, 500000];
const defaultParticleCount = particleCounts[0];

async function createAccretionRenderer(canvas, model, initialCount=defaultParticleCount) {
  const gl = canvas.getContext('webgl2', {alpha:false, antialias:false, depth:false});
  if (!gl || !gl.getExtension('EXT_color_buffer_float'))
    throw new Error('The simulation requires WebGL 2 with floating-point render targets.');
  if (!particleCounts.includes(initialCount)) throw new RangeError('Unsupported particle count');
  let count = initialCount;
  const horizon = 9, fiducialRadius = 84;
  const speeds = model.patternSpeeds;
  const upper = speeds.findIndex(([r]) => r >= fiducialRadius);
  const [r1,w1] = speeds[upper-1], [r2,w2] = speeds[upper];
  const omega = w1+(w2-w1)*(fiducialRadius-r1)/(r2-r1);
  const gravity = omega**2*fiducialRadius*(fiducialRadius-horizon)**2;
  const magnetic = createMagneticField(gravity, horizon);

  function program(vertex, fragment, outputs) {
    const p = gl.createProgram();
    for (const [stage, source] of [[gl.VERTEX_SHADER,vertex],[gl.FRAGMENT_SHADER,fragment]]) {
      const shader = gl.createShader(stage);
      gl.shaderSource(shader, '#version 300 es\nprecision highp float;\n'+source);
      gl.compileShader(shader);
      if (!gl.getShaderParameter(shader,gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(shader));
      gl.attachShader(p,shader);
      gl.deleteShader(shader);
    }
    if (outputs) gl.transformFeedbackVaryings(p,outputs,gl.INTERLEAVED_ATTRIBS);
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p,gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
    const uniforms = {};
    for (let i=0;i<gl.getProgramParameter(p,gl.ACTIVE_UNIFORMS);i++) {
      const name = gl.getActiveUniform(p,i).name;
      uniforms[name.replace(/\[0\]$/,'')] = gl.getUniformLocation(p,name);
    }
    return {p,u:uniforms};
  }
  function texture(unit, target, wrap=false) {
    gl.activeTexture(gl.TEXTURE0+unit);
    const tex = gl.createTexture();
    gl.bindTexture(target,tex);
    gl.texParameteri(target,gl.TEXTURE_MIN_FILTER,gl.LINEAR);
    gl.texParameteri(target,gl.TEXTURE_MAG_FILTER,gl.LINEAR);
    gl.texParameteri(target,gl.TEXTURE_WRAP_S,wrap ? gl.REPEAT : gl.CLAMP_TO_EDGE);
    gl.texParameteri(target,gl.TEXTURE_WRAP_T,gl.CLAMP_TO_EDGE);
    return tex;
  }
  function sharpen(bytes, channels) {
    // Unsharp mask over a few source pixels, with periodic angular neighbors.
    const n=model.angles, rows=model.radii, original=bytes.slice();
    const layers=bytes.length/(n*rows*channels);
    for(let f=0;f<layers;f++) for(let r=0;r<rows;r++) for(let a=0;a<n;a++) {
      const row=(f*rows+r)*n, i=(row+a)*channels;
      for(let c=0;c<(channels===4?2:1);c++) {
        const mid=original[i+c];
        const blur=(original[(row+(a+2)%n)*channels+c]+original[(row+(a+n-2)%n)*channels+c]
          +original[i+c+Math.min(3,rows-1-r)*n*channels]+original[i+c-Math.min(3,r)*n*channels])/4;
        bytes[i+c]=Math.min(255,Math.max(0,Math.round(mid+0.68*(mid-blur))));
      }
    }
  }

  const data=Uint8Array.from(atob(model.packed), c=>c.charCodeAt(0));
  const packed=Uint8Array.from(atob(accretionField.packed), c=>c.charCodeAt(0));
  const stream=new Blob([packed]).stream().pipeThrough(new DecompressionStream('gzip'));
  const opening=new Uint8Array(await new Response(stream).arrayBuffer()).subarray(768);
  gl.pixelStorei(gl.UNPACK_ALIGNMENT,1);
  sharpen(data.subarray(768),4);
  const openingFrames=model.openingFrames;
  const physicalOpening=opening.subarray(0,openingFrames*model.angles*model.radii*3);
  sharpen(physicalOpening,3);
  // A brightness pattern can propagate through the rotating gas. Store its
  // measured radial drift in the spare channel of the static disk texture.
  for (let row=0; row<model.radii; row++) {
    const r=2*Math.min(row,128)+5*Math.max(0,row-128);
    const band=Math.max(1,speeds.findIndex(([radius])=>radius>=r));
    let ratio=1;
    if (r>=speeds[0][0] && r<=speeds.at(-1)[0]) {
      const [lo,left]=speeds[band-1], [hi,right]=speeds[band];
      const pattern=left+(right-left)*(r-lo)/(hi-lo);
      const circular=Math.sqrt(gravity/r)/(r-horizon);
      ratio=Math.max(1.05,pattern/circular*1.035);
    } else if(r<speeds[0][0]) ratio=1.05;
    else {
      const [edge,rate]=speeds.at(-1);
      ratio=1.035*rate*(edge-horizon)/Math.sqrt(gravity/edge);
    }
    for (let a=0; a<model.angles; a++) data[768+4*(row*model.angles+a)+3]=Math.round(127*ratio);
  }
  texture(0,gl.TEXTURE_2D,true);
  gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA8,model.angles,model.radii,0,gl.RGBA,gl.UNSIGNED_BYTE,data.subarray(768));
  texture(1,gl.TEXTURE_2D);
  gl.texImage2D(gl.TEXTURE_2D,0,gl.RGB8,256,1,0,gl.RGB,gl.UNSIGNED_BYTE,data.subarray(0,768));
  const light=texture(2,gl.TEXTURE_2D);
  const lightTarget=gl.createFramebuffer();
  texture(3,gl.TEXTURE_2D_ARRAY,true);
  gl.texImage3D(gl.TEXTURE_2D_ARRAY,0,gl.RGB8,model.angles,model.radii,openingFrames,0,gl.RGB,gl.UNSIGNED_BYTE,physicalOpening);
  const lastPhysicalFrame=openingFrames-1;
  const duration=lastPhysicalFrame/accretionField.fps;
  const turbulence=createTurbulence(gl,turbulenceModel,duration);
  const mapSize=192, mapExtent=256;
  const heating=texture(4,gl.TEXTURE_2D);
  gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA16F,mapSize,mapSize,0,gl.RGBA,gl.HALF_FLOAT,null);
  const heatTarget=gl.createFramebuffer();
  gl.bindFramebuffer(gl.FRAMEBUFFER,heatTarget);
  gl.framebufferTexture2D(gl.FRAMEBUFFER,gl.COLOR_ATTACHMENT0,gl.TEXTURE_2D,heating,0);

  const density=texture(8,gl.TEXTURE_2D);
  const floatMass=gl.getExtension('EXT_float_blend') && gl.getExtension('OES_texture_float_linear');
  gl.texImage2D(gl.TEXTURE_2D,0,floatMass ? gl.R32F : gl.R16F,mapSize,mapSize,0,gl.RED,floatMass ? gl.FLOAT : gl.HALF_FLOAT,null);
  const densityTarget=gl.createFramebuffer();
  gl.bindFramebuffer(gl.FRAMEBUFFER,densityTarget);
  gl.framebufferTexture2D(gl.FRAMEBUFFER,gl.COLOR_ATTACHMENT0,gl.TEXTURE_2D,density,0);

  // Each parcel deposits its mass into the disk plane. A uniform sheet has
  // density one regardless of camera zoom, inclination or display resolution.
  const measureDensity=program(`
    layout(location=0) in vec4 orbit;
    layout(location=1) in vec4 parcel;
    flat out float massPerArea;
    void main(){
      float r=length(orbit.xy);
      float sigma=6.5+4.0*smoothstep(140.0,240.0,r);
      massPerArea=parcel.y/(0.9510565*6.283185307*sigma*sigma);
      gl_PointSize=6.0*sigma*${mapSize}.0/${2*mapExtent}.0;
      gl_Position=vec4(orbit.xy/${mapExtent}.0,0,1);
      if(r>280.0) gl_Position=vec4(3,3,0,1);
    }`,`
    flat in float massPerArea;
    out vec4 mass;
    void main(){
      vec2 q=(gl_PointCoord-0.5)*6.0;
      mass=vec4(massPerArea*exp(-0.5*dot(q,q)),0,0,0);
    }`);

  // Project each sheet and its outward pressure gradient onto the disk.
  const sheets=program(`
    uniform vec4 cells[${magnetic.count*2}];
    out vec2 uv;
    flat out vec2 radial, widths;
    flat out float power, shear;
    void main(){
      vec4 a=cells[gl_InstanceID*2], b=cells[gl_InstanceID*2+1];
      radial=a.zw; widths=b.xy; power=b.z; shear=-0.8*b.w;
      uv=3.25*(2.0*vec2(gl_VertexID&1,gl_VertexID>>1)-1.0);
      vec2 tangent=vec2(-radial.y,radial.x);
      vec2 pos=a.xy+radial*uv.x*widths.x+tangent*(uv.y*widths.y+shear*uv.x*widths.x);
      gl_Position=vec4(pos/${mapExtent}.0,0,1);
      if(power==0.0) gl_Position=vec4(3,3,0,1);
    }`,`
    in vec2 uv;
    flat in vec2 radial,widths;
    flat in float power,shear;
    out vec4 outHeat;
    void main(){
      float sheath=exp(-0.5*dot(uv,uv));
      // A narrow current sheet deposits heat inside a broader exhaust.
      vec2 core=uv*vec2(2.8,1.25);
      float sheet=exp(-0.5*dot(core,core));
      float q=power*(0.72*sheath+0.93*sheet);
      vec2 tangent=vec2(-radial.y,radial.x);
      vec2 g=power*uv*(vec2(0.72*sheath)+0.93*sheet*vec2(2.8*2.8,1.25*1.25))/widths;
      vec2 pressure=radial*(g.x-shear*g.y)+tangent*g.y;
      outHeat=vec4(q,pressure,0);
    }`);
  const stateInputs=`
    layout(location=0) in vec4 orbit;
    layout(location=1) in vec4 parcel; // turbulent emission, parcel area, material variation, generation
    layout(location=2) in vec2 thermal; // log density at last step, log temperature relative to ambient
  `;
  const evolve=program(stateInputs+`
    uniform sampler2D disk, deposition, surfaceMass;
    uniform highp sampler2DArray guide;
    uniform sampler2D eddyA, eddyB;
    uniform highp sampler3D quantiles;
    uniform vec4 eddyTime; // two material ages, fractional renewal, buffer parity
    uniform vec4 cue; // frame A, frame B, fraction, opening active
    uniform float dt, freshParcelArea;
    out vec4 nextOrbit,nextParcel;
    out vec2 nextThermal;
    float logDensity(vec2 q){
      float d=textureLod(surfaceMass,q/${2*mapExtent}.0+0.5,0.0).r;
      return log(clamp(d,0.1,12.0))*(1.0-smoothstep(210.0,245.0,length(q)));
    }
    const float gm=${gravity.toFixed(6)}, rs=${horizon}.0;
    vec2 force(vec2 q){
      float r=max(length(q),20.0), d=r-rs;
      return -gm*q/(r*d*d);
    }
    float hash(uint v){
      v ^= v>>16; v*=0x21f0aaadu; v^=v>>15; v*=0x735a2d97u; v^=v>>15;
      return float(v>>8)*(1.0/16777216.0);
    }
    vec2 polar(vec2 p){
      float r=length(p);
      return vec2(atan(p.y,p.x)/6.283185307+0.5/${model.angles}.0,
        (min(r,256.0)*0.5+max(r-256.0,0.0)*0.2+0.5)/${model.radii}.0);
    }
    void main(){
      vec2 q=orbit.xy, v=orbit.zw;
      float radius=max(length(q),20.0);
      vec3 deposit=textureLod(deposition,q/${mapExtent*2}.0+0.5,0.0).rgb;
      // Symmetric gravity kicks, gas pressure, and slow angular momentum transport.
      float inflow=0.46*sqrt(84.0/radius);
      // A steady viscous inflow loses angular momentum at v_r dL_circular/dr.
      float drag=inflow*max(0.08,1.5-radius/(radius-rs))/radius;
      // Collisional relaxation damps epicycles relative to the mean circular flow.
      vec2 normal=q/radius;
      vec2 circular=vec2(-normal.y,normal.x)*sqrt(gm*radius)/(radius-rs);
      float plunge=1.0-smoothstep(28.0,40.0,radius);
      // Taper the circular-flow closure at the inner edge. Residual magnetic
      // stress removes support there and allows gas to plunge across the sink.
      v=mix(v,circular-inflow*normal,1.0-exp(-0.55*(1.0-plunge)*dt));
      drag+=0.22*plunge;
      v+=0.5*dt*force(q);
      v*=exp(-dt*drag);
      // The local pressure gradient resists further compression of crowded gas.
      float h=${2*mapExtent/mapSize};
      vec2 gx=vec2(h,0);
      vec2 grad=vec2(logDensity(q+gx)-logDensity(q-gx),logDensity(q+gx.yx)-logDensity(q-gx.yx))/(2.0*h);
      v+=dt*(deposit.gb*2.0-13.0*exp(thermal.y)*grad);
      q+=dt*v;
      v+=0.5*dt*force(q);
      float r=length(q), generation=parcel.w, material=parcel.z, area=parcel.y;
      float temp=parcel.x;
      bool replaced=dt>0.0 && (r<27.5 || r>${model.extent*1.18});
      if(replaced){
        // Gas crossing the inner boundary is replaced by fresh outer supply.
        generation=mod(generation+1.0,65536.0);
        uint key=uint(gl_VertexID)+uint(generation)*747796405u;
        float a=6.283185307*hash(key+13u);
        r=201.0+330.0*pow(hash(key+35u),3.4);
        vec2 n=vec2(cos(a),sin(a)); q=r*n;
        v=vec2(-n.y,n.x)*sqrt(gm*r)/(r-rs)-0.46*sqrt(84.0/r)*n;
        material=hash(key+71u); temp=0.0; area=freshParcelArea;
      }
      vec2 p=q*vec2(1,0.9510565);
      vec2 uv=polar(p);
      vec4 fit=textureLod(disk,uv,0.0);
      float base=fit.g;
      float contrast=max(0.02,0.5*fit.b);
      float ambient;
      if(cue.w>0.5){
        vec3 a=textureLod(guide,vec3(uv,cue.x),0.0).rgb;
        vec2 flow=(a.gb*255.0-128.0)/4.0;
        float start=textureLod(guide,vec3(polar(p-cue.z*flow),cue.x),0.0).r;
        float end=textureLod(guide,vec3(polar(p+(1.0-cue.z)*flow),cue.y),0.0).r;
        ambient=mix(start,end,cue.z);
      } else {
        float projectedR=length(p);
        float bin=min(projectedR,256.0)*0.5+max(projectedR-256.0,0.0)*0.2;
        vec2 grid=vec2(atan(p.y,p.x)/6.283185307+0.5/${turbulenceModel.angles}.0,
                      (bin*0.5+0.5)/${turbulenceModel.radii}.0);
        // A material eddy shears at the parcel's angular velocity. The texture
        // contains zero-mean unit-scale fluctuations, independent of viewing angle.
        float omega=clamp((q.x*v.y-q.y*v.x)/max(r*r,1.0),0.15,3.6);
        vec2 travel=omega*(fit.a*255.0/127.0)*eddyTime.xy/6.283185307;
        bool swapped=eddyTime.w>0.5;
        float a=textureLod(eddyA,grid-vec2(swapped ? travel.y : travel.x,0),0.0).r;
        float b=textureLod(eddyB,grid-vec2(swapped ? travel.x : travel.y,0),0.0).r;
        vec2 fluctuations=((swapped ? vec2(b,a) : vec2(a,b))*255.0-128.0)/40.0;
        float phase=eddyTime.z*1.570796327;
        float z=dot(fluctuations,vec2(cos(phase),sin(phase)));
        float level=clamp(0.5+z/(2.0*${turbulenceModel.limit}),0.0,1.0);
        ambient=textureLod(quantiles,vec3(grid,(level*${turbulenceModel.levels-1}.0+0.5)/${turbulenceModel.levels}.0),0.0).r;
      }
      // Both the opening and the continuation heat and cool the same parcel.
      float tc=0.037+0.018*material;
      float equilibrium=(ambient-base)/contrast;
      float logRho=logDensity(q);
      // Ideal-gas compressional work along the moving parcel: d ln T =
      // (gamma-1) d ln rho. Surface density assumes a fixed local scale height.
      float dLogRho=clamp(logRho-thermal.x,-0.35,0.35);
      float cooling=0.36+0.15*material;
      float keep=exp(-dt/cooling);
      float specificPower=0.075*deposit.r*exp(-0.5*logRho);
      float ratio=exp(thermal.y+0.5*dLogRho);
      ratio=1.0+(ratio-1.0)*keep+specificPower*cooling*(1.0-keep);
      nextThermal=vec2(logRho,(dt==0.0 || replaced) ? 0.0 : log(clamp(ratio,0.42,3.5)));
      float next=equilibrium+(temp-equilibrium)*exp(-dt/tc);
      if(dt==0.0) next=(fit.r-base)/contrast;
      nextOrbit=vec4(q,v);
      nextParcel=vec4(clamp(next,-9.0,12.0),area,material,generation);
      gl_Position=vec4(0);
    }`,'out vec4 c; void main(){c=vec4(0);}', ['nextOrbit','nextParcel','nextThermal']);
  const gas=program(stateInputs+`
    uniform vec2 clipScale,centerOffset;
    uniform float pixelsPerUnit,kernelWidth;
    uniform sampler2D disk;
    flat out float heat,areaWeight;
    void main(){
      vec2 p=orbit.xy*vec2(1,0.9510565);
      float radius=length(p);
      float sigma=kernelWidth*(1.0+max(0.0,radius-180.0)/95.0);
      float size=max(2.0,6.0*sigma*pixelsPerUnit);
      areaWeight=parcel.y/(6.2831853*pow(size/(6.0*pixelsPerUnit),2.0));
      float radialBin=min(radius,256.0)*0.5+max(radius-256.0,0.0)*0.2;
      vec2 uv=vec2(atan(p.y,p.x)/6.283185307+0.5/${model.angles}.0,(radialBin+0.5)/${model.radii}.0);
      vec4 fit=textureLod(disk,uv,0.0);
      // Keep the persistent viewing asymmetry in the camera frame, while the
      // thermal excess travels with the gas passing through its bright side.
      heat=max(0.0,fit.g+max(0.02,0.5*fit.b)*parcel.x)*(0.99+0.02*parcel.z);
      // The observed palette maps emissivity into display color. Denser gas
      // adds emitters, and a hotter parcel emits more in that same palette.
      heat*=exp(0.14*tanh(thermal.x)+0.49*tanh(thermal.y));
      gl_Position=vec4((p+centerOffset)*clipScale,0,1);
      gl_PointSize=size;
      if(heat<0.002) gl_Position=vec4(3,3,0,1);
    }`,`
    flat in float heat,areaWeight;
    out vec4 energy;
    void main(){
      vec2 q=(gl_PointCoord-0.5)*6.0;
      float mass=exp(-0.5*dot(q,q))*areaWeight;
      energy=vec4(heat*mass,mass,0,0);
    }`);
  const resolve=program(`
    void main(){
      vec2 c=vec2((gl_VertexID<<1)&2,gl_VertexID&2);
      gl_Position=vec4(c*2.0-1.0,0,1);
    }`,`
    uniform sampler2D energy,palette;

    out vec4 color;
    void main(){
      vec2 gas=texelFetch(energy,ivec2(gl_FragCoord.xy),0).rg;
      float temp=gas.r/max(gas.g,0.08);

      // Soft shoulder keeps structure inside a newly heated, white-hot knot.
      if(temp>0.92) temp=0.92+0.08*(1.0-exp(-(temp-0.92)/0.08));
      color=vec4(texture(palette,vec2((temp*255.0+0.5)/256.0,0.5)).rgb,1);
    }`);
  gl.useProgram(evolve.p); gl.uniform1i(evolve.u.disk,0); gl.uniform1i(evolve.u.guide,3); gl.uniform1i(evolve.u.deposition,4);
  gl.uniform1i(evolve.u.eddyA,5); gl.uniform1i(evolve.u.eddyB,6); gl.uniform1i(evolve.u.quantiles,7); gl.uniform1i(evolve.u.surfaceMass,8);
  gl.useProgram(gas.p); gl.uniform1i(gas.u.disk,0);
  gl.useProgram(resolve.p); gl.uniform1i(resolve.u.energy,2); gl.uniform1i(resolve.u.palette,1);

  const stride=10;
  function seedDisk(){
    const initial=new Float32Array(count*stride);
    const bands=count===500000 ? 500 : 250, perBand=count/bands;
    const outer=model.extent*1.08;
    const span=180+95*Math.log1p((outer-180)/95);
    function radial(s){ return s<=180?s:180+95*Math.expm1((s-180)/95); }
    for(let i=0;i<count;i++){
      const band=Math.floor(i/perBand), j=i%perBand;
      const lo=radial(band*span/bands), hi=radial((band+1)*span/bands);
      const stagger=(j*0.618033989+band*0.414213562)%1;
      const r=lo+(hi-lo)*(0.04+0.92*stagger);
      const a=2*Math.PI*(j/perBand+band*0.618033989);
      const x=Math.cos(a), y=Math.sin(a);
      const v=Math.sqrt(gravity*Math.max(r,20))/(Math.max(r,20)-horizon);
      const mat=0.5+0.5*Math.sin(2.8*r+2.3*Math.sin(3*a+r*0.07));
      initial.set([r*x,r*y,-y*v-0.16*x,x*v-0.16*y,0,
        Math.PI*(hi*hi-lo*lo)*0.9510565/perBand,mat,0,0,0],i*stride);
    }
    return initial;
  }
  const buffers=[], vaos=[];
  for(let i=0;i<2;i++){
    const buf=gl.createBuffer(), vao=gl.createVertexArray();
    buffers.push(buf); vaos.push(vao);
    gl.bindVertexArray(vao); gl.bindBuffer(gl.ARRAY_BUFFER,buf);
    for(let j=0;j<3;j++){
      gl.enableVertexAttribArray(j);
      gl.vertexAttribPointer(j,j===2 ? 2 : 4,gl.FLOAT,false,stride*4,j*16);
    }
  }
  gl.bindVertexArray(null);
  let front=0, seconds=0, width=0, height=0;
  function integrate(dt){
    gl.useProgram(evolve.p);
    gl.uniform1f(evolve.u.dt,dt);
    gl.uniform1f(evolve.u.freshParcelArea,1.8*131072/count);
    const frame=Math.min(seconds*accretionField.fps,lastPhysicalFrame);
    const guide=seconds<=duration ? 1 : 0;
    turbulence.setTime(seconds,evolve.u.eddyTime);
    gl.uniform4f(evolve.u.cue,Math.floor(frame),Math.min(Math.floor(frame)+1,lastPhysicalFrame),frame%1,guide);
    gl.enable(gl.RASTERIZER_DISCARD);
    gl.bindVertexArray(vaos[front]);
    gl.bindBufferBase(gl.TRANSFORM_FEEDBACK_BUFFER,0,buffers[1-front]);
    gl.beginTransformFeedback(gl.POINTS);
    gl.drawArrays(gl.POINTS,0,count);
    gl.endTransformFeedback();
    gl.bindBufferBase(gl.TRANSFORM_FEEDBACK_BUFFER,0,null);
    gl.bindVertexArray(null);
    gl.disable(gl.RASTERIZER_DISCARD);
    front=1-front;
  }
  let massStep=0;
  function measureMass(){
    gl.bindFramebuffer(gl.FRAMEBUFFER,densityTarget);
    gl.viewport(0,0,mapSize,mapSize);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.useProgram(measureDensity.p);
    gl.enable(gl.BLEND); gl.blendFunc(gl.ONE,gl.ONE);
    gl.bindVertexArray(vaos[front]); gl.drawArrays(gl.POINTS,0,count);
    gl.bindVertexArray(null); gl.disable(gl.BLEND);
    gl.bindFramebuffer(gl.FRAMEBUFFER,null);
  }
  function loadState(state){
    front=0;
    gl.bindBuffer(gl.ARRAY_BUFFER,buffers[0]);
    gl.bufferData(gl.ARRAY_BUFFER,state,gl.DYNAMIC_COPY);
    gl.bindBuffer(gl.ARRAY_BUFFER,buffers[1]);
    gl.bufferData(gl.ARRAY_BUFFER,state.byteLength,gl.DYNAMIC_COPY);
    gl.bindBuffer(gl.ARRAY_BUFFER,null);
    massStep=0;
    measureMass();
  }
  function setParticleCount(nextCount){
    if (!particleCounts.includes(nextCount)) throw new RangeError('Unsupported particle count');
    if (nextCount===count) return;
    if (seconds===0) {
      count=nextCount;
      reset();
      return;
    }
    const current=new Float32Array(count*stride);
    gl.bindBuffer(gl.ARRAY_BUFFER,buffers[front]);
    gl.getBufferSubData(gl.ARRAY_BUFFER,0,current);
    gl.bindBuffer(gl.ARRAY_BUFFER,null);
    const next=new Float32Array(nextCount*stride);
    if(nextCount>count){
      const copies=nextCount/count;
      for(let i=0;i<count;i++){
        const source=i*stride;
        const radius=Math.min(0.8,0.42*Math.sqrt(current[source+5]));
        const phase=2*Math.PI*((i*0.618033989)%1);
        for(let j=0;j<copies;j++){
          const dest=(i*copies+j)*stride;
          for(let k=0;k<stride;k++) next[dest+k]=current[source+k];
          const phi=phase+2*Math.PI*j/copies;
          next[dest]+=radius*Math.cos(phi);
          next[dest+1]+=radius*Math.sin(phi);
          next[dest+5]/=copies;
        }
      }
    } else {
      // Select a mass-weighted representative per group. Preserve group mass;
      // averaging positions across separated orbits would put gas in the shadow.
      const group=count/nextCount;
      for(let i=0;i<nextCount;i++){
        const start=i*group*stride;
        let mass=0;
        for(let j=0;j<group;j++) mass+=current[start+j*stride+5];
        let offset=mass*((0.5+i*0.618033989)%1), source=start;
        while(source<start+(group-1)*stride && offset>current[source+5]){
          offset-=current[source+5];
          source+=stride;
        }
        for(let k=0;k<stride;k++) next[i*stride+k]=current[source+k];
        next[i*stride+5]=mass;
      }
    }
    count=nextCount;
    loadState(next);
  }
  function reset(){
    seconds=0; front=0; magnetic.reset(); turbulence.reset();
    gl.bindFramebuffer(gl.FRAMEBUFFER,heatTarget); gl.clear(gl.COLOR_BUFFER_BIT);
    gl.bindFramebuffer(gl.FRAMEBUFFER,null);
    loadState(seedDisk()); integrate(0);
  }
  function advance(dt){
    seconds+=dt;
    if(massStep===0) measureMass();
    massStep=(massStep+1)%3;
    const cells=magnetic.advance(dt);
    gl.bindFramebuffer(gl.FRAMEBUFFER,heatTarget); gl.viewport(0,0,mapSize,mapSize);
    gl.clear(gl.COLOR_BUFFER_BIT); gl.useProgram(sheets.p);
    gl.uniform4fv(sheets.u.cells,cells); gl.enable(gl.BLEND); gl.blendFunc(gl.ONE,gl.ONE);
    gl.drawArraysInstanced(gl.TRIANGLE_STRIP,0,4,magnetic.count);
    gl.disable(gl.BLEND); gl.bindFramebuffer(gl.FRAMEBUFFER,null);
    integrate(dt);
  }
  function draw(fieldSize,w,h){
    if(w!==width || h!==height){
      canvas.width=width=w; canvas.height=height=h;
      gl.activeTexture(gl.TEXTURE2); gl.bindTexture(gl.TEXTURE_2D,light);
      gl.texImage2D(gl.TEXTURE_2D,0,gl.RG16F,w,h,0,gl.RG,gl.HALF_FLOAT,null);
      gl.bindFramebuffer(gl.FRAMEBUFFER,lightTarget);
      gl.framebufferTexture2D(gl.FRAMEBUFFER,gl.COLOR_ATTACHMENT0,gl.TEXTURE_2D,light,0);
      if(gl.checkFramebufferStatus(gl.FRAMEBUFFER)!==gl.FRAMEBUFFER_COMPLETE)
        throw new Error('Cannot allocate the particle render target.');
    }
    const scale=Math.min(w,h)/fieldSize;
    gl.viewport(0,0,w,h); gl.bindFramebuffer(gl.FRAMEBUFFER,lightTarget);
    gl.clear(gl.COLOR_BUFFER_BIT); gl.useProgram(gas.p);
    gl.uniform2f(gas.u.clipScale,2*scale/w,-2*scale/h);
    gl.uniform2f(gas.u.centerOffset,model.center[0]-model.fieldCenter[0],model.center[1]-model.fieldCenter[1]);
    gl.uniform1f(gas.u.kernelWidth,Math.max(1.65,2.6*Math.pow(131072/count,0.22)));
    gl.uniform1f(gas.u.pixelsPerUnit,scale); gl.enable(gl.BLEND); gl.blendFunc(gl.ONE,gl.ONE);
    gl.bindVertexArray(vaos[front]); gl.drawArrays(gl.POINTS,0,count);
    gl.bindVertexArray(null); gl.disable(gl.BLEND);
    gl.bindFramebuffer(gl.FRAMEBUFFER,null); gl.useProgram(resolve.p);
    gl.drawArrays(gl.TRIANGLES,0,3);
  }
  reset();
  return {reset,advance,draw,setParticleCount};
}
