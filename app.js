// Earth Song - a spinning globe that plays its elevation as music.
//
// data/elev.bin    Int16 metres, the full native ETOPO 10 arc-min grid
//                  (1081 x 2161), used as ONE texture for displacement,
//                  colour and per-pixel normals.
// data/voices.bin  pluck-engine control tables per 10' longitude column
//                  (note, amplitude, relief weight per latitude voice; ocean
//                  drone). Same tables as audio F2 of reel 174.
// data/meta.json   ramp stops, engine constants, lights.
// All three come from ../tools/export_webapp_data.py.
import * as THREE from "three";

// ---------------------------------------------------------------- tuning --
// Steady rotation: one turn per 60 s (6 deg/s), the slow end of globe viewers
// and the exact pace of the 60 s reel cut.
const AUTO_RATE = 360 / 60;
// Arrow hold: 8 deg/s at once, eased in-out (cubic) to 180 deg/s over 3 s;
// omega chases the target with a 0.15 s time constant.
const ARROW_START = 8, ARROW_MAX = 180, ARROW_RAMP_S = 3.0, ARROW_TAU = 0.15;
// Release / fling: exponential relaxation to rest (or to AUTO_RATE) with a
// 0.35 s time constant (1.0 s felt too slow to settle).
const RELAX_TAU = 0.35;
// Fling ceiling: 9 deg per frame at 60 fps; faster strobes.
const FLING_MAX = 540;
// Lighting: the kit's studio rig, gains tuned by eye against the reel.
const LIGHT_GAINS = [1.6, 0.4, 0.4], AMBIENT = 0.04;

const $ = (id) => document.getElementById(id);
const clamp = (x, a, b) => Math.min(b, Math.max(a, x));
const smoothstep = (a, b, x) => { const t = clamp((x - a) / (b - a), 0, 1); return t * t * (3 - 2 * t); };
const easeInOutCubic = (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);
const DEG = Math.PI / 180;
const DATA_V = 3;   // bump with the data files so browsers do not use stale copies

function fail(msg) {
  const e = $("err");
  e.textContent = msg;
  e.style.display = "grid";
}

// ------------------------------------------------------------- the globe --
// One texture carries the elevation. The vertex shader displaces a dense
// sphere by max(e, 0) (sea geometry clamped, so the ocean is a true sphere);
// the fragment shader colours every PIXEL from the raw, unclamped elevation
// (so shelves and ridges show) with the kit's ramp pinned to metres, and
// derives a per-pixel normal from the clamped relief. Lights are fixed in
// world space, so shading changes as the globe turns.
const VERT = /* glsl */ `
  uniform sampler2D uElev;
  uniform float uExag;
  varying vec2 vUv;
  varying vec3 vDir;
  void main() {
    float e = texture2D(uElev, uv).r;
    vUv = uv;
    vDir = position;
    vec3 p = position * (1.0 + max(e, 0.0) * uExag);
    gl_Position = projectionMatrix * modelViewMatrix * vec4(p, 1.0);
  }
`;

function fragShader(nStops) {
  return /* glsl */ `
  uniform sampler2D uElev;
  uniform float uExag;
  uniform vec2 uTexel;
  uniform mat3 uRot;
  uniform vec3 uLightDir[3];
  uniform float uLightI[3];
  uniform float uAmbient;
  uniform float uStopM[${nStops}];
  uniform vec3 uStopC[${nStops}];
  varying vec2 vUv;
  varying vec3 vDir;

  vec3 toLin(vec3 c) {
    return mix(c / 12.92, pow((c + 0.055) / 1.055, vec3(2.4)), step(0.04045, c));
  }
  vec3 toSrgb(vec3 c) {
    c = clamp(c, 0.0, 1.0);
    return mix(c * 12.92, 1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055, step(0.0031308, c));
  }
  // piecewise-linear in sRGB between metre-pinned stops (matplotlib-style)
  vec3 ramp(float e) {
    vec3 c = uStopC[0];
    for (int k = 1; k < ${nStops}; k++) {
      float t = clamp((e - uStopM[k - 1]) / (uStopM[k] - uStopM[k - 1]), 0.0, 1.0);
      if (e >= uStopM[k - 1]) c = mix(uStopC[k - 1], uStopC[k], t);
    }
    return c;
  }
  float h(vec2 uv) { return max(texture2D(uElev, uv).r, 0.0) * uExag; }

  void main() {
    float e = texture2D(uElev, vUv).r;
    vec3 albedo = toLin(ramp(e));

    // per-pixel normal from the CLAMPED relief (sea stays a smooth sphere)
    vec3 d = normalize(vDir);
    float lat = asin(clamp(d.y, -1.0, 1.0));
    float lon = atan(d.x, d.z);
    vec3 east = vec3(cos(lon), 0.0, -sin(lon));
    vec3 north = vec3(-sin(lat) * sin(lon), cos(lat), -sin(lat) * cos(lon));
    float du = max(uTexel.x, fwidth(vUv.x));
    float dv = max(uTexel.y, fwidth(vUv.y));
    const float RAD_U = 6.2831853 * 2161.0 / 2160.0;
    const float RAD_V = 3.1415927 * 1081.0 / 1080.0;
    float dhdx = (h(vUv + vec2(du, 0.0)) - h(vUv - vec2(du, 0.0))) / (2.0 * du * RAD_U * max(cos(lat), 0.05));
    float dhdy = (h(vUv - vec2(0.0, dv)) - h(vUv + vec2(0.0, dv))) / (2.0 * dv * RAD_V);
    vec3 n = normalize(uRot * normalize(d - east * dhdx - north * dhdy));

    float light = uAmbient;
    for (int i = 0; i < 3; i++) light += uLightI[i] * max(dot(n, uLightDir[i]), 0.0);
    // three's Lambert convention (BRDF albedo / pi), kept so the look matches
    gl_FragColor = vec4(toSrgb(albedo * light / 3.1415927), 1.0);
  }`;
}

function rampStops(meta) {
  let stops = meta.colour_stops.map(([m, hex], i) => [
    m === null ? (i === 0 ? meta.vmin : meta.vmax) : m,
    new THREE.Vector3(...[1, 3, 5].map((k) => parseInt(hex.slice(k, k + 2), 16) / 255)),
  ]);
  // drop interior stops outside the data range, as colors.earth_colormap does
  stops = stops.filter(([m], i) => i === 0 || i === stops.length - 1 || (m > meta.vmin && m < meta.vmax));
  return stops;
}

function buildGlobe(meta, elev, segLon, segLat) {
  const W = meta.n_lon, H = meta.n_lat;
  // Int16 metres -> half-float red texture (filterable + mipmapped on WebGL2)
  const half = new Uint16Array(W * H);
  for (let k = 0; k < W * H; k++) half[k] = THREE.DataUtils.toHalfFloat(elev[k]);
  const tex = new THREE.DataTexture(half, W, H, THREE.RedFormat, THREE.HalfFloatType);
  tex.minFilter = THREE.LinearMipmapLinearFilter;
  tex.magFilter = THREE.LinearFilter;
  tex.wrapS = tex.wrapT = THREE.ClampToEdgeWrapping;
  tex.generateMipmaps = true;
  tex.flipY = false;
  tex.needsUpdate = true;

  // lat/lon sphere; uv addresses texel CENTRES (col 0 = 0E ... col 2160 = 360E)
  const NX = segLon + 1, NY = segLat + 1;
  const pos = new Float32Array(NX * NY * 3);
  const uvs = new Float32Array(NX * NY * 2);
  for (let i = 0; i < NY; i++) {
    const latDeg = 90 - (180 * i) / segLat;
    const lat = latDeg * DEG, cl = Math.cos(lat), sl = Math.sin(lat);
    for (let j = 0; j < NX; j++) {
      const lonDeg = (360 * j) / segLon, lon = lonDeg * DEG;
      const k = i * NX + j;
      // lon 0 faces +z (camera), east = +x (screen right), north = +y
      pos[3 * k] = cl * Math.sin(lon);
      pos[3 * k + 1] = sl;
      pos[3 * k + 2] = cl * Math.cos(lon);
      uvs[2 * k] = ((lonDeg / 360) * (W - 1) + 0.5) / W;
      uvs[2 * k + 1] = (((90 - latDeg) / 180) * (H - 1) + 0.5) / H;
    }
  }
  const idx = new Uint32Array(segLon * segLat * 6);
  let p = 0;
  for (let i = 0; i < segLat; i++) {
    for (let j = 0; j < segLon; j++) {
      const a = i * NX + j, b = a + 1, c = a + NX, d = c + 1;
      idx[p++] = a; idx[p++] = c; idx[p++] = b;
      idx[p++] = b; idx[p++] = c; idx[p++] = d;
    }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  g.setAttribute("uv", new THREE.BufferAttribute(uvs, 2));
  g.setIndex(new THREE.BufferAttribute(idx, 1));
  g.boundingSphere = new THREE.Sphere(new THREE.Vector3(), 1.07);

  const stops = rampStops(meta);
  const lightDirs = meta.lights.map((L) => {
    // kit frame (x toward camera, y right, z up) -> three (z toward camera, x right, y up)
    const az = L.azimuth * DEG, el = L.elevation * DEG;
    return new THREE.Vector3(Math.cos(el) * Math.sin(az), Math.sin(el), Math.cos(el) * Math.cos(az)).normalize();
  });
  const mat = new THREE.ShaderMaterial({
    vertexShader: VERT,
    fragmentShader: fragShader(stops.length),
    uniforms: {
      uElev: { value: tex },
      uExag: { value: meta.exaggeration / meta.earth_radius_m },
      uTexel: { value: new THREE.Vector2(1 / W, 1 / H) },
      uRot: { value: new THREE.Matrix3() },
      uLightDir: { value: lightDirs },
      uLightI: { value: meta.lights.map((L, i) => L.intensity * LIGHT_GAINS[i]) },
      uAmbient: { value: AMBIENT },
      uStopM: { value: stops.map((s) => s[0]) },
      uStopC: { value: stops.map((s) => s[1]) },
    },
    extensions: { derivatives: true },
  });
  return new THREE.Mesh(g, mat);
}

// ----------------------------------------------------------------- sound --
// The pluck engine of reel 174's audio F2 (tools/sonify_musical2.py), live:
// on a 90 BPM clock, the loudest land voice under the line plays the melody
// (one pentatonic octave up, lift and ceiling lowered over high relief so the
// Himalaya never hammers C7); in F the accompaniment plays only on the
// off-beats. Kalimba pluck (4 harmonics, upper ones decay faster, +3 cent
// twin), seeded-noise reverb, a quiet ocean-depth drone and a C3+G3 pad.
// Notes are pre-rendered sample buffers played at fixed pitch: a pitch change
// is always a NEW note, never a glide.
function mulberry32(a) {
  return () => {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

class Song {
  constructor(meta, table, variant) {
    this.v = meta.voices;
    this.P = this.v.variants[variant];
    this.variant = variant;
    this.table = table;
    this.ctx = null;
    this.on = false;
    this.paused = false;
    this.s = 0;                      // grid step counter
    this.nextTick = 0;
    this.lastCol = null;
    this.swept = 0;
    this.lastPitch = new Map();
    this.lastStep = new Map();
    this.cache = new Map();          // note index -> [fundamental, upper] buffers
    this.log = [];                   // recent notes, for checks
  }

  get stepDur() { return (this.P.step / 30); }   // 10 frames = eighth, 20 = quarter at 90 BPM

  start() {
    const ctx = (this.ctx = new (window.AudioContext || window.webkitAudioContext)());
    this.master = ctx.createGain(); this.master.gain.value = 0;
    this.pauseGain = ctx.createGain(); this.pauseGain.gain.value = 1;
    const comp = ctx.createDynamicsCompressor();
    comp.threshold.value = -14; comp.knee.value = 8; comp.ratio.value = 3;
    comp.attack.value = 0.005; comp.release.value = 0.3;
    this.bus = ctx.createGain(); this.bus.gain.value = 0.32;
    this.bus.connect(comp).connect(this.pauseGain).connect(this.master).connect(ctx.destination);

    // reverb: seeded noise IR, ~2 s RT60, one-pole darkened, stereo, 20 ms predelay;
    // wet level set so wet rms ~ wet_frac x dry rms, as in the offline engine
    const sr = ctx.sampleRate, rt = 2.0, n = Math.floor(rt * sr), pre = Math.floor(0.02 * sr);
    const ir = ctx.createBuffer(2, n + pre, sr);
    let energy = 0;
    for (let ch = 0; ch < 2; ch++) {
      const rnd = mulberry32(174 + ch), d = ir.getChannelData(ch);
      let y = 0;
      for (let i = 0; i < n; i++) {
        const g = Math.sqrt(-2 * Math.log(rnd() + 1e-12)) * Math.cos(2 * Math.PI * rnd());
        y = 0.35 * g * Math.exp((-6.91 * i) / sr / rt) + 0.65 * y;
        d[pre + i] = y; energy += y * y / 2;
      }
    }
    this.verb = ctx.createConvolver(); this.verb.normalize = false; this.verb.buffer = ir;
    this.wet = ctx.createGain(); this.wet.gain.value = this.P.wet / Math.sqrt(energy);
    this.send = ctx.createGain(); this.send.gain.value = 1;
    this.send.connect(this.verb).connect(this.wet).connect(this.bus);
    this.dry = ctx.createGain(); this.dry.gain.value = 1;
    this.dry.connect(this.bus);
    this.dry.connect(this.send);

    // continuous layer: ocean drone (sine + 0.3 x 2nd) and, for E/F, the C3+G3 pad
    this.contGain = ctx.createGain(); this.contGain.gain.value = 0;
    this.contGain.connect(this.dry);
    this.drone = [ctx.createOscillator(), ctx.createOscillator()];
    this.droneG = ctx.createGain(); this.droneG.gain.value = 0;
    const d2 = ctx.createGain(); d2.gain.value = 0.3;
    this.drone[0].connect(this.droneG); this.drone[1].connect(d2).connect(this.droneG);
    this.droneG.connect(this.contGain);
    this.drone.forEach((o, i) => { o.frequency.value = 80 * (i + 1); o.start(); });
    if (this.P.pad > 0) {
      const padG = ctx.createGain(); padG.gain.value = this.P.pad / 2;
      const lfo = ctx.createOscillator(), lfoG = ctx.createGain();
      lfo.frequency.value = 1 / 18; lfoG.gain.value = 0.1 * this.P.pad / 2;
      lfo.connect(lfoG).connect(padG.gain); lfo.start();
      for (const [f, a] of [[130.81, 1], [261.62, 0.2], [196.0, 0.8], [392.0, 0.16]]) {
        const o = ctx.createOscillator(), g = ctx.createGain();
        o.frequency.value = f; g.gain.value = a; o.connect(g).connect(padG); o.start();
      }
      padG.connect(this.contGain);
    }
  }

  // kalimba pluck, split into fundamental and upper partials so the relief
  // softening (upper x h) needs no extra renders
  buffers(ni) {
    if (this.cache.has(ni)) return this.cache.get(ni);
    const ctx = this.ctx, sr = ctx.sampleRate, tau = this.P.tau, H = this.v.harmonics;
    const n = Math.floor(Math.min(4.5 * tau, 6.0) * sr);
    const f = this.v.note_freqs[ni];
    const out = [ctx.createBuffer(1, n, sr), ctx.createBuffer(1, n, sr)];
    const a = out[0].getChannelData(0), b = out[1].getChannelData(0);
    for (let i = 0; i < n; i++) {
      const t = i / sr, att = 1 - Math.exp(-t / 0.004);
      for (let k = 1; k <= H.length; k++) {
        const env = att * Math.exp(-t / (tau / Math.pow(k, 1.3)));
        const s = H[k - 1] * env * (Math.sin(2 * Math.PI * k * f * t) + 0.5 * Math.sin(2 * Math.PI * k * f * 1.0017 * t)) / 1.5;
        if (k === 1) a[i] += s; else b[i] += s;
      }
    }
    this.cache.set(ni, out);
    return out;
  }

  play(ni, vel, h, lat, when) {
    const ctx = this.ctx, [bf, bu] = this.buffers(ni);
    const pan = ctx.createStereoPanner(); pan.pan.value = 0.35 * (lat / 78);
    pan.connect(this.dry);
    for (const [buf, g] of [[bf, vel], [bu, vel * h]]) {
      if (g <= 0) continue;
      const src = ctx.createBufferSource(), gn = ctx.createGain();
      src.buffer = buf; gn.gain.value = g;
      src.connect(gn).connect(pan);
      src.start(when);
    }
    this.log.push({ t: when, ni, hz: this.v.note_freqs[ni], vel, h, lat });
    if (this.log.length > 400) this.log.shift();
  }

  async toggle() {
    if (!this.ctx) this.start();
    this.on = !this.on;
    this.swept = 0; this.lastCol = null;
    const t = this.ctx.currentTime;
    this.master.gain.cancelScheduledValues(t);
    if (this.on) {
      await this.ctx.resume();
      this.nextTick = this.ctx.currentTime + 0.05;
      this.master.gain.setTargetAtTime(1, this.ctx.currentTime, 0.08);
    } else {
      this.master.gain.setTargetAtTime(0, t, 0.06);
      clearTimeout(this.suspendTimer);
      this.suspendTimer = setTimeout(() => { if (!this.on) this.ctx.suspend(); }, 600);
    }
    return this.on;
  }

  setPaused(p) {
    this.paused = p;
    if (!this.ctx) return;
    const t = this.ctx.currentTime;
    this.pauseGain.gain.cancelScheduledValues(t);
    this.pauseGain.gain.setTargetAtTime(p ? 0 : 1, t, p ? 0.05 : 0.12);
    this.nextTick = t + 0.05;
    this.swept = 0;
  }

  col(c, field, v) { return this.table[c * this.v.bytes_per_col + field * this.v.n_voices + v]; }

  // facing: deg east under the line; omega: deg/s
  update(facing, omega) {
    if (!this.on || !this.ctx || this.paused) return;
    const NC = this.v.n_cols, NV = this.v.n_voices, BPC = this.v.bytes_per_col;
    const c = ((Math.round((facing * NC) / 360) % NC) + NC) % NC;
    if (this.lastCol === null) this.lastCol = c;
    let d = c - this.lastCol;
    if (d > NC / 2) d -= NC; else if (d < -NC / 2) d += NC;
    this.swept += d; this.lastCol = c;

    const sAbs = Math.abs(omega);
    // still -> silent; full from about the auto rate; a little softer when flung
    const speedGain = smoothstep(0.3, AUTO_RATE, sAbs) * (1 - 0.35 * smoothstep(120, FLING_MAX, sAbs));
    const now = this.ctx.currentTime;
    this.contGain.gain.setTargetAtTime(speedGain, now, 0.25);
    const df = 55 + (this.table[c * BPC + 72] / 255) * 55;
    this.drone[0].frequency.setTargetAtTime(df, now, 0.3);
    this.drone[1].frequency.setTargetAtTime(2 * df, now, 0.3);
    this.droneG.gain.setTargetAtTime((this.table[c * BPC + 73] / 255) * this.v.drone_gain, now, 0.3);

    if (now + 0.03 < this.nextTick) return;
    const when = Math.max(this.nextTick, now + 0.01);
    this.nextTick = (this.nextTick < now - 0.5 ? now : this.nextTick) + this.stepDur;
    const s = this.s++;
    const span = Math.min(Math.abs(this.swept), NC), dir = Math.sign(this.swept) || 1;
    this.swept = 0;
    if (speedGain < 0.02) return;

    // pool the longitudes swept since the last beat: mean amplitude and
    // relief, and each voice plays the note where it was loudest
    const amps = new Float32Array(NV), rw = new Float32Array(NV), notes = new Int16Array(NV), peak = new Float32Array(NV).fill(-1);
    const n = Math.max(span, 1), start = c - dir * (n - 1);
    for (let k = 0; k < n; k++) {
      const cc = (((start + dir * k) % NC) + NC) % NC;
      for (let v = 0; v < NV; v++) {
        const a = this.col(cc, 1, v);
        amps[v] += a / 255 / n; rw[v] += this.col(cc, 2, v) / 255 / n;
        if (a > peak[v]) { peak[v] = a; notes[v] = this.col(cc, 0, v); }
      }
    }
    const floor = this.v.amp_floor / this.v.gmax;
    const cand = Array.from(amps.keys()).filter((v) => amps[v] >= floor).sort((a, b) => amps[b] - amps[a]);
    if (!cand.length) return;

    const chosen = [];
    const mel = cand[0];
    const lift = Math.round(this.v.mel_lift - this.v.mel_lift_cut * rw[mel]);
    const ceil = 15 - Math.round(this.v.mel_ceil_cut * rw[mel]);
    chosen.push([mel, Math.min(notes[mel] + lift, ceil), true]);
    // high-speed thinning: accompaniment drops to one voice above 60 deg/s
    // and to none above 180 deg/s; the beat itself never speeds up
    const accomp = sAbs < 60 ? this.P.accomp : sAbs < 180 ? Math.min(1, this.P.accomp) : 0;
    const pool = cand.slice(1, 7);
    if (pool.length && !(this.P.offbeat && s % 2 === 0)) {
      for (let j = 0; j < accomp; j++) {
        const vo = pool[(s + j) % pool.length];
        chosen.push([vo, notes[vo], false]);
      }
    }
    for (const [vo, ni, isMel] of chosen) {
      const refire = isMel ? 1 : 2;
      if (this.lastPitch.get(vo) === ni && s - (this.lastStep.get(vo) ?? -9) < refire) continue;
      this.lastPitch.set(vo, ni); this.lastStep.set(vo, s);
      const a = amps[vo];
      let vel = (0.15 + 0.85 * Math.pow(a, 1.3)) * (0.88 + 0.12 * Math.sin(2.3999 * (s + 7 * vo)));
      if (!isMel) vel *= 0.7;
      const h = 1 - this.v.pluck_harm_cut * rw[vo];
      this.play(ni, vel * speedGain, h, this.v.voice_lat[vo], when);
    }
  }
}

// ------------------------------------------------------------------ main --
async function main() {
  let meta, elev, table;
  try {
    const get = (u, kind) => fetch(u).then((r) => { if (!r.ok) throw new Error(`${u}: ${r.status}`); return r[kind](); });
    const [m, e, v] = await Promise.all([get(`data/meta.json?v=${DATA_V}`, "json"), get(`data/elev.bin?v=${DATA_V}`, "arrayBuffer"), get(`data/voices.bin?v=${DATA_V}`, "arrayBuffer")]);
    meta = m; elev = new Int16Array(e); table = new Uint8Array(v);
  } catch (err) {
    fail("Could not load the elevation data. Serve this folder over http (python3 -m http.server), not from file://.");
    return;
  }
  if (elev.length !== meta.n_lat * meta.n_lon) { fail("Elevation file has the wrong size."); return; }
  const qv = new URLSearchParams(location.search).get("voice");
  const variant = qv && meta.voices.variants[qv.toUpperCase()] ? qv.toUpperCase() : meta.voices.default_variant;

  const stage = $("stage");
  let renderer;
  try {
    renderer = new THREE.WebGLRenderer({ antialias: true });
    if (!renderer.capabilities.isWebGL2) throw new Error("webgl2");
  } catch (err) {
    fail("This browser cannot run WebGL 2.");
    return;
  }
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  stage.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(meta.background);
  const camera = new THREE.PerspectiveCamera(28, 1, 0.1, 50);

  // phones get a lighter mesh; the texture (colour, normals) is the same
  const small = matchMedia("(pointer: coarse)").matches || Math.min(screen.width, screen.height) < 700;
  const globe = buildGlobe(meta, elev, small ? 1024 : 1440, small ? 512 : 720);
  scene.add(globe);

  // the fixed meridian: not in the globe, so it never turns
  const arc = new THREE.Curve();
  const RL = 1.068;   // just above the max x60 bulge (1.0615 R)
  arc.getPoint = (t, out = new THREE.Vector3()) => {
    const lat = (88 - 176 * t) * DEG;
    return out.set(0, RL * Math.sin(lat), RL * Math.cos(lat));
  };
  scene.add(new THREE.Mesh(new THREE.TubeGeometry(arc, 160, 0.0045, 8, false),
    new THREE.MeshBasicMaterial({ color: 0xfff1d6 })));
  scene.add(new THREE.Mesh(new THREE.TubeGeometry(arc, 160, 0.016, 8, false),
    new THREE.MeshBasicMaterial({ color: 0xffd9a0, transparent: true, opacity: 0.16,
      blending: THREE.AdditiveBlending, depthWrite: false })));

  // ------------------------------------------------------------- layout --
  let W = 1, H = 1, radiusPx = 1, centrePx = 0;
  const lonEl = $("lon");
  function resize() {
    W = window.innerWidth; H = window.innerHeight;
    renderer.setSize(W, H);
    camera.aspect = W / H;
    const top = document.querySelector("header").getBoundingClientRect().bottom + 30;
    const bottom = document.querySelector(".controls").getBoundingClientRect().top - 18;
    const halfPx = Math.max(60, Math.min(W * 0.46, (bottom - top) / 2));
    const centreY = (top + bottom) / 2;
    const tanHalf = Math.tan((camera.fov * DEG) / 2);
    camera.position.set(0, 0, (1.07 * (H / 2)) / (halfPx * tanHalf));
    camera.lookAt(0, 0, 0);
    camera.setViewOffset(W, H, 0, H / 2 - centreY, W, H);
    camera.updateProjectionMatrix();
    radiusPx = halfPx / 1.07;
    centrePx = centreY;
    lonEl.style.top = `${Math.max(centreY - halfPx - 30, 4)}px`;
  }
  window.addEventListener("resize", resize);
  resize();

  // -------------------------------------------------------------- state --
  let theta = 0;          // globe rotation (rad); facing longitude = -theta
  let omega = 0;          // deg/s, + = surface moves right = eastward spin
  let auto = false;
  let paused = false;
  const held = { left: false, right: false };
  let holdT = 0, holdDir = 0;
  const drag = { on: false, x: 0, t: 0, v: 0, id: null };
  const song = new Song(meta, table, variant);

  // ------------------------------------------------------------- input --
  const caps = { left: $("k-left"), right: $("k-right"), auto: $("k-auto"), pause: $("k-pause"), sound: $("k-sound") };
  // keyboard OR pointer can hold an arrow; it stays orange while either does
  const holders = { left: new Set(), right: new Set() };
  const press = (k, src, down) => {
    if (down) holders[k].add(src); else holders[k].delete(src);
    held[k] = holders[k].size > 0;
    caps[k].classList.toggle("down", held[k]);
  };
  const flash = (el) => { el.classList.add("down"); setTimeout(() => el.classList.remove("down"), 110); };

  function setAuto(v) {
    auto = v;
    caps.auto.classList.toggle("on", auto);
    caps.auto.setAttribute("aria-pressed", String(auto));
  }
  // Pause freezes ALL rotation (momentum and auto included) and silences the
  // sound; resume continues exactly where it was (same speed, same mode).
  function setPaused(v) {
    paused = v;
    if (paused) { drag.on = false; stage.classList.remove("dragging"); }
    caps.pause.classList.toggle("on", paused);
    caps.pause.setAttribute("aria-pressed", String(paused));
    $("i-pause").style.display = paused ? "none" : "";
    $("i-play").style.display = paused ? "" : "none";
    lonEl.classList.toggle("paused", paused);
    song.setPaused(paused);
  }
  async function toggleSound() {
    const on = await song.toggle();
    if (paused) song.setPaused(true);
    caps.sound.classList.toggle("on", on);
    caps.sound.setAttribute("aria-pressed", String(on));
    $("i-waves").setAttribute("opacity", on ? "1" : "0.25");
  }

  window.addEventListener("keydown", (e) => {
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    if (e.key === "ArrowLeft" || e.key === "ArrowRight") {
      e.preventDefault();
      press(e.key === "ArrowLeft" ? "left" : "right", "kbd", true);
    } else if (e.code === "Space" || e.key === "p" || e.key === "P") {
      e.preventDefault();
      if (!e.repeat) { setPaused(!paused); flash(caps.pause); }
    } else if ((e.key === "r" || e.key === "R") && !e.repeat) {
      setAuto(!auto); flash(caps.auto);
    } else if ((e.key === "s" || e.key === "S" || e.key === "m" || e.key === "M") && !e.repeat) {
      toggleSound(); flash(caps.sound);
    }
  });
  window.addEventListener("keyup", (e) => {
    if (e.key === "ArrowLeft") press("left", "kbd", false);
    if (e.key === "ArrowRight") press("right", "kbd", false);
  });
  window.addEventListener("blur", () => {
    for (const k of ["left", "right"]) { holders[k].clear(); press(k, "kbd", false); }
  });

  for (const k of ["left", "right"]) {
    const b = caps[k];
    b.addEventListener("pointerdown", (e) => {
      e.preventDefault();
      try { b.setPointerCapture(e.pointerId); } catch (_) { /* synthetic pointer */ }
      press(k, "ptr" + e.pointerId, true);
    });
    for (const ev of ["pointerup", "pointercancel", "lostpointercapture"]) {
      b.addEventListener(ev, (e) => press(k, "ptr" + e.pointerId, false));
    }
    b.addEventListener("contextmenu", (e) => e.preventDefault());
    b.addEventListener("keydown", (e) => { if (e.code === "Space" || e.key === "Enter") e.preventDefault(); });
  }
  caps.auto.addEventListener("click", (e) => { setAuto(!auto); e.currentTarget.blur(); });
  caps.pause.addEventListener("click", (e) => { setPaused(!paused); e.currentTarget.blur(); });
  caps.sound.addEventListener("click", (e) => { toggleSound(); e.currentTarget.blur(); });
  for (const b of [caps.auto, caps.pause, caps.sound]) {
    b.addEventListener("pointerdown", () => b.classList.add("down"));
    for (const ev of ["pointerup", "pointercancel", "pointerleave"]) b.addEventListener(ev, () => b.classList.remove("down"));
  }

  // drag to spin: the surface under the pointer follows it; release throws
  // the globe with the pointer's recent velocity, then it relaxes
  stage.addEventListener("pointerdown", (e) => {
    if (paused) return;
    drag.on = true; drag.id = e.pointerId; drag.x = e.clientX; drag.t = performance.now(); drag.v = 0;
    try { stage.setPointerCapture(e.pointerId); } catch (_) { /* synthetic or lost pointer */ }
    stage.classList.add("dragging");
  });
  stage.addEventListener("pointermove", (e) => {
    if (!drag.on || e.pointerId !== drag.id) return;
    const now = performance.now();
    const dt = Math.max((now - drag.t) / 1000, 1e-3);
    const dth = (e.clientX - drag.x) / radiusPx;          // rad
    theta += dth;
    const a = 1 - Math.exp(-dt / 0.05);                    // ~50 ms velocity filter
    drag.v += (dth / DEG / dt - drag.v) * a;
    drag.x = e.clientX; drag.t = now;
  });
  const endDrag = (e) => {
    if (!drag.on || (e && e.pointerId !== drag.id)) return;
    drag.on = false;
    stage.classList.remove("dragging");
    const idle = (performance.now() - drag.t) / 1000;
    omega = idle > 0.08 ? 0 : clamp(drag.v, -FLING_MAX, FLING_MAX);   // stopped before letting go -> no throw
  };
  stage.addEventListener("pointerup", endDrag);
  stage.addEventListener("pointercancel", endDrag);

  // ---------------------------------------------------------- the loop --
  let last = performance.now();
  let lastLonText = "";
  function tick(dt, now) {
    if (paused) return;               // frozen: theta, omega and mode are kept
    if (drag.on) {
      const idle = (now - drag.t) / 1000;
      if (idle > 0.05) drag.v *= Math.exp(-dt / 0.06);
      omega = clamp(drag.v, -FLING_MAX, FLING_MAX);
    } else {
      const dir = (held.right ? 1 : 0) - (held.left ? 1 : 0);
      if (dir !== 0) {
        if (dir !== holdDir) { holdT = 0; holdDir = dir; }
        holdT += dt;
        const target = dir * (ARROW_START + (ARROW_MAX - ARROW_START) * easeInOutCubic(clamp(holdT / ARROW_RAMP_S, 0, 1)));
        omega += (target - omega) * (1 - Math.exp(-dt / ARROW_TAU));
      } else {
        holdDir = 0; holdT = 0;
        const rest = auto ? AUTO_RATE : 0;
        omega += (rest - omega) * (1 - Math.exp(-dt / RELAX_TAU));
        if (Math.abs(omega - rest) < 0.02) omega = rest;
      }
      theta += omega * DEG * dt;
    }
    theta = ((theta % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI);
  }
  function facingDeg() { return (((-theta / DEG) % 360) + 360) % 360; }
  function sync() {
    globe.rotation.y = theta;
    globe.updateMatrixWorld();
    globe.material.uniforms.uRot.value.setFromMatrix4(globe.matrixWorld);
    const lonE = facingDeg() > 180 ? facingDeg() - 360 : facingDeg();
    const txt = `${Math.abs(Math.round(lonE))}° ${lonE >= 0 ? "E" : "W"}`;
    if (txt !== lastLonText) { lonEl.textContent = txt; lastLonText = txt; }
  }
  function frame(now) {
    tick(Math.min((now - last) / 1000, 0.05), now);
    last = now;
    sync();
    song.update(facingDeg(), paused ? 0 : omega);
    renderer.render(scene, camera);
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);

  // for headless checks (a hidden tab does not run requestAnimationFrame)
  window.__earthsong = {
    get theta() { return theta; }, set theta(v) { theta = v; }, get omega() { return omega; },
    get auto() { return auto; }, get paused() { return paused; }, song, setAuto, setPaused, variant,
    step(ms) {
      const t0 = performance.now();
      for (let k = 0; k < ms / 16.667; k++) { tick(1 / 60, t0 + k * 16.667); sync(); }
    },
    render() { sync(); renderer.render(scene, camera); },
    pixel(x, y) {
      sync(); renderer.render(scene, camera);
      const gl = renderer.getContext(), b = new Uint8Array(4), pr = renderer.getPixelRatio();
      gl.readPixels(Math.round(x * pr), Math.round((H - y) * pr), 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, b);
      return Array.from(b.slice(0, 3));
    },
    get geom() { return { W, H, radiusPx, centrePx, pr: renderer.getPixelRatio() }; },
    get lonText() { return lonEl.textContent; },
  };
}

main();
