/**
 * WebCloth 2 — WGSL compute shaders for the WebGPU backend.
 * Pipeline per substep: integrate -> (solve x iterations) -> collide -> updateVelocity.
 *
 * Data layout notes:
 *  - positions/prev/velocities: packed f32 xyz per particle (no vec3 padding waste);
 *  - constraints: 4 x u32 [iA, iB, bitcast(rest), bitcast(compliance)];
 *  - triAdj: per particle ADJ_SLOTS pairs (b,c) of adjacent-triangle vertices;
 *  - static particles have invMass == 0 and are skipped by kernels
 *    (their positions are driven by the engine via writeBuffer).
 *
 * Constraint solving uses one thread per constraint with relaxed memory
 * semantics (standard GPU-PBD approximation; convergence is restored by
 * iterating the Jacobi-like relaxations).
 */

export const WGSL = /* wgsl */ `
const SENTINEL: u32 = 0xffffffffu;
const ADJ_STRIDE: u32 = 16u;

struct Params {
  n: u32,
  nc: u32,
  dt: f32,
  invDt: f32,
  gravity: vec4<f32>,
  windDir: vec4<f32>,    // xyz normalized, w = strength
  windParam: vec4<f32>,  // x freq, y turbulence, z gust, w time
  dragParam: vec4<f32>,  // x aeroDrag, y damping, z maxVelocity, w friction
  colliderCount: u32,
  pad0: u32,
  pad1: u32,
  pad2: u32,
  colliders: array<vec4<f32>, 48>, // 3 vec4 per collider, max 16
}

@group(0) @binding(0) var<storage, read_write> positions: array<f32>;
@group(0) @binding(1) var<storage, read_write> prevPositions: array<f32>;
@group(0) @binding(2) var<storage, read_write> velocities: array<f32>;
@group(0) @binding(3) var<storage, read> invMass: array<f32>;
@group(0) @binding(4) var<storage, read> constraints: array<u32>;
@group(0) @binding(5) var<storage, read> triAdj: array<u32>;
@group(0) @binding(6) var<uniform> params: Params;
@group(0) @binding(7) var<storage, read> consOffset: array<u32>;
@group(0) @binding(8) var<storage, read> consOther: array<u32>;
@group(0) @binding(9) var<storage, read> consCon: array<u32>;

fn loadPos(i: u32) -> vec3<f32> {
  let i3 = i * 3u;
  return vec3<f32>(positions[i3], positions[i3 + 1u], positions[i3 + 2u]);
}

fn storePos(i: u32, p: vec3<f32>) {
  let i3 = i * 3u;
  positions[i3] = p.x;
  positions[i3 + 1u] = p.y;
  positions[i3 + 2u] = p.z;
}

fn loadVel(i: u32) -> vec3<f32> {
  let i3 = i * 3u;
  return vec3<f32>(velocities[i3], velocities[i3 + 1u], velocities[i3 + 2u]);
}

fn storeVel(i: u32, v: vec3<f32>) {
  let i3 = i * 3u;
  velocities[i3] = v.x;
  velocities[i3 + 1u] = v.y;
  velocities[i3 + 2u] = v.z;
}

fn loadPrev(i: u32) -> vec3<f32> {
  let i3 = i * 3u;
  return vec3<f32>(prevPositions[i3], prevPositions[i3 + 1u], prevPositions[i3 + 2u]);
}

fn storePrev(i: u32, p: vec3<f32>) {
  let i3 = i * 3u;
  prevPositions[i3] = p.x;
  prevPositions[i3 + 1u] = p.y;
  prevPositions[i3 + 2u] = p.z;
}

// ---- wind field (mirrors CPU noise.ts) ----
fn hash13(p: vec3<f32>) -> f32 {
  var q = fract(p * 0.1031);
  q = q + vec3<f32>(dot(q, q.zyx + vec3<f32>(31.32)));
  return fract((q.x + q.y) * q.z);
}

fn vnoise(p: vec3<f32>) -> f32 {
  let i = floor(p);
  let f = fract(p);
  let u = f * f * (vec3<f32>(3.0) - 2.0 * f);
  let n000 = hash13(i);
  let n100 = hash13(i + vec3<f32>(1.0, 0.0, 0.0));
  let n010 = hash13(i + vec3<f32>(0.0, 1.0, 0.0));
  let n110 = hash13(i + vec3<f32>(1.0, 1.0, 0.0));
  let n001 = hash13(i + vec3<f32>(0.0, 0.0, 1.0));
  let n101 = hash13(i + vec3<f32>(1.0, 0.0, 1.0));
  let n011 = hash13(i + vec3<f32>(0.0, 1.0, 1.0));
  let n111 = hash13(i + vec3<f32>(1.0, 1.0, 1.0));
  let nx00 = n000 + (n100 - n000) * u.x;
  let nx10 = n010 + (n110 - n010) * u.x;
  let nx01 = n001 + (n101 - n001) * u.x;
  let nx11 = n011 + (n111 - n011) * u.x;
  let a0 = nx00 + (nx10 - nx00) * u.y;
  let a1 = nx01 + (nx11 - nx01) * u.y;
  return a0 + (a1 - a0) * u.z;
}

fn windVel(p: vec3<f32>, v: vec3<f32>) -> vec3<f32> {
  let t = params.windParam.w;
  let strength = params.windDir.w;
  if (strength <= 0.0) {
    return vec3<f32>(0.0);
  }
  let gust = 1.0 + params.windParam.z * sin(t * 0.9 + p.x * 0.15 + p.y * 0.1);
  // windParam.y is already turbulence * strength (packed by the CPU) —
  // do NOT multiply by strength again or the noise explodes.
  let amp = params.windParam.y;
  let fp = p * params.windParam.x;
  let tx = (vnoise(fp + vec3<f32>(t * 1.7, 0.0, 0.0)) - 0.5) * 2.0 * amp;
  let ty = (vnoise(fp + vec3<f32>(0.0, t * 1.3 + 3.7, 0.0)) - 0.5) * 2.0 * amp;
  let tz = (vnoise(fp + vec3<f32>(5.1, 0.0, t * 2.3)) - 0.5) * 2.0 * amp;
  return params.windDir.xyz * strength * gust + vec3<f32>(tx, ty, tz) - v * 0.0;
}

@compute @workgroup_size(64)
fn integrate(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.n) { return; }
  let w = invMass[i];
  if (w <= 0.0) { return; }
  let p = loadPos(i);
  var v = loadVel(i);

  var f = vec3<f32>(0.0);
  if (params.dragParam.x > 0.0 && params.windDir.w > 0.0) {
    let wv = windVel(p, v);
    // per-particle viscous streaming drag (works even edge-on to the wind)
    let lk = params.dragParam.x * 2.2;
    f = (wv - v) * lk;
    let base = i * ADJ_STRIDE;
    for (var k: u32 = 0u; k < ADJ_STRIDE; k = k + 2u) {
      let ib = triAdj[base + k];
      if (ib == SENTINEL) { break; }
      let ic = triAdj[base + k + 1u];
      let pb = loadPos(ib);
      let pc = loadPos(ic);
      let cr = cross(pb - p, pc - p);
      let area2 = length(cr);
      if (area2 > 1e-9) {
        let nn = cr / area2;
        let relV = wv - v;
        let d = dot(nn, relV);
        // AERO_SCALE mirrors the CPU backend (acceleration units)
        f = f + nn * (d * area2 * 0.5 * params.dragParam.x / 3.0 * 600.0);
      }
    }
  }

  v = v + (f * w + params.gravity.xyz) * params.dt;
  let vmax = params.dragParam.z;
  let vlen = length(v);
  if (vlen > vmax) {
    v = v * (vmax / vlen);
  }
  storeVel(i, v);
  storePrev(i, p);
  storePos(i, p + v * params.dt);
}

// Race-free per-particle XPBD relaxation: each thread owns exactly one
// particle, iterates only its own constraints (CSR adjacency) and writes ONLY
// that particle. Reads of neighbours may be stale (Jacobi/Gauss-Seidel hybrid)
// but no correction is ever lost. Summing FULL per-constraint corrections
// diverges on stiff lattices (classic Jacobi overshoot), so the sum is
// under-relaxed by the particle degree 1/k.
@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.n) { return; }
  let w = invMass[i];
  if (w <= 0.0) { return; }
  let p = loadPos(i);
  var dp = vec3<f32>(0.0);
  let start = consOffset[i];
  let end = consOffset[i + 1u];
  let k = max(1u, end - start);
  for (var e = start; e < end; e = e + 1u) {
    let j = consOther[e];
    let wj = invMass[j];
    let wsum = w + wj;
    if (wsum <= 0.0) { continue; }
    let ci = consCon[e];
    let rest = bitcast<f32>(constraints[ci * 4u + 2u]);
    let compliance = bitcast<f32>(constraints[ci * 4u + 3u]);
    let pj = loadPos(j);
    let d = pj - p;
    let len = length(d);
    if (len < 1e-9) { continue; }
    let C = len - rest;
    let alphaT = compliance * params.invDt * params.invDt;
    // XPBD: dl = C / (wsum + compliance/dt^2); this particle moves dl * w
    let dl = C / (wsum + alphaT);
    dp = dp + (d / len) * (dl * w);
  }
  storePos(i, p + dp / f32(k));
}

@compute @workgroup_size(64)
fn collide(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.n) { return; }
  if (invMass[i] <= 0.0) { return; }
  var p = loadPos(i);
  var touched = false;
  var nrm = vec3<f32>(0.0);
  for (var c: u32 = 0u; c < params.colliderCount; c = c + 1u) {
    let c0 = params.colliders[c * 3u];
    let c1 = params.colliders[c * 3u + 1u];
    let c2 = params.colliders[c * 3u + 2u];
    let typ = u32(c1.w + 0.5);
    var cn = vec3<f32>(0.0);
    var pen = 0.0;
    if (typ == 0u) {
      // sphere: center c0.xyz, radius c0.w
      let d = p - c0.xyz;
      let dist = length(d);
      if (dist < c0.w && dist > 1e-9) {
        cn = d / dist;
        pen = c0.w - dist;
      }
    } else if (typ == 1u) {
      // capsule: segment c0.xyz..c1.xyz, radius c0.w
      let ab = c1.xyz - c0.xyz;
      let ap = p - c0.xyz;
      let ab2 = dot(ab, ab);
      var t = 0.0;
      if (ab2 > 1e-9) {
        t = clamp(dot(ap, ab) / ab2, 0.0, 1.0);
      }
      let d = p - (c0.xyz + ab * t);
      let dist = length(d);
      if (dist < c0.w && dist > 1e-9) {
        cn = d / dist;
        pen = c0.w - dist;
      }
    } else {
      // plane: point c0.xyz, normal c2.xyz
      let dist = dot(p - c0.xyz, c2.xyz);
      if (dist < 0.0) {
        cn = c2.xyz;
        pen = -dist;
      }
    }
    if (pen > 0.0) {
      p = p + cn * pen;
      nrm = cn;
      touched = true;
    }
  }
  if (touched) {
    storePos(i, p);
    var v = loadVel(i);
    let vn = dot(v, nrm);
    let tangent = v - nrm * vn;
    let fric = max(0.0, 1.0 - params.dragParam.w * 0.9);
    v = tangent * fric + nrm * max(vn, 0.0);
    storeVel(i, v);
  }
}

@compute @workgroup_size(64)
fn updateVelocity(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.n) { return; }
  var v = (loadPos(i) - loadPrev(i)) * params.invDt;
  if (invMass[i] > 0.0) {
    let damp = max(0.0, 1.0 - params.dragParam.y * params.dt);
    v = v * damp;
    let vmax = params.dragParam.z;
    let vlen = length(v);
    if (vlen > vmax) {
      v = v * (vmax / vlen);
    }
  }
  storeVel(i, v);
}
`;
