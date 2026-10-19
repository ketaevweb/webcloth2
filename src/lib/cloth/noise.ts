/**
 * WebCloth 2 — deterministic 3D value noise shared by CPU solver (mirrored in WGSL).
 * Used by the turbulent wind field.
 */

function fract(x: number): number {
  return x - Math.floor(x);
}

function hash3(x: number, y: number, z: number): number {
  let qx = fract(x * 0.1031);
  let qy = fract(y * 0.1031);
  let qz = fract(z * 0.1031);
  // q += dot(q, q.zyx + 31.32)
  const d = qx * (qz + 31.32) + qy * (qy + 31.32) + qz * (qx + 31.32);
  qx += d; qy += d; qz += d;
  return fract((qx + qy) * qz);
}

export function vnoise(x: number, y: number, z: number): number {
  const ix = Math.floor(x), iy = Math.floor(y), iz = Math.floor(z);
  const fx = x - ix, fy = y - iy, fz = z - iz;
  const ux = fx * fx * (3 - 2 * fx);
  const uy = fy * fy * (3 - 2 * fy);
  const uz = fz * fz * (3 - 2 * fz);
  const n000 = hash3(ix, iy, iz);
  const n100 = hash3(ix + 1, iy, iz);
  const n010 = hash3(ix, iy + 1, iz);
  const n110 = hash3(ix + 1, iy + 1, iz);
  const n001 = hash3(ix, iy, iz + 1);
  const n101 = hash3(ix + 1, iy, iz + 1);
  const n011 = hash3(ix, iy + 1, iz + 1);
  const n111 = hash3(ix + 1, iy + 1, iz + 1);
  const nx00 = n000 + (n100 - n000) * ux;
  const nx10 = n010 + (n110 - n010) * ux;
  const nx01 = n001 + (n101 - n001) * ux;
  const nx11 = n011 + (n111 - n011) * ux;
  const a0 = nx00 + (nx10 - nx00) * uy;
  const a1 = nx01 + (nx11 - nx01) * uy;
  return a0 + (a1 - a0) * uz;
}
