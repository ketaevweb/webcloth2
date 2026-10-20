/**
 * WebCloth 2 — CPU XPBD solver (WebGL2 / fallback backend).
 * Structure-of-arrays layout, zero per-frame allocations, mirroring the WGSL
 * compute pipeline. Handles ~10-40k particles at 60 Hz on a modern desktop.
 */
import { vnoise } from "./noise";
import { ADJ_STRIDE, SENTINEL } from "./ClothBuilder";
import { AERO_SCALE, STREAM_DRAG, type ColliderLike, type EngineSettings } from "./types";

export interface SolverState {
  n: number;
  nc: number;
  positions: Float32Array; // xyz * N
  prevPositions: Float32Array;
  velocities: Float32Array;
  invMass: Float32Array;
  constraintA: Uint32Array;
  constraintB: Uint32Array;
  restLength: Float32Array;
  compliance: Float32Array; // precomputed per constraint from stiffness
  triAdj: Uint32Array;
  /** CSR per-particle constraint adjacency (used by the WebGPU solve kernel) */
  consOffset: Uint32Array; // n + 1, entry ranges per particle
  consOther: Uint32Array; // 2 * nc, opposite particle (pool-global)
  consCon: Uint32Array; // 2 * nc, constraint index (pool-global)
  /** self-collision radius per particle (0 => disabled for that particle) */
  radius: Float32Array;
  /** tearing: 1 = constraint broken (skipped by solvers, grown monotonically) */
  broken: Uint8Array;
  /** 1 = structural (tearable) constraint */
  breakable: Uint8Array;
}

export class CpuSolver {
  private s!: SolverState;
  time = 0;
  // spatial-hash scratch (allocated on demand, reused every step)
  private cellCount: Int32Array = new Int32Array(0);
  private cellStart: Int32Array = new Int32Array(0);
  private cellEntries: Int32Array = new Int32Array(0);

  setState(s: SolverState) {
    this.s = s;
  }

  resetTime() {
    this.time = 0;
  }

  /** One full simulation step: substeps x (integrate, solve*, selfCollide, collide, velocity) */
  step(dt: number, settings: EngineSettings, colliders: ColliderLike[]) {
    const s = this.s;
    const substeps = Math.max(1, settings.substeps | 0);
    const h = dt / substeps;
    for (let i = 0; i < substeps; i++) {
      this.time += h;
      this.integrate(h, settings);
      const iters = Math.max(1, settings.iterations | 0);
      for (let k = 0; k < iters; k++) this.solve(h, settings);
      if (settings.selfCollision) this.selfCollide();
      this.collide(settings, colliders);
      this.updateVelocity(h, settings);
    }
  }

  /** @internal debug entry: run phases individually */
  stepPhase(phase: "integrate" | "solve" | "collide" | "updateVelocity", h: number, settings: EngineSettings, colliders: ColliderLike[]) {
    if (phase === "integrate") this.integrate(h, settings);
    else if (phase === "solve") this.solve(h, settings);
    else if (phase === "collide") this.collide(settings, colliders);
    else this.updateVelocity(h, settings);
  }

  private integrate(h: number, st: EngineSettings) {
    const { positions, prevPositions, velocities, invMass, triAdj, n } = this.s;
    const gy = st.gravityY;
    const drag = st.aeroDrag;
    const maxV = st.maxVelocity;
    const t = this.time;
    // wind field parameters
    const wdx = st.windDirX, wdz = st.windDirZ;
    const wlen = Math.sqrt(wdx * wdx + wdz * wdz) || 1;
    const dirX = wdx / wlen, dirZ = wdz / wlen;
    const strength = st.windStrength;
    const turbAmp = st.windTurbulence * strength;
    const freq = st.windFrequency;
    const gustAmp = st.windGust;

    for (let i = 0; i < n; i++) {
      const w = invMass[i];
      if (w <= 0) continue;
      const i3 = i * 3;
      const px = positions[i3], py = positions[i3 + 1], pz = positions[i3 + 2];
      let vx = velocities[i3], vy = velocities[i3 + 1], vz = velocities[i3 + 2];

      // aerodynamic wind force through adjacent triangles + streaming drag
      let fx = 0, fy = 0, fz = 0;
      if (drag > 0 && strength > 0) {
        const gust = 1 + gustAmp * Math.sin(t * 0.9 + px * 0.15 + py * 0.1);
        const wf = freq;
        const tx = (vnoise(px * wf + t * 1.7, py * wf, pz * wf) - 0.5) * 2 * turbAmp;
        const ty = (vnoise(px * wf, py * wf + t * 1.3 + 3.7, pz * wf) - 0.5) * 2 * turbAmp;
        const tz = (vnoise(px * wf + 5.1, py * wf, pz * wf + t * 2.3) - 0.5) * 2 * turbAmp;
        const wvx = dirX * strength * gust + tx;
        const wvy = ty;
        const wvz = dirZ * strength * gust + tz;
        // per-particle viscous streaming drag (works even edge-on to the wind)
        const lk = drag * STREAM_DRAG;
        fx = (wvx - vx) * lk;
        fy = (wvy - vy) * lk;
        fz = (wvz - vz) * lk;
        const base = i * ADJ_STRIDE;
        for (let k = 0; k < ADJ_STRIDE; k += 2) {
          const ib = triAdj[base + k];
          if (ib === SENTINEL) break;
          const ic = triAdj[base + k + 1];
          const ib3 = ib * 3, ic3 = ic * 3;
          const e1x = positions[ib3] - px, e1y = positions[ib3 + 1] - py, e1z = positions[ib3 + 2] - pz;
          const e2x = positions[ic3] - px, e2y = positions[ic3 + 1] - py, e2z = positions[ic3 + 2] - pz;
          let nx = e1y * e2z - e1z * e2y;
          let ny = e1z * e2x - e1x * e2z;
          let nz = e1x * e2y - e1y * e2x;
          const area2 = Math.sqrt(nx * nx + ny * ny + nz * nz);
          if (area2 < 1e-9) continue;
          nx /= area2; ny /= area2; nz /= area2;
          // relative airflow through the surface
          const d = nx * (wvx - vx) + ny * (wvy - vy) + nz * (wvz - vz);
          const c = d * area2 * 0.5 * drag * AERO_SCALE / 3;
          fx += nx * c; fy += ny * c; fz += nz * c;
        }
      }

      vx += (fx * w) * h;
      vy += (fy * w + gy) * h;
      vz += (fz * w) * h;
      const v2 = vx * vx + vy * vy + vz * vz;
      if (v2 > maxV * maxV) {
        const s = maxV / Math.sqrt(v2);
        vx *= s; vy *= s; vz *= s;
      }
      velocities[i3] = vx; velocities[i3 + 1] = vy; velocities[i3 + 2] = vz;
      prevPositions[i3] = px; prevPositions[i3 + 1] = py; prevPositions[i3 + 2] = pz;
      positions[i3] = px + vx * h;
      positions[i3 + 1] = py + vy * h;
      positions[i3 + 2] = pz + vz * h;
    }
  }

  private solve(h: number, st: EngineSettings) {
    const { positions, invMass, constraintA, constraintB, restLength, compliance, nc, broken, breakable } = this.s;
    const tear = st.tearFactor;
    const invDt2 = 1 / (h * h);
    for (let c = 0; c < nc; c++) {
      if (broken[c]) continue;
      const ia = constraintA[c], ib = constraintB[c];
      const wa = invMass[ia], wb = invMass[ib];
      const wsum = wa + wb;
      if (wsum <= 0) continue;
      const ia3 = ia * 3, ib3 = ib * 3;
      const dx = positions[ib3] - positions[ia3];
      const dy = positions[ib3 + 1] - positions[ia3 + 1];
      const dz = positions[ib3 + 2] - positions[ia3 + 2];
      const len = Math.sqrt(dx * dx + dy * dy + dz * dz);
      if (len < 1e-9) continue;
      // tearing: structural overstretch beyond the break threshold
      if (tear > 0 && breakable[c] && len > restLength[c] * tear) {
        broken[c] = 1;
        continue;
      }
      const C = len - restLength[c];
      const aT = compliance[c] * invDt2;
      // XPBD: dl = C / (wsum + compliance/dt^2); pa += n*dl*wa; pb -= n*dl*wb
      const dl = C / (wsum + aT);
      const nx = dx / len, ny = dy / len, nz = dz / len;
      positions[ia3] += nx * dl * wa;
      positions[ia3 + 1] += ny * dl * wa;
      positions[ia3 + 2] += nz * dl * wa;
      positions[ib3] -= nx * dl * wb;
      positions[ib3 + 1] -= ny * dl * wb;
      positions[ib3 + 2] -= nz * dl * wb;
    }
  }

  /**
   * Self / cross-cloth particle collision via a uniform spatial hash
   * (counting sort, zero allocations after warm-up). Particles repel when
   * closer than (r_i + r_j); each of the two moves by its own inverse-mass
   * share. Only j > i pairs are processed.
   */
  private selfCollide() {
    const { positions, invMass, radius, n } = this.s;
    // find pool bounds + max radius
    let minX = Infinity, minY = Infinity, minZ = Infinity;
    let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
    let maxR = 0;
    for (let i = 0; i < n; i++) {
      const i3 = i * 3;
      const x = positions[i3], y = positions[i3 + 1], z = positions[i3 + 2];
      if (x < minX) minX = x; if (x > maxX) maxX = x;
      if (y < minY) minY = y; if (y > maxY) maxY = y;
      if (z < minZ) minZ = z; if (z > maxZ) maxZ = z;
      if (radius[i] > maxR) maxR = radius[i];
    }
    if (!(maxR > 0)) return;
    const cell = maxR * 2;
    const margin = maxR * 2 + 0.01;
    const dx0 = minX - margin, dy0 = minY - margin, dz0 = minZ - margin;
    const dimsX = Math.max(1, Math.ceil((maxX + margin - dx0) / cell));
    const dimsY = Math.max(1, Math.ceil((maxY + margin - dy0) / cell));
    const dimsZ = Math.max(1, Math.ceil((maxZ + margin - dz0) / cell));
    const cells = dimsX * dimsY * dimsZ;
    if (cells > 4_000_000) return; // safety valve for pathological bounds
    if (this.cellCount.length < cells) {
      this.cellCount = new Int32Array(cells);
      this.cellStart = new Int32Array(cells + 1);
    }
    if (this.cellEntries.length < n) this.cellEntries = new Int32Array(n);
    const count = this.cellCount;
    count.fill(0, 0, cells);
    const start = this.cellStart;
    const entries = this.cellEntries;

    const cellOf = (x: number, y: number, z: number): number => {
      const cx = Math.min(dimsX - 1, Math.max(0, ((x - dx0) / cell) | 0));
      const cy = Math.min(dimsY - 1, Math.max(0, ((y - dy0) / cell) | 0));
      const cz = Math.min(dimsZ - 1, Math.max(0, ((z - dz0) / cell) | 0));
      return cx + cy * dimsX + cz * dimsX * dimsY;
    };

    // counting sort into cells
    for (let i = 0; i < n; i++) count[cellOf(positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2])]++;
    let acc = 0;
    for (let c = 0; c < cells; c++) {
      start[c] = acc;
      acc += count[c];
    }
    start[cells] = acc;
    count.fill(0, 0, cells); // reuse count as per-cell fill cursor
    const cursor = count;
    for (let i = 0; i < n; i++) {
      const c = cellOf(positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2]);
      entries[start[c] + cursor[c]++] = i;
    }

    // neighbour sweep, 27 cells per particle, j > i only
    const dX = dimsX, dY = dimsY;
    for (let i = 0; i < n; i++) {
      const wi = invMass[i];
      if (wi <= 0) continue;
      const ri = radius[i];
      const i3 = i * 3;
      const px = positions[i3], py = positions[i3 + 1], pz = positions[i3 + 2];
      const cx = Math.min(dX - 1, Math.max(0, ((px - dx0) / cell) | 0));
      const cy = Math.min(dY - 1, Math.max(0, ((py - dy0) / cell) | 0));
      const cz = Math.min(dimsZ - 1, Math.max(0, ((pz - dz0) / cell) | 0));
      for (let oz = -1; oz <= 1; oz++) {
        const zz = cz + oz;
        if (zz < 0 || zz >= dimsZ) continue;
        for (let oy = -1; oy <= 1; oy++) {
          const yy = cy + oy;
          if (yy < 0 || yy >= dY) continue;
          const rowBase = (cx - 1) + yy * dX + zz * dX * dY;
          for (let ox = -1; ox <= 1; ox++) {
            const xx = cx + ox;
            if (xx < 0 || xx >= dX) continue;
            const c = rowBase + 1 + ox; // = xx + yy*dX + zz*dX*dY
            const s0 = start[c], s1 = start[c + 1];
            for (let e = s0; e < s1; e++) {
              const j = entries[e];
              if (j <= i) continue;
              const wj = invMass[j];
              const rr = ri + radius[j];
              if (rr <= 0) continue;
              const wsum = wi + wj;
              if (wsum <= 0) continue;
              const j3 = j * 3;
              const ddx = positions[j3] - px;
              const ddy = positions[j3 + 1] - py;
              const ddz = positions[j3 + 2] - pz;
              const d2 = ddx * ddx + ddy * ddy + ddz * ddz;
              if (d2 >= rr * rr || d2 < 1e-12) continue;
              const d = Math.sqrt(d2);
              // i moves by its own inverse-mass share (softened 0.5x);
              // j will move by its share when its own pass visits this pair
              const coef = ((rr - d) / d) * 0.5 * (wi / wsum);
              positions[i3] -= ddx * coef;
              positions[i3 + 1] -= ddy * coef;
              positions[i3 + 2] -= ddz * coef;
            }
          }
        }
      }
    }
  }

  private collide(st: EngineSettings, colliders: ColliderLike[]) {
    const { positions, velocities, invMass, n } = this.s;
    if (colliders.length === 0) return;
    const friction = st.friction;
    for (let i = 0; i < n; i++) {
      if (invMass[i] <= 0) continue;
      const i3 = i * 3;
      let px = positions[i3], py = positions[i3 + 1], pz = positions[i3 + 2];
      let touched = false;
      let nx = 0, ny = 0, nz = 0;
      for (let c = 0; c < colliders.length; c++) {
        const col = colliders[c];
        let cnx = 0, cny = 0, cnz = 0, pen = 0;
        if (col.kind === "sphere") {
          const dx = px - col.ax, dy = py - col.ay, dz = pz - col.az;
          const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
          if (dist < col.radius && dist > 1e-9) {
            cnx = dx / dist; cny = dy / dist; cnz = dz / dist;
            pen = col.radius - dist;
          }
        } else if (col.kind === "capsule") {
          const abx = col.bx - col.ax, aby = col.by - col.ay, abz = col.bz - col.az;
          const apx = px - col.ax, apy = py - col.ay, apz = pz - col.az;
          const ab2 = abx * abx + aby * aby + abz * abz;
          let t = 0;
          if (ab2 > 1e-9) t = Math.min(1, Math.max(0, (apx * abx + apy * aby + apz * abz) / ab2));
          const cx2 = col.ax + abx * t, cy2 = col.ay + aby * t, cz2 = col.az + abz * t;
          const dx = px - cx2, dy = py - cy2, dz = pz - cz2;
          const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
          if (dist < col.radius && dist > 1e-9) {
            cnx = dx / dist; cny = dy / dist; cnz = dz / dist;
            pen = col.radius - dist;
          }
        } else {
          // plane: point (ax,ay,az), normal (nx,ny,nz)
          const d = (px - col.ax) * col.nx + (py - col.ay) * col.ny + (pz - col.az) * col.nz;
          if (d < 0) {
            cnx = col.nx; cny = col.ny; cnz = col.nz;
            pen = -d;
          }
        }
        if (pen > 0) {
          px += cnx * pen; py += cny * pen; pz += cnz * pen;
          nx = cnx; ny = cny; nz = cnz;
          touched = true;
        }
      }
      if (touched) {
        positions[i3] = px; positions[i3 + 1] = py; positions[i3 + 2] = pz;
        // friction: damp tangential velocity, keep outward normal component
        let vx = velocities[i3], vy = velocities[i3 + 1], vz = velocities[i3 + 2];
        const vn = vx * nx + vy * ny + vz * nz;
        const tx = vx - nx * vn, ty = vy - ny * vn, tz = vz - nz * vn;
        const f = 1 - friction * 0.9;
        velocities[i3] = tx * f + nx * Math.max(vn, 0);
        velocities[i3 + 1] = ty * f + ny * Math.max(vn, 0);
        velocities[i3 + 2] = tz * f + nz * Math.max(vn, 0);
      }
    }
  }

  private updateVelocity(h: number, st: EngineSettings) {
    const { positions, prevPositions, velocities, invMass, n } = this.s;
    const invH = 1 / h;
    const damp = Math.max(0, 1 - st.damping * h);
    const maxV = st.maxVelocity;
    for (let i = 0; i < n; i++) {
      const i3 = i * 3;
      let vx = (positions[i3] - prevPositions[i3]) * invH;
      let vy = (positions[i3 + 1] - prevPositions[i3 + 1]) * invH;
      let vz = (positions[i3 + 2] - prevPositions[i3 + 2]) * invH;
      if (invMass[i] > 0) {
        vx *= damp; vy *= damp; vz *= damp;
        const v2 = vx * vx + vy * vy + vz * vz;
        if (v2 > maxV * maxV) {
          const s = maxV / Math.sqrt(v2);
          vx *= s; vy *= s; vz *= s;
        }
      }
      velocities[i3] = vx; velocities[i3 + 1] = vy; velocities[i3 + 2] = vz;
    }
  }
}
