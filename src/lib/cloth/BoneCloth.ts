/**
 * WebCloth 2 — Bone Cloth.
 * Magica-style bone chains: a root transform drives a chain of simulated
 * joints; after solving, bone transforms (quaternion per segment) drive any
 * skinned rig. XPBD distance constraints along + across the chain, collider
 * pushout, the same turbulent wind field as mesh cloth.
 *
 * The solver is deliberately tiny (few dozen joints per chain) and runs on
 * the CPU every frame before the mesh-cloth step.
 */
import { vnoise } from "./noise";
import { STREAM_DRAG, type ColliderLike, type EngineSettings } from "./types";

export interface BoneChainOptions {
  /** Segment lengths; the chain has lengths.length + 1 joints */
  lengths: number[];
  /** Initial joint positions, world space: root first, then one per segment */
  joints: number[]; // xyz * (N+1)
  /** 0..1, resistance to folding across two segments (default 0.35) */
  bendStiffness?: number;
  /** collision radius against colliders (default 0.04) */
  radius?: number;
  /** per-joint aerodynamic drag multiplier (default 1) */
  dragScale?: number;
  /**
   * Extra linear damping (1/s) added on top of settings.damping. Chains are
   * light and fully-snapped, so they resonate easily — the extra damping
   * keeps driven chains stable (default 1.2).
   */
  damping?: number;
  /** velocity clamp for chain joints, m/s (default 8) */
  maxJointSpeed?: number;
}

export class BoneChain {
  readonly jointCount: number;
  readonly positions: Float32Array; // xyz * (N+1), world
  readonly prev: Float32Array;
  readonly vel: Float32Array;
  readonly lengths: number[];
  /** rest span between joint i-2 and i (straight chain) for bend constraints */
  readonly bendRest: Float32Array;
  readonly bendStiffness: number;
  readonly radius: number;
  readonly dragScale: number;
  readonly damping: number;
  readonly maxJointSpeed: number;
  private time = 0;

  constructor(opts: BoneChainOptions) {
    this.lengths = opts.lengths.slice();
    this.jointCount = this.lengths.length + 1;
    const n = this.jointCount;
    this.positions = new Float32Array(n * 3);
    this.prev = new Float32Array(n * 3);
    this.vel = new Float32Array(n * 3);
    this.positions.set(opts.joints.slice(0, n * 3));
    this.prev.set(this.positions);
    this.bendStiffness = opts.bendStiffness ?? 0.35;
    this.radius = opts.radius ?? 0.04;
    this.dragScale = opts.dragScale ?? 1;
    this.damping = opts.damping ?? 1.2;
    this.maxJointSpeed = opts.maxJointSpeed ?? 8;
    // straight-chain rest spans: bendRest[i-2] = len[i-2] + len[i-1].
    // Measuring spans on the (curved) initial pose instead makes length and
    // bend constraints fight each other on a hanging chain — the solver
    // jackhammers and injects huge velocities. With straight spans the bend
    // term is exactly neutral on a straight chain and only opposes folding.
    this.bendRest = new Float32Array(Math.max(0, n - 2));
    for (let i = 2; i < n; i++) {
      this.bendRest[i - 2] = this.lengths[i - 2] + this.lengths[i - 1];
    }
  }

  /** Teleport the root (driven end). Root velocity is derived automatically. */
  setRoot(x: number, y: number, z: number) {
    this.positions[0] = x;
    this.positions[1] = y;
    this.positions[2] = z;
  }

  reset() {
    // restore straight-down-the-initial-shape pose: rebuild from lengths along
    // the original root->tip direction is scene-specific; instead park joints
    // stacked at the root and let constraints straighten them next frames.
    const n = this.jointCount;
    for (let i = 1; i < n; i++) {
      this.positions[i * 3] = this.positions[0];
      this.positions[i * 3 + 1] = this.positions[1] - this.lengths[i - 1];
      this.positions[i * 3 + 2] = this.positions[2];
    }
    this.prev.set(this.positions);
    this.vel.fill(0);
    this.time = 0;
  }

  /** Advance the chain one frame (same substep scheme as the mesh solver). */
  step(dt: number, st: EngineSettings, colliders: ColliderLike[]) {
    const n = this.jointCount;
    const substeps = Math.max(1, st.substeps | 0);
    const iters = Math.max(1, st.iterations | 0);
    const h = dt / substeps;

    // wind field setup (mirrors CpuSolver.integrate)
    const wdx = st.windDirX, wdz = st.windDirZ;
    const wlen = Math.sqrt(wdx * wdx + wdz * wdz) || 1;
    const dirX = wdx / wlen, dirZ = wdz / wlen;
    const strength = st.windStrength;
    const turbAmp = st.windTurbulence * strength;
    const freq = st.windFrequency;
    const dragK = st.aeroDrag * STREAM_DRAG * this.dragScale;

    for (let s = 0; s < substeps; s++) {
      this.time += h;

      // integrate (root is kinematic: velocity from root motion is injected
      // implicitly because prev is not reset for joint 0)
      for (let i = 1; i < n; i++) {
        const i3 = i * 3;
        let vx = this.vel[i3], vy = this.vel[i3 + 1], vz = this.vel[i3 + 2];
        const px = this.positions[i3], py = this.positions[i3 + 1], pz = this.positions[i3 + 2];
        let fx = 0, fy = 0, fz = 0;
        if (dragK > 0 && strength > 0) {
          const gust = 1 + st.windGust * Math.sin(this.time * 0.9 + px * 0.15 + py * 0.1);
          const tx = (vnoise(px * freq + this.time * 1.7, py * freq, pz * freq) - 0.5) * 2 * turbAmp;
          const ty = (vnoise(px * freq, py * freq + this.time * 1.3 + 3.7, pz * freq) - 0.5) * 2 * turbAmp;
          const tz = (vnoise(px * freq + 5.1, py * freq, pz * freq + this.time * 2.3) - 0.5) * 2 * turbAmp;
          const wvx = dirX * strength * gust + tx;
          const wvy = ty;
          const wvz = dirZ * strength * gust + tz;
          fx = (wvx - vx) * dragK;
          fy = (wvy - vy) * dragK;
          fz = (wvz - vz) * dragK;
        }
        vx += fx * h;
        vy += (fy + st.gravityY) * h;
        vz += fz * h;
        const v2 = vx * vx + vy * vy + vz * vz;
        const maxV0 = Math.min(st.maxVelocity, this.maxJointSpeed);
        if (v2 > maxV0 * maxV0) {
          const k = maxV0 / Math.sqrt(v2);
          vx *= k; vy *= k; vz *= k;
        }
        this.vel[i3] = vx; this.vel[i3 + 1] = vy; this.vel[i3 + 2] = vz;
        this.prev[i3] = px; this.prev[i3 + 1] = py; this.prev[i3 + 2] = pz;
        this.positions[i3] = px + vx * h;
        this.positions[i3 + 1] = py + vy * h;
        this.positions[i3 + 2] = pz + vz * h;
      }
      // root follows the driven transform; its motion feeds inertia via prev
      this.prev[0] = this.positions[0];
      this.prev[1] = this.positions[1];
      this.prev[2] = this.positions[2];

    // relax — bend FIRST (XPBD compliance), length LAST: every iteration must
    // END with exact segment lengths. The bend pass MUST be XPBD (compliance
    // scaled by 1/h²): a plain PBD bend leaks its corrections into the Verlet
    // velocity every substep and pumps the chain apart exponentially.
    // Compliance mapping: bendStiffness 1 -> rigid (1e-4), 0 -> soft (2).
    const bendW = Math.min(1, Math.max(0, this.bendStiffness));
    const bendCompliance = Math.pow(1 - bendW, 2) * 2 + 1e-4;
    const alphaT = bendCompliance / (h * h);
    for (let k = 0; k < iters; k++) {
      // across constraints — XPBD distance between joints i-2 and i;
      // the kinematic root (i-2 === 0) has inverse mass 0 and never moves
      for (let i = 2; i < n; i++) {
        const i3 = i * 3, p3 = (i - 2) * 3;
        const dx = this.positions[i3] - this.positions[p3];
        const dy = this.positions[i3 + 1] - this.positions[p3 + 1];
        const dz = this.positions[i3 + 2] - this.positions[p3 + 2];
        const len = Math.hypot(dx, dy, dz) || 1e-9;
        const C = len - this.bendRest[i - 2];
        const wParent = i - 2 === 0 ? 0 : 1;
        const wsum = wParent + 1;
        const lam = C / (wsum + alphaT);
        const cCorr = lam / len;
        const pCorr = (lam * wParent) / len;
        this.positions[i3] -= dx * cCorr;
        this.positions[i3 + 1] -= dy * cCorr;
        this.positions[i3 + 2] -= dz * cCorr;
        this.positions[p3] += dx * pCorr;
        this.positions[p3 + 1] += dy * pCorr;
        this.positions[p3 + 2] += dz * pCorr;
      }
      // segment lengths — joint i owns the constraint to i-1
      for (let i = 1; i < n; i++) {
        const i3 = i * 3, p3 = (i - 1) * 3;
        const dx = this.positions[i3] - this.positions[p3];
        const dy = this.positions[i3 + 1] - this.positions[p3 + 1];
        const dz = this.positions[i3 + 2] - this.positions[p3 + 2];
        const len = Math.hypot(dx, dy, dz) || 1e-9;
        const rest = this.lengths[i - 1];
        // root side is kinematic: joint i takes the full correction
        const dl = (len - rest) / len;
        this.positions[i3] -= dx * dl;
        this.positions[i3 + 1] -= dy * dl;
        this.positions[i3 + 2] -= dz * dl;
      }
    }

      // velocity update BEFORE colliders: collider pushout must not leak into
      // velocity, otherwise light chain joints keep inflating away from the body
      const invH = 1 / h;
      const damp = Math.max(0, 1 - (st.damping + this.damping) * h);
      const maxV = Math.min(st.maxVelocity, this.maxJointSpeed);
      for (let i = 1; i < n; i++) {
        const i3 = i * 3;
        let vx = (this.positions[i3] - this.prev[i3]) * invH;
        let vy = (this.positions[i3 + 1] - this.prev[i3 + 1]) * invH;
        let vz = (this.positions[i3 + 2] - this.prev[i3 + 2]) * invH;
        vx *= damp; vy *= damp; vz *= damp;
        const v2 = vx * vx + vy * vy + vz * vz;
        if (v2 > maxV * maxV) {
          const k = maxV / Math.sqrt(v2);
          vx *= k; vy *= k; vz *= k;
        }
        this.vel[i3] = vx; this.vel[i3 + 1] = vy; this.vel[i3 + 2] = vz;
      }

      // colliders (sphere / capsule / plane) + friction; position-only pushout,
      // velocity keeps its pre-push direction (friction projects it)
      for (let i = 1; i < n; i++) {
        const i3 = i * 3;
        let px = this.positions[i3], py = this.positions[i3 + 1], pz = this.positions[i3 + 2];
        let touched = false, nx = 0, ny = 0, nz = 0;
        for (let c = 0; c < colliders.length; c++) {
          const col = colliders[c];
          let cnx = 0, cny = 0, cnz = 0, pen = 0;
          if (col.kind === "sphere") {
            const dx = px - col.ax, dy = py - col.ay, dz = pz - col.az;
            const rr = col.radius + this.radius;
            const dist = Math.hypot(dx, dy, dz);
            if (dist < rr && dist > 1e-9) {
              cnx = dx / dist; cny = dy / dist; cnz = dz / dist;
              pen = rr - dist;
            }
          } else if (col.kind === "capsule") {
            const abx = col.bx - col.ax, aby = col.by - col.ay, abz = col.bz - col.az;
            const apx = px - col.ax, apy = py - col.ay, apz = pz - col.az;
            const ab2 = abx * abx + aby * aby + abz * abz;
            let t = 0;
            if (ab2 > 1e-9) t = Math.min(1, Math.max(0, (apx * abx + apy * aby + apz * abz) / ab2));
            const cx = col.ax + abx * t, cy = col.ay + aby * t, cz = col.az + abz * t;
            const dx = px - cx, dy = py - cy, dz = pz - cz;
            const rr = col.radius + this.radius;
            const dist = Math.hypot(dx, dy, dz);
            if (dist < rr && dist > 1e-9) {
              cnx = dx / dist; cny = dy / dist; cnz = dz / dist;
              pen = rr - dist;
            }
          } else {
            const d = (px - col.ax) * col.nx + (py - col.ay) * col.ny + (pz - col.az) * col.nz;
            if (d < this.radius) {
              cnx = col.nx; cny = col.ny; cnz = col.nz;
              pen = this.radius - d;
            }
          }
          if (pen > 0) {
            px += cnx * pen; py += cny * pen; pz += cnz * pen;
            nx = cnx; ny = cny; nz = cnz;
            touched = true;
          }
        }
        if (touched) {
          this.positions[i3] = px; this.positions[i3 + 1] = py; this.positions[i3 + 2] = pz;
          let vx = this.vel[i3], vy = this.vel[i3 + 1], vz = this.vel[i3 + 2];
          const vn = vx * nx + vy * ny + vz * nz;
          const tx = vx - nx * vn, ty = vy - ny * vn, tz = vz - nz * vn;
          const f = 1 - st.friction * 0.9;
          this.vel[i3] = tx * f + nx * Math.max(vn, 0);
          this.vel[i3 + 1] = ty * f + ny * Math.max(vn, 0);
          this.vel[i3 + 2] = tz * f + nz * Math.max(vn, 0);
        }
      }
    }
  }
}

/**
 * Collection of chains; the engine steps it before mesh cloth so both share
 * the same colliders and settings within a frame.
 */
export class BoneClothSystem {
  chains: BoneChain[] = [];

  add(chain: BoneChain): BoneChain {
    this.chains.push(chain);
    return chain;
  }

  clear() {
    this.chains = [];
  }

  get jointCount(): number {
    let n = 0;
    for (const c of this.chains) n += c.jointCount;
    return n;
  }

  step(dt: number, st: EngineSettings, colliders: ColliderLike[]) {
    for (const c of this.chains) c.step(dt, st, colliders);
  }

  reset() {
    for (const c of this.chains) c.reset();
  }
}
