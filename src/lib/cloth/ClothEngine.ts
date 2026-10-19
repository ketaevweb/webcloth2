/**
 * WebCloth 2 — engine facade.
 * Manages the particle/constraint pool, backend selection (WebGPU compute or
 * CPU XPBD), kinematic targets, colliders, wind settings and geometry sync.
 */
import * as THREE from "three";
import { buildSimMesh, stiffnessToCompliance, ADJ_STRIDE, type BuiltMesh } from "./ClothBuilder";
import { CpuSolver, type SolverState } from "./CpuSolver";
import { WebGpuSolver } from "./WebGpuSolver";
import { DEFAULT_SETTINGS, MAX_COLLIDERS, type BackendKind, type ClothBuildOptions, type ColliderLike, type EngineSettings } from "./types";

export class Cloth {
  readonly mesh: BuiltMesh;
  readonly geometry: THREE.BufferGeometry;
  offset = 0;
  constraintOffset = 0;
  kinPrev: Float32Array;
  kinCurr: Float32Array;
  private kinTmp: Float32Array;
  smoothNormals = true;
  /** @internal */
  kinDirty = false;

  constructor(geometry: THREE.BufferGeometry, mesh: BuiltMesh) {
    this.geometry = geometry;
    this.mesh = mesh;
    const k = mesh.kinIndices.length;
    this.kinPrev = new Float32Array(k * 3);
    this.kinCurr = new Float32Array(k * 3);
    this.kinTmp = new Float32Array(k * 3);
    this.kinPrev.set(mesh.kinRest);
    this.kinCurr.set(mesh.kinRest);
  }

  /** Compute this frame's kinematic target positions from particle rest positions. */
  updateKinematicTargets(fn: (restX: number, restY: number, restZ: number, out: Float32Array, k: number) => void) {
    const rest = this.mesh.kinRest;
    for (let i = 0; i < rest.length; i += 3) {
      fn(rest[i], rest[i + 1], rest[i + 2], this.kinTmp, i);
    }
    this.kinPrev.set(this.kinCurr);
    this.kinCurr.set(this.kinTmp);
    this.kinDirty = true;
  }

  /** @internal consume dirty flag */
  consumeDirty(): boolean {
    const d = this.kinDirty;
    this.kinDirty = false;
    return d;
  }
}

export interface AddClothOptions extends ClothBuildOptions {
  smoothNormals?: boolean;
}

export class ClothEngine {
  settings: EngineSettings = { ...DEFAULT_SETTINGS };
  backend: BackendKind = "cpu";
  cloths: Cloth[] = [];
  colliders: ColliderLike[] = [];
  stats = { particles: 0, constraints: 0, triangles: 0, simMs: 0, adapter: "" };
  onFallback: ((reason: string) => void) | null = null;

  private state!: SolverState;
  private kinds!: Uint8Array;
  private cpu = new CpuSolver();
  private gpu: WebGpuSolver | null = null;
  private gpuBroken = false;
  private time = 0;

  static async create(): Promise<ClothEngine> {
    const e = new ClothEngine();
    e.gpu = await WebGpuSolver.create(() => e.gpuFailed("GPU device error"));
    e.backend = e.gpu ? "webgpu" : "cpu";
    return e;
  }

  get adapterName(): string {
    if (this.backend === "webgpu") return this.gpu?.adapterName || "WebGPU";
    return "CPU (WebGL2 fallback)";
  }

  addCloth(geometry: THREE.BufferGeometry, opts: AddClothOptions = {}): Cloth {
    const mesh = buildSimMesh(geometry, opts);
    const cloth = new Cloth(geometry, mesh);
    cloth.smoothNormals = opts.smoothNormals !== false;
    this.cloths.push(cloth);
    this.rebuildPool();
    return cloth;
  }

  addCollider(c: ColliderLike) {
    if (this.colliders.length < MAX_COLLIDERS) this.colliders.push(c);
  }

  clearColliders() {
    this.colliders = [];
  }

  clearCloths() {
    this.cloths = [];
    this.rebuildPool();
  }

  setBackend(b: BackendKind) {
    if (b === this.backend) return;
    if (b === "webgpu" && (!this.gpu || this.gpuBroken)) return;
    this.backend = b;
    this.uploadAll();
  }

  /** Recompute XPBD compliance after stiffness sliders change. */
  refreshCompliance() {
    const st = this.state;
    const dS = stiffnessToCompliance(this.settings.distanceStiffness, false);
    const dB = stiffnessToCompliance(this.settings.bendStiffness, true);
    for (let c = 0; c < st.nc; c++) {
      st.compliance[c] = this.kinds[c] === 1 ? dB : dS;
    }
    if (this.backend === "webgpu") this.gpu?.uploadState(st);
  }

  reset() {
    for (const cloth of this.cloths) {
      const m = cloth.mesh;
      const n3 = m.particleCount * 3;
      const o3 = cloth.offset * 3;
      this.state.positions.set(m.restPositions, o3);
      this.state.prevPositions.set(m.restPositions, o3);
      this.state.velocities.fill(0, o3, o3 + n3);
      cloth.kinPrev.set(m.kinRest);
      cloth.kinCurr.set(m.kinRest);
    }
    this.time = 0;
    this.cpu.resetTime();
    this.gpu?.resetTime();
    this.uploadAll();
  }

  async update(dtSeconds: number): Promise<void> {
    const st = this.settings;
    if (st.paused) return;
    let dt = Math.min(dtSeconds, 1 / 30) * st.timeScale;
    if (dt <= 0) return;

    // kinematic targets -> pool (+ GPU buffers)
    for (const cloth of this.cloths) {
      if (!cloth.consumeDirty()) continue;
      const idx = cloth.mesh.kinIndices;
      const o3 = cloth.offset * 3;
      for (let k = 0; k < idx.length; k++) {
        const p3 = (o3 + idx[k]) * 3;
        const k3 = k * 3;
        this.state.prevPositions[p3] = cloth.kinPrev[k3];
        this.state.prevPositions[p3 + 1] = cloth.kinPrev[k3 + 1];
        this.state.prevPositions[p3 + 2] = cloth.kinPrev[k3 + 2];
        this.state.positions[p3] = cloth.kinCurr[k3];
        this.state.positions[p3 + 1] = cloth.kinCurr[k3 + 1];
        this.state.positions[p3 + 2] = cloth.kinCurr[k3 + 2];
      }
      if (this.backend === "webgpu" && this.gpu) {
        // GPU keeps its own copies of the driven particles — upload them here,
        // otherwise the belt/skirt never follows the character on WebGPU.
        this.gpu.writeKinematics(idx, cloth.offset, cloth.kinCurr, cloth.kinPrev);
      }
    }

    const t0 = performance.now();
    if (this.backend === "webgpu" && this.gpu) {
      const gen = this.gpu.generation;
      this.gpu.step(dt, st, this.colliders, false);
      const ok = await this.gpu.readback(this.state.positions);
      if (!ok) {
        if (this.gpu.generation === gen) {
          this.gpuFailed("NaN detected in GPU output");
        }
        // NaN already poisoned the shared state — restore clean rest positions
        // before running the CPU solver, otherwise NaN propagates forever.
        this.restoreFromRest();
        this.cpu.step(dt, st, this.colliders);
      }
    } else {
      this.cpu.step(dt, st, this.colliders);
    }
    this.time += dt;
    this.stats.simMs = this.stats.simMs * 0.9 + (performance.now() - t0) * 0.1;
    this.syncGeometries();
  }

  private syncGeometries() {
    const pos = this.state.positions;
    for (const cloth of this.cloths) {
      const attr = cloth.geometry.getAttribute("position") as THREE.BufferAttribute;
      if (!attr || !(attr.array instanceof Float32Array)) continue;
      const arr = attr.array as Float32Array;
      const map = cloth.mesh.attrToParticle;
      const o = cloth.offset;
      for (let i = 0; i < map.length; i++) {
        const p3 = (o + map[i]) * 3;
        const i3 = i * 3;
        arr[i3] = pos[p3];
        arr[i3 + 1] = pos[p3 + 1];
        arr[i3 + 2] = pos[p3 + 2];
      }
      attr.needsUpdate = true;
      if (cloth.smoothNormals) {
        cloth.geometry.computeVertexNormals();
      }
    }
  }

  private uploadAll() {
    if (this.backend === "webgpu" && this.gpu) {
      this.gpu.uploadState(this.state);
    } else {
      this.cpu.setState(this.state);
      this.cpu.resetTime();
    }
  }

  /** Wipe poisoned sim state back to rest pose (per cloth, offset aware). */
  private restoreFromRest() {
    for (const cloth of this.cloths) {
      const m = cloth.mesh;
      const o3 = cloth.offset * 3;
      const n3 = m.particleCount * 3;
      this.state.positions.set(m.restPositions, o3);
      this.state.prevPositions.set(m.restPositions, o3);
      this.state.velocities.fill(0, o3, o3 + n3);
    }
    this.cpu.setState(this.state);
    this.cpu.resetTime();
  }

  private gpuFailed(reason: string) {
    if (this.backend === "cpu") return;
    console.warn("[WebCloth] GPU fallback:", reason);
    this.gpuBroken = true;
    this.backend = "cpu";
    this.uploadAll();
    this.onFallback?.(reason);
  }

  private rebuildPool() {
    let n = 0, nc = 0, tris = 0;
    for (const c of this.cloths) {
      c.offset = n;
      c.constraintOffset = nc;
      n += c.mesh.particleCount;
      nc += c.mesh.constraintCount;
      tris += c.mesh.triangles.length / 3;
    }
    const state: SolverState = {
      n,
      nc,
      positions: new Float32Array(n * 3),
      prevPositions: new Float32Array(n * 3),
      velocities: new Float32Array(n * 3),
      invMass: new Float32Array(n),
      constraintA: new Uint32Array(nc),
      constraintB: new Uint32Array(nc),
      restLength: new Float32Array(nc),
      compliance: new Float32Array(nc),
      triAdj: new Uint32Array(n * ADJ_STRIDE).fill(0xffffffff),
      consOffset: new Uint32Array(n + 1),
      consOther: new Uint32Array(nc * 2),
      consCon: new Uint32Array(nc * 2),
    };
    this.kinds = new Uint8Array(nc);
    const dS = stiffnessToCompliance(this.settings.distanceStiffness, false);
    const dB = stiffnessToCompliance(this.settings.bendStiffness, true);
    for (const cloth of this.cloths) {
      const m = cloth.mesh;
      const o = cloth.offset, o3 = o * 3;
      const co = cloth.constraintOffset;
      state.positions.set(m.restPositions, o3);
      state.prevPositions.set(m.restPositions, o3);
      state.invMass.set(m.invMass, o);
      state.triAdj.set(m.triAdj, o * ADJ_STRIDE);
      state.constraintA.set(m.constraintA, co);
      state.constraintB.set(m.constraintB, co);
      state.restLength.set(m.restLength, co);
      for (let c = 0; c < m.constraintCount; c++) {
        const kind = m.constraintKind[c];
        this.kinds[co + c] = kind;
        state.compliance[co + c] = kind === 1 ? dB : dS;
      }
      // pool-global CSR: shift particle indices by cloth offset
      const coN = m.particleCount;
      for (let p = 0; p <= coN; p++) {
        state.consOffset[o + p] = m.consOffset[p] + co * 2;
      }
      for (let e = 0; e < m.consOther.length; e++) {
        state.consOther[co * 2 + e] = m.consOther[e] + o;
        state.consCon[co * 2 + e] = m.consCon[e] + co;
      }
    }
    this.state = state;
    this.stats.particles = n;
    this.stats.constraints = nc;
    this.stats.triangles = tris;
    this.uploadAll();
  }

  dispose() {
    this.gpu?.destroy();
    this.gpu = null;
    this.cloths = [];
  }
}
