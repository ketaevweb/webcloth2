/// <reference types="@webgpu/types" />
/**
 * WebCloth 2 — WebGPU compute backend.
 * Owns GPU buffers + compute pipelines, packs collider/param uniforms,
 * dispatches the XPBD pipeline and reads positions back for rendering.
 */
import { WGSL } from "./shaders";
import { ADJ_STRIDE } from "./ClothBuilder";
import type { ColliderLike, EngineSettings } from "./types";
import type { SolverState } from "./CpuSolver";

const UNIFORM_SIZE = 864; // 96 bytes header + 48 vec4 colliders
const STAGING_COUNT = 2;

export class WebGpuSolver {
  device!: GPUDevice;
  private module!: GPUShaderModule;
  private posBuf!: GPUBuffer;
  private prevBuf!: GPUBuffer;
  private velBuf!: GPUBuffer;
  private invMassBuf!: GPUBuffer;
  private constraintBuf!: GPUBuffer;
  private consOffsetBuf!: GPUBuffer;
  private consOtherBuf!: GPUBuffer;
  private consConBuf!: GPUBuffer;
  private triAdjBuf!: GPUBuffer;
  private uniformBuf!: GPUBuffer;
  private staging: GPUBuffer[] = [];
  private stagingMapped = [false, false];
  private frameIndex = 0;

  private pipeIntegrate!: GPUComputePipeline;
  private pipeSolve!: GPUComputePipeline;
  private pipeCollide!: GPUComputePipeline;
  private pipeUpdateVel!: GPUComputePipeline;
  private bgIntegrate!: GPUBindGroup;
  private bgSolve!: GPUBindGroup;
  private bgCollide!: GPUBindGroup;
  private bgUpdateVel!: GPUBindGroup;

  private uniformData = new ArrayBuffer(UNIFORM_SIZE);
  private uniformU32 = new Uint32Array(this.uniformData);
  private uniformF32 = new Float32Array(this.uniformData);

  n = 0;
  nc = 0;
  adapterName = "";
  /** incremented on every pool rebuild; used to discard stale readbacks */
  generation = 0;
  private onFail: () => void;
  private simTimeF32 = new Float32Array(1);

  private constructor(onFail: () => void) {
    this.onFail = onFail;
  }

  static async create(onFail: () => void): Promise<WebGpuSolver | null> {
    const nav = navigator as unknown as { gpu?: GPU };
    if (!nav.gpu) return null;
    let adapter: GPUAdapter | null = null;
    try {
      adapter = await nav.gpu.requestAdapter({ powerPreference: "high-performance" });
    } catch {
      return null;
    }
    if (!adapter) return null;
    let device: GPUDevice;
    try {
      device = await adapter.requestDevice();
    } catch {
      return null;
    }
    const solver = new WebGpuSolver(onFail);
    solver.device = device;
    let info = "";
    try {
      const ai = adapter.info as GPUAdapterInfo | undefined;
      info = [ai?.vendor, ai?.architecture].filter(Boolean).join(" ") || "";
    } catch {
      /* adapter.info unavailable */
    }
    solver.adapterName = info || "WebGPU";
    void device.lost.then(() => solver.onFail());
    solver.module = device.createShaderModule({ code: WGSL });
    return solver;
  }

  get available() {
    return !!this.device;
  }

  /** (Re)create pool buffers for a state and upload everything. */
  uploadState(s: SolverState) {
    const d = this.device;
    this.n = s.n;
    this.nc = s.nc;
    this.generation++;
    const posBytes = Math.max(4, s.n * 3 * 4);
    const conBytes = Math.max(4, s.nc * 4 * 4);
    const adjBytes = Math.max(4, s.n * ADJ_STRIDE * 4);

    this.posBuf?.destroy(); this.prevBuf?.destroy(); this.velBuf?.destroy();
    this.invMassBuf?.destroy(); this.constraintBuf?.destroy(); this.triAdjBuf?.destroy();
    this.consOffsetBuf?.destroy(); this.consOtherBuf?.destroy(); this.consConBuf?.destroy();
    for (const st of this.staging) { try { st.unmap(); } catch { /* not mapped */ } st.destroy(); }
    this.staging = [];

    this.posBuf = d.createBuffer({ size: posBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    this.prevBuf = d.createBuffer({ size: posBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.velBuf = d.createBuffer({ size: posBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.invMassBuf = d.createBuffer({ size: Math.max(4, s.n * 4), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.constraintBuf = d.createBuffer({ size: Math.max(4, conBytes), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.triAdjBuf = d.createBuffer({ size: Math.max(4, adjBytes), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.consOffsetBuf = d.createBuffer({ size: Math.max(4, s.consOffset.byteLength), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.consOtherBuf = d.createBuffer({ size: Math.max(4, s.consOther.byteLength), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.consConBuf = d.createBuffer({ size: Math.max(4, s.consCon.byteLength), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    for (let i = 0; i < STAGING_COUNT; i++) {
      this.staging.push(d.createBuffer({ size: posBytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }));
    }

    // pack constraints: [ia, ib, restBits, complianceBits] * nc
    const conData = new Uint32Array(s.nc * 4);
    const scratch = new ArrayBuffer(4);
    const sf = new Float32Array(scratch);
    const su = new Uint32Array(scratch);
    for (let c = 0; c < s.nc; c++) {
      conData[c * 4] = s.constraintA[c];
      conData[c * 4 + 1] = s.constraintB[c];
      sf[0] = s.restLength[c]; conData[c * 4 + 2] = su[0];
      sf[0] = s.compliance[c]; conData[c * 4 + 3] = su[0];
    }
    // NOTE: plain 3-argument writeBuffer calls only. Trailing undefined
    // dataOffset/size arguments (or ArrayBuffer-style partial writes) proved
    // unreliable on some Dawn/SwiftShader stacks — uploads silently landed as
    // zeros there, collapsing the whole simulation.
    const q = d.queue;
    q.writeBuffer(this.posBuf, 0, s.positions as unknown as GPUAllowSharedBufferSource);
    q.writeBuffer(this.prevBuf, 0, s.prevPositions as unknown as GPUAllowSharedBufferSource);
    q.writeBuffer(this.velBuf, 0, s.velocities as unknown as GPUAllowSharedBufferSource);
    q.writeBuffer(this.invMassBuf, 0, s.invMass as unknown as GPUAllowSharedBufferSource);
    if (s.nc > 0) q.writeBuffer(this.constraintBuf, 0, conData);
    if (s.n > 0) q.writeBuffer(this.triAdjBuf, 0, s.triAdj as unknown as GPUAllowSharedBufferSource);
    q.writeBuffer(this.consOffsetBuf, 0, s.consOffset as unknown as GPUAllowSharedBufferSource);
    if (s.consOther.byteLength > 0) {
      q.writeBuffer(this.consOtherBuf, 0, s.consOther as unknown as GPUAllowSharedBufferSource);
      q.writeBuffer(this.consConBuf, 0, s.consCon as unknown as GPUAllowSharedBufferSource);
    }

    this.ensureUniform();
    this.buildPipelines();
  }

  private buildPipelines() {
    const d = this.device;
    const mk = (entry: string) =>
      d.createComputePipeline({ layout: "auto", compute: { module: this.module, entryPoint: entry } });
    this.pipeIntegrate = mk("integrate");
    this.pipeSolve = mk("solve");
    this.pipeCollide = mk("collide");
    this.pipeUpdateVel = mk("updateVelocity");

    const group = (pipe: GPUComputePipeline, bindings: Array<[number, GPUBuffer]>) =>
      d.createBindGroup({
        layout: pipe.getBindGroupLayout(0),
        entries: bindings.map(([binding, buffer]) => ({ binding, resource: { buffer } })),
      });

    this.bgIntegrate = group(this.pipeIntegrate, [
      [0, this.posBuf], [1, this.prevBuf], [2, this.velBuf],
      [3, this.invMassBuf], [5, this.triAdjBuf], [6, this.uniformBuf!],
    ]);
    this.bgSolve = group(this.pipeSolve, [
      [0, this.posBuf], [3, this.invMassBuf], [4, this.constraintBuf], [6, this.uniformBuf!],
      [7, this.consOffsetBuf], [8, this.consOtherBuf], [9, this.consConBuf],
    ]);
    this.bgCollide = group(this.pipeCollide, [
      [0, this.posBuf], [2, this.velBuf], [3, this.invMassBuf], [6, this.uniformBuf!],
    ]);
    this.bgUpdateVel = group(this.pipeUpdateVel, [
      [0, this.posBuf], [1, this.prevBuf], [2, this.velBuf], [3, this.invMassBuf], [6, this.uniformBuf!],
    ]);
  }

  private ensureUniform() {
    if (!this.uniformBuf) {
      this.uniformBuf = this.device.createBuffer({
        size: UNIFORM_SIZE,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
    }
  }

  /** Write kinematic targets: positions <- curr, prevPositions <- prev (per index). */
  writeKinematics(indices: Uint32Array, indexOffset: number, curr: Float32Array, prev: Float32Array) {
    const q = this.device.queue;
    const c = curr as unknown as GPUAllowSharedBufferSource;
    const p = prev as unknown as GPUAllowSharedBufferSource;
    for (let k = 0; k < indices.length; k++) {
      const i3 = (indexOffset + indices[k]) * 3 * 4;
      q.writeBuffer(this.posBuf, i3, c, k * 3, 3);
      q.writeBuffer(this.prevBuf, i3, p, k * 3, 3);
    }
  }

  /** Dispatch one full simulation step (all substeps). */
  step(dt: number, settings: EngineSettings, colliders: ColliderLike[], hasKinematics: boolean) {
    this.ensureUniform();
    const d = this.device;
    const substeps = Math.max(1, settings.substeps | 0);
    const iters = Math.max(1, settings.iterations | 0);
    const h = dt / substeps;

    // pack uniform header
    const wdx = settings.windDirX, wdz = settings.windDirZ;
    const wlen = Math.sqrt(wdx * wdx + wdz * wdz) || 1;
    const u32 = this.uniformU32, f32 = this.uniformF32;
    u32[0] = this.n;
    u32[1] = this.nc;
    f32[2] = h;
    f32[3] = 1 / h;
    f32[4] = 0; f32[5] = settings.gravityY; f32[6] = 0; f32[7] = 0;
    f32[8] = wdx / wlen; f32[9] = 0; f32[10] = wdz / wlen; f32[11] = settings.windStrength;
    f32[12] = settings.windFrequency; f32[13] = settings.windTurbulence * settings.windStrength;
    f32[14] = settings.windGust; f32[15] = this.simTime;
    f32[16] = settings.aeroDrag; f32[17] = settings.damping; f32[18] = settings.maxVelocity; f32[19] = settings.friction;
    u32[20] = Math.min(colliders.length, 16);
    u32[21] = 0; u32[22] = 0; u32[23] = 0;
    const maxC = Math.min(colliders.length, 16);
    for (let c = 0; c < maxC; c++) {
      const col = colliders[c];
      const o = 24 + c * 12;
      f32[o] = col.ax; f32[o + 1] = col.ay; f32[o + 2] = col.az; f32[o + 3] = col.radius;
      f32[o + 4] = col.bx; f32[o + 5] = col.by; f32[o + 6] = col.bz;
      f32[o + 7] = col.kind === "sphere" ? 0 : col.kind === "capsule" ? 1 : 2;
      f32[o + 8] = col.nx; f32[o + 9] = col.ny; f32[o + 10] = col.nz; f32[o + 11] = 0;
    }
    d.queue.writeBuffer(this.uniformBuf, 0, this.uniformData);

    const nWg = Math.ceil(this.n / 64) || 1;
    const encoder = d.createCommandEncoder();
    const pass = encoder.beginComputePass();
    for (let s = 0; s < substeps; s++) {
      this.simTime += h;
      this.simTimeF32[0] = this.simTime;
      d.queue.writeBuffer(this.uniformBuf, 60, this.simTimeF32); // time only
      pass.setPipeline(this.pipeIntegrate);
      pass.setBindGroup(0, this.bgIntegrate);
      pass.dispatchWorkgroups(nWg);
      // per-particle relaxation (race-free) — one sweep per iteration
      pass.setPipeline(this.pipeSolve);
      pass.setBindGroup(0, this.bgSolve);
      for (let k = 0; k < iters; k++) pass.dispatchWorkgroups(nWg);
      pass.setPipeline(this.pipeCollide);
      pass.setBindGroup(0, this.bgCollide);
      pass.dispatchWorkgroups(nWg);
      pass.setPipeline(this.pipeUpdateVel);
      pass.setBindGroup(0, this.bgUpdateVel);
      pass.dispatchWorkgroups(nWg);
    }
    pass.end();
    // kick readback for THIS frame into staging[frameIndex % 2].
    // NOTE: copy commands MUST be encoded after pass.end() — the command
    // encoder is locked while a pass is open, and any copy encoded there
    // invalidates the whole command buffer (silently dropped on submit).
    const slot = this.frameIndex % STAGING_COUNT;
    encoder.copyBufferToBuffer(this.posBuf, 0, this.staging[slot], 0, this.n * 3 * 4);
    d.queue.submit([encoder.finish()]);
    this.frameIndex++;
    void hasKinematics;
  }

  private simTime = 0;

  /**
   * Await mapped results of the previous frame (double buffered) and copy them
   * into `dst`. Returns false when NaN corruption is detected (triggers fallback).
   */
  async readback(dst: Float32Array): Promise<boolean> {
    if (this.n === 0) return true;
    // data of the just-submitted frame lives in staging[(frameIndex-1) % 2]
    const slot = (this.frameIndex - 1 + STAGING_COUNT * 4) % STAGING_COUNT;
    const buf = this.staging[slot];
    if (!buf) return true;
    try {
      if (!this.stagingMapped[slot]) {
        await buf.mapAsync(GPUMapMode.READ);
        this.stagingMapped[slot] = true;
      }
      const src = new Float32Array(buf.getMappedRange());
      let ok = true;
      let firstNaN = -1;
      for (let i = 0; i < src.length; i++) {
        const v = src[i];
        dst[i] = v;
        if (v !== v && firstNaN < 0) { ok = false; firstNaN = i; }
      }
      if (!ok) {
        const p = Math.floor(firstNaN / 3);
        const c = firstNaN % 3;
        console.warn(
          `[WebCloth] NaN in readback: particle ${p} comp ${c}` +
          ` | pos0-2: ${Array.from(src.slice(0, 9)).map((x) => x.toPrecision(4))}` +
          ` | prev@p: ${Array.from(dst.slice(Math.max(0, firstNaN - 3), firstNaN + 6)).map((x) => x.toPrecision(4))}` +
          ` | simTime ${this.simTime.toFixed(3)}`
        );
      }
      buf.unmap();
      this.stagingMapped[slot] = false;
      return ok;
    } catch (e) {
      console.warn("[WebCloth] staging map failed:", String(e).slice(0, 120));
      this.stagingMapped[slot] = false;
      return false;
    }
  }

  resetTime() {
    this.simTime = 0;
  }

  destroy() {
    this.posBuf?.destroy(); this.prevBuf?.destroy(); this.velBuf?.destroy();
    this.invMassBuf?.destroy(); this.constraintBuf?.destroy(); this.triAdjBuf?.destroy();
    this.uniformBuf?.destroy();
    for (const st of this.staging) { try { st.unmap(); } catch { /* ignore */ } st.destroy(); }
    this.staging = [];
    try { this.device.destroy(); } catch { /* ignore */ }
  }
}
