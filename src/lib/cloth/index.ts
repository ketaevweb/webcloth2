/**
 * WebCloth 2 — public API.
 * Original high-performance cloth simulation for WebGPU / WebGL 2+.
 */
export { ClothEngine, Cloth, type AddClothOptions } from "./ClothEngine";
export { CpuSolver, type SolverState } from "./CpuSolver";
export { WebGpuSolver } from "./WebGpuSolver";
export { buildSimMesh, stiffnessToCompliance, ADJ_SLOTS, ADJ_STRIDE, SENTINEL, type BuiltMesh } from "./ClothBuilder";
export { vnoise } from "./noise";
export * from "./types";
