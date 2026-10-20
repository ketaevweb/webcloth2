/**
 * WebCloth 2 — core types
 * Original high-performance cloth simulation engine for WebGPU / WebGL 2+
 * based on XPBD (Extended Position Based Dynamics).
 */

export type BackendKind = "webgpu" | "cpu";

/**
 * Aerodynamic force scale: converts triangle wind loading into particle
 * acceleration. Physically equivalent to a very light cloth density with
 * unit particle mass; keeps XPBD compliance behaviour predictable.
 */
export const AERO_SCALE = 600;

/**
 * Per-particle streaming (linear) drag: f = (windVel - v) * aeroDrag * STREAM_DRAG.
 * Pure normal-pressure aerodynamics goes blind when a surface faces the wind
 * edge-on (flag hanging from its pole). This viscous term keeps the cloth
 * streaming with the airflow — mirrors Magica-style per-vertex air drag.
 */
export const STREAM_DRAG = 2.2;

/** Static particle categories */
export const STATIC_PINNED = 0; // never moves (invMass = 0, no external writes)
export const STATIC_KINEMATIC = 1; // driven by engine targets each frame

export interface ClothBuildOptions {
  /** Particles matching this predicate become pinned (invMass = 0, never moved) */
  pinnedPredicate?: (x: number, y: number, z: number) => boolean;
  /** Particles matching this predicate become kinematic (driven by targets) */
  kinematicPredicate?: (x: number, y: number, z: number) => boolean;
}

export interface EngineSettings {
  gravityY: number;
  windStrength: number;
  windTurbulence: number;
  windFrequency: number;
  windGust: number;
  windDirX: number;
  windDirZ: number;
  substeps: number;
  iterations: number;
  damping: number;
  aeroDrag: number;
  friction: number;
  maxVelocity: number;
  distanceStiffness: number;
  bendStiffness: number;
  timeScale: number;
  paused: boolean;
  /** Particle-vs-particle self / cross collision (spatial hash, CPU + GPU) */
  selfCollision: boolean;
  /**
   * Tearing: a structural constraint breaks when stretched beyond
   * restLength * tearFactor. 0 disables tearing.
   */
  tearFactor: number;
}

export const DEFAULT_SETTINGS: EngineSettings = {
  gravityY: -9.81,
  windStrength: 0,
  windTurbulence: 0.55,
  windFrequency: 0.9,
  windGust: 0.35,
  windDirX: 0.18,
  windDirZ: 1,
  substeps: 2,
  iterations: 8,
  damping: 0.06,
  aeroDrag: 0.55,
  friction: 0.35,
  maxVelocity: 24,
  distanceStiffness: 0.92,
  bendStiffness: 0.55,
  timeScale: 1,
  paused: false,
  selfCollision: true,
  tearFactor: 0,
};

export interface EngineStats {
  fps: number;
  frameMs: number;
  simMs: number;
  particles: number;
  constraints: number;
  triangles: number;
  bones: number;
  backend: BackendKind;
  adapter: string;
}

/**
 * Self-collision: per-cloth particle radius = SELF_RADIUS_SCALE * avg edge
 * length. At rest, adjacent particles sit 1/0.45 ≈ 2.2 radii apart, so the
 * rest state never self-repels; folds and cross-cloth contacts do.
 */
export const SELF_RADIUS_SCALE = 0.45;

/** Packed collider data — 3 vec4 per collider */
export const MAX_COLLIDERS = 16;
export const COLLIDER_SPHERE = 0;
export const COLLIDER_CAPSULE = 1;
export const COLLIDER_PLANE = 2;

export interface ColliderLike {
  kind: "sphere" | "capsule" | "plane";
  ax: number; ay: number; az: number;
  bx: number; by: number; bz: number;
  radius: number;
  nx: number; ny: number; nz: number;
}
