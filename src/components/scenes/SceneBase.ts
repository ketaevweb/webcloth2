/**
 * WebCloth 2 — collider primitives shared by scenes.
 * Coordinates are mutable so scenes can animate them in place.
 */
import * as THREE from "three";
import type { ColliderLike } from "@/lib/cloth";

export class SphereCollider implements ColliderLike {
  kind = "sphere" as const;
  bx = 0; by = 0; bz = 0;
  nx = 0; ny = 1; nz = 0;
  constructor(
    public ax = 0, public ay = 0, public az = 0,
    public radius = 0.1
  ) {}
  set(x: number, y: number, z: number) {
    this.ax = x; this.ay = y; this.az = z;
  }
}

export class CapsuleCollider implements ColliderLike {
  kind = "capsule" as const;
  nx = 0; ny = 1; nz = 0;
  constructor(
    public ax = 0, public ay = 0, public az = 0,
    public bx = 0, public by = 0, public bz = 0,
    public radius = 0.1
  ) {}
  setA(x: number, y: number, z: number) { this.ax = x; this.ay = y; this.az = z; }
  setB(x: number, y: number, z: number) { this.bx = x; this.by = y; this.bz = z; }
}

export class PlaneCollider implements ColliderLike {
  kind = "plane" as const;
  bx = 0; by = 0; bz = 0;
  radius = 0;
  constructor(
    public ax = 0, public ay = 0, public az = 0,
    public nx = 0, public ny = 1, public nz = 0
  ) {}
}

export interface ClothScene {
  id: string;
  name: string;
  description: string;
  defaults: Partial<import("@/lib/cloth").EngineSettings>;
  camera: { pos: [number, number, number]; target: [number, number, number] };
  build(engine: import("@/lib/cloth").ClothEngine, scene: THREE.Scene): void;
  update(elapsed: number, dt: number): void;
  setWireframe(on: boolean): void;
  dispose(scene: THREE.Scene): void;
  /** optional per-scene controls key (e.g. stress cloth count) */
  param?: number;
}

/** Shared environment: lights + ground. */
export function addEnvironment(scene: THREE.Scene) {
  const hemi = new THREE.HemisphereLight(0xf4efe8, 0x2a2622, 1.05);
  scene.add(hemi);
  const dir = new THREE.DirectionalLight(0xfff2e0, 1.6);
  dir.position.set(4, 7, 3);
  scene.add(dir);
  const fill = new THREE.DirectionalLight(0xffe0cc, 0.5);
  fill.position.set(-5, 3, -4);
  scene.add(fill);

  const ground = new THREE.Mesh(
    new THREE.CircleGeometry(14, 48),
    new THREE.MeshStandardMaterial({ color: 0x23211f, roughness: 0.95, metalness: 0 })
  );
  ground.rotation.x = -Math.PI / 2;
  ground.position.y = -0.005;
  scene.add(ground);

  const grid = new THREE.GridHelper(28, 28, 0x4a4640, 0x35322e);
  (grid.material as THREE.Material).transparent = true;
  (grid.material as THREE.Material).opacity = 0.5;
  scene.add(grid);
}

export const FOG_COLOR = 0x171513;
