/**
 * WebCloth 2 — converts a THREE.BufferGeometry into a simulation mesh:
 *  - welds duplicate vertices (UV seams / hard edges) so cloth does not tear;
 *  - builds structural (edge) distance constraints;
 *  - builds bending constraints (opposite vertices of adjacent triangle pairs);
 *  - builds per-particle triangle adjacency used by the aerodynamic wind model.
 */
import * as THREE from "three";
import type { ClothBuildOptions } from "./types";

export const ADJ_SLOTS = 8; // max triangles per particle
export const ADJ_STRIDE = ADJ_SLOTS * 2;
export const SENTINEL = 0xffffffff;
const PACK = 0x40000000; // assumes particleCount < 2^30

export interface BuiltMesh {
  particleCount: number;
  constraintCount: number;
  /** number of structural (edge) constraints; the rest are bending */
  structuralCount: number;
  /** pool-local particle rest positions, xyz * N */
  restPositions: Float32Array;
  /** inverse mass per particle (0 => static) */
  invMass: Float32Array;
  /** structural + bending constraints */
  constraintA: Uint32Array;
  constraintB: Uint32Array;
  restLength: Float32Array;
  constraintKind: Uint8Array; // 0 structural, 1 bending
  triangles: Uint32Array; // particle indices, 3 * T
  /** particle index (pool-local) for every attribute vertex */
  attrToParticle: Uint32Array;
  /** per particle adjacent triangles as (b,c) vertex pairs */
  triAdj: Uint32Array;
  /** indices of kinematic particles + their rest positions */
  kinIndices: Uint32Array;
  kinRest: Float32Array; // xyz * K
  attrVertexCount: number;
}

export function buildSimMesh(
  geometry: THREE.BufferGeometry,
  opts: ClothBuildOptions = {}
): BuiltMesh {
  const posAttr = geometry.getAttribute("position") as THREE.BufferAttribute;
  const attrVertexCount = posAttr.count;
  if (!attrVertexCount) throw new Error("Cloth: geometry has no positions");

  let index: ArrayLike<number>;
  if (geometry.index) {
    index = geometry.index.array;
  } else {
    const trivial = new Uint32Array(attrVertexCount);
    for (let i = 0; i < attrVertexCount; i++) trivial[i] = i;
    index = trivial;
  }
  const triCount = index.length / 3 | 0;

  // ---- weld duplicate attribute vertices ----
  const attrToParticle = new Uint32Array(attrVertexCount);
  const weldMap = new Map<string, number>();
  const rest: number[] = []; // growable xyz
  const px = new Float32Array(attrVertexCount);
  const py = new Float32Array(attrVertexCount);
  const pz = new Float32Array(attrVertexCount);
  for (let i = 0; i < attrVertexCount; i++) {
    const x = posAttr.getX(i), y = posAttr.getY(i), z = posAttr.getZ(i);
    px[i] = x; py[i] = y; pz[i] = z;
    const key =
      Math.round(x * 8192) + "_" + Math.round(y * 8192) + "_" + Math.round(z * 8192);
    const found = weldMap.get(key);
    if (found !== undefined) {
      attrToParticle[i] = found;
    } else {
      const id = rest.length / 3 | 0;
      rest.push(x, y, z);
      weldMap.set(key, id);
      attrToParticle[i] = id;
    }
  }
  const n = rest.length / 3 | 0;

  // ---- particles: dynamic / pinned / kinematic ----
  const invMass = new Float32Array(n).fill(1);
  const kinIdx: number[] = [];
  for (let p = 0; p < n; p++) {
    const x = rest[p * 3], y = rest[p * 3 + 1], z = rest[p * 3 + 2];
    if (opts.kinematicPredicate && opts.kinematicPredicate(x, y, z)) {
      invMass[p] = 0;
      kinIdx.push(p);
    } else if (opts.pinnedPredicate && opts.pinnedPredicate(x, y, z)) {
      invMass[p] = 0;
    }
  }
  const kinIndices = new Uint32Array(kinIdx);
  const kinRest = new Float32Array(kinIdx.length * 3);
  for (let k = 0; k < kinIdx.length; k++) {
    kinRest[k * 3] = rest[kinIdx[k] * 3];
    kinRest[k * 3 + 1] = rest[kinIdx[k] * 3 + 1];
    kinRest[k * 3 + 2] = rest[kinIdx[k] * 3 + 2];
  }

  // ---- constraints ----
  const edgeMap = new Map<number, number>(); // packed edge -> exists
  const bendMap = new Map<number, number>(); // packed edge -> first opposite vertex
  const a: number[] = [], b: number[] = [];
  for (let t = 0; t < triCount; t++) {
    const i0 = attrToParticle[index[t * 3]];
    const i1 = attrToParticle[index[t * 3 + 1]];
    const i2 = attrToParticle[index[t * 3 + 2]];
    if (i0 === i1 || i1 === i2 || i0 === i2) continue; // degenerate after weld
    addEdge(edgeMap, a, b, i0, i1);
    addEdge(edgeMap, a, b, i1, i2);
    addEdge(edgeMap, a, b, i2, i0);
  }
  // bending: opposite vertices across every interior edge
  for (let t = 0; t < triCount; t++) {
    const v0 = attrToParticle[index[t * 3]];
    const v1 = attrToParticle[index[t * 3 + 1]];
    const v2 = attrToParticle[index[t * 3 + 2]];
    addBend(bendMap, a, b, v0, v1, v2);
    addBend(bendMap, a, b, v1, v2, v0);
    addBend(bendMap, a, b, v2, v0, v1);
  }
  const nc = a.length;
  const structuralCount = edgeMap.size;
  const constraintA = new Uint32Array(a);
  const constraintB = new Uint32Array(b);
  const constraintKind = new Uint8Array(nc);
  const restLength = new Float32Array(nc);
  for (let c = 0; c < nc; c++) {
    const isBend = c >= structuralCount;
    constraintKind[c] = isBend ? 1 : 0;
    const dx = rest[constraintA[c] * 3] - rest[constraintB[c] * 3];
    const dy = rest[constraintA[c] * 3 + 1] - rest[constraintB[c] * 3 + 1];
    const dz = rest[constraintA[c] * 3 + 2] - rest[constraintB[c] * 3 + 2];
    restLength[c] = Math.sqrt(dx * dx + dy * dy + dz * dz);
  }

  // ---- triangles (particle space) ----
  const triangles = new Uint32Array(triCount * 3);
  for (let t = 0; t < triCount; t++) {
    triangles[t * 3] = attrToParticle[index[t * 3]];
    triangles[t * 3 + 1] = attrToParticle[index[t * 3 + 1]];
    triangles[t * 3 + 2] = attrToParticle[index[t * 3 + 2]];
  }

  // ---- per particle triangle adjacency (for aerodynamic wind) ----
  const triAdj = new Uint32Array(n * ADJ_STRIDE).fill(SENTINEL);
  const cursor = new Uint32Array(n);
  for (let t = 0; t < triCount; t++) {
    const v0 = triangles[t * 3], v1 = triangles[t * 3 + 1], v2 = triangles[t * 3 + 2];
    tryAdj(triAdj, cursor, v0, v1, v2);
    tryAdj(triAdj, cursor, v1, v2, v0);
    tryAdj(triAdj, cursor, v2, v0, v1);
  }

  return {
    particleCount: n,
    constraintCount: nc,
    structuralCount,
    restPositions: new Float32Array(rest),
    invMass,
    constraintA,
    constraintB,
    restLength,
    constraintKind,
    triangles,
    attrToParticle,
    triAdj,
    kinIndices,
    kinRest,
    attrVertexCount,
  };
}

function addEdge(
  edgeMap: Map<number, number>,
  a: number[], b: number[],
  i0: number, i1: number
) {
  const lo = Math.min(i0, i1), hi = Math.max(i0, i1);
  const key = lo * PACK + hi;
  if (edgeMap.has(key)) return;
  edgeMap.set(key, 1);
  a.push(lo); b.push(hi);
}

function addBend(
  bendMap: Map<number, number>,
  a: number[], b: number[],
  v0: number, v1: number, v2: number
) {
  // edge v0-v1, opposite vertex v2; link opposite vertices of two triangles sharing the edge
  const lo = Math.min(v0, v1), hi = Math.max(v0, v1);
  const key = lo * PACK + hi;
  const prev = bendMap.get(key);
  if (prev === undefined) {
    bendMap.set(key, v2); // remember first opposite vertex
    return;
  }
  if (prev === v2) return; // degenerate
  bendMap.set(key, v2); // chain (non-manifold safety)
  a.push(prev); b.push(v2);
}

function tryAdj(
  triAdj: Uint32Array, cursor: Uint32Array, p: number, nb1: number, nb2: number
) {
  const slot = cursor[p];
  if (slot >= ADJ_SLOTS) return; // overflow ignored (dense meshes)
  triAdj[p * ADJ_STRIDE + slot * 2] = nb1;
  triAdj[p * ADJ_STRIDE + slot * 2 + 1] = nb2;
  cursor[p] = slot + 1;
}

/** XPBD compliance from stiffness 0..1 (exponential mapping) */
export function stiffnessToCompliance(stiffness: number, bend: boolean): number {
  const s = Math.min(1, Math.max(0, stiffness));
  return bend
    ? Math.pow(10, -1.0 - 4.0 * s)
    : Math.pow(10, -2.0 - 5.0 * s);
}
