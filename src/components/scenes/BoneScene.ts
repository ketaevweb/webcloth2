/**
 * WebCloth 2 — Bone Cloth demo scene.
 * A mannequin wears three simulated bone chains:
 *  - a long back ponytail rendered as a SKINNED ribbon (THREE.SkinnedMesh
 *    driven by real THREE.Bone transforms — the classic Bone Cloth use case:
 *    solve chains, deform skinned geometry);
 *  - two side ponytails rendered as tapered capsule segments per bone.
 * The chains are pure BoneChain XPBD simulators: root driven by the head,
 * colliders on the torso/head/back, the shared turbulent wind field.
 */
import * as THREE from "three";
import { ClothEngine, type BoneChain } from "@/lib/cloth";
import {
  FOG_COLOR, SphereCollider, CapsuleCollider, PlaneCollider,
  type ClothScene,
} from "./SceneBase";

const HAIR = 0x3b2f28;
const DOWN = new THREE.Vector3(0, -1, 0);
const UP = new THREE.Vector3(0, 1, 0);

export class BoneScene implements ClothScene {
  id = "bones";
  name = "Кости";
  description = "Bone Cloth: косы на скелетных цепях, скиннинг и коллайдеры";
  defaults = {
    windStrength: 1.2,
    windTurbulence: 0.55,
    windGust: 0.4,
    windDirX: 0.35,
    windDirZ: -1, // ветер от лица к затылку — волосы летят назад, как от бега
    aeroDrag: 0.5,
    distanceStiffness: 0.95,
    bendStiffness: 0.6,
    substeps: 3,
    iterations: 10,
    friction: 0.35,
    selfCollision: true,
    tearFactor: 0,
  };

  camera = { pos: [-1.5, 1.9, -2.6] as [number, number, number], target: [0, 1.15, -0.1] as [number, number, number] };

  private group = new THREE.Group();
  private bodyMat = new THREE.MeshStandardMaterial({ color: 0x7a818b, roughness: 0.5, metalness: 0.15 });
  private hairMat = new THREE.MeshStandardMaterial({ color: HAIR, roughness: 0.55, metalness: 0.08, side: THREE.DoubleSide });
  private head!: THREE.Mesh;
  private torso!: THREE.Mesh;
  private pelvis!: THREE.Mesh;
  private limbs: THREE.Mesh[] = [];
  private skinned: THREE.SkinnedMesh | null = null;
  private ribbonRoot: THREE.Bone | null = null;
  private skeletonBones: THREE.Bone[] = [];
  private restLocalDirs: THREE.Vector3[] = [];
  private backChain: BoneChain | null = null;
  private sideChains: BoneChain[] = [];
  private sideBones: Array<{ chain: BoneChain; meshes: THREE.Mesh[]; lens: number[] }> = [];

  // colliders (world, animated)
  private colHead = new SphereCollider(0, 1.62, 0, 0.14);
  private colTorso = new CapsuleCollider(0, 1.05, 0.02, 0, 1.5, 0.04, 0.15);
  private colBack = new CapsuleCollider(0, 1.1, -0.14, 0, 1.45, -0.16, 0.12);
  private colThigh: CapsuleCollider[] = [
    new CapsuleCollider(0, 0.95, 0, 0, 0.62, 0, 0.08),
    new CapsuleCollider(0, 0.95, 0, 0, 0.62, 0, 0.08),
  ];
  private tmpA = new THREE.Vector3();
  private tmpB = new THREE.Vector3();

  build(engine: ClothEngine, scene: THREE.Scene) {
    scene.fog = new THREE.Fog(FOG_COLOR, 8, 24);

    // ---- mannequin (light jog animation in update) ----
    this.head = new THREE.Mesh(new THREE.SphereGeometry(0.135, 20, 16), this.bodyMat);
    this.pelvis = new THREE.Mesh(new THREE.SphereGeometry(0.19, 20, 16), this.bodyMat);
    const torsoGeo = new THREE.CapsuleGeometry(0.15, 0.34, 6, 14);
    this.torso = new THREE.Mesh(torsoGeo, this.bodyMat);
    this.group.add(this.head, this.pelvis, this.torso);
    for (let i = 0; i < 2; i++) {
      const thigh = new THREE.Mesh(new THREE.CapsuleGeometry(0.075, 0.4, 6, 12), this.bodyMat);
      const shin = new THREE.Mesh(new THREE.CapsuleGeometry(0.06, 0.36, 6, 12), this.bodyMat);
      const arm = new THREE.Mesh(new THREE.CapsuleGeometry(0.05, 0.4, 6, 12), this.bodyMat);
      thigh.userData.boneLen = 0.55; shin.userData.boneLen = 0.48; arm.userData.boneLen = 0.5;
      this.limbs.push(thigh, shin, arm);
      this.group.add(thigh, shin, arm);
    }
    scene.add(this.group);

    // ---- chains ----
    const headBack = new THREE.Vector3(0, 1.7, -0.12);
    // back ponytail: 6 segments -> 7 joints, starts flowing down the back
    const backJoints: number[] = [];
    const backLens = [0.17, 0.18, 0.18, 0.17, 0.16, 0.14];
    {
      let p = headBack.clone();
      let dir = new THREE.Vector3(0, -0.85, -0.53).normalize();
      backJoints.push(p.x, p.y, p.z);
      for (const l of backLens) {
        dir.y -= 0.28; // curve towards vertical as it hangs
        dir.normalize();
        p = p.clone().addScaledVector(dir, l);
        backJoints.push(p.x, p.y, p.z);
      }
    }
    this.backChain = engine.addBoneChain({
      lengths: backLens,
      joints: backJoints,
      bendStiffness: 0.32,
      radius: 0.035,
      dragScale: 0.85,
    });
    const ribbon = this.buildRibbon(this.backChain, 0.11);
    this.skinned = ribbon.mesh;
    this.ribbonRoot = ribbon.root;
    // bone hierarchy lives under the SkinnedMesh (ribbon.root === bones[0],
    // already added inside buildRibbon); mesh sits at identity, so bone local
    // space == world space
    scene.add(ribbon.mesh);

    // side ponytails: 4 segments each, rendered as capsule-per-bone
    for (const side of [-1, 1]) {
      const root = new THREE.Vector3(side * 0.13, 1.66, -0.05);
      const lens = [0.14, 0.14, 0.13, 0.12];
      const joints: number[] = [root.x, root.y, root.z];
      let p = root.clone();
      let dir = new THREE.Vector3(side * 0.55, -0.5, -0.35).normalize();
      for (const l of lens) {
        dir.y -= 0.34;
        dir.normalize();
        p = p.clone().addScaledVector(dir, l);
        joints.push(p.x, p.y, p.z);
      }
      const chain = engine.addBoneChain({
        lengths: lens,
        joints,
        bendStiffness: 0.42,
        radius: 0.03,
        dragScale: 0.9,
      });
      this.sideChains.push(chain);
      // one tapered capsule per segment
      const meshes: THREE.Mesh[] = [];
      for (let s = 0; s < lens.length; s++) {
        const r = 0.036 * (1 - (s / lens.length) * 0.55);
        const geo = new THREE.CapsuleGeometry(r, Math.max(0.01, lens[s] - 2 * r), 4, 10);
        geo.translate(0, -lens[s] / 2, 0); // origin at TOP of the segment
        const m = new THREE.Mesh(geo, this.hairMat);
        m.userData.baseLen = lens[s];
        meshes.push(m);
        this.group.add(m);
      }
      this.sideBones.push({ chain, meshes, lens });
    }

    // colliders
    engine.addCollider(this.colHead);
    engine.addCollider(this.colTorso);
    engine.addCollider(this.colBack);
    engine.addCollider(this.colThigh[0]);
    engine.addCollider(this.colThigh[1]);
    engine.addCollider(new PlaneCollider(0, 0, 0, 0, 1, 0));
  }

  /** Build a skinned ribbon (SkinnedMesh + real bones) bound to a chain. */
  private buildRibbon(chain: BoneChain, width: number): { root: THREE.Bone; mesh: THREE.SkinnedMesh } {
    const n = chain.jointCount;
    const bones: THREE.Bone[] = [];
    for (let i = 0; i < n; i++) {
      const b = new THREE.Bone();
      if (i > 0) {
        bones[i - 1].add(b);
        // rest local offset = child joint - parent joint (initial pose)
        b.position.set(
          chain.positions[i * 3] - chain.positions[(i - 1) * 3],
          chain.positions[i * 3 + 1] - chain.positions[(i - 1) * 3 + 1],
          chain.positions[i * 3 + 2] - chain.positions[(i - 1) * 3 + 2]
        );
      } else {
        b.position.set(chain.positions[0], chain.positions[1], chain.positions[2]);
      }
      bones.push(b);
    }
    // rest local directions for the per-frame quaternion update
    this.restLocalDirs = [];
    for (let i = 1; i < n; i++) {
      this.restLocalDirs.push(bones[i].position.clone().normalize());
    }

    // ribbon geometry: rings at every joint, width across X-ish, tapering tip
    const rings = n;
    const pos = new Float32Array(rings * 2 * 3);
    const uv = new Float32Array(rings * 2 * 2);
    const skinIndex = new Uint16Array(rings * 2 * 4);
    const skinWeight = new Float32Array(rings * 2 * 4);
    const root = chain.positions;
    // side vector: perpendicular to chain direction, horizontal-ish
    const side = new THREE.Vector3(1, 0, 0);
    for (let i = 0; i < rings; i++) {
      const taper = 1 - (i / (rings - 1)) * 0.82; // tip narrows
      const w = (width / 2) * taper;
      const px = root[i * 3], py = root[i * 3 + 1], pz = root[i * 3 + 2];
      pos[(i * 2) * 3] = px - side.x * w; pos[(i * 2) * 3 + 1] = py; pos[(i * 2) * 3 + 2] = pz - 0;
      pos[(i * 2 + 1) * 3] = px + side.x * w; pos[(i * 2 + 1) * 3 + 1] = py; pos[(i * 2 + 1) * 3 + 2] = pz + 0;
      uv[(i * 2) * 2] = 0; uv[(i * 2) * 2 + 1] = i / (rings - 1);
      uv[(i * 2 + 1) * 2] = 1; uv[(i * 2 + 1) * 2 + 1] = i / (rings - 1);
      for (let v = 0; v < 2; v++) {
        const o = (i * 2 + v) * 4;
        skinIndex[o] = i; skinWeight[o] = 1;
      }
    }
    const idx: number[] = [];
    for (let i = 0; i < rings - 1; i++) {
      const a = i * 2, b = i * 2 + 1, c = (i + 1) * 2, d = (i + 1) * 2 + 1;
      idx.push(a, c, b, b, c, d);
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    geo.setAttribute("uv", new THREE.BufferAttribute(uv, 2));
    geo.setAttribute("skinIndex", new THREE.Uint16BufferAttribute(skinIndex, 4));
    geo.setAttribute("skinWeight", new THREE.Uint16BufferAttribute(skinWeight, 4));
    geo.setIndex(idx);
    geo.computeVertexNormals();

    const mesh = new THREE.SkinnedMesh(geo, this.hairMat);
    const skeleton = new THREE.Skeleton(bones);
    mesh.add(bones[0]);
    mesh.bind(skeleton);
    mesh.frustumCulled = false;
    this.skeletonBones = bones;
    return { root: bones[0], mesh };
  }

  update(elapsed: number, _dt: number) {
    const t = elapsed * 4.2;
    // smooth bob (no |sin| kinks — kinks kick the driven chains each period)
    const bob = 0.03 * (0.5 - 0.5 * Math.cos(t * 2));
    const hipY = 0.98 + bob;
    const sway = 0.06 * Math.sin(t * 0.5);

    // body pose
    this.pelvis.position.set(0, hipY, 0);
    this.tmpA.set(0, hipY + 0.14, 0.01);
    this.tmpB.set(0, hipY + 0.58, 0.03);
    this.torso.position.copy(this.tmpA).add(this.tmpB).multiplyScalar(0.5);
    this.torso.quaternion.setFromUnitVectors(UP, this.tmpB.clone().sub(this.tmpA).normalize());
    this.head.position.set(sway * 0.4, hipY + 0.72 + 0.012 * Math.sin(t * 2), 0.04);

    // colliders follow
    this.colHead.set(this.head.position.x, this.head.position.y, this.head.position.z);
    this.colTorso.setA(this.tmpA.x, this.tmpA.y, this.tmpA.z);
    this.colTorso.setB(this.tmpB.x, this.tmpB.y, this.tmpB.z);
    this.colBack.setA(0, hipY + 0.18, -0.13);
    this.colBack.setB(0, hipY + 0.52, -0.15);
    for (let i = 0; i < 2; i++) {
      const s = i === 0 ? 1 : -1;
      const ph = t + (s > 0 ? 0 : Math.PI);
      const swing = 0.5 * Math.sin(ph);
      const hipX = s * 0.09;
      this.tmpA.set(hipX, hipY - 0.02, 0);
      const knee = this.tmpB.set(hipX, this.tmpA.y - Math.cos(swing) * 0.4, Math.sin(swing) * 0.4);
      this.colThigh[i].setA(this.tmpA.x, this.tmpA.y, this.tmpA.z);
      this.colThigh[i].setB(knee.x, knee.y, knee.z);
      this.orient(this.limbs[i * 3], this.tmpA, knee);
      const foot = new THREE.Vector3(knee.x, knee.y - 0.42, knee.z - 0.06);
      this.orient(this.limbs[i * 3 + 1], knee, foot);
      const shoulder = new THREE.Vector3(s * 0.2, hipY + 0.52, 0.02);
      const hand = new THREE.Vector3(s * 0.24, hipY + 0.14, 0.1 * Math.sin(ph + 1));
      this.orient(this.limbs[i * 3 + 2], shoulder, hand);
    }

    // ---- drive chain roots from the head (kept OUTSIDE the head collider:
    // head r 0.14 + chain r ~0.035 -> clearance ≥ 0.19 from head centre) ----
    const hp = this.head.position;
    if (this.backChain) {
      this.backChain.setRoot(hp.x, hp.y - 0.04, hp.z - 0.19);
    }
    for (let i = 0; i < 2; i++) {
      const s = i === 0 ? 1 : -1;
      this.sideChains[i].setRoot(hp.x + s * 0.17, hp.y - 0.04, hp.z - 0.08);
    }

    // ---- bone transforms from solved joints ----
    if (this.backChain && this.skinned) this.applyChainToBones(this.backChain);
    for (const sb of this.sideBones) {
      const { chain, meshes } = sb;
      const p = chain.positions;
      for (let s = 0; s < meshes.length; s++) {
        const m = meshes[s];
        m.position.set(p[s * 3], p[s * 3 + 1], p[s * 3 + 2]);
        const dir = this.tmpB.set(
          p[(s + 1) * 3] - p[s * 3],
          p[(s + 1) * 3 + 1] - p[s * 3 + 1],
          p[(s + 1) * 3 + 2] - p[s * 3 + 2]
        );
        const len = dir.length();
        if (len > 1e-6) m.quaternion.setFromUnitVectors(DOWN, dir.divideScalar(len));
        const baseLen = m.userData.baseLen as number;
        m.scale.set(1, len / baseLen, 1);
      }
    }
  }

  /** Copy solved chain joints into the THREE.Bone hierarchy (skinning). */
  private applyChainToBones(chain: BoneChain) {
    const bones = this.skeletonBones;
    const p = chain.positions;
    const n = bones.length;
    if (n === 0) return;
    bones[0].position.set(p[0], p[1], p[2]);
    const parentQ = new THREE.Quaternion();
    const cur = new THREE.Vector3();
    const local = new THREE.Vector3();
    const inv = new THREE.Quaternion();
    for (let i = 1; i < n; i++) {
      cur.set(p[i * 3] - p[(i - 1) * 3], p[i * 3 + 1] - p[(i - 1) * 3 + 1], p[i * 3 + 2] - p[(i - 1) * 3 + 2]);
      const len = cur.length();
      if (len < 1e-9) continue;
      cur.divideScalar(len);
      bones[i - 1].getWorldQuaternion(parentQ);
      inv.copy(parentQ).invert();
      local.copy(cur).applyQuaternion(inv);
      bones[i].quaternion.setFromUnitVectors(this.restLocalDirs[i - 1], local);
    }
  }

  private orient(mesh: THREE.Mesh, a: THREE.Vector3, b: THREE.Vector3) {
    mesh.position.copy(a).add(b).multiplyScalar(0.5);
    const dir = b.clone().sub(a);
    const len = dir.length();
    mesh.quaternion.setFromUnitVectors(UP, dir.normalize());
    const base = (mesh.userData.boneLen as number) || len;
    mesh.scale.set(1, len / base, 1);
  }

  setWireframe(on: boolean) {
    this.hairMat.wireframe = on;
  }

  dispose(scene: THREE.Scene) {
    scene.remove(this.group);
    if (this.skinned) {
      scene.remove(this.skinned);
      this.skinned.geometry.dispose();
    }
    this.group.traverse((o) => {
      if (o instanceof THREE.Mesh) o.geometry.dispose();
    });
    this.bodyMat.dispose();
    this.hairMat.dispose();
    this.skinned = null;
    this.skeletonBones = [];
    this.sideBones = [];
    this.sideChains = [];
    this.backChain = null;
    scene.fog = null;
  }
}
