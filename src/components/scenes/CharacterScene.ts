/**
 * WebCloth 2 — Character demo scene.
 * Procedural mannequin running in place, wearing a simulated skirt.
 * Demonstrates kinematic attachments (skirt belt follows the hips),
 * capsule/sphere colliders on animated limbs and movement inertia.
 */
import * as THREE from "three";
import { ClothEngine, type Cloth, type EngineSettings } from "@/lib/cloth";
import {
  FOG_COLOR,
  SphereCollider, CapsuleCollider, PlaneCollider,
  type ClothScene,
} from "./SceneBase";

const UP = new THREE.Vector3(0, 1, 0);

function capsuleMesh(r: number, boneLen: number, mat: THREE.Material): THREE.Mesh {
  const cyl = Math.max(0.01, boneLen - 2 * r);
  const mesh = new THREE.Mesh(new THREE.CapsuleGeometry(r, cyl, 6, 14), mat);
  return mesh;
}

function orientBone(mesh: THREE.Mesh, a: THREE.Vector3, b: THREE.Vector3) {
  mesh.position.copy(a).add(b).multiplyScalar(0.5);
  const dir = b.clone().sub(a);
  const len = dir.length();
  mesh.quaternion.setFromUnitVectors(UP, dir.normalize());
  mesh.scale.set(1, len / Math.max(1e-6, mesh.userData.boneLen ?? len), 1);
  void len;
}

export class CharacterScene implements ClothScene {
  id = "character";
  name = "Персонаж";
  description = "Юбка на бегущем манекене: кинематический пояс, коллайдеры на конечностях";
  defaults: Partial<EngineSettings> = {
    windStrength: 2.4,
    windTurbulence: 0.5,
    windGust: 0.3,
    windDirX: 0.3,
    windDirZ: 1,
    aeroDrag: 0.5,
    distanceStiffness: 0.95,
    bendStiffness: 0.6,
    substeps: 3,
    iterations: 10,
    friction: 0.4,
  };
  camera = { pos: [2.1, 1.5, 2.9] as [number, number, number], target: [0, 1.0, 0] as [number, number, number] };

  private group = new THREE.Group();
  private bodyMat = new THREE.MeshStandardMaterial({ color: 0x50565e, roughness: 0.55, metalness: 0.2 });
  private skirtMat = new THREE.MeshStandardMaterial({
    color: 0xd94f3d, side: THREE.DoubleSide, roughness: 0.8, metalness: 0.03,
  });
  private torso!: THREE.Mesh;
  private head!: THREE.Mesh;
  private thighs: THREE.Mesh[] = [];
  private shins: THREE.Mesh[] = [];
  private upArms: THREE.Mesh[] = [];
  private foreArms: THREE.Mesh[] = [];
  private skirtMesh: THREE.Mesh | null = null;
  private cloth: Cloth | null = null;

  // colliders
  private colPelvis = new SphereCollider(0, 1, 0, 0.235);
  private colTorso = new CapsuleCollider(0, 1.1, 0, 0, 1.7, 0, 0.165);
  private colThigh: CapsuleCollider[] = [
    new CapsuleCollider(0, 1, 0, 0, 0.6, 0, 0.085),
    new CapsuleCollider(0, 1, 0, 0, 0.6, 0, 0.085),
  ];
  private colShin: CapsuleCollider[] = [
    new CapsuleCollider(0, 0.6, 0, 0, 0.2, 0, 0.068),
    new CapsuleCollider(0, 0.6, 0, 0, 0.2, 0, 0.068),
  ];
  private colUpArm: CapsuleCollider[] = [
    new CapsuleCollider(0, 1.5, 0, 0, 1.2, 0, 0.055),
    new CapsuleCollider(0, 1.5, 0, 0, 1.2, 0, 0.055),
  ];
  private colForeArm: CapsuleCollider[] = [
    new CapsuleCollider(0, 1.2, 0, 0, 0.95, 0, 0.05),
    new CapsuleCollider(0, 1.2, 0, 0, 0.95, 0, 0.05),
  ];
  private tmpA = new THREE.Vector3();
  private tmpB = new THREE.Vector3();
  private tmpC = new THREE.Vector3();

  build(engine: ClothEngine, scene: THREE.Scene) {
    scene.fog = new THREE.Fog(FOG_COLOR, 9, 26);

    this.torso = capsuleMesh(0.165, 0.6, this.bodyMat);
    this.torso.userData.boneLen = 0.6;
    this.head = new THREE.Mesh(new THREE.SphereGeometry(0.125, 20, 16), this.bodyMat);
    for (let i = 0; i < 2; i++) {
      const th = capsuleMesh(0.085, 0.44, this.bodyMat); th.userData.boneLen = 0.44;
      const sh = capsuleMesh(0.068, 0.42, this.bodyMat); sh.userData.boneLen = 0.42;
      const ua = capsuleMesh(0.055, 0.3, this.bodyMat); ua.userData.boneLen = 0.3;
      const fa = capsuleMesh(0.05, 0.26, this.bodyMat); fa.userData.boneLen = 0.26;
      this.thighs.push(th); this.shins.push(sh); this.upArms.push(ua); this.foreArms.push(fa);
      this.group.add(th, sh, ua, fa);
    }
    this.group.add(this.torso, this.head);
    scene.add(this.group);

    // skirt: open cylinder, top ring at local y = 0 (kinematic belt)
    const skirtGeo = new THREE.CylinderGeometry(0.35, 0.54, 0.62, 36, 6, true);
    skirtGeo.translate(0, -0.31, 0);
    this.skirtMesh = new THREE.Mesh(skirtGeo, this.skirtMat);
    scene.add(this.skirtMesh);

    this.cloth = engine.addCloth(skirtGeo, {
      kinematicPredicate: (_x, y) => y > -0.005,
    });

    engine.addCollider(this.colPelvis);
    engine.addCollider(this.colTorso);
    for (let i = 0; i < 2; i++) {
      engine.addCollider(this.colThigh[i]);
      engine.addCollider(this.colShin[i]);
      engine.addCollider(this.colUpArm[i]);
      engine.addCollider(this.colForeArm[i]);
    }
    engine.addCollider(new PlaneCollider(0, 0, 0, 0, 1, 0));
  }

  update(elapsed: number, _dt: number) {
    const phase = elapsed * 4.6;
    const hipY = 0.99 + 0.05 * Math.abs(Math.sin(phase));
    const hipRotY = 0.14 * Math.sin(phase);

    // pelvis & torso & head
    this.colPelvis.set(0, hipY + 0.03, 0);
    this.tmpA.set(0, hipY + 0.16, 0.02);
    this.tmpB.set(0, hipY + 0.76, 0.05);
    this.colTorso.setA(this.tmpA.x, this.tmpA.y, this.tmpA.z);
    this.colTorso.setB(this.tmpB.x, this.tmpB.y, this.tmpB.z);
    orientBone(this.torso, this.tmpA, this.tmpB);
    this.head.position.set(0, hipY + 0.95, 0.06);

    // legs & arms (s = +1 left, -1 right)
    for (let i = 0; i < 2; i++) {
      const s = i === 0 ? 1 : -1;
      const ph = phase + (s > 0 ? 0 : Math.PI);
      const swing = Math.sin(ph);
      const thighA = 0.78 * swing;
      const flex = 1.15 * Math.max(0, -Math.sin(ph + 0.65)) + 0.1;

      const hipX = s * 0.11;
      this.tmpA.set(hipX, hipY, 0); // hip joint
      const knee = this.tmpB.set(
        hipX,
        hipY - Math.cos(thighA) * 0.44,
        Math.sin(thighA) * 0.44
      );
      this.colThigh[i].setA(this.tmpA.x, this.tmpA.y, this.tmpA.z);
      this.colThigh[i].setB(knee.x, knee.y, knee.z);
      orientBone(this.thighs[i], this.tmpA, knee);

      const shinA = thighA - flex;
      const foot = this.tmpC.set(
        knee.x,
        knee.y - Math.cos(shinA) * 0.42,
        knee.z + Math.sin(shinA) * 0.42
      );
      this.colShin[i].setA(knee.x, knee.y, knee.z);
      this.colShin[i].setB(foot.x, foot.y, foot.z);
      orientBone(this.shins[i], knee, foot);

      // arms swing opposite to legs
      const armPh = ph + Math.PI;
      const armA = 0.55 * Math.sin(armPh);
      this.tmpA.set(s * 0.26, hipY + 0.62, 0.03); // shoulder
      const elbow = this.tmpB.set(
        s * 0.28,
        this.tmpA.y - Math.cos(armA) * 0.3,
        this.tmpA.z + Math.sin(armA) * 0.22
      );
      this.colUpArm[i].setA(this.tmpA.x, this.tmpA.y, this.tmpA.z);
      this.colUpArm[i].setB(elbow.x, elbow.y, elbow.z);
      orientBone(this.upArms[i], this.tmpA, elbow);

      const foreA = armA - 1.0;
      const hand = this.tmpC.set(
        elbow.x + s * 0.02,
        elbow.y - Math.cos(foreA) * 0.26,
        elbow.z + Math.sin(foreA) * 0.2
      );
      this.colForeArm[i].setA(elbow.x, elbow.y, elbow.z);
      this.colForeArm[i].setB(hand.x, hand.y, hand.z);
      orientBone(this.foreArms[i], elbow, hand);
    }

    // skirt belt follows hips with a subtle flare
    const cloth = this.cloth;
    if (cloth) {
      const flare = 1.0 + 0.045 * Math.sin(phase * 2 + 1.2);
      const cosY = Math.cos(hipRotY), sinY = Math.sin(hipRotY);
      const lift = hipY - 0.045;
      cloth.updateKinematicTargets((rx, _ry, rz, out, o) => {
        out[o] = rx * cosY * flare - rz * sinY * flare;
        out[o + 1] = lift;
        out[o + 2] = rx * sinY * flare + rz * cosY * flare;
      });
    }
  }

  setWireframe(on: boolean) {
    if (this.skirtMesh) (this.skirtMesh.material as THREE.MeshStandardMaterial).wireframe = on;
  }

  dispose(scene: THREE.Scene) {
    scene.remove(this.group, this.skirtMesh!);
    this.group.traverse((o) => {
      if (o instanceof THREE.Mesh) { o.geometry.dispose(); }
    });
    this.skirtMesh?.geometry.dispose();
    this.skirtMat.dispose();
    this.bodyMat.dispose();
    this.cloth = null;
    this.skirtMesh = null;
    scene.fog = null;
  }
}
