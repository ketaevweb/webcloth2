/**
 * WebCloth 2 — Stress test scene.
 * A grid of independent cloth panels; the count is adjustable to benchmark
 * the engine from a few thousand up to tens of thousands of particles.
 */
import * as THREE from "three";
import { ClothEngine, type EngineSettings } from "@/lib/cloth";
import { FOG_COLOR, type ClothScene } from "./SceneBase";

export class StressScene implements ClothScene {
  id = "stress";
  name = "Стресс-тест";
  description = "Сетка полотнищ для замера производительности бэкенда";
  defaults: Partial<EngineSettings> = {
    windStrength: 1.8,
    windTurbulence: 0.5,
    windGust: 0.4,
    windDirX: 1,
    windDirZ: -0.3,
    aeroDrag: 0.35,
    distanceStiffness: 0.9,
    bendStiffness: 0.4,
    substeps: 2,
    iterations: 6,
    friction: 0.2,
  };
  camera = { pos: [0.5, 7.5, 14.5] as [number, number, number], target: [0, 1.3, 0] as [number, number, number] };

  private meshes: THREE.Mesh[] = [];
  private poleMat = new THREE.MeshStandardMaterial({ color: 0x8b857c, roughness: 0.45, metalness: 0.6 });
  private clothMat = new THREE.MeshBasicMaterial({ color: 0xd97b3d, side: THREE.DoubleSide });
  private poleGeo = new THREE.CylinderGeometry(0.035, 0.05, 3.1, 10);
  private group = new THREE.Group();
  private wireOn = false;
  param = 12;

  build(engine: ClothEngine, scene: THREE.Scene) {
    scene.fog = new THREE.Fog(FOG_COLOR, 16, 40);
    const count = Math.max(1, Math.min(24, this.param | 0));
    const cols = 6;
    const rows = Math.ceil(count / cols);

    const poleGeo = new THREE.CylinderGeometry(0.035, 0.05, 3.1, 10);
    for (let i = 0; i < count; i++) {
      const cx = ((i % cols) - (cols - 1) / 2) * 2.2;
      const cz = (Math.floor(i / cols) - (rows - 1) / 2) * 2.6;
      // panels start nearly edge-on to the wind (wind ~ (1,0,-0.3)):
      // each banner flutters around its own pole instead of blanketing neighbours
      const yaw = 0.29 + ((i % cols) / cols - 0.5) * 0.7 + (Math.floor(i / cols) - 0.5) * 0.15;

      const pole = new THREE.Mesh(this.poleGeo, this.poleMat);
      pole.position.set(cx, 1.55, cz);
      this.group.add(pole);

      // cloth panel baked in world space, rotated around its pole
      const geo = new THREE.PlaneGeometry(0.9, 0.65, 18, 12);
      geo.translate(0.45, 0, 0);
      geo.translate(0, -0.325, 0); // top edge (pinned) at pole, cloth hangs down
      geo.rotateY(yaw);
      geo.translate(cx, 2.75, cz);
      const mesh = new THREE.Mesh(geo, this.clothMat);
      this.meshes.push(mesh);
      this.group.add(mesh);

      engine.addCloth(geo, {
        smoothNormals: false,
        pinnedPredicate: (x, y, z) => {
          const dx = x - cx, dz = z - cz;
          return dx * dx + dz * dz < 0.02 && y > 1.7;
        },
      });
    }
    scene.add(this.group);
    this.applyWireframe();
  }

  update(_elapsed: number, _dt: number) {}

  setWireframe(on: boolean) {
    this.wireOn = on;
    this.applyWireframe();
  }

  private applyWireframe() {
    this.clothMat.wireframe = this.wireOn;
  }

  dispose(scene: THREE.Scene) {
    scene.remove(this.group);
    for (const m of this.meshes) m.geometry.dispose();
    this.meshes = [];
    this.poleGeo.dispose();
    this.poleMat.dispose();
    this.clothMat.dispose();
    scene.fog = null;
  }
}
