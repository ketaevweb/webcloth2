/**
 * WebCloth 2 — Flag demo scene.
 * A large cloth pinned to a pole, driven by turbulent aerodynamic wind.
 */
import * as THREE from "three";
import { ClothEngine, type Cloth, type EngineSettings } from "@/lib/cloth";
import { FOG_COLOR, type ClothScene } from "./SceneBase";

const POLE = new THREE.Vector3(0, 0, 0);

export class FlagScene implements ClothScene {
  id = "flag";
  name = "Флаг";
  description = "Большое полотнище на мачте: аэродинамика ветра и турбулентность";
  defaults: Partial<EngineSettings> = {
    windStrength: 9,
    windTurbulence: 0.55,
    windGust: 0.4,
    windDirX: 0.22,
    windDirZ: 1,
    aeroDrag: 0.85,
    distanceStiffness: 0.92,
    bendStiffness: 0.55,
    substeps: 2,
    iterations: 8,
    friction: 0.2,
  };
  camera = { pos: [5.2, 3.4, 6.0] as [number, number, number], target: [1.2, 2.4, 0] as [number, number, number] };

  private cloth: Cloth | null = null;
  private mesh: THREE.Mesh | null = null;
  private poleGroup = new THREE.Group();

  build(engine: ClothEngine, scene: THREE.Scene) {
    scene.fog = new THREE.Fog(FOG_COLOR, 12, 30);

    // pole
    const poleMat = new THREE.MeshStandardMaterial({ color: 0x8b857c, roughness: 0.4, metalness: 0.7 });
    const pole = new THREE.Mesh(new THREE.CylinderGeometry(0.045, 0.07, 3.6, 16), poleMat);
    pole.position.set(0, 1.8, 0);
    const base = new THREE.Mesh(new THREE.CylinderGeometry(0.4, 0.48, 0.1, 24), poleMat);
    base.position.set(0, 0.05, 0);
    const finial = new THREE.Mesh(new THREE.SphereGeometry(0.07, 16, 12), new THREE.MeshStandardMaterial({ color: 0xd9a441, roughness: 0.3, metalness: 0.8 }));
    finial.position.set(0, 3.62, 0);
    this.poleGroup.add(pole, base, finial);
    scene.add(this.poleGroup);

    // flag cloth: vertical plane from pole outward (+X)
    const geo = new THREE.PlaneGeometry(2.6, 1.6, 40, 24);
    geo.translate(1.3, 0, 0);
    geo.translate(0, 2.75, 0);
    const mat = new THREE.MeshStandardMaterial({
      color: 0xd94f3d,
      side: THREE.DoubleSide,
      roughness: 0.82,
      metalness: 0.02,
    });
    this.mesh = new THREE.Mesh(geo, mat);
    scene.add(this.mesh);

    this.cloth = engine.addCloth(geo, {
      pinnedPredicate: (x) => x < 0.06,
    });
  }

  update(_elapsed: number, _dt: number) {
    // wind is static; nothing to animate
  }

  setWireframe(on: boolean) {
    const m = this.mesh?.material as THREE.MeshStandardMaterial | undefined;
    if (m) m.wireframe = on;
  }

  dispose(scene: THREE.Scene) {
    scene.remove(this.poleGroup, this.mesh!);
    this.poleGroup.traverse((o) => {
      if (o instanceof THREE.Mesh) { o.geometry.dispose(); (o.material as THREE.Material).dispose(); }
    });
    if (this.mesh) { this.mesh.geometry.dispose(); (this.mesh.material as THREE.Material).dispose(); }
    this.cloth = null;
    this.mesh = null;
    scene.fog = null;
    void POLE;
  }
}
