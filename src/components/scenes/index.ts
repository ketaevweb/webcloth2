/**
 * WebCloth 2 — demo scene registry.
 */
import type { ClothScene } from "./SceneBase";
import { FlagScene } from "./FlagScene";
import { CharacterScene } from "./CharacterScene";
import { StressScene } from "./StressScene";
import { BoneScene } from "./BoneScene";

export const SCENES: Array<() => ClothScene> = [() => new FlagScene(), () => new CharacterScene(), () => new BoneScene(), () => new StressScene()];

export function createScene(id: string, param?: number): ClothScene {
  const factory = SCENES.find((f) => f().id === id) || SCENES[0];
  const scene = factory();
  if (param !== undefined) scene.param = param;
  return scene;
}

export type { ClothScene } from "./SceneBase";
export { addEnvironment, FOG_COLOR } from "./SceneBase";
