"use client";
/**
 * WebCloth 2 — interactive playground.
 * Hosts the three.js scene, the ClothEngine RAF loop, live settings HUD
 * and backend statistics (Russian UI).
 */
import { useCallback, useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import {
  ClothEngine,
  DEFAULT_SETTINGS,
  type BackendKind,
  type EngineSettings,
  type EngineStats,
} from "@/lib/cloth";
import { addEnvironment, createScene, FOG_COLOR, type ClothScene } from "./scenes";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { Badge } from "@/components/ui/badge";
import {
  Pause, Play, RotateCcw, Gauge, Grid3x3, Settings2, X, TriangleAlert,
} from "lucide-react";

interface Runtime {
  engine: ClothEngine;
  renderer: THREE.WebGLRenderer;
  scene: THREE.Scene;
  camera: THREE.PerspectiveCamera;
  controls: OrbitControls;
  activeScene: ClothScene | null;
  raf: number;
  lastT: number;
  fpsEma: number;
  busy: boolean;
  statT: number;
}

const SCENE_TABS = [
  { id: "flag", label: "Флаг" },
  { id: "character", label: "Персонаж" },
  { id: "bones", label: "Кости" },
  { id: "stress", label: "Стресс" },
];

export default function Playground() {
  const holderRef = useRef<HTMLDivElement>(null);
  const R = useRef<Runtime | null>(null);
  const [ready, setReady] = useState(false);
  const [sceneId, setSceneId] = useState("flag");
  const [stressCount, setStressCount] = useState(12);
  const [settings, setSettings] = useState<EngineSettings>({ ...DEFAULT_SETTINGS });
  const [stats, setStats] = useState<EngineStats>({
    fps: 0, frameMs: 0, simMs: 0, particles: 0, constraints: 0, triangles: 0, bones: 0,
    backend: "cpu", adapter: "",
  });
  const [backend, setBackend] = useState<BackendKind>("cpu");
  const [gpuOk, setGpuOk] = useState(false);
  const [fallbackMsg, setFallbackMsg] = useState<string | null>(null);
  const [wireframe, setWireframe] = useState(false);
  const [panelOpen, setPanelOpen] = useState(true);
  const [fatalError, setFatalError] = useState<string | null>(null);
  const buildId = (process.env.NEXT_PUBLIC_BUILD_ID as string | undefined) || "dev";

  // ---- init once ----
  useEffect(() => {
    let disposed = false;
    const holder = holderRef.current;
    if (!holder) return;

    (async () => {
      let engine: ClothEngine;
      try {
        // Guard against adapters that never resolve requestAdapter() (rare
        // driver bug): abort into a visible error instead of an eternal spinner.
        engine = await Promise.race([
          ClothEngine.create(),
          new Promise<never>((_, reject) =>
            setTimeout(() => reject(new Error("Не удалось инициализировать GPU/CPU-движок за 12 с")), 12000)
          ),
        ]);
      } catch (e) {
        if (!disposed) setFatalError(e instanceof Error ? e.message : String(e));
        return;
      }
      if (disposed) { engine.dispose(); return; }
      engine.onFallback = (reason) => {
        setBackend("cpu");
        setFallbackMsg(`WebGPU отключён, включён CPU-режим: ${reason}`);
      };
      engine.settings = { ...settings };

      let renderer: THREE.WebGLRenderer;
      try {
        renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
      } catch (e) {
        if (!disposed) {
          setFatalError(
            "Не удалось создать WebGL 2 контекст — браузер блокирует аппаратный ускоритель. " +
            "Включите аппаратное ускорение в настройках или откройте демо в другом браузере."
          );
        }
        engine.dispose();
        return;
      }
      renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
      renderer.setSize(holder.clientWidth, holder.clientHeight);
      renderer.toneMapping = THREE.ACESFilmicToneMapping;
      renderer.domElement.style.display = "block";
      holder.appendChild(renderer.domElement);

      const scene = new THREE.Scene();
      scene.background = new THREE.Color(FOG_COLOR);
      addEnvironment(scene);

      const camera = new THREE.PerspectiveCamera(45, holder.clientWidth / Math.max(1, holder.clientHeight), 0.1, 100);
      const controls = new OrbitControls(camera, renderer.domElement);
      controls.enableDamping = true;
      controls.dampingFactor = 0.08;
      controls.maxPolarAngle = Math.PI * 0.55;
      controls.minDistance = 1.2;
      controls.maxDistance = 24;

      R.current = {
        engine, renderer, scene, camera, controls,
        activeScene: null, raf: 0, lastT: performance.now(), fpsEma: 60, busy: false,
        statT: 0,
      };
      // debug hook (used by E2E diagnostics)
      (window as unknown as { __wc?: ClothEngine }).__wc = engine;
      setReady(true);
      setBackend(engine.backend);
      setGpuOk(engine.backend === "webgpu");
      setStats((s) => ({ ...s, backend: engine.backend, adapter: engine.adapterName }));

      const loop = (t: number) => {
        const r = R.current;
        if (!r || disposed) return;
        r.raf = requestAnimationFrame(loop);
        const dt = Math.min(0.05, Math.max(0.0001, (t - r.lastT) / 1000));
        r.lastT = t;
        if (r.activeScene) r.activeScene.update(t / 1000, dt);
        if (!r.busy) {
          r.busy = true;
          r.engine
            .update(dt)
            .catch(() => undefined)
            .finally(() => { r.busy = false; });
        }
        r.controls.update();
        r.renderer.render(r.scene, r.camera);
        r.fpsEma = r.fpsEma * 0.93 + (1 / dt) * 0.07;
        if (t - r.statT > 300) {
          r.statT = t;
          setStats({
            fps: Math.round(r.fpsEma),
            frameMs: Math.round((1000 / Math.max(1, r.fpsEma)) * 10) / 10,
            simMs: Math.round(r.engine.stats.simMs * 10) / 10,
            particles: r.engine.stats.particles,
            constraints: r.engine.stats.constraints,
            triangles: r.engine.stats.triangles,
            bones: r.engine.stats.bones,
            backend: r.engine.backend,
            adapter: r.engine.adapterName,
          });
        }
      };
      R.current.raf = requestAnimationFrame(loop);
    })();

    return () => {
      disposed = true;
      const r = R.current;
      if (r) {
        cancelAnimationFrame(r.raf);
        r.activeScene?.dispose(r.scene);
        r.engine.dispose();
        r.controls.dispose();
        r.renderer.dispose();
        r.renderer.domElement.remove();
        R.current = null;
      }
    };
  }, []);

  // ---- (re)build scene ----
  useEffect(() => {
    const r = R.current;
    if (!r || !ready) return;
    r.activeScene?.dispose(r.scene);
    // remove leftover environment-unrelated objects? scenes clean themselves
    r.engine.clearCloths();
    r.engine.clearColliders();
    r.engine.clearBoneChains();
    const s = createScene(sceneId, stressCount);
    s.build(r.engine, r.scene);
    s.setWireframe(wireframe);
    r.activeScene = s;
    r.camera.position.set(...s.camera.pos);
    r.controls.target.set(...s.camera.target);
    r.controls.update();
    setSettings((prev) => {
      const next = { ...DEFAULT_SETTINGS, ...s.defaults, paused: prev.paused, timeScale: prev.timeScale };
      r.engine.settings = next;
      r.engine.refreshCompliance();
      return next;
    });
  }, [ready, sceneId, stressCount]);

  // ---- apply settings ----
  useEffect(() => {
    const r = R.current;
    if (!r) return;
    r.engine.settings = settings;
    r.engine.refreshCompliance();
  }, [settings]);

  const patch = useCallback((p: Partial<EngineSettings>) => {
    setSettings((prev) => ({ ...prev, ...p }));
  }, []);

  const toggleBackend = useCallback(() => {
    const r = R.current;
    if (!r) return;
    const target: BackendKind = r.engine.backend === "webgpu" ? "cpu" : "webgpu";
    r.engine.setBackend(target);
    setBackend(r.engine.backend);
  }, []);

  const onReset = useCallback(() => {
    R.current?.engine.reset();
  }, []);

  const toggleWireframe = useCallback(() => {
    const r = R.current;
    if (!r) return;
    const next = !wireframe;
    setWireframe(next);
    r.activeScene?.setWireframe(next);
  }, [wireframe]);

  // ---- resize ----
  useEffect(() => {
    const holder = holderRef.current;
    if (!holder) return;
    const ro = new ResizeObserver(() => {
      const r = R.current;
      if (!r) return;
      const w = holder.clientWidth, h = Math.max(1, holder.clientHeight);
      r.renderer.setSize(w, h);
      r.camera.aspect = w / h;
      r.camera.updateProjectionMatrix();
    });
    ro.observe(holder);
    return () => ro.disconnect();
  }, []);

  return (
    <div className="relative h-[72vh] min-h-[540px] w-full overflow-hidden rounded-2xl border border-border bg-[#171513] shadow-xl">
      <div ref={holderRef} className="absolute inset-0" aria-label="3D сцена симуляции ткани" />

      {/* loading */}
      {!ready && (
        <div className="absolute inset-0 z-20 flex items-center justify-center bg-background/80">
          <div className="flex flex-col items-center gap-3">
            <div className="h-9 w-9 animate-spin rounded-full border-2 border-primary border-t-transparent" />
            <p className="text-sm text-muted-foreground">Инициализация движка…</p>
          </div>
        </div>
      )}

      {/* stats HUD */}
      <div className="absolute left-3 top-3 z-10 select-none rounded-xl border border-white/10 bg-black/55 px-3 py-2 font-mono text-[11px] leading-relaxed text-neutral-200 backdrop-blur-md">
        <div className="flex items-center gap-2">
          <span className="text-[13px] font-semibold text-emerald-400">{stats.fps} FPS</span>
          <span className="text-neutral-400">кадр {stats.frameMs} мс</span>
        </div>
        <div className="text-neutral-400">симуляция {stats.simMs} мс</div>
        <div className="text-neutral-400">частицы {stats.particles.toLocaleString("ru-RU")} · связи {stats.constraints.toLocaleString("ru-RU")}</div>
        {stats.bones > 0 && (
          <div className="text-neutral-400">кости {stats.bones}</div>
        )}
        <div className="mt-1 flex items-center gap-1.5">
          <Badge variant={stats.backend === "webgpu" ? "default" : "secondary"} className="h-5 px-1.5 text-[10px]">
            {stats.backend === "webgpu" ? "WebGPU Compute" : "CPU · WebGL2"}
          </Badge>
          <span className="max-w-[130px] truncate text-neutral-500">{stats.adapter}</span>
        </div>
        <div className="text-neutral-600" title="Идентификатор сборки — сверяйте, что видите свежую версию">
          сборка {buildId}
        </div>
      </div>

      {/* panel toggle (mobile) */}
      <button
        onClick={() => setPanelOpen((v) => !v)}
        className="absolute right-3 top-3 z-30 rounded-lg border border-white/10 bg-black/55 p-2 text-neutral-200 backdrop-blur-md md:hidden"
        aria-label="Показать/скрыть панель управления"
      >
        <Settings2 className="h-4 w-4" />
      </button>

      {/* control panel */}
      {panelOpen && (
        <div className="absolute right-3 top-3 z-20 max-h-[calc(100%-24px)] w-[280px] overflow-y-auto rounded-xl border border-white/10 bg-black/60 p-3 text-neutral-200 backdrop-blur-md max-md:left-3 max-md:w-auto">
          <div className="mb-2 flex items-center justify-between">
            <h3 className="text-xs font-semibold uppercase tracking-wider text-neutral-400">Сцена</h3>
            <button className="md:hidden" onClick={() => setPanelOpen(false)} aria-label="Закрыть панель">
              <X className="h-4 w-4 text-neutral-400" />
            </button>
          </div>
          <div className="mb-3 grid grid-cols-3 gap-1.5">
            {SCENE_TABS.map((t) => (
              <button
                key={t.id}
                onClick={() => setSceneId(t.id)}
                className={`rounded-lg px-2 py-1.5 text-xs font-medium transition-colors ${
                  sceneId === t.id
                    ? "bg-primary text-primary-foreground"
                    : "bg-white/5 text-neutral-300 hover:bg-white/10"
                }`}
              >
                {t.label}
              </button>
            ))}
          </div>

          {sceneId === "stress" && (
            <div className="mb-3">
              <div className="mb-1 flex justify-between text-[11px] text-neutral-400">
                <span>Полотнищ</span>
                <span className="font-mono text-neutral-200">{stressCount}</span>
              </div>
              <Slider
                value={[stressCount]}
                min={1} max={24} step={1}
                onValueChange={([v]) => setStressCount(v)}
                aria-label="Количество полотнищ"
              />
            </div>
          )}

          <div className="mb-3 flex items-center justify-between rounded-lg bg-white/5 px-2.5 py-2">
            <div className="text-[11px] leading-tight">
              <div className="font-medium text-neutral-200">Бэкенд</div>
              <div className="text-neutral-500">{gpuOk ? "WebGPU доступен" : "WebGPU недоступен"}</div>
            </div>
            <Button
              size="sm"
              variant={backend === "webgpu" ? "default" : "secondary"}
              className="h-7 px-2.5 text-[11px]"
              disabled={!gpuOk}
              onClick={toggleBackend}
            >
              {backend === "webgpu" ? "WebGPU" : "CPU"}
            </Button>
          </div>

          <div className="mb-3 grid grid-cols-2 gap-1.5">
            <button
              onClick={() => patch({ selfCollision: !settings.selfCollision })}
              className={`rounded-lg px-2 py-1.5 text-[11px] font-medium transition-colors ${
                settings.selfCollision
                  ? "bg-primary text-primary-foreground"
                  : "bg-white/5 text-neutral-300 hover:bg-white/10"
              }`}
              aria-pressed={settings.selfCollision}
            >
              Самоколлизия
            </button>
            <div className="rounded-lg bg-white/5 px-2 py-1">
              <div className="flex justify-between text-[10px] text-neutral-400">
                <span>Прочность</span>
                <span className="font-mono text-neutral-200">
                  {settings.tearFactor > 0 ? `×${settings.tearFactor.toFixed(1)}` : "выкл"}
                </span>
              </div>
              <Slider
                value={[settings.tearFactor]}
                min={0} max={4} step={0.1}
                onValueChange={([v]) => patch({ tearFactor: v })}
                aria-label="Прочность на разрыв"
              />
            </div>
          </div>

          <h3 className="mb-2 text-xs font-semibold uppercase tracking-wider text-neutral-400">Физика</h3>
          <div className="space-y-2.5">
            <SliderRow label="Ветер" value={settings.windStrength} min={0} max={20} step={0.5}
              onChange={(v) => patch({ windStrength: v })} format={(v) => v.toFixed(1)} />
            <SliderRow label="Турбулентность" value={settings.windTurbulence} min={0} max={1.5} step={0.05}
              onChange={(v) => patch({ windTurbulence: v })} format={(v) => v.toFixed(2)} />
            <SliderRow label="Гравитация" value={settings.gravityY} min={-20} max={0} step={0.1}
              onChange={(v) => patch({ gravityY: v })} format={(v) => v.toFixed(1)} />
            <SliderRow label="Жёсткость" value={settings.distanceStiffness} min={0.3} max={1} step={0.01}
              onChange={(v) => patch({ distanceStiffness: v })} format={(v) => v.toFixed(2)} />
            <SliderRow label="Изгибы" value={settings.bendStiffness} min={0} max={1} step={0.01}
              onChange={(v) => patch({ bendStiffness: v })} format={(v) => v.toFixed(2)} />
            <SliderRow label="Трение" value={settings.friction} min={0} max={1} step={0.05}
              onChange={(v) => patch({ friction: v })} format={(v) => v.toFixed(2)} />
            <SliderRow label="Подшаги" value={settings.substeps} min={1} max={4} step={1}
              onChange={(v) => patch({ substeps: v })} format={(v) => String(v)} />
            <SliderRow label="Итерации" value={settings.iterations} min={1} max={16} step={1}
              onChange={(v) => patch({ iterations: v })} format={(v) => String(v)} />
          </div>

          <div className="mt-3 grid grid-cols-2 gap-1.5">
            <Button size="sm" variant="secondary" className="h-8 text-[11px]"
              onClick={() => patch({ paused: !settings.paused })}>
              {settings.paused ? <Play className="mr-1 h-3.5 w-3.5" /> : <Pause className="mr-1 h-3.5 w-3.5" />}
              {settings.paused ? "Пуск" : "Пауза"}
            </Button>
            <Button size="sm" variant="secondary" className="h-8 text-[11px]" onClick={onReset}>
              <RotateCcw className="mr-1 h-3.5 w-3.5" /> Сброс
            </Button>
            <Button size="sm"
              variant={settings.timeScale < 1 ? "default" : "secondary"}
              className="h-8 text-[11px]"
              onClick={() => patch({ timeScale: settings.timeScale < 1 ? 1 : 0.3 })}>
              <Gauge className="mr-1 h-3.5 w-3.5" /> Slow-mo
            </Button>
            <Button size="sm" variant={wireframe ? "default" : "secondary"} className="h-8 text-[11px]"
              onClick={toggleWireframe}>
              <Grid3x3 className="mr-1 h-3.5 w-3.5" /> Каркас
            </Button>
          </div>
        </div>
      )}

      {/* fatal init error */}
      {fatalError && (
        <div className="absolute inset-0 z-30 flex items-center justify-center bg-background/90 p-6">
          <div className="max-w-md rounded-xl border border-red-500/30 bg-red-950/70 p-5 text-center">
            <TriangleAlert className="mx-auto mb-2 h-7 w-7 text-red-400" />
            <h3 className="mb-1 text-sm font-semibold text-red-100">Демо не смогло запуститься</h3>
            <p className="text-xs leading-relaxed text-red-200/80">{fatalError}</p>
            <p className="mt-3 text-[11px] text-neutral-400">
              Попробуйте жёсткое обновление (Ctrl+F5 / Cmd+Shift+R) или другой браузер:
              Chrome / Edge / Firefox 125+.
            </p>
            <Button size="sm" variant="secondary" className="mt-3 h-8" onClick={() => window.location.reload()}>
              <RotateCcw className="mr-1 h-3.5 w-3.5" /> Перезагрузить
            </Button>
          </div>
        </div>
      )}

      {/* fallback toast */}
      {fallbackMsg && (
        <div className="absolute bottom-3 left-3 right-3 z-30 flex items-start gap-2 rounded-xl border border-amber-500/30 bg-amber-950/80 px-3 py-2.5 text-xs text-amber-100 backdrop-blur-md">
          <TriangleAlert className="mt-0.5 h-4 w-4 shrink-0" />
          <span className="flex-1">{fallbackMsg}</span>
          <button onClick={() => setFallbackMsg(null)} aria-label="Закрыть предупреждение">
            <X className="h-4 w-4" />
          </button>
        </div>
      )}
    </div>
  );
}

function SliderRow({
  label, value, min, max, step, onChange, format,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
  format: (v: number) => string;
}) {
  return (
    <div>
      <div className="mb-1 flex justify-between text-[11px] text-neutral-400">
        <span>{label}</span>
        <span className="font-mono text-neutral-200">{format(value)}</span>
      </div>
      <Slider
        value={[value]}
        min={min} max={max} step={step}
        onValueChange={([v]) => onChange(v)}
        aria-label={label}
      />
    </div>
  );
}
