import Link from "next/link";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import DemoSection from "@/components/DemoSection";
import {
  Cpu, GitBranch, Hand, Layers, Sparkles, Waves, Wind, Zap,
} from "lucide-react";

const GITHUB_URL = "https://github.com/ketaevweb/webcloth2";

const FEATURES = [
  {
    icon: Zap,
    title: "GPU-вычисления на WebGPU",
    text: "Полный XPBD-пайплайн в compute-шейдерах WGSL: интеграция, сотни тысяч связей, коллизии — без возврата данных на CPU между подшагами.",
  },
  {
    icon: Cpu,
    title: "CPU-бэкенд для WebGL2",
    text: "Оптимизированный решатель на типизированных массивах (SoA, ноль аллокаций в кадре). Работает везде, где есть WebGL 2+.",
  },
  {
    icon: Wind,
    title: "Аэродинамический ветер",
    text: "Сила ветра считается по треугольникам: нормаль, площадь и относительный поток, плюс многооктавный шум турбулентности и порывы.",
  },
  {
    icon: Hand,
    title: "Коллайдеры",
    text: "Сферы, капсулы и плоскости с трением. Капсулы анимируются каждый кадр — ткань скользит по конечностям персонажа.",
  },
  {
    icon: Layers,
    title: "Кинематические привязки",
    text: "Вершины пояса юбки или петель флага управляются внешними трансформами и тянут ткань за движущимся телом с инерцией.",
  },
  {
    icon: Waves,
    title: "Структурные и изгибные связи",
    text: "Дистанционные XPBD-связи по рёбрам сетки плюс изгибные связи между противоположными вершинами смежных треугольников.",
  },
];

const COMPARISON: Array<{ feature: string; wc: string; mc: string; ok?: boolean; partial?: boolean }> = [
  { feature: "Платформа", wc: "Браузер (WebGPU / WebGL 2+)", mc: "Unity (DirectX / Vulkan / Metal)", ok: true },
  { feature: "Вершинная ткань (Mesh Cloth)", wc: "Да", mc: "Да", ok: true },
  { feature: "Костная ткань (Bone Cloth)", wc: "В планах", mc: "Да", partial: true },
  { feature: "GPU compute-решатель", wc: "Да (WGSL)", mc: "Да (Compute Shaders)", ok: true },
  { feature: "Аэродинамика и ветер", wc: "Да (по треугольникам + шум)", mc: "Да", ok: true },
  { feature: "Коллайдеры сфера/капсула/плоскость", wc: "Да", mc: "Да", ok: true },
  { feature: "Кинематические привязки", wc: "Да", mc: "Да", ok: true },
  { feature: "Инерция движения", wc: "Да", mc: "Да", ok: true },
  { feature: "Самоколлизии ткани", wc: "В планах", mc: "Да", partial: true },
  { feature: "Разрывы ткани", wc: "В планах", mc: "Да", partial: true },
  { feature: "Лицензия", wc: "MIT, открытый код", mc: "Коммерческий ассет", ok: true },
];

const PIPELINE = [
  { step: "1", title: "Подшаги", text: "Кадр делится на 1–4 подшага с фиксированным dt — устойчивость при любом FPS." },
  { step: "2", title: "Интеграция", text: "Полуявный Эйлер: гравитация + аэродинамические силы ветра по треугольникам, ограничение скорости." },
  { step: "3", title: "Решение связей", text: "1–16 итераций XPBD: дистанционные связи с compliance (мягкость) по рёбрам и диагоналям." },
  { step: "4", title: "Коллизии", text: "Выталкивание из сфер, капсул и плоскостей с тангенциальным трением." },
  { step: "5", title: "Скорости", text: "v = (p − p_prev)/dt, демпфирование, клэмп максимальной скорости." },
  { step: "6", title: "Рендер", text: "Позиции возвращаются в геометрию three.js: WebGPU — через staged readback, CPU — напрямую." },
];

export default function Home() {
  return (
    <div className="dark min-h-screen bg-[#121110] text-neutral-100">
      {/* header */}
      <header className="sticky top-0 z-40 border-b border-white/5 bg-[#121110]/85 backdrop-blur-md">
        <div className="mx-auto flex h-14 max-w-6xl items-center justify-between px-4">
          <div className="flex items-center gap-2">
            <span className="flex h-7 w-7 items-center justify-center rounded-lg bg-gradient-to-br from-orange-500 to-red-500 text-[13px] font-black text-white">W</span>
            <span className="font-semibold tracking-tight">WebCloth 2</span>
            <Badge variant="secondary" className="ml-1 hidden h-5 px-1.5 text-[10px] sm:inline-flex">v2.0.1</Badge>
          </div>
          <nav className="flex items-center gap-4 text-sm text-neutral-400">
            <a href="#demo" className="transition-colors hover:text-neutral-100">Демо</a>
            <a href="#features" className="hidden transition-colors hover:text-neutral-100 sm:inline">Возможности</a>
            <a href="#compare" className="hidden transition-colors hover:text-neutral-100 sm:inline">Сравнение</a>
            <a href="#pipeline" className="hidden transition-colors hover:text-neutral-100 md:inline">Как работает</a>
            <a href={GITHUB_URL} target="_blank" rel="noreferrer" className="flex items-center gap-1 transition-colors hover:text-neutral-100">
              <GitBranch className="h-4 w-4" /> GitHub
            </a>
          </nav>
        </div>
      </header>

      <main className="mx-auto max-w-6xl px-4">
        {/* hero */}
        <section className="py-14 text-center sm:py-20">
          <div className="mx-auto mb-5 flex flex-wrap items-center justify-center gap-2">
            <Badge className="bg-orange-500/15 text-orange-400 hover:bg-orange-500/15">WebGPU Compute</Badge>
            <Badge className="bg-orange-500/15 text-orange-400 hover:bg-orange-500/15">WebGL 2+</Badge>
            <Badge className="bg-orange-500/15 text-orange-400 hover:bg-orange-500/15">XPBD</Badge>
            <Badge className="bg-orange-500/15 text-orange-400 hover:bg-orange-500/15">MIT</Badge>
          </div>
          <h1 className="mx-auto max-w-3xl text-4xl font-extrabold leading-tight tracking-tight sm:text-6xl">
            Производительный аналог{" "}
            <span className="bg-gradient-to-r from-orange-400 to-red-400 bg-clip-text text-transparent">
              Magica Cloth 2
            </span>{" "}
            для веба
          </h1>
          <p className="mx-auto mt-5 max-w-2xl text-base text-neutral-400 sm:text-lg">
            Физика ткани в браузере: оригинальный движок на XPBD с compute-шейдерами WebGPU
            и CPU-фолбэком для WebGL 2+. Десятки тысяч частиц в реальном времени — без плагинов и сборки.
          </p>
          <div className="mt-8 flex flex-wrap items-center justify-center gap-3">
            <Button asChild size="lg" className="bg-orange-500 text-white hover:bg-orange-600">
              <a href="#demo"><Sparkles className="mr-2 h-4 w-4" />Открыть демо</a>
            </Button>
            <Button asChild size="lg" variant="secondary">
              <a href={GITHUB_URL} target="_blank" rel="noreferrer"><GitBranch className="mr-2 h-4 w-4" />Исходный код</a>
            </Button>
          </div>
        </section>

        <DemoSection />

        {/* features */}
        <section id="features" className="scroll-mt-20 py-12">
          <h2 className="text-center text-2xl font-bold tracking-tight sm:text-3xl">Возможности движка</h2>
          <p className="mx-auto mt-2 max-w-2xl text-center text-sm text-muted-foreground sm:text-base">
            Ядро <code className="rounded bg-white/5 px-1.5 py-0.5 font-mono text-[13px]">src/lib/cloth</code> не
            зависит от фреймворков: движок можно подключить к three.js, Babylon.js или собственному рендереру.
          </p>
          <div className="mt-8 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {FEATURES.map((f) => (
              <div key={f.title} className="rounded-2xl border border-white/5 bg-white/[0.03] p-5 transition-colors hover:bg-white/[0.05]">
                <f.icon className="mb-3 h-6 w-6 text-orange-400" />
                <h3 className="mb-1.5 font-semibold">{f.title}</h3>
                <p className="text-sm leading-relaxed text-neutral-400">{f.text}</p>
              </div>
            ))}
          </div>
        </section>

        {/* comparison */}
        <section id="compare" className="scroll-mt-20 py-12">
          <h2 className="text-center text-2xl font-bold tracking-tight sm:text-3xl">Сравнение с Magica Cloth 2</h2>
          <p className="mx-auto mt-2 max-w-2xl text-center text-sm text-muted-foreground sm:text-base">
            Честная таблица: WebCloth 2 — самостоятельная реализация публично известных методик
            (XPBD / Position Based Dynamics), не использующая код оригинала.
          </p>
          <div className="mt-8 overflow-x-auto rounded-2xl border border-white/5">
            <table className="w-full min-w-[560px] text-sm">
              <thead>
                <tr className="border-b border-white/5 bg-white/[0.04] text-left">
                  <th className="px-4 py-3 font-medium text-neutral-300">Возможность</th>
                  <th className="px-4 py-3 font-medium text-orange-400">WebCloth 2</th>
                  <th className="px-4 py-3 font-medium text-neutral-300">Magica Cloth 2</th>
                </tr>
              </thead>
              <tbody>
                {COMPARISON.map((row) => (
                  <tr key={row.feature} className="border-b border-white/5 last:border-0">
                    <td className="px-4 py-2.5 text-neutral-300">{row.feature}</td>
                    <td className={`px-4 py-2.5 font-medium ${row.partial ? "text-amber-400" : "text-emerald-400"}`}>{row.wc}</td>
                    <td className="px-4 py-2.5 text-neutral-500">{row.mc}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>

        {/* pipeline */}
        <section id="pipeline" className="scroll-mt-20 py-12">
          <h2 className="text-center text-2xl font-bold tracking-tight sm:text-3xl">Как работает симуляция</h2>
          <div className="mt-8 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {PIPELINE.map((p) => (
              <div key={p.step} className="rounded-2xl border border-white/5 bg-white/[0.03] p-5">
                <div className="mb-2 flex h-7 w-7 items-center justify-center rounded-lg bg-orange-500/15 text-sm font-bold text-orange-400">
                  {p.step}
                </div>
                <h3 className="mb-1.5 font-semibold">{p.title}</h3>
                <p className="text-sm leading-relaxed text-neutral-400">{p.text}</p>
              </div>
            ))}
          </div>
          <div className="mt-6 rounded-2xl border border-white/5 bg-white/[0.03] p-5">
            <h3 className="mb-2 font-semibold">Минимальный пример кода</h3>
            <pre className="overflow-x-auto rounded-xl bg-black/50 p-4 font-mono text-[12.5px] leading-relaxed text-neutral-300">
{`import { ClothEngine } from "@/lib/cloth";

const engine = await ClothEngine.create();     // WebGPU или CPU-фолбэк
const cloth = engine.addCloth(flagGeometry, {
  pinnedPredicate: (x) => x < 0.05,            // закреплённая кромка
});
engine.settings.windStrength = 9;              // живые настройки

function frame(dt) {
  await engine.update(dt);                     // физика + синхронизация геометрии
  renderer.render(scene, camera);
}`}
            </pre>
          </div>
        </section>
      </main>

      {/* footer */}
      <footer className="mt-8 border-t border-white/5 bg-black/30">
        <div className="mx-auto flex max-w-6xl flex-col items-center justify-between gap-3 px-4 py-6 text-sm text-neutral-500 sm:flex-row">
          <p>WebCloth 2 — открытая физика ткани для веба. Лицензия MIT.</p>
          <p>Оригинальная реализация XPBD. Не связан с MagicaSoft.</p>
        </div>
      </footer>
    </div>
  );
}
