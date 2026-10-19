"use client";
/**
 * Client boundary: loads the playground on the client only
 * (WebGL2/WebGPU require the DOM).
 */
import dynamic from "next/dynamic";

const Playground = dynamic(() => import("./Playground"), {
  ssr: false,
  loading: () => (
    <div className="flex h-[72vh] min-h-[540px] w-full items-center justify-center rounded-2xl border border-border bg-[#171513]">
      <div className="flex flex-col items-center gap-3">
        <div className="h-9 w-9 animate-spin rounded-full border-2 border-neutral-400 border-t-transparent" />
        <p className="text-sm text-neutral-400">Загрузка 3D-движка…</p>
      </div>
    </div>
  ),
});

export default function DemoSection() {
  return (
    <section id="demo" className="scroll-mt-20 py-10">
      <div className="mb-6 text-center">
        <h2 className="text-2xl font-bold tracking-tight sm:text-3xl">Интерактивная демонстрация</h2>
        <p className="mx-auto mt-2 max-w-2xl text-sm text-muted-foreground sm:text-base">
          Живая симуляция прямо в браузере: переключайте сцены, бэкенды и параметры физики.
          Наведите мышь на сцену и вращайте камеру колесом или перетаскиванием.
        </p>
      </div>
      <Playground />
    </section>
  );
}
