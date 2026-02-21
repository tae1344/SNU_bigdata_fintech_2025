
export default function Section({ title, subtitle, children }: { title: string; subtitle?: string; children: React.ReactNode }) {
  return (
    <section className="w-full max-w-7xl mx-auto px-4 md:px-6 py-10">
      <div className="mb-6">
        <h2 className="text-2xl md:text-3xl font-bold tracking-tight text-white">{title}</h2>
        {subtitle && <p className="text-sm md:text-base text-slate-300 mt-1">{subtitle}</p>}
      </div>
      {children}
    </section>
  );
}