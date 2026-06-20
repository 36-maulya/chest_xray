import { Link } from "react-router-dom";

const features = [
  {
    icon: "🧠",
    title: "AI-Powered Detection",
    desc: "EfficientNet-B0 CNN model trained on annotated chest X-rays to detect rotation errors with high accuracy.",
    color: "cyan",
  },
  {
    icon: "📐",
    title: "Geometric Analysis",
    desc: "Measures clavicle symmetry from spine to determine direction and severity of patient mispositioning.",
    color: "blue",
  },
  {
    icon: "🔄",
    title: "Auto Correction",
    desc: "Affine transformation automatically rotates and realigns the X-ray to standard PA positioning.",
    color: "purple",
  },
  {
    icon: "📥",
    title: "Instant Download",
    desc: "Download the corrected X-ray instantly for use in diagnosis or reporting workflows.",
    color: "green",
  },
];

const steps = [
  { step: "01", title: "Upload X-Ray",       desc: "Drag and drop or browse your PA chest X-ray image.",              icon: "📤" },
  { step: "02", title: "AI Analysis",        desc: "CNN model detects rotation. Geometric analysis measures asymmetry.", icon: "🧠" },
  { step: "03", title: "View Results",       desc: "See confidence score, direction, severity and correction angle.",   icon: "📊" },
  { step: "04", title: "Download Corrected", desc: "Get the corrected X-ray image ready for diagnostic use.",          icon: "📥" },
];

const stats = [
  { value: "99+",      label: "X-Rays Analyzed"    },
  { value: "73.7%",    label: "Model Accuracy"      },
  { value: "EfficientNet-B0", label: "CNN Architecture" },
  { value: "<2s",      label: "Analysis Time"       },
];

export default function Home() {
  return (
    <div className="text-white">

      <style>{`
        .hero-bg {
          background: radial-gradient(ellipse at 50% 0%, rgba(0,212,255,0.08) 0%, transparent 60%),
                      radial-gradient(ellipse at 80% 50%, rgba(0,119,255,0.06) 0%, transparent 50%);
        }
        .feature-card {
          background: rgba(255,255,255,0.03);
          border: 1px solid rgba(255,255,255,0.07);
          transition: all 0.3s ease;
        }
        .feature-card:hover {
          border-color: rgba(0,212,255,0.3);
          background: rgba(0,212,255,0.05);
          transform: translateY(-4px);
        }
        .stat-card {
          background: rgba(255,255,255,0.03);
          border: 1px solid rgba(255,255,255,0.07);
        }
        .glow-text {
          text-shadow: 0 0 40px rgba(0,212,255,0.4);
        }
        .step-line::after {
          content: '';
          position: absolute;
          top: 24px;
          left: 60%;
          width: 80%;
          height: 1px;
          background: linear-gradient(90deg, rgba(0,212,255,0.3), transparent);
        }
        .btn-primary {
          background: linear-gradient(135deg, #00d4ff, #0077ff);
          transition: all 0.3s ease;
        }
        .btn-primary:hover {
          transform: translateY(-2px);
          box-shadow: 0 8px 25px rgba(0,212,255,0.4);
        }
        .scan-line {
          position: absolute;
          top: 0; left: 0; right: 0;
          height: 2px;
          background: linear-gradient(90deg, transparent, #00d4ff, transparent);
          animation: scanDown 3s linear infinite;
          box-shadow: 0 0 8px #00d4ff;
        }
        @keyframes scanDown {
          0%   { top: 0%;   opacity: 1; }
          90%  { opacity: 1; }
          100% { top: 100%; opacity: 0; }
        }
        .float-anim {
          animation: floatUp 4s ease-in-out infinite;
        }
        @keyframes floatUp {
          0%, 100% { transform: translateY(0px);   }
          50%      { transform: translateY(-10px);  }
        }
      `}</style>

      {/* ── Hero Section ── */}
      <section className="hero-bg min-h-screen flex items-center justify-center px-6 py-20 relative overflow-hidden">

        {/* Background grid */}
        <div className="absolute inset-0 opacity-5"
          style={{
            backgroundImage: "linear-gradient(rgba(0,212,255,0.3) 1px, transparent 1px), linear-gradient(90deg, rgba(0,212,255,0.3) 1px, transparent 1px)",
            backgroundSize: "50px 50px"
          }}
        />

        <div className="max-w-6xl mx-auto grid md:grid-cols-2 gap-16 items-center relative z-10">

          {/* Left Text */}
          <div>
            <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-cyan-500/10 border border-cyan-500/20 mb-6">
              <div className="w-2 h-2 bg-cyan-400 rounded-full animate-pulse" />
              <span className="text-cyan-400 text-sm font-medium">Major Project — AI in Radiology</span>
            </div>

            <h1 className="text-5xl md:text-6xl font-black leading-tight mb-6">
              Smarter<br />
              <span className="text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-blue-500 glow-text">
                Chest X-Ray
              </span><br />
              Analysis
            </h1>

            <p className="text-gray-400 text-lg leading-relaxed mb-8 max-w-lg">
              AI-powered system that detects rotation and positioning errors
              in PA chest X-rays and automatically corrects them — helping
              radiographers work faster and more accurately.
            </p>

            <div className="flex flex-wrap gap-4">
              <Link
                to="/analyze"
                className="btn-primary px-8 py-4 rounded-xl font-bold text-white text-lg flex items-center gap-2"
              >
                🔬 Start Analysis
              </Link>
              <Link
                to="/about"
                className="px-8 py-4 rounded-xl font-semibold text-gray-300 border border-white/10 hover:border-cyan-500/40 hover:text-cyan-400 transition-all"
              >
                Learn More →
              </Link>
            </div>

            {/* Mini Stats */}
            <div className="flex gap-6 mt-10">
              {[
                { val: "99+",   lab: "Images"   },
                { val: "73.7%", lab: "Accuracy" },
                { val: "<2s",   lab: "Speed"    },
              ].map((s) => (
                <div key={s.lab}>
                  <p className="text-2xl font-black text-cyan-400">{s.val}</p>
                  <p className="text-xs text-gray-500">{s.lab}</p>
                </div>
              ))}
            </div>
          </div>

          {/* Right — X-Ray Visual */}
          <div className="flex justify-center">
            <div className="relative w-80 h-80 float-anim">
              {/* Outer glow ring */}
              <div className="absolute inset-0 rounded-full border border-cyan-500/20 animate-pulse" />
              <div className="absolute inset-4 rounded-full border border-cyan-500/10" />

              {/* X-Ray box */}
              <div className="absolute inset-8 rounded-2xl overflow-hidden border border-cyan-500/30"
                style={{ background: "rgba(0,212,255,0.05)" }}>
                <div className="scan-line" />
                <div className="w-full h-full flex items-center justify-center">
                  <div className="text-center">
                    <div className="text-8xl mb-2">🫁</div>
                    <p className="text-cyan-400/60 text-xs font-mono">PA CHEST X-RAY</p>
                    <p className="text-cyan-400/40 text-xs font-mono">ANALYZING...</p>
                  </div>
                </div>
              </div>

              {/* Corner brackets */}
              {["top-6 left-6", "top-6 right-6", "bottom-6 left-6", "bottom-6 right-6"].map((pos, i) => (
                <div key={i} className={`absolute ${pos} w-4 h-4 border-cyan-400/40`}
                  style={{
                    borderTop:    i < 2 ? "2px solid" : "none",
                    borderBottom: i >= 2 ? "2px solid" : "none",
                    borderLeft:   i % 2 === 0 ? "2px solid" : "none",
                    borderRight:  i % 2 === 1 ? "2px solid" : "none",
                  }}
                />
              ))}

              {/* Floating badges */}
              <div className="absolute -right-4 top-16 px-3 py-2 rounded-lg bg-green-500/20 border border-green-500/30 text-xs text-green-400 font-semibold">
                ✅ Normal
              </div>
              <div className="absolute -left-6 bottom-16 px-3 py-2 rounded-lg bg-red-500/20 border border-red-500/30 text-xs text-red-400 font-semibold">
                ⚠️ Rotated
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ── Stats Bar ── */}
      <section className="border-y border-white/5 py-10 px-6"
        style={{ background: "rgba(0,212,255,0.03)" }}>
        <div className="max-w-5xl mx-auto grid grid-cols-2 md:grid-cols-4 gap-6">
          {stats.map((s) => (
            <div key={s.label} className="stat-card rounded-xl p-5 text-center">
              <p className="text-2xl font-black text-cyan-400">{s.value}</p>
              <p className="text-xs text-gray-500 mt-1">{s.label}</p>
            </div>
          ))}
        </div>
      </section>

      {/* ── Features ── */}
      <section className="py-20 px-6">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-14">
            <p className="text-cyan-400 text-sm font-medium uppercase tracking-wider mb-3">
              What We Do
            </p>
            <h2 className="text-4xl font-black mb-4">
              Powerful Features
            </h2>
            <p className="text-gray-500 max-w-xl mx-auto">
              A complete pipeline from detection to correction — built for
              real clinical radiography workflows.
            </p>
          </div>

          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
            {features.map((f) => (
              <div key={f.title} className="feature-card rounded-2xl p-6">
                <div className="text-4xl mb-4">{f.icon}</div>
                <h3 className="text-white font-bold mb-2">{f.title}</h3>
                <p className="text-gray-500 text-sm leading-relaxed">{f.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── How It Works ── */}
      <section className="py-20 px-6 border-t border-white/5"
        style={{ background: "rgba(255,255,255,0.01)" }}>
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-14">
            <p className="text-cyan-400 text-sm font-medium uppercase tracking-wider mb-3">
              Simple Process
            </p>
            <h2 className="text-4xl font-black mb-4">How It Works</h2>
            <p className="text-gray-500 max-w-xl mx-auto">
              Four simple steps from upload to corrected X-ray output.
            </p>
          </div>

          <div className="grid md:grid-cols-4 gap-6">
            {steps.map((s, i) => (
              <div key={s.step} className="relative text-center">
                <div className="w-14 h-14 rounded-2xl mx-auto mb-4 flex items-center justify-center text-2xl border border-cyan-500/20"
                  style={{ background: "rgba(0,212,255,0.08)" }}>
                  {s.icon}
                </div>
                <div className="text-cyan-400/40 text-xs font-mono mb-1">STEP {s.step}</div>
                <h3 className="text-white font-bold mb-2">{s.title}</h3>
                <p className="text-gray-500 text-sm">{s.desc}</p>
                {i < steps.length - 1 && (
                  <div className="hidden md:block absolute top-7 left-[65%] w-[70%] h-px"
                    style={{ background: "linear-gradient(90deg, rgba(0,212,255,0.3), transparent)" }}
                  />
                )}
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── CTA ── */}
      <section className="py-20 px-6">
        <div className="max-w-3xl mx-auto text-center">
          <div className="rounded-3xl p-12 border border-cyan-500/20 relative overflow-hidden"
            style={{ background: "radial-gradient(ellipse at center, rgba(0,212,255,0.08) 0%, transparent 70%)" }}>
            <div className="absolute top-0 left-0 right-0 h-px"
              style={{ background: "linear-gradient(90deg, transparent, rgba(0,212,255,0.5), transparent)" }}
            />
            <h2 className="text-4xl font-black mb-4">
              Ready to Analyze?
            </h2>
            <p className="text-gray-400 mb-8 text-lg">
              Upload your chest X-ray and get instant AI analysis with
              automatic correction in under 2 seconds.
            </p>
            <Link
              to="/analyze"
              className="btn-primary inline-flex items-center gap-2 px-10 py-4 rounded-xl font-bold text-white text-lg"
            >
              🔬 Start Free Analysis
            </Link>
          </div>
        </div>
      </section>

    </div>
  );
}