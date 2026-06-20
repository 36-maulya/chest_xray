import { Link } from "react-router-dom";

const technologies = [
  { icon: "🧠", name: "EfficientNet-B0",  role: "CNN Model",         desc: "Lightweight deep learning model for rotation detection."     },
  { icon: "🔥", name: "PyTorch",           role: "ML Framework",      desc: "Deep learning framework used for model training."            },
  { icon: "⚡", name: "FastAPI",           role: "Backend API",       desc: "High-performance Python backend serving the AI model."       },
  { icon: "⚛️", name: "React",             role: "Frontend",          desc: "Modern UI library for building the doctor-facing interface." },
  { icon: "🎨", name: "Tailwind CSS",      role: "Styling",           desc: "Utility-first CSS framework for stunning UI design."        },
  { icon: "👁️", name: "OpenCV",            role: "Image Processing",  desc: "Affine transformation for automatic X-ray correction."      },
  { icon: "📐", name: "Geometric Analysis",role: "Measurement",       desc: "Clavicle symmetry analysis to detect rotation direction."   },
  { icon: "📊", name: "Pandas + openpyxl", role: "Data Processing",   desc: "Reading and processing annotation data from Excel sheets."  },
];

const workflow = [
  { step: "01", icon: "📤", title: "Image Input",         desc: "Doctor uploads a PA chest X-ray image through the web interface."                          },
  { step: "02", icon: "🔍", title: "Preprocessing",       desc: "Image is resized to 224×224 and normalized for CNN input."                                 },
  { step: "03", icon: "🧠", title: "CNN Detection",       desc: "EfficientNet-B0 classifies the image as Normal or Rotated with a confidence score."        },
  { step: "04", icon: "📐", title: "Geometric Analysis",  desc: "Clavicle distances are measured to find asymmetry, direction, and severity of rotation."   },
  { step: "05", icon: "🔄", title: "Affine Correction",   desc: "If rotated, OpenCV applies affine transformation to correct the image alignment."          },
  { step: "06", icon: "📥", title: "Output",              desc: "Corrected X-ray is displayed with full report and available for download."                  },
];

const team = [
  { name: "Project Lead",     role: "AI Model Development",      icon: "👨‍💻" },
  { name: "Team Member",      role: "Dataset & Annotation",      icon: "👨‍⚕️" },
  { name: "Team Member",      role: "Frontend Development",      icon: "👩‍💻" },
  { name: "Team Member",      role: "Backend & Deployment",      icon: "👨‍🔬" },
];

export default function About() {
  return (
    <div className="text-white min-h-screen px-6 py-16">
      <style>{`
        .card { background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); }
        .tech-card {
          background: rgba(255,255,255,0.02);
          border: 1px solid rgba(255,255,255,0.07);
          transition: all 0.3s ease;
        }
        .tech-card:hover {
          border-color: rgba(0,212,255,0.3);
          background: rgba(0,212,255,0.05);
          transform: translateY(-3px);
        }
        .btn-primary { background: linear-gradient(135deg, #00d4ff, #0077ff); transition: all 0.3s; }
        .btn-primary:hover { transform: translateY(-2px); box-shadow: 0 8px 25px rgba(0,212,255,0.4); }
        .hero-bg {
          background: radial-gradient(ellipse at 50% 0%, rgba(0,212,255,0.07) 0%, transparent 60%);
        }
      `}</style>

      <div className="max-w-6xl mx-auto">

        {/* Hero */}
        <div className="hero-bg rounded-3xl p-12 text-center mb-14 border border-white/5 relative overflow-hidden">
          <div className="absolute top-0 left-0 right-0 h-px"
            style={{ background: "linear-gradient(90deg, transparent, rgba(0,212,255,0.5), transparent)" }}
          />
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-cyan-500/10 border border-cyan-500/20 mb-6">
            <span className="text-cyan-400 text-sm">🎓 Major Project — Computer Science Engineering</span>
          </div>
          <h1 className="text-5xl font-black mb-4">
            About{" "}
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-blue-500">
              ChestAlign AI
            </span>
          </h1>
          <p className="text-gray-400 text-lg max-w-2xl mx-auto leading-relaxed">
            An AI-based system for automatic detection and correction of
            positioning errors in PA chest X-rays — built to assist
            radiographers and improve diagnostic accuracy.
          </p>
        </div>

        {/* Project Overview */}
        <div className="grid md:grid-cols-2 gap-8 mb-14">
          <div className="card rounded-2xl p-8">
            <h2 className="text-2xl font-black mb-4">🎯 Project Goal</h2>
            <p className="text-gray-400 leading-relaxed mb-4">
              In chest radiography, proper patient positioning is critical.
              Incorrectly positioned X-rays cause clavicle asymmetry, improper
              mediastinal alignment, and inaccurate cardiothoracic ratio
              measurements — leading to misdiagnosis.
            </p>
            <p className="text-gray-400 leading-relaxed">
              Our system automates the detection and correction of these
              errors using deep learning and geometric analysis, reducing
              repeat radiographs and radiation exposure to patients.
            </p>
          </div>

          <div className="card rounded-2xl p-8">
            <h2 className="text-2xl font-black mb-4">📋 Project Status</h2>
            <div className="space-y-3">
              {[
                { label: "Dataset Collection",      done: true  },
                { label: "Image Annotation (99)",   done: true  },
                { label: "Geometric Analysis",      done: true  },
                { label: "CNN Model Training",      done: true  },
                { label: "Rotation Correction",     done: true  },
                { label: "Web Interface",           done: true  },
                { label: "HRNet Keypoint Detection",done: false },
                { label: "Real-time Deployment",    done: false },
              ].map((item) => (
                <div key={item.label} className="flex items-center gap-3">
                  <div className={`w-5 h-5 rounded-full flex items-center justify-center text-xs flex-shrink-0
                    ${item.done
                      ? "bg-green-500/20 border border-green-500/40 text-green-400"
                      : "bg-gray-700/50 border border-gray-600/40 text-gray-600"}`}
                  >
                    {item.done ? "✓" : "○"}
                  </div>
                  <span className={`text-sm ${item.done ? "text-white" : "text-gray-600"}`}>
                    {item.label}
                  </span>
                  {!item.done && (
                    <span className="text-xs px-2 py-0.5 rounded-full bg-yellow-500/10 text-yellow-500 border border-yellow-500/20">
                      Future
                    </span>
                  )}
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Dataset Stats */}
        <div className="mb-14">
          <h2 className="text-2xl font-black mb-6 text-center">📊 Dataset & Model</h2>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {[
              { icon: "🖼️",  value: "99",            label: "Annotated Images"   },
              { icon: "✅",  value: "27",             label: "Normal X-Rays"      },
              { icon: "⚠️",  value: "72",             label: "Rotated X-Rays"     },
              { icon: "🎯",  value: "73.7%",          label: "Model Accuracy"     },
              { icon: "🧠",  value: "EfficientNet-B0",label: "Architecture"       },
              { icon: "📏",  value: "224×224",        label: "Input Size"         },
              { icon: "🔄",  value: "10",             label: "Training Epochs"    },
              { icon: "⚡",  value: "<2s",            label: "Inference Time"     },
            ].map((s) => (
              <div key={s.label} className="card rounded-xl p-4 text-center">
                <p className="text-2xl mb-1">{s.icon}</p>
                <p className="text-lg font-black text-cyan-400">{s.value}</p>
                <p className="text-xs text-gray-500 mt-1">{s.label}</p>
              </div>
            ))}
          </div>
        </div>

        {/* Workflow */}
        <div className="mb-14">
          <h2 className="text-2xl font-black mb-6 text-center">⚙️ System Workflow</h2>
          <div className="space-y-3">
            {workflow.map((w, i) => (
              <div key={w.step} className="flex gap-5 items-start">
                <div className="flex flex-col items-center flex-shrink-0">
                  <div className="w-12 h-12 rounded-xl flex items-center justify-center text-xl border border-cyan-500/20"
                    style={{ background: "rgba(0,212,255,0.08)" }}>
                    {w.icon}
                  </div>
                  {i < workflow.length - 1 && (
                    <div className="w-px flex-1 mt-1 mb-1 min-h-[20px]"
                      style={{ background: "rgba(0,212,255,0.15)" }}
                    />
                  )}
                </div>
                <div className="pb-4">
                  <div className="flex items-center gap-2 mb-1">
                    <span className="text-cyan-400/50 text-xs font-mono">STEP {w.step}</span>
                    <h3 className="text-white font-bold">{w.title}</h3>
                  </div>
                  <p className="text-gray-500 text-sm">{w.desc}</p>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Technologies */}
        <div className="mb-14">
          <h2 className="text-2xl font-black mb-6 text-center">🛠️ Technologies Used</h2>
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-4">
            {technologies.map((t) => (
              <div key={t.name} className="tech-card rounded-xl p-5">
                <div className="text-3xl mb-3">{t.icon}</div>
                <p className="text-white font-bold text-sm">{t.name}</p>
                <p className="text-cyan-400/60 text-xs mb-2">{t.role}</p>
                <p className="text-gray-600 text-xs leading-relaxed">{t.desc}</p>
              </div>
            ))}
          </div>
        </div>

        {/* Why PA Xray */}
        <div className="card rounded-2xl p-8 mb-14">
          <h2 className="text-2xl font-black mb-4">🫁 Why PA Chest X-Rays?</h2>
          <div className="grid md:grid-cols-3 gap-6">
            {[
              { icon: "📏", title: "Standard View",         desc: "PA view is the international standard for chest radiography, providing consistent anatomical alignment." },
              { icon: "❤️", title: "Accurate Heart Size",   desc: "PA view gives the most accurate representation of cardiac silhouette for cardiothoracic ratio measurement." },
              { icon: "🔬", title: "Consistent Analysis",   desc: "Consistent chest structure in PA view makes it ideal for AI model training and geometric analysis." },
            ].map((item) => (
              <div key={item.title} className="text-center p-4">
                <div className="text-4xl mb-3">{item.icon}</div>
                <h3 className="text-white font-bold mb-2">{item.title}</h3>
                <p className="text-gray-500 text-sm">{item.desc}</p>
              </div>
            ))}
          </div>
        </div>

        {/* CTA */}
        <div className="text-center rounded-3xl p-12 border border-cyan-500/20 relative overflow-hidden"
          style={{ background: "radial-gradient(ellipse at center, rgba(0,212,255,0.07) 0%, transparent 70%)" }}>
          <div className="absolute top-0 left-0 right-0 h-px"
            style={{ background: "linear-gradient(90deg, transparent, rgba(0,212,255,0.5), transparent)" }}
          />
          <h2 className="text-3xl font-black mb-3">Try It Yourself</h2>
          <p className="text-gray-400 mb-8 max-w-lg mx-auto">
            Upload a PA chest X-ray and see the AI detection and
            correction system in action.
          </p>
          <Link
            to="/analyze"
            className="btn-primary inline-flex items-center gap-2 px-10 py-4 rounded-xl font-bold text-white text-lg"
          >
            🔬 Start Analysis
          </Link>
        </div>

      </div>
    </div>
  );
}