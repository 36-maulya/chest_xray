import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { useApp } from "../context/AppContext";

export default function Results() {
  const { current } = useApp();
  const navigate    = useNavigate();
  const [sliderPos, setSliderPos] = useState(50);

  if (!current) {
    return (
      <div className="min-h-screen flex items-center justify-center text-white px-6">
        <div className="text-center">
          <div className="text-6xl mb-4">🔬</div>
          <h2 className="text-2xl font-bold mb-2">No Analysis Yet</h2>
          <p className="text-gray-500 mb-6">Upload a chest X-ray to see results here.</p>
          <Link
            to="/analyze"
            className="px-6 py-3 rounded-xl font-bold text-white"
            style={{ background: "linear-gradient(135deg, #00d4ff, #0077ff)" }}
          >
            Go to Analyze
          </Link>
        </div>
      </div>
    );
  }

  const isRotated = current.status === "ROTATED";

  const downloadCorrected = () => {
    if (!current.corrected_img) return;
    const link    = document.createElement("a");
    link.href     = `data:image/jpeg;base64,${current.corrected_img}`;
    link.download = `corrected_${current.filename || "xray"}.jpg`;
    link.click();
  };

  const downloadOriginal = () => {
    if (!current.original_img) return;
    const link    = document.createElement("a");
    link.href     = `data:image/jpeg;base64,${current.original_img}`;
    link.download = `original_${current.filename || "xray"}.jpg`;
    link.click();
  };

  return (
    <div className="text-white min-h-screen px-6 py-16">
      <style>{`
        .card { background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); }
        .glow-green { box-shadow: 0 0 40px rgba(34,197,94,0.15);  }
        .glow-red   { box-shadow: 0 0 40px rgba(239,68,68,0.15);  }
        .btn-primary { background: linear-gradient(135deg, #00d4ff, #0077ff); transition: all 0.3s; }
        .btn-primary:hover { transform: translateY(-2px); box-shadow: 0 8px 25px rgba(0,212,255,0.4); }
        .slider-wrap { position: relative; overflow: hidden; border-radius: 12px; cursor: col-resize; user-select: none; }
      `}</style>

      <div className="max-w-6xl mx-auto space-y-8">

        {/* Top Bar */}
        <div className="flex items-center justify-between flex-wrap gap-4">
          <div>
            <p className="text-gray-500 text-sm mb-1">📁 {current.filename}</p>
            <h1 className="text-3xl font-black">Analysis Results</h1>
          </div>
          <div className="flex gap-3 flex-wrap">
            <button
              onClick={() => navigate("/analyze")}
              className="px-4 py-2 rounded-xl text-sm font-semibold text-gray-400 border border-white/10 hover:border-cyan-500/30 hover:text-cyan-400 transition-all"
            >
              ← Analyze Another
            </button>
            <Link
              to="/history"
              className="px-4 py-2 rounded-xl text-sm font-semibold text-gray-400 border border-white/10 hover:border-cyan-500/30 hover:text-cyan-400 transition-all"
            >
              View History
            </Link>
          </div>
        </div>

        {/* Status Banner */}
        <div className={`p-6 rounded-2xl flex items-center gap-5 ${isRotated ? "bg-red-500/10 border border-red-500/20 glow-red" : "bg-green-500/10 border border-green-500/20 glow-green"}`}>
          <div className="text-5xl">{isRotated ? "⚠️" : "✅"}</div>
          <div className="flex-1">
            <h2 className="text-2xl font-black mb-1">
              {isRotated ? "Positioning Error Detected" : "Properly Positioned X-Ray"}
            </h2>
            <p className="text-gray-400">
              {isRotated
                ? `${current.direction} detected with ${current.severity?.toLowerCase()} severity. Correction of ${current.angle}° applied.`
                : "No rotation error found. The chest X-ray meets standard PA positioning criteria."}
            </p>
          </div>
          <div className={`px-4 py-2 rounded-full text-sm font-bold border ${isRotated ? "bg-red-500/20 text-red-400 border-red-500/30" : "bg-green-500/20 text-green-400 border-green-500/30"}`}>
            {current.status}
          </div>
        </div>

        {/* Stats Row */}
        <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
          {[
            { icon: "🧠", label: "Final Result",    value: current?.status === "ROTATED" ? "ROTATED" : "NORMAL"        },
            { icon: "📊", label: "Confidence",    value: current 
    ? (current.normal_conf > current.rotated_conf
        ? `${current.normal_conf.toFixed(1)}% (${current.status === "ROTATED" ? "Rotated" : "Normal"})`
        : `${current.rotated_conf.toFixed(1)}% (${current.status === "ROTATED" ? "Rotated" : "Normal"})`)
    : "No Data"  },
            { icon: "📏", label: "Rotation Ratio",     value: `${(current.rotation_ratio ).toFixed(3)}`         },
            { icon: "🔄", label: "Correction",    value: `${current.angle}°`               },
            { icon: "⚡", label: "Severity",      value: current.severity                  },
          ].map((s) => (
            <div key={s.label} className="card rounded-xl p-4 text-center">
              <p className="text-2xl mb-1">{s.icon}</p>
              <p className="text-lg font-black text-white">{s.value}</p>
              <p className="text-xs text-gray-500 mt-1">{s.label}</p>
            </div>
          ))}
        </div>

        {/* Main Content */}
        <div className="grid md:grid-cols-5 gap-6">

          {/* Image Section — 3 cols */}
          <div className="md:col-span-3 card rounded-2xl p-6 space-y-4">
            <h3 className="font-bold text-gray-300 text-lg">📸 Image Comparison</h3>

            {isRotated && current.corrected_img ? (
              <>
                <div
                  className="slider-wrap h-80"
                  onMouseMove={(e) => {
                    const rect = e.currentTarget.getBoundingClientRect();
                    const pos  = ((e.clientX - rect.left) / rect.width) * 100;
                    setSliderPos(Math.min(Math.max(pos, 5), 95));
                  }}
                >
                  {/* Corrected (back) */}
                  <img
                    src={`data:image/jpeg;base64,${current.corrected_img}`}
                    alt="corrected"
                    className="absolute inset-0 w-full h-full object-cover"
                  />
                  {/* Original (front clipped) */}
                  <div
                    className="absolute inset-0 overflow-hidden"
                    style={{ width: `${sliderPos}%` }}
                  >
                    <img
                      src={`data:image/jpeg;base64,${current.original_img}`}
                      alt="original"
                      className="h-full object-cover"
                      style={{ width: `${10000 / sliderPos}%`, maxWidth: "none" }}
                    />
                  </div>
                  {/* Divider */}
                  <div
                    className="absolute top-0 bottom-0 w-0.5 bg-cyan-400 pointer-events-none"
                    style={{ left: `${sliderPos}%` }}
                  >
                    <div className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 w-9 h-9 rounded-full bg-cyan-400 flex items-center justify-center text-gray-900 font-black text-sm shadow-xl">
                      ↔
                    </div>
                  </div>
                  {/* Labels */}
                  <div className="absolute top-3 left-3 px-2 py-1 rounded-lg bg-black/70 text-xs text-white font-medium">
                    Original
                  </div>
                  <div className="absolute top-3 right-3 px-2 py-1 rounded-lg bg-black/70 text-xs text-cyan-400 font-medium">
                    Corrected
                  </div>
                </div>
                <p className="text-xs text-gray-600 text-center">← Move mouse to compare Original vs Corrected →</p>
              </>
            ) : (
              <div className="h-80 rounded-xl overflow-hidden">
                <img
                  src={`data:image/jpeg;base64,${current.original_img}`}
                  alt="original"
                  className="w-full h-full object-cover"
                />
              </div>
            )}

            {/* Download Buttons */}
            <div className="flex gap-3">
              {isRotated && current.corrected_img && (
                <button
                  onClick={downloadCorrected}
                  className="btn-primary flex-1 py-3 rounded-xl font-semibold text-white text-sm flex items-center justify-center gap-2"
                >
                  📥 Download Corrected
                </button>
              )}
              <button
                onClick={downloadOriginal}
                className="flex-1 py-3 rounded-xl font-semibold text-sm text-gray-400 border border-white/10 hover:border-white/20 hover:text-white transition-all flex items-center justify-center gap-2"
              >
                📄 Download Original
              </button>
            </div>
          </div>

          {/* Analysis Panel — 2 cols */}
          <div className="md:col-span-2 space-y-4">

            {/* Clinical Decision & Action Plan */}
<div className="card rounded-2xl p-5 border border-slate-800/80 bg-slate-900/30">
  <div className="flex items-center gap-2 mb-4">
    <span className="text-xl">🔬</span>
    <h4 className="font-bold text-gray-300">Quality Assurance Breakdown</h4>
  </div>

  {current?.status === "ROTATED" ? (
    /* LAYOUT FOR ROTATED IMAGES */
    <div className="space-y-4">
      <div className="py-2.5 px-4 rounded-xl bg-rose-500/10 border border-rose-500/20 text-center">
        <span className="text-rose-400 font-black text-sm tracking-wider">
          🚨 REJECTED: ROTATION DETECTED
        </span>
      </div>

      <div className="space-y-2.5 text-xs text-gray-400 border-t border-b border-white/5 py-3">
        <div className="flex justify-between">
          <span>Observed Defect:</span>
          <span className="text-amber-400 font-semibold">{current?.direction}</span>
        </div>
        <div className="flex justify-between">
          <span>Severity Level:</span>
          <span className="text-amber-500 font-medium">{current?.severity}</span>
        </div>
        <div className="flex justify-between">
          <span>Rotation Ratio:</span>
          <span className="text-white font-medium">{(current?.rotation_ratio ).toFixed(3)} </span>
        </div>
      </div>

      <div className="bg-amber-500/5 border border-amber-500/10 rounded-xl p-3 text-xs text-amber-300/90 leading-relaxed">
        <strong>💡 Action Required:</strong> Patient positioning is compromised. Review the geometric warp matrix output below or order a projection re-take.
      </div>
    </div>
  ) : (
    /* COMPLETELY DIFFERENT LAYOUT FOR NORMAL IMAGES */
    <div className="space-y-4">
      <div className="py-2.5 px-4 rounded-xl bg-emerald-500/10 border border-emerald-500/20 text-center">
        <span className="text-emerald-400 font-black text-sm tracking-wider">
          ✅ APPROVED: PERFECT ALIGNMENT
        </span>
      </div>

      {/* Pulling the extra clinical metrics sent by your backend script */}
      <div className="space-y-2.5 text-xs text-gray-400 border-t border-b border-white/5 py-3">
        <div className="flex justify-between">
          <span>Mediastinal Shift:</span>
          <span className={`font-semibold ${current?.mediastinal_status === "Normal" ? "text-emerald-400" : "text-amber-400"}`}>
            {current?.mediastinal_status || "Normal"}
          </span>
        </div>
        <div className="flex justify-between">
          <span>Scapular Retraction:</span>
          <span className="text-white font-medium">
            {current?.scapular_status || "Acceptable Layout"}
          </span>
        </div>
        <div className="flex justify-between">
          <span>Symmetry Variance:</span>
          <span className="text-emerald-500 font-medium">{((current?.rotation_ratio || 0) ).toFixed(3)}</span>
        </div>
      </div>

      <div className="bg-emerald-500/5 border border-emerald-500/10 rounded-xl p-3 text-xs text-emerald-400/90 leading-relaxed">
        <strong>💡 Next Step:</strong> Thoracic anatomy is well-centered. This image is completely pristine and ready for secondary pathology diagnostic pipelines (Lungs/Heart).
      </div>
    </div>
  )}
</div>

            {/* Geometric */}
            <div className="card rounded-2xl p-5">
              <h4 className="font-bold text-gray-300 mb-4">📐 Geometric Measurements</h4>
              <div className="space-y-3">
                {[
                  { label: "Right Clavicle", value: `${current.right_cm} cm` },
                  { label: "Left Clavicle",  value: `${current.left_cm} cm`  },
                  { label: "Rotation Ratio",     value: `${(current.rotation_ratio ).toFixed(3)}`},
                  { label: "Direction",      value: current.direction         },
                  { label: "Severity",       value: current.severity          },
                  { label: "Angle",          value: `${current.angle}°`       },
                ].map((item) => (
                  <div key={item.label} className="flex justify-between items-center py-2 border-b border-white/5 last:border-0">
                    <span className="text-gray-500 text-sm">{item.label}</span>
                    <span className="text-white text-sm font-semibold">{item.value}</span>
                  </div>
                ))}
              </div>
            </div>

            {/* Timestamp */}
            <div className="card rounded-xl p-4">
              <p className="text-xs text-gray-600">🕐 Analyzed at</p>
              <p className="text-white text-sm font-semibold mt-1">{current.timestamp}</p>
            </div>

          </div>
        </div>
      </div>
    </div>
  );
}