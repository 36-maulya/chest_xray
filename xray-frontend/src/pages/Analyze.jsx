import { useState, useCallback } from "react";
import { useDropzone } from "react-dropzone";
import { useNavigate } from "react-router-dom";
import axios from "axios";
import { useApp } from "../context/AppContext";

const API_URL = "http://127.0.0.1:8000";

export default function Analyze() {
  const [file,    setFile]    = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error,   setError]   = useState(null);

  const { addResult } = useApp();
  const navigate      = useNavigate();

  const onDrop = useCallback((accepted) => {
    if (!accepted.length) return;
    const f = accepted[0];
    setFile(f);
    setPreview(URL.createObjectURL(f));
    setError(null);
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { "image/*": [".jpg", ".jpeg", ".png"] },
    maxFiles: 1,
  });

  const analyze = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    const form = new FormData();
    form.append("file", file);
    try {
      const res = await axios.post(`${API_URL}/analyze`, form);
      if (!res.data.valid) {
        setError(res.data.message);
      } else {
        const entry = addResult({ ...res.data, filename: file.name });
        navigate("/results");
      }
    } catch {
      setError("Cannot connect to backend. Make sure the server is running on port 8000.");
    }
    setLoading(false);
  };

  const reset = () => {
    setFile(null);
    setPreview(null);
    setError(null);
  };

  return (
    <div className="text-white min-h-screen px-6 py-16">
      <style>{`
        .upload-zone {
          background: radial-gradient(ellipse at center, rgba(0,212,255,0.04) 0%, transparent 70%);
          border: 2px dashed rgba(0,212,255,0.25);
          transition: all 0.3s ease;
        }
        .upload-zone:hover, .upload-zone.active {
          border-color: rgba(0,212,255,0.7);
          background: radial-gradient(ellipse at center, rgba(0,212,255,0.08) 0%, transparent 70%);
        }
        .btn-primary {
          background: linear-gradient(135deg, #00d4ff, #0077ff);
          transition: all 0.3s ease;
        }
        .btn-primary:hover  { transform: translateY(-2px); box-shadow: 0 8px 25px rgba(0,212,255,0.4); }
        .btn-primary:disabled { opacity: 0.5; transform: none; cursor: not-allowed; }
        .scan-line {
          position: absolute; top: 0; left: 0; right: 0;
          height: 2px;
          background: linear-gradient(90deg, transparent, #00d4ff, transparent);
          animation: scanDown 2s linear infinite;
          box-shadow: 0 0 8px #00d4ff;
        }
        @keyframes scanDown {
          0%   { top: 0%;   }
          100% { top: 100%; }
        }
        .tip-card {
          background: rgba(255,255,255,0.02);
          border: 1px solid rgba(255,255,255,0.06);
        }
        .pulse-ring {
          animation: pulseRing 2s ease-out infinite;
        }
        @keyframes pulseRing {
          0%   { transform: scale(0.95); opacity: 0.5; }
          100% { transform: scale(1.3);  opacity: 0;   }
        }
      `}</style>

      <div className="max-w-5xl mx-auto">

        {/* Header */}
        <div className="text-center mb-12">
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-cyan-500/10 border border-cyan-500/20 mb-4">
            <span className="text-cyan-400 text-sm">🔬 AI Analysis Engine</span>
          </div>
          <h1 className="text-4xl font-black mb-3">
            Analyze Chest <span className="text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-blue-500">X-Ray</span>
          </h1>
          <p className="text-gray-400 max-w-lg mx-auto">
            Upload a PA chest X-ray image. Our AI will detect rotation errors
            and automatically correct the positioning.
          </p>
        </div>

        <div className="grid md:grid-cols-3 gap-8">

          {/* Upload Area — takes 2 cols */}
          <div className="md:col-span-2 space-y-5">

            {/* Dropzone */}
            <div
              {...getRootProps()}
              className={`upload-zone rounded-2xl cursor-pointer ${isDragActive ? "active" : ""}`}
              style={{ minHeight: "320px", display: "flex", alignItems: "center", justifyContent: "center" }}
            >
              <input {...getInputProps()} />

              {preview ? (
                <div className="p-6 text-center w-full">
                  <div className="relative w-56 h-56 mx-auto rounded-xl overflow-hidden border border-cyan-500/30 mb-4">
                    <img src={preview} alt="preview" className="w-full h-full object-cover" />
                    {loading && <div className="absolute inset-0 bg-gray-950/60"><div className="scan-line" /></div>}
                  </div>
                  <p className="text-cyan-400 font-semibold">{file?.name}</p>
                  <p className="text-gray-500 text-sm mt-1">Click or drag to change image</p>
                </div>
              ) : (
                <div className="p-12 text-center">
                  <div className="relative w-24 h-24 mx-auto mb-6">
                    <div className="pulse-ring absolute inset-0 rounded-full border-2 border-cyan-500/30" />
                    <div className="w-24 h-24 rounded-full bg-cyan-500/10 border-2 border-cyan-500/20 flex items-center justify-center relative z-10">
                      <span className="text-4xl">📤</span>
                    </div>
                  </div>
                  <p className="text-white font-bold text-xl mb-2">
                    {isDragActive ? "Drop X-Ray here..." : "Drag & Drop Chest X-Ray"}
                  </p>
                  <p className="text-gray-500 text-sm mb-4">or click to browse files</p>
                  <div className="flex justify-center gap-2">
                    {["JPG", "JPEG", "PNG"].map((fmt) => (
                      <span key={fmt} className="px-3 py-1 rounded-full bg-white/5 border border-white/10 text-xs text-gray-400">
                        {fmt}
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>

            {/* Error */}
            {error && (
              <div className="p-4 rounded-xl bg-red-500/10 border border-red-500/30 flex items-start gap-3">
                <span className="text-2xl">⚠️</span>
                <div>
                  <p className="text-red-400 font-semibold">Invalid Image</p>
                  <p className="text-red-300/70 text-sm mt-1">{error}</p>
                </div>
              </div>
            )}

            {/* Buttons */}
            <div className="flex gap-3">
              <button
                onClick={analyze}
                disabled={!file || loading}
                className="btn-primary flex-1 py-4 rounded-xl font-bold text-white text-lg flex items-center justify-center gap-2"
              >
                {loading ? (
                  <>
                    <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                    Analyzing X-Ray...
                  </>
                ) : (
                  <> 🔍 Analyze X-Ray </>
                )}
              </button>
              {file && !loading && (
                <button
                  onClick={reset}
                  className="px-6 py-4 rounded-xl font-semibold text-gray-400 border border-white/10 hover:border-white/20 hover:text-white transition-all"
                >
                  Reset
                </button>
              )}
            </div>
          </div>

          {/* Right — Tips */}
          <div className="space-y-4">
            <h3 className="text-white font-bold text-sm uppercase tracking-wider">
              📋 Upload Guidelines
            </h3>

            {[
              { icon: "✅", title: "PA View Only",       desc: "Use Postero-Anterior chest X-rays only for accurate analysis." },
              { icon: "✅", title: "Clear Image",         desc: "Ensure the image is not blurry or overexposed." },
              { icon: "✅", title: "Full Chest Visible",  desc: "Both lungs and clavicles should be fully visible." },
              { icon: "❌", title: "No Other X-Rays",    desc: "Knee, spine, or limb X-rays will be rejected." },
              { icon: "❌", title: "No Screenshots",     desc: "Avoid screenshots or heavily compressed images." },
            ].map((tip) => (
              <div key={tip.title} className="tip-card rounded-xl p-4 flex gap-3">
                <span className="text-lg">{tip.icon}</span>
                <div>
                  <p className="text-white text-sm font-semibold">{tip.title}</p>
                  <p className="text-gray-500 text-xs mt-0.5">{tip.desc}</p>
                </div>
              </div>
            ))}

            {/* What happens */}
            <div className="mt-6 p-4 rounded-xl border border-cyan-500/20"
              style={{ background: "rgba(0,212,255,0.04)" }}>
              <p className="text-cyan-400 text-sm font-semibold mb-3">⚡ What happens next?</p>
              {[
                "CNN detects rotation",
                "Geometric analysis runs",
                "Affine correction applied",
                "Results page opens",
              ].map((step, i) => (
                <div key={step} className="flex items-center gap-2 mb-2">
                  <div className="w-5 h-5 rounded-full bg-cyan-500/20 border border-cyan-500/30 flex items-center justify-center text-xs text-cyan-400 font-bold flex-shrink-0">
                    {i + 1}
                  </div>
                  <p className="text-gray-400 text-xs">{step}</p>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}