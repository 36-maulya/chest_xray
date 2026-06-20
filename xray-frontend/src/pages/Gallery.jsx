import { useState } from "react";
import { Link } from "react-router-dom";
import { useApp } from "../context/AppContext";

export default function Gallery() {
  const { results, setCurrent }   = useApp();
  const [filter, setFilter]       = useState("all");
  const [selected, setSelected]   = useState(null);

  const filtered = results.filter((r) => {
    if (filter === "all")      return true;
    if (filter === "normal")   return r.status === "NORMAL";
    if (filter === "rotated")  return r.status === "ROTATED";
    if (filter === "corrected") return r.corrected_img != null;
    return true;
  });

  const downloadImage = (base64, filename) => {
    const link    = document.createElement("a");
    link.href     = `data:image/jpeg;base64,${base64}`;
    link.download = filename;
    link.click();
  };

  if (results.length === 0) {
    return (
      <div className="min-h-screen flex items-center justify-center text-white px-6">
        <div className="text-center">
          <div className="text-6xl mb-4">🖼️</div>
          <h2 className="text-2xl font-bold mb-2">No Images Yet</h2>
          <p className="text-gray-500 mb-6">Analyzed X-rays will appear in the gallery.</p>
          <Link
            to="/analyze"
            className="px-6 py-3 rounded-xl font-bold text-white"
            style={{ background: "linear-gradient(135deg, #00d4ff, #0077ff)" }}
          >
            Analyze First X-Ray
          </Link>
        </div>
      </div>
    );
  }

  return (
    <div className="text-white min-h-screen px-6 py-16">
      <style>{`
        .card { background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); }
        .img-card {
          background: rgba(255,255,255,0.02);
          border: 1px solid rgba(255,255,255,0.07);
          transition: all 0.3s ease;
          overflow: hidden;
        }
        .img-card:hover {
          border-color: rgba(0,212,255,0.35);
          transform: translateY(-4px);
          box-shadow: 0 12px 30px rgba(0,0,0,0.4);
        }
        .filter-btn {
          transition: all 0.2s ease;
        }
        .overlay {
          background: linear-gradient(to top, rgba(0,0,0,0.85) 0%, transparent 60%);
        }
        .modal-bg {
          background: rgba(0,0,0,0.85);
          backdrop-filter: blur(8px);
        }
        .btn-primary { background: linear-gradient(135deg, #00d4ff, #0077ff); }
      `}</style>

      <div className="max-w-7xl mx-auto">

        {/* Header */}
        <div className="flex items-center justify-between flex-wrap gap-4 mb-8">
          <div>
            <h1 className="text-3xl font-black mb-1">Image Gallery</h1>
            <p className="text-gray-500">All original and corrected X-ray images from this session.</p>
          </div>
          <Link
            to="/analyze"
            className="btn-primary px-5 py-2.5 rounded-xl font-bold text-white text-sm"
          >
            + New Analysis
          </Link>
        </div>

        {/* Filter Tabs */}
        <div className="flex gap-2 mb-8 flex-wrap">
          {[
            { key: "all",       label: `All (${results.length})`                                    },
            { key: "normal",    label: `Normal (${results.filter(r => r.status === "NORMAL").length})`   },
            { key: "rotated",   label: `Rotated (${results.filter(r => r.status === "ROTATED").length})` },
            { key: "corrected", label: `Corrected (${results.filter(r => r.corrected_img).length})`      },
          ].map((f) => (
            <button
              key={f.key}
              onClick={() => setFilter(f.key)}
              className={`filter-btn px-4 py-2 rounded-xl text-sm font-semibold border
                ${filter === f.key
                  ? "bg-cyan-500/15 text-cyan-400 border-cyan-500/30"
                  : "text-gray-500 border-white/10 hover:text-white hover:border-white/20"
                }`}
            >
              {f.label}
            </button>
          ))}
        </div>

        {/* Grid */}
        {filtered.length === 0 ? (
          <div className="text-center py-20 text-gray-600">
            <p className="text-4xl mb-3">🔍</p>
            <p>No images match this filter.</p>
          </div>
        ) : (
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-5">
            {filtered.map((entry, i) => (
              <div key={entry.id}>

                {/* Original */}
                <div
                  className="img-card rounded-2xl cursor-pointer mb-3"
                  onClick={() => setSelected({ img: entry.original_img, label: "Original", entry })}
                >
                  <div className="relative h-44">
                    <img
                      src={`data:image/jpeg;base64,${entry.original_img}`}
                      alt="original"
                      className="w-full h-full object-cover"
                    />
                    <div className="overlay absolute inset-0" />
                    <div className="absolute top-2 left-2 px-2 py-1 rounded-lg bg-black/60 text-xs text-white font-medium">
                      Original
                    </div>
                    <div className={`absolute top-2 right-2 px-2 py-1 rounded-lg text-xs font-bold
                      ${entry.status === "NORMAL"
                        ? "bg-green-500/30 text-green-400"
                        : "bg-red-500/30 text-red-400"}`}
                    >
                      {entry.status === "NORMAL" ? "✅" : "⚠️"} {entry.status}
                    </div>
                    <div className="absolute bottom-0 left-0 right-0 p-3">
                      <p className="text-white text-xs font-semibold truncate">
                        {entry.filename || `Image ${i + 1}`}
                      </p>
                      <p className="text-gray-400 text-xs">{entry.confidence}% confidence</p>
                    </div>
                  </div>
                  <div className="p-3 flex justify-between items-center">
                    <span className="text-gray-500 text-xs">{entry.timestamp?.split(",")[0]}</span>
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        downloadImage(entry.original_img, `original_${entry.filename || i}`);
                      }}
                      className="text-xs text-cyan-400 hover:text-cyan-300 transition-colors"
                    >
                      📥 Save
                    </button>
                  </div>
                </div>

                {/* Corrected (if exists) */}
                {entry.corrected_img && (
                  <div
                    className="img-card rounded-2xl cursor-pointer border-cyan-500/20"
                    onClick={() => setSelected({ img: entry.corrected_img, label: "Corrected", entry })}
                  >
                    <div className="relative h-44">
                      <img
                        src={`data:image/jpeg;base64,${entry.corrected_img}`}
                        alt="corrected"
                        className="w-full h-full object-cover"
                      />
                      <div className="overlay absolute inset-0" />
                      <div className="absolute top-2 left-2 px-2 py-1 rounded-lg bg-cyan-500/40 text-xs text-cyan-300 font-medium">
                        ✨ Corrected
                      </div>
                      <div className="absolute top-2 right-2 px-2 py-1 rounded-lg bg-black/60 text-xs text-white">
                        {entry.angle}° fixed
                      </div>
                      <div className="absolute bottom-0 left-0 right-0 p-3">
                        <p className="text-white text-xs font-semibold truncate">
                          corrected_{entry.filename || `Image ${i + 1}`}
                        </p>
                        <p className="text-cyan-400/70 text-xs">{entry.direction}</p>
                      </div>
                    </div>
                    <div className="p-3 flex justify-between items-center">
                      <span className="text-cyan-500/50 text-xs">Auto-corrected</span>
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          downloadImage(entry.corrected_img, `corrected_${entry.filename || i}`);
                        }}
                        className="text-xs text-cyan-400 hover:text-cyan-300 transition-colors"
                      >
                        📥 Save
                      </button>
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>

      {/* ── Lightbox Modal ── */}
      {selected && (
        <div
          className="fixed inset-0 modal-bg z-50 flex items-center justify-center p-6"
          onClick={() => setSelected(null)}
        >
          <div
            className="relative max-w-2xl w-full rounded-2xl overflow-hidden border border-white/10"
            onClick={(e) => e.stopPropagation()}
          >
            {/* Modal Header */}
            <div className="flex items-center justify-between px-5 py-4 border-b border-white/10"
              style={{ background: "rgba(15,23,42,0.95)" }}>
              <div>
                <p className="text-white font-bold">{selected.label} X-Ray</p>
                <p className="text-gray-500 text-xs">{selected.entry.filename}</p>
              </div>
              <div className="flex items-center gap-3">
                <button
                  onClick={() => downloadImage(selected.img, `${selected.label.toLowerCase()}_${selected.entry.filename || "xray"}.jpg`)}
                  className="btn-primary px-4 py-2 rounded-lg text-sm font-semibold text-white"
                >
                  📥 Download
                </button>
                <button
                  onClick={() => setSelected(null)}
                  className="w-8 h-8 rounded-lg bg-white/10 hover:bg-white/20 flex items-center justify-center text-gray-400 hover:text-white transition-all"
                >
                  ✕
                </button>
              </div>
            </div>

            {/* Modal Image */}
            <img
              src={`data:image/jpeg;base64,${selected.img}`}
              alt={selected.label}
              className="w-full object-contain max-h-[70vh]"
              style={{ background: "#000" }}
            />

            {/* Modal Footer */}
            <div className="px-5 py-3 flex gap-4 flex-wrap border-t border-white/10"
              style={{ background: "rgba(15,23,42,0.95)" }}>
              {[
                { label: "Status",     value: selected.entry.status           },
                { label: "Confidence", value: `${selected.entry.confidence}%` },
                { label: "Angle",      value: `${selected.entry.angle}°`      },
                { label: "Severity",   value: selected.entry.severity         },
              ].map((info) => (
                <div key={info.label}>
                  <p className="text-gray-600 text-xs">{info.label}</p>
                  <p className="text-white text-sm font-bold">{info.value}</p>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}