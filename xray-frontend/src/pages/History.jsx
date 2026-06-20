import { Link, useNavigate } from "react-router-dom";
import { useApp } from "../context/AppContext";

export default function History() {
  const { results, setCurrent } = useApp();
  const navigate                = useNavigate();

  const viewResult = (entry) => {
    setCurrent(entry);
    navigate("/results");
  };

  if (results.length === 0) {
    return (
      <div className="min-h-screen flex items-center justify-center text-white px-6">
        <div className="text-center">
          <div className="text-6xl mb-4">🕐</div>
          <h2 className="text-2xl font-bold mb-2">No History Yet</h2>
          <p className="text-gray-500 mb-6">Your analyzed X-rays will appear here.</p>
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

  const totalNormal  = results.filter((r) => r.status === "NORMAL").length;
  const totalRotated = results.filter((r) => r.status === "ROTATED").length;
  const avgConf      = (results.reduce((a, r) => a + r.confidence, 0) / results.length).toFixed(1);

  return (
    <div className="text-white min-h-screen px-6 py-16">
      <style>{`
        .card { background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); }
        .row-card {
          background: rgba(255,255,255,0.02);
          border: 1px solid rgba(255,255,255,0.06);
          transition: all 0.2s ease;
        }
        .row-card:hover {
          background: rgba(0,212,255,0.05);
          border-color: rgba(0,212,255,0.2);
          transform: translateX(4px);
        }
        .btn-primary { background: linear-gradient(135deg, #00d4ff, #0077ff); }
      `}</style>

      <div className="max-w-5xl mx-auto">

        {/* Header */}
        <div className="flex items-center justify-between flex-wrap gap-4 mb-10">
          <div>
            <h1 className="text-3xl font-black mb-1">Analysis History</h1>
            <p className="text-gray-500">All your previous X-ray analyses in this session.</p>
          </div>
          <Link
            to="/analyze"
            className="btn-primary px-5 py-2.5 rounded-xl font-bold text-white text-sm"
          >
            + New Analysis
          </Link>
        </div>

        {/* Summary Stats */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
          {[
            { icon: "📋", label: "Total Analyzed", value: results.length,  color: "text-white"       },
            { icon: "✅", label: "Normal",          value: totalNormal,     color: "text-green-400"   },
            { icon: "⚠️", label: "Rotated",         value: totalRotated,    color: "text-red-400"     },
            { icon: "🎯", label: "Avg Confidence",  value: `${avgConf}%`,   color: "text-cyan-400"    },
          ].map((s) => (
            <div key={s.label} className="card rounded-xl p-4 text-center">
              <p className="text-2xl mb-1">{s.icon}</p>
              <p className={`text-2xl font-black ${s.color}`}>{s.value}</p>
              <p className="text-xs text-gray-500 mt-1">{s.label}</p>
            </div>
          ))}
        </div>

        {/* Table Header */}
        <div className="hidden md:grid grid-cols-6 gap-4 px-4 py-2 mb-2">
          {["Image", "Time", "CNN Result", "Confidence", "Status", "Action"].map((h) => (
            <p key={h} className="text-xs text-gray-600 uppercase tracking-wider font-semibold">{h}</p>
          ))}
        </div>

        {/* Rows */}
        <div className="space-y-3">
          {results.map((entry, i) => (
            <div key={entry.id} className="row-card rounded-xl px-4 py-4 grid grid-cols-2 md:grid-cols-6 gap-4 items-center cursor-pointer"
              onClick={() => viewResult(entry)}
            >
              {/* Image thumb */}
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-lg overflow-hidden border border-white/10 flex-shrink-0">
                  {entry.original_img
                    ? <img src={`data:image/jpeg;base64,${entry.original_img}`} alt="thumb" className="w-full h-full object-cover" />
                    : <div className="w-full h-full bg-gray-800 flex items-center justify-center text-xs">🫁</div>
                  }
                </div>
                <div>
                  <p className="text-white text-sm font-semibold truncate max-w-[100px]">{entry.filename || `Image ${i + 1}`}</p>
                  <p className="text-gray-600 text-xs">#{i + 1}</p>
                </div>
              </div>

              {/* Time */}
              <p className="text-gray-500 text-xs">{entry.timestamp}</p>

              {/* CNN */}
              <div className={`hidden md:flex items-center gap-2 px-3 py-1 rounded-full w-fit text-xs font-bold
                ${entry.cnn_prediction === "NORMAL"
                  ? "bg-green-500/15 text-green-400 border border-green-500/25"
                  : "bg-red-500/15 text-red-400 border border-red-500/25"}`}
              >
                {entry.cnn_prediction === "NORMAL" ? "✅" : "⚠️"} {entry.cnn_prediction}
              </div>

              {/* Confidence */}
              <div className="hidden md:block">
                <div className="flex items-center gap-2">
                  <div className="flex-1 bg-gray-800 rounded-full h-1.5">
                    <div
                      className="h-1.5 rounded-full bg-cyan-400"
                      style={{ width: `${entry.confidence}%` }}
                    />
                  </div>
                  <span className="text-cyan-400 text-xs font-bold">{entry.confidence}%</span>
                </div>
              </div>

              {/* Status */}
              <div className={`hidden md:flex items-center gap-1.5 px-3 py-1 rounded-full w-fit text-xs font-bold
                ${entry.status === "NORMAL"
                  ? "bg-green-500/15 text-green-400"
                  : "bg-red-500/15 text-red-400"}`}
              >
                {entry.status}
              </div>

              {/* Action */}
              <button
                onClick={(e) => { e.stopPropagation(); viewResult(entry); }}
                className="hidden md:flex items-center gap-1 px-3 py-1.5 rounded-lg text-xs font-semibold text-cyan-400 border border-cyan-500/20 hover:bg-cyan-500/10 transition-all w-fit"
              >
                View →
              </button>
            </div>
          ))}
        </div>

        {/* Note */}
        <p className="text-center text-gray-700 text-xs mt-8">
          ℹ️ History is stored for this session only. Refresh page to clear.
        </p>
      </div>
    </div>
  );
}