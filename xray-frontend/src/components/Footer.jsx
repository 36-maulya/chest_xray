import { Link } from "react-router-dom";

export default function Footer() {
  return (
    <footer className="border-t border-white/5 bg-gray-950 mt-16">
      <div className="max-w-7xl mx-auto px-6 py-12">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-10">

          {/* Brand */}
          <div className="col-span-1 md:col-span-2">
            <div className="flex items-center gap-3 mb-4">
              <div className="w-10 h-10 rounded-xl bg-cyan-500/20 flex items-center justify-center border border-cyan-500/30">
                <span className="text-xl">🫁</span>
              </div>
              <div>
                <h2 className="text-lg font-bold text-white">
                  ChestAlign <span className="text-cyan-400">AI</span>
                </h2>
                <p className="text-xs text-gray-500">X-Ray Positioning Analyzer</p>
              </div>
            </div>
            <p className="text-gray-500 text-sm leading-relaxed max-w-sm">
              An AI-powered system to detect and correct positioning errors
              in PA chest X-rays using EfficientNet-B0 deep learning and
              geometric analysis.
            </p>
            <div className="flex items-center gap-2 mt-4 px-3 py-1.5 rounded-full bg-green-500/10 border border-green-500/20 w-fit">
              <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse" />
              <span className="text-xs text-green-400 font-medium">System Online</span>
            </div>
          </div>

          {/* Quick Links */}
          <div>
            <h3 className="text-white font-semibold mb-4 text-sm uppercase tracking-wider">
              Quick Links
            </h3>
            <ul className="space-y-2">
              {[
                { path: "/",        label: "Home"     },
                { path: "/analyze", label: "Analyze"  },
                { path: "/history", label: "History"  },
                { path: "/gallery", label: "Gallery"  },
                { path: "/about",   label: "About"    },
              ].map((link) => (
                <li key={link.path}>
                  <Link
                    to={link.path}
                    className="text-gray-500 hover:text-cyan-400 text-sm transition-colors"
                  >
                    {link.label}
                  </Link>
                </li>
              ))}
            </ul>
          </div>

          {/* Technology */}
          <div>
            <h3 className="text-white font-semibold mb-4 text-sm uppercase tracking-wider">
              Technology
            </h3>
            <ul className="space-y-2">
              {[
                "EfficientNet-B0",
                "PyTorch",
                "FastAPI",
                "React + Tailwind",
                "OpenCV",
                "Geometric Analysis",
              ].map((tech) => (
                <li key={tech} className="flex items-center gap-2">
                  <div className="w-1.5 h-1.5 bg-cyan-500 rounded-full" />
                  <span className="text-gray-500 text-sm">{tech}</span>
                </li>
              ))}
            </ul>
          </div>
        </div>

        {/* Bottom Bar */}
        <div className="border-t border-white/5 mt-10 pt-6 flex flex-col md:flex-row items-center justify-between gap-4">
          <p className="text-gray-600 text-sm">
            © 2025 ChestAlign AI — Major Project Demo
          </p>
          <div className="flex items-center gap-2">
            <span className="text-gray-600 text-sm">Built with</span>
            <span className="text-red-400">❤️</span>
            <span className="text-gray-600 text-sm">for better radiology</span>
          </div>
        </div>
      </div>
    </footer>
  );
}