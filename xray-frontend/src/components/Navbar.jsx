import { useState } from "react";
import { Link, useLocation } from "react-router-dom";

const navLinks = [
  { path: "/",        label: "Home",    icon: "🏠" },
  { path: "/analyze", label: "Analyze", icon: "🔬" },
  { path: "/history", label: "History", icon: "🕐" },
  { path: "/gallery", label: "Gallery", icon: "🖼️" },
  { path: "/about",   label: "About",   icon: "ℹ️"  },
];

export default function Navbar() {
  const location = useLocation();
  const [menuOpen, setMenuOpen] = useState(false);

  return (
    <nav className="border-b border-white/5 bg-gray-950/95 backdrop-blur-md sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">

        {/* Logo */}
        <Link to="/" className="flex items-center gap-3 group">
          <div className="relative">
            <div className="w-10 h-10 rounded-xl bg-cyan-500/20 flex items-center justify-center border border-cyan-500/30 group-hover:border-cyan-400/60 transition-all">
              <span className="text-xl">🫁</span>
            </div>
            <div className="absolute -top-1 -right-1 w-3 h-3 bg-green-400 rounded-full border-2 border-gray-950 animate-pulse" />
          </div>
          <div>
            <h1 className="text-lg font-bold text-white">
              ChestAlign <span className="text-cyan-400">AI</span>
            </h1>
            <p className="text-xs text-gray-500">X-Ray Positioning Analyzer</p>
          </div>
        </Link>

        {/* Desktop Links */}
        <div className="hidden md:flex items-center gap-1">
          {navLinks.map((link) => {
            const isActive = location.pathname === link.path;
            return (
              <Link
                key={link.path}
                to={link.path}
                className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all
                  ${isActive
                    ? "bg-cyan-500/15 text-cyan-400 border border-cyan-500/30"
                    : "text-gray-400 hover:text-white hover:bg-white/5"
                  }`}
              >
                <span>{link.icon}</span>
                {link.label}
              </Link>
            );
          })}
        </div>

        {/* Analyze Button */}
        <div className="hidden md:flex items-center gap-3">
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-green-500/10 border border-green-500/20">
            <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse" />
            <span className="text-xs text-green-400 font-medium">System Online</span>
          </div>
          <Link
            to="/analyze"
            className="px-4 py-2 rounded-xl text-sm font-bold text-white"
            style={{ background: "linear-gradient(135deg, #00d4ff, #0077ff)" }}
          >
            Start Analysis
          </Link>
        </div>

        {/* Mobile Menu Button */}
        <button
          className="md:hidden text-gray-400 hover:text-white"
          onClick={() => setMenuOpen(!menuOpen)}
        >
          {menuOpen ? "✕" : "☰"}
        </button>
      </div>

      {/* Mobile Menu */}
      {menuOpen && (
        <div className="md:hidden border-t border-white/5 px-6 py-4 space-y-1">
          {navLinks.map((link) => {
            const isActive = location.pathname === link.path;
            return (
              <Link
                key={link.path}
                to={link.path}
                onClick={() => setMenuOpen(false)}
                className={`flex items-center gap-3 px-4 py-3 rounded-lg text-sm font-medium transition-all
                  ${isActive
                    ? "bg-cyan-500/15 text-cyan-400"
                    : "text-gray-400 hover:text-white hover:bg-white/5"
                  }`}
              >
                <span>{link.icon}</span>
                {link.label}
              </Link>
            );
          })}
        </div>
      )}
    </nav>
  );
}