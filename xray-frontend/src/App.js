import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import { AppProvider } from "./context/AppContext";
import Navbar from "./components/Navbar";
import Footer from "./components/Footer";
import Home from "./pages/Home";
import Analyze from "./pages/Analyze";
import Results from "./pages/Results";
import History from "./pages/History";
import Gallery from "./pages/Gallery";
import About from "./pages/About";

export default function App() {
  return (
    <AppProvider>
      <Router>
        <div className="min-h-screen bg-gray-950 flex flex-col">
          <Navbar />
          <main className="flex-1">
            <Routes>
              <Route path="/"        element={<Home />}    />
              <Route path="/analyze" element={<Analyze />} />
              <Route path="/results" element={<Results />} />
              <Route path="/history" element={<History />} />
              <Route path="/gallery" element={<Gallery />} />
              <Route path="/about"   element={<About />}   />
            </Routes>
          </main>
          <Footer />
        </div>
      </Router>
    </AppProvider>
  );
}