import { Routes, Route, Navigate } from "react-router-dom";
import Navbar from "./components/Navbar";
import HomePage from "./pages/HomePage";
import DictionaryPage from "./pages/DictionaryPage";
import SettingsPage from "./pages/SettingsPage";

export default function App() {
  return (
    <div className="app">
      <Navbar />
      <main style={{ paddingTop: 80 }}>
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/dictionary" element={<DictionaryPage />} />
          <Route path="/settings" element={<SettingsPage />} />
<<<<<<< HEAD
          <Route path="/collection" element={<Navigate to="/" replace />} />
=======
>>>>>>> 4ab33f2ebbae39c6697c71347ceccec003d2dd78
        </Routes>
      </main>
    </div>
  );
}