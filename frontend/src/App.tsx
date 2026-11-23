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
          <Route path="/collection" element={<Navigate to="/" replace />} />
        </Routes>
      </main>
    </div>
  );
}
