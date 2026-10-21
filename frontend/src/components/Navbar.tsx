
import { NavLink } from "react-router-dom";
import { Home, BookOpen, Settings } from "lucide-react";
import styles from "./Navbar.module.css";

export default function Navbar() {
  return (
    <header className={styles.header}>
      <div className={styles.inner}>
        <NavLink to="/" className={styles.logo} aria-label="Sinaliza - Início">
          <img
            src="/logos/Logo%20Maos.jpeg"
            alt="Logo Sinaliza"
            className={styles.logoImage}
          />
          <span className={styles.logoText}>Sinaliza</span>
        </NavLink>

        <nav className={styles.nav} aria-label="Navegação principal">
          <NavLink
            to="/"
            end
            className={({ isActive }) =>
              `${styles.link} ${isActive ? styles.active : ""}`
            }
          >
            <Home size={16} />
            <span>Início</span>
          </NavLink>

          <NavLink
            to="/dictionary"
            className={({ isActive }) =>
              `${styles.link} ${isActive ? styles.active : ""}`
            }
          >
            <BookOpen size={16} />
            <span>Dicionário</span>
          </NavLink>

          <NavLink
            to="/settings"
            className={({ isActive }) =>
              `${styles.link} ${isActive ? styles.active : ""}`
            }
          >
            <Settings size={16} />
            <span>Configurações</span>
          </NavLink>
        </nav>
      </div>
    </header>
  );
}
