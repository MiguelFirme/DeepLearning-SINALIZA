import { NavLink } from "react-router-dom";
import { Home, BookOpen, Settings } from "lucide-react";
import styles from "./Navbar.module.css";

export default function Navbar() {
  return (
    <header className={styles.header}>
      <div className={styles.inner}>
        {/* Logo */}
        <div className={styles.logo}>
          <span className={styles.logoIcon}>✋</span>
          <span className={styles.logoText}>Sinaliza</span>
        </div>

        {/* Navigation */}
        <nav className={styles.nav}>
          <NavLink
            to="/"
            className={({ isActive }) =>
              `${styles.link} ${isActive ? styles.active : ""}`
            }
          >
            <Home size={16} />
            Home
          </NavLink>
          <NavLink
            to="/dictionary"
            className={({ isActive }) =>
              `${styles.link} ${isActive ? styles.active : ""}`
            }
          >
            <BookOpen size={16} />
            Dictionary
          </NavLink>
          <NavLink
            to="/settings"
            className={({ isActive }) =>
              `${styles.link} ${isActive ? styles.active : ""}`
            }
          >
            <Settings size={16} />
            Settings
          </NavLink>
        </nav>
      </div>
    </header>
  );
}
