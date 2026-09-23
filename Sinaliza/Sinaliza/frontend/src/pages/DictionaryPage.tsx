import { useState, useEffect, useCallback } from "react";
import { Search } from "lucide-react";
import api, { type DictionaryEntry } from "../services/api";
import SignCard from "../components/SignCard";
import styles from "./DictionaryPage.module.css";

const PAGE_SIZE = 12;

/* Categorias padrão caso a API não retorne */
const DEFAULT_CATEGORIES = [
  "Saudações",
  "Família",
  "Tempo",
  "Emoções",
  "Educação",
];

const CATEGORY_ICONS: Record<string, string> = {
  Saudações: "👋",
  Família: "👨‍👩‍👧",
  Tempo: "⏰",
  Emoções: "😊",
  Educação: "📚",
};

export default function DictionaryPage() {
  const [entries, setEntries] = useState<DictionaryEntry[]>([]);
  const [categories, setCategories] = useState<string[]>(DEFAULT_CATEGORIES);
  const [activeCategory, setActiveCategory] = useState(DEFAULT_CATEGORIES[0]);
  const [searchTerm, setSearchTerm] = useState("");
  const [total, setTotal] = useState(0);
  const [skip, setSkip] = useState(0);
  const [loading, setLoading] = useState(false);

  const fetchEntries = useCallback(
    async (reset = false) => {
      setLoading(true);
      try {
        const offset = reset ? 0 : skip;
        const res = await api.dictionary({
          category: searchTerm ? undefined : activeCategory,
          search: searchTerm || undefined,
          skip: offset,
          limit: PAGE_SIZE,
        });
        if (reset) {
          setEntries(res.entries);
        } else {
          setEntries((prev) => [...prev, ...res.entries]);
        }
        setTotal(res.total);
        if (res.categories.length > 0) setCategories(res.categories);
        setSkip(offset + res.entries.length);
      } catch (err) {
        console.error("Erro ao buscar dicionário:", err);
      } finally {
        setLoading(false);
      }
    },
    [activeCategory, searchTerm, skip]
  );

  /* Reload quando muda categoria ou busca */
  useEffect(() => {
    setSkip(0);
    fetchEntries(true);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeCategory, searchTerm]);

  const categoryDesc: Record<string, string> = {
    Saudações: "Sinais essenciais para iniciar e terminar conversas de forma educada",
    Família: "Sinais para membros da família e relações pessoais",
    Tempo: "Sinais relacionados a tempo, datas e horários",
    Emoções: "Sinais para expressar sentimentos e emoções",
    Educação: "Sinais do ambiente escolar e acadêmico",
  };

  return (
    <div className={`container ${styles.page}`}>
      <h1 className={styles.title}>Dicionário de Libras</h1>

      {/* Search */}
      <div className={styles.searchWrapper}>
        <Search size={18} className={styles.searchIcon} />
        <input
          type="text"
          placeholder="Buscar um sinal..."
          className={styles.searchInput}
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
        />
      </div>

      <div className={styles.grid}>
        {/* Sidebar — Categorias */}
        <aside className={styles.sidebar}>
          <h3 className={styles.sideTitle}>Categorias</h3>
          <ul className={styles.catList}>
            {categories.map((cat) => (
              <li key={cat}>
                <button
                  className={`${styles.catBtn} ${
                    activeCategory === cat ? styles.catActive : ""
                  }`}
                  onClick={() => {
                    setSearchTerm("");
                    setActiveCategory(cat);
                  }}
                >
                  <span className={styles.catIcon}>
                    {CATEGORY_ICONS[cat] ?? "📁"}
                  </span>
                  {cat}
                </button>
              </li>
            ))}
          </ul>
        </aside>

        {/* Conteúdo principal */}
        <section className={styles.main}>
          <div className={styles.catHeader}>
            <h2 className={styles.catName}>
              <span>{CATEGORY_ICONS[activeCategory] ?? "📁"}</span>
              {searchTerm ? `Resultados: "${searchTerm}"` : activeCategory}
            </h2>
            <p className={styles.catDesc}>
              {searchTerm
                ? `${total} sinal(is) encontrado(s)`
                : categoryDesc[activeCategory] ?? ""}
            </p>
          </div>

          {/* Grid de cards */}
          <div className={styles.cards}>
            {entries.map((entry) => (
              <SignCard key={entry.sign_id} entry={entry} />
            ))}
          </div>

          {entries.length === 0 && !loading && (
            <p className={styles.empty}>Nenhum sinal encontrado.</p>
          )}

          {/* Load more */}
          {entries.length < total && (
            <div className={styles.loadMore}>
              <button
                className="btn btn-outline"
                onClick={() => fetchEntries(false)}
                disabled={loading}
              >
                {loading ? "Carregando..." : "Carregar mais sinais ▼"}
              </button>
            </div>
          )}
        </section>
      </div>
    </div>
  );
}
