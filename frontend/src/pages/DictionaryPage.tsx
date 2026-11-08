import { useState, useEffect, useCallback, useRef } from "react";
import { Mic, MicOff, Search } from "lucide-react";
import api, { type DictionaryEntry } from "../services/api";
import SignCard from "../components/SignCard";
import { useSpeechRecognition } from "../hooks/useSpeechRecognition";
import styles from "./DictionaryPage.module.css";

const PAGE_SIZE = 12;

/* Categorias padrão caso a API não retorne */
const DEFAULT_CATEGORIES = [
  "Alfabeto",
  "Educação",
  "Emoções",
  "Família",
  "Números",
  "Saudações",
  "Tempo",
];

const CATEGORY_ICONS: Record<string, string> = {
  Saudações: "👋",
  Família: "👨‍👩‍👧",
  Tempo: "⏰",
  Emoções: "😊",
  Educação: "📚",
  Alfabeto: "🔤",
  Números: "🔢",
};

export default function DictionaryPage() {
  const [entries, setEntries] = useState<DictionaryEntry[]>([]);
  const [categories, setCategories] = useState<string[]>(DEFAULT_CATEGORIES);
  const [activeCategory, setActiveCategory] = useState(DEFAULT_CATEGORIES[0]);
  const [searchTerm, setSearchTerm] = useState("");
  const [total, setTotal] = useState(0);
  const [skip, setSkip] = useState(0);
  const [loading, setLoading] = useState(false);
  const [loadError, setLoadError] = useState(false);
  /* Marcador no fim da lista: quando ele aparece na tela, carrega mais sinais */
  const sentinelRef = useRef<HTMLDivElement>(null);
  /* Evita misturar categorias: cada vez que a lista recomeça (troca de categoria ou busca)
     o "epoch" muda, e qualquer resposta de uma lista antiga que chegar atrasada é descartada. */
  const epochRef = useRef(0);
  const busyRef = useRef(false); // há uma carga em andamento agora?

  /* Busca por voz: o texto reconhecido vira o termo de busca */
  const voice = useSpeechRecognition({
    onResult: (text) => setSearchTerm(text),
  });

  const fetchEntries = useCallback(
    async (reset = false) => {
      if (!reset && busyRef.current) return; // nunca duas cargas de "mais" ao mesmo tempo
      if (reset) epochRef.current += 1; // nova lista: invalida respostas antigas
      const epoch = epochRef.current;
      busyRef.current = true;
      setLoading(true);
      setLoadError(false);
      try {
        const offset = reset ? 0 : skip;
        const res = await api.dictionary({
          category: searchTerm ? undefined : activeCategory,
          search: searchTerm || undefined,
          skip: offset,
          limit: PAGE_SIZE,
        });
        if (epoch !== epochRef.current) return; // resposta de outra categoria/busca: ignora
        if (reset) {
          setEntries(res.entries);
        } else {
          setEntries((prev) => [...prev, ...res.entries]);
        }
        setTotal(res.total);
        if (res.categories.length > 0) {
          setCategories(res.categories);
          // se a categoria atual não existe (ex.: dicionário sem a pasta), vai para a primeira
          if (!res.categories.includes(activeCategory)) {
            setActiveCategory(res.categories[0]);
          }
        }
        setSkip(offset + res.entries.length);
      } catch (err) {
        if (epoch !== epochRef.current) return;
        console.error("Erro ao buscar dicionário:", err);
        setLoadError(true);
      } finally {
        if (epoch === epochRef.current) {
          busyRef.current = false;
          setLoading(false);
        }
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

  /* Rolagem infinita: ao chegar perto do fim da lista, carrega a próxima página.
     Não dispara enquanto já está carregando, nem fica repetindo se deu erro. */
  const hasMore = entries.length < total;
  // navegadores muito antigos não têm IntersectionObserver: nesse caso o botão volta a aparecer
  const canObserve = typeof IntersectionObserver !== "undefined";
  useEffect(() => {
    const el = sentinelRef.current;
    if (!canObserve || !el || !hasMore || loading || loadError) return;
    const observer = new IntersectionObserver(
      ([item]) => {
        if (item.isIntersecting) fetchEntries(false);
      },
      { rootMargin: "400px 0px" } // começa a carregar um pouco antes de chegar no fim
    );
    observer.observe(el);
    return () => observer.disconnect();
  }, [canObserve, hasMore, loading, loadError, fetchEntries]);

  const categoryDesc: Record<string, string> = {
    Saudações: "Sinais essenciais para iniciar e terminar conversas de forma educada",
    Família: "Sinais para membros da família e relações pessoais",
    Tempo: "Sinais relacionados a tempo, datas e horários",
    Emoções: "Sinais para expressar sentimentos e emoções",
    Educação: "Sinais do ambiente escolar e acadêmico",
    Alfabeto: "O alfabeto manual: uma letra por vez",
    Números: "Sinais dos números",
  };

  return (
    <div className={`container ${styles.page}`}>
      <h1 className={styles.title}>Dicionário de Libras</h1>

      {/* Search */}
      <div className={styles.searchWrapper}>
        <Search size={18} className={styles.searchIcon} />
        <input
          type="text"
          placeholder={voice.listening ? "Ouvindo... fale o sinal" : "Buscar um sinal..."}
          className={`${styles.searchInput} ${voice.supported ? styles.withMic : ""}`}
          value={voice.listening && voice.interim ? voice.interim : searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
        />
        {voice.supported && (
          <button
            type="button"
            className={`${styles.micBtn} ${voice.listening ? styles.micActive : ""}`}
            onClick={voice.toggle}
            aria-label={voice.listening ? "Parar de ouvir" : "Buscar por voz"}
            aria-pressed={voice.listening}
            title={voice.listening ? "Parar de ouvir" : "Buscar por voz"}
          >
            {voice.listening ? <MicOff size={18} /> : <Mic size={18} />}
          </button>
        )}
      </div>
      {voice.error && (
        <p className={styles.voiceError} role="alert">
          {voice.error}
        </p>
      )}
      <span className={styles.srOnly} aria-live="polite">
        {voice.listening ? "Ouvindo" : ""}
      </span>

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
            <div className={styles.catTitleRow}>
              <h2 className={styles.catName}>
                <span>{CATEGORY_ICONS[activeCategory] ?? "📁"}</span>
                {searchTerm ? `Resultados: "${searchTerm}"` : activeCategory}
              </h2>
              {!searchTerm && total > 0 && (
                <span className={styles.count}>
                  {total} {total === 1 ? "sinal" : "sinais"}
                </span>
              )}
            </div>
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

          {/* Fim da lista: carrega mais sozinho ao rolar */}
          {hasMore && (
            <div ref={sentinelRef} className={styles.loadMore}>
              {!loadError && (
                <p className={styles.progress}>
                  {loading
                    ? "Carregando mais sinais..."
                    : `Mostrando ${entries.length} de ${total} sinais`}
                </p>
              )}
              {(loadError || !canObserve) && (
                <button
                  className="btn btn-outline"
                  onClick={() => fetchEntries(false)}
                  disabled={loading}
                >
                  {loadError ? "Tentar novamente" : "Carregar mais sinais ▼"}
                </button>
              )}
            </div>
          )}
        </section>
      </div>
    </div>
  );
}