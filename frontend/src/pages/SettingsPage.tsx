import { useState, type KeyboardEvent } from "react";
import {
  Info,
  Headphones,
  Mail,
  Languages,
  ScanEye,
  HeartHandshake,
  MapPin,
  ArrowUpRight,
  BookOpen,
  CircleHelp,
  ChevronDown,
  Sun,
  UserRound,
  Hand,
  Frame,
  Hourglass,
} from "lucide-react";
import styles from "./SettingsPage.module.css";
 
type Tab = "sobre" | "guia" | "ajuda" | "suporte";
 
const TABS: { id: Tab; label: string; icon: typeof Info }[] = [
  { id: "sobre", label: "Sobre", icon: Info },
  { id: "guia", label: "Guia", icon: BookOpen },
  { id: "ajuda", label: "Ajuda", icon: CircleHelp },
  { id: "suporte", label: "Suporte", icon: Headphones },
];
 
const FEATURES = [
  {
    icon: Languages,
    title: "Tradução em tempo real",
    text: "De Libras para Português.",
  },
  {
    icon: ScanEye,
    title: "Tecnologia de visão computacional",
    text: "De ponta, para reconhecimento preciso de sinais.",
  },
  {
    icon: HeartHandshake,
    title: "Acessibilidade para todos",
    text: "Promovendo a inclusão e autonomia da comunidade surda.",
  },
];
 
const CONTACTS = [
  {
    icon: Mail,
    title: "Contato Geral",
    email: "contato@sinaliza.com",
  },
  {
    icon: Headphones,
    title: "Suporte Técnico",
    email: "suporte@sinaliza.com",
  },
];
 
/**
 * Sinais que o tradutor reconhece (as 25 classes do modelo treinado).
 * Vem de "Modelos salvos/transformer_mixed25_full/data_split/label_map.json".
 * Se o modelo for retreinado com outros sinais, atualize esta lista.
 */
const RECOGNIZED_SIGNS = [
  "Acontecer", "Ajudar", "Aluno", "Amarelo", "Amigo", "Aprender", "Aproveitar",
  "Banco", "Banheiro", "Barulho", "Beber", "Brincar", "Cansado", "Certo",
  "Cinco", "Comer", "Conhecer", "Diretor", "Espelho", "Filho", "Medo", "Ruim",
  "Vacina", "Vontade", "Água",
].sort((a, b) => a.localeCompare(b, "pt-BR"));
 
const CAMERA_TIPS = [
  {
    icon: Sun,
    title: "Boa iluminação",
    text: "Prefira um ambiente bem iluminado, com a luz vindo de frente para você.",
  },
  {
    icon: UserRound,
    title: "Tronco e rosto visíveis",
    text: "O sistema usa o corpo e o rosto para traduzir. Mantenha os dois dentro da imagem.",
  },
  {
    icon: Hand,
    title: "Mãos dentro do quadro",
    text: "Faça o sinal à frente do corpo, sem tirar as mãos da imagem.",
  },
  {
    icon: Frame,
    title: "Fundo simples",
    text: "Evite pessoas ou objetos se movendo atrás de você.",
  },
  {
    icon: Hourglass,
    title: "Um sinal por vez",
    text: "Faça o sinal com calma e espere o resultado antes de começar o próximo.",
  },
];
 
/* Mesmas cores e nomes da legenda que aparece sobre a câmera */
const POINT_LEGEND = [
  { color: "#00e676", name: "Mão esquerda", text: "Pontos verdes sobre a mão esquerda." },
  { color: "#ff4081", name: "Mão direita", text: "Pontos rosa sobre a mão direita." },
  { color: "#38bdf8", name: "Corpo", text: "Pontos azuis nos ombros, braços e tronco." },
  { color: "#fbbf24", name: "Face", text: "Pontos amarelos em sobrancelhas, olhos, nariz e boca." },
];
 
const FAQ = [
  {
    q: "A câmera não abre",
    a: "Confira se o navegador tem permissão para usar a câmera: clique no cadeado ao lado do endereço do site e permita o acesso. Feche outros programas ou abas que estejam usando a câmera e recarregue a página.",
  },
  {
    q: "Aparece “Rastreamento insuficiente”",
    a: "O sistema não está enxergando bem o seu tronco e o seu rosto. Afaste-se um pouco da câmera, fique centralizado na imagem e melhore a iluminação até os pontos coloridos aparecerem.",
  },
  {
    q: "Fica em “Analisando sinal” e não traduz",
    a: "Isso quer dizer que o sistema ainda não tem confiança no sinal. Repita o sinal com calma, de forma bem visível e com as mãos dentro da imagem. Confira também se ele está na lista de sinais reconhecidos, na aba Guia.",
  },
  {
    q: "O meu sinal não é reconhecido",
    a: "O tradutor reconhece apenas os 25 sinais listados na aba Guia; os outros não são traduzidos. Já o Dicionário tem muitos sinais para você consultar e aprender.",
  },
  {
    q: "Aparece “Conectando” ou “Erro na tradução”",
    a: "O aplicativo não conseguiu falar com o serviço de tradução. Verifique a sua internet, recarregue a página e tente de novo. Se continuar, fale com o suporte na aba Suporte.",
  },
  {
    q: "A busca por voz do Dicionário não funciona",
    a: "Use o Google Chrome e permita o microfone quando o navegador pedir; é preciso estar conectado à internet. Alguns navegadores não suportam esse recurso (o Brave, por exemplo, e o Edge pode falhar). Se o botão de microfone nem aparece, o navegador não é compatível.",
  },
  {
    q: "Dá para ouvir o sinal traduzido?",
    a: "Sim. Quando um sinal é identificado, use o botão “Leitura de voz” ao lado do resultado para ouvir a palavra em português.",
  },
];
 
function PanelHero({
  icon: Icon,
  title,
  text,
}: {
  icon: typeof Info;
  title: string;
  text: string;
}) {
  return (
    <div className={styles.hero}>
      <div className={styles.heroIcon}>
        <Icon size={22} />
      </div>
      <div>
        <h2 className={styles.heroTitle}>{title}</h2>
        <p className={styles.heroText}>{text}</p>
      </div>
    </div>
  );
}
 
export default function SettingsPage() {
  const [tab, setTab] = useState<Tab>("sobre");
 
  /* Setas do teclado trocam de aba (padrão de acessibilidade para abas) */
  const handleKeyDown = (e: KeyboardEvent<HTMLElement>) => {
    const forward = e.key === "ArrowDown" || e.key === "ArrowRight";
    const backward = e.key === "ArrowUp" || e.key === "ArrowLeft";
    if (!forward && !backward) return;
    e.preventDefault();
    const current = TABS.findIndex((t) => t.id === tab);
    const next = (current + (forward ? 1 : -1) + TABS.length) % TABS.length;
    setTab(TABS[next].id);
    document.getElementById(`tab-${TABS[next].id}`)?.focus();
  };
 
  const panelProps = (id: Tab) => ({
    id: `panel-${id}`,
    role: "tabpanel" as const,
    "aria-labelledby": `tab-${id}`,
    className: styles.panel,
  });
 
  return (
    <div className={`container ${styles.page}`}>
      <header className={styles.header}>
        <h1 className={styles.title}>Configurações</h1>
        <p className={styles.subtitle}>
          Conheça o projeto, aprenda a usar e fale com a nossa equipe.
        </p>
      </header>
 
      <div className={styles.grid}>
        {/* Sidebar */}
        <aside
          className={styles.sidebar}
          role="tablist"
          aria-label="Seções de configurações"
          onKeyDown={handleKeyDown}
        >
          {TABS.map(({ id, label, icon: Icon }) => (
            <button
              key={id}
              id={`tab-${id}`}
              type="button"
              role="tab"
              aria-selected={tab === id}
              aria-controls={`panel-${id}`}
              tabIndex={tab === id ? 0 : -1}
              className={`${styles.tabBtn} ${tab === id ? styles.tabActive : ""}`}
              onClick={() => setTab(id)}
            >
              <Icon size={18} />
              {label}
            </button>
          ))}
        </aside>
 
        {/* Content */}
        <section className={styles.content}>
          {tab === "sobre" && (
            <div {...panelProps("sobre")}>
              <PanelHero
                icon={Info}
                title="Sobre o Sinaliza"
                text="O Sinaliza é uma plataforma dedicada a quebrar barreiras de comunicação. Nossas principais funcionalidades incluem:"
              />
 
              <ul className={styles.features}>
                {FEATURES.map(({ icon: Icon, title, text }) => (
                  <li key={title} className={`card ${styles.feature}`}>
                    <div className={styles.featureIcon}>
                      <Icon size={20} />
                    </div>
                    <strong>{title}</strong>
                    <span>{text}</span>
                  </li>
                ))}
              </ul>
 
              <div className={styles.origin}>
                <MapPin size={18} className={styles.originIcon} />
                <div>
                  <strong>Origem do projeto</strong>
                  <p>
                    Projeto originado na EMEB Polo de Surdos Profª Maria de
                    Lourdes Carneiro — Criciúma, SC.
                  </p>
                </div>
              </div>
            </div>
          )}
 
          {tab === "guia" && (
            <div {...panelProps("guia")}>
              <PanelHero
                icon={BookOpen}
                title="Guia de uso"
                text="Veja como conseguir uma boa tradução e quais sinais o Sinaliza reconhece."
              />
 
              <div className={`card ${styles.section}`}>
                <h3 className={styles.sectionTitle}>Dicas para a câmera</h3>
                <ul className={styles.tips}>
                  {CAMERA_TIPS.map(({ icon: Icon, title, text }) => (
                    <li key={title} className={styles.tip}>
                      <div className={styles.featureIcon}>
                        <Icon size={20} />
                      </div>
                      <div>
                        <strong>{title}</strong>
                        <span>{text}</span>
                      </div>
                    </li>
                  ))}
                </ul>
              </div>
 
              <div className={`card ${styles.section}`}>
                <h3 className={styles.sectionTitle}>O que são os pontos coloridos?</h3>
                <p className={styles.sectionText}>
                  Quando a câmera está ligada, o sistema marca o que enxerga de
                  você. Se os pontos aparecem, o rastreamento está funcionando.
                </p>
                <ul className={styles.legend}>
                  {POINT_LEGEND.map(({ color, name, text }) => (
                    <li key={name}>
                      <span
                        className={styles.legendDot}
                        style={{ background: color }}
                        aria-hidden="true"
                      />
                      <div>
                        <strong>{name}</strong>
                        <span>{text}</span>
                      </div>
                    </li>
                  ))}
                </ul>
              </div>
 
              <div className={`card ${styles.section}`}>
                <div className={styles.sectionHead}>
                  <h3 className={styles.sectionTitle}>Sinais reconhecidos</h3>
                  <span className={styles.count}>{RECOGNIZED_SIGNS.length} sinais</span>
                </div>
                <p className={styles.sectionText}>
                  O tradutor foi treinado para reconhecer estes sinais. O
                  Dicionário tem muitos outros para consultar e aprender.
                </p>
                <ul className={styles.chips}>
                  {RECOGNIZED_SIGNS.map((sign) => (
                    <li key={sign} className={styles.chip}>
                      {sign}
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          )}
 
          {tab === "ajuda" && (
            <div {...panelProps("ajuda")}>
              <PanelHero
                icon={CircleHelp}
                title="Perguntas frequentes"
                text="Toque em uma pergunta para ver a resposta. Se não encontrar o que procura, fale com o suporte."
              />
 
              <div className={styles.faq}>
                {FAQ.map(({ q, a }) => (
                  <details key={q} className={`card ${styles.faqItem}`}>
                    <summary className={styles.faqQuestion}>
                      <span>{q}</span>
                      <ChevronDown size={18} className={styles.faqChevron} />
                    </summary>
                    <p className={styles.faqAnswer}>{a}</p>
                  </details>
                ))}
              </div>
            </div>
          )}
 
          {tab === "suporte" && (
            <div {...panelProps("suporte")}>
              <PanelHero
                icon={Headphones}
                title="Suporte e Contato"
                text="Precisa de ajuda ou tem alguma dúvida? Entre em contato com a nossa equipe através dos e-mails abaixo."
              />
 
              <div className={styles.contactList}>
                {CONTACTS.map(({ icon: Icon, title, email }) => (
                  <a
                    key={email}
                    href={`mailto:${email}`}
                    className={`card ${styles.contactItem}`}
                  >
                    <div className={styles.contactIcon}>
                      <Icon size={20} />
                    </div>
                    <div className={styles.contactInfo}>
                      <strong>{title}</strong>
                      <span>{email}</span>
                    </div>
                    <ArrowUpRight size={18} className={styles.contactArrow} />
                  </a>
                ))}
              </div>
            </div>
          )}
        </section>
      </div>
    </div>
  );
}