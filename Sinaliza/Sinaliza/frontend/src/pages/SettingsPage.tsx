import { useState } from "react";
import { Info, Headphones, Mail } from "lucide-react";
import styles from "./SettingsPage.module.css";

type Tab = "sobre" | "suporte";

export default function SettingsPage() {
  const [tab, setTab] = useState<Tab>("sobre");

  return (
    <div className={`container ${styles.page}`}>
      <h1 className={styles.title}>Configurações</h1>

      <div className={styles.grid}>
        {/* Sidebar */}
        <aside className={styles.sidebar}>
          <button
            className={`${styles.tabBtn} ${tab === "sobre" ? styles.tabActive : ""}`}
            onClick={() => setTab("sobre")}
          >
            <Info size={16} />
            Sobre
          </button>
          <button
            className={`${styles.tabBtn} ${tab === "suporte" ? styles.tabActive : ""}`}
            onClick={() => setTab("suporte")}
          >
            <Headphones size={16} />
            Suporte
          </button>
        </aside>

        {/* Content */}
        <section className={styles.content}>
          {tab === "sobre" && (
            <div className={`card ${styles.card}`}>
              <h2 className={styles.cardTitle}>Sobre o Sinaliza</h2>
              <p className={styles.text}>
                O Sinaliza é uma plataforma dedicada a quebrar barreiras de
                comunicação. Nossas principais funcionalidades incluem:
              </p>
              <ul className={styles.features}>
                <li>
                  <strong>Tradução em tempo real</strong> de Libras para Português.
                </li>
                <li>
                  <strong>Tecnologia de visão computacional</strong> de ponta para
                  reconhecimento preciso de sinais.
                </li>
                <li>
                  <strong>Acessibilidade para todos</strong>, promovendo a inclusão
                  e autonomia da comunidade surda.
                </li>
              </ul>
              <p className={styles.origin}>
                Projeto originado na EMEB Polo de Surdos Profª Maria de Lourdes
                Carneiro — Criciúma, SC.
              </p>
            </div>
          )}

          {tab === "suporte" && (
            <div className={`card ${styles.card}`}>
              <h2 className={styles.cardTitle}>Suporte e Contato</h2>
              <p className={styles.text}>
                Precisa de ajuda ou tem alguma dúvida? Entre em contato com a
                nossa equipe através dos e-mails abaixo.
              </p>

              <div className={styles.contactList}>
                <a href="mailto:contato@sinaliza.com" className={styles.contactItem}>
                  <div className={styles.contactIcon}>
                    <Mail size={18} />
                  </div>
                  <div>
                    <strong>Contato Geral</strong>
                    <span>contato@sinaliza.com</span>
                  </div>
                </a>

                <a href="mailto:suporte@sinaliza.com" className={styles.contactItem}>
                  <div className={styles.contactIcon}>
                    <Headphones size={18} />
                  </div>
                  <div>
                    <strong>Suporte Técnico</strong>
                    <span>suporte@sinaliza.com</span>
                  </div>
                </a>
              </div>
            </div>
          )}
        </section>
      </div>
    </div>
  );
}
