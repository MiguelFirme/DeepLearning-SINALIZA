# Vocabulário dos treinos de 120 épocas — 398 classes

Este é o `label_map.json` usado tanto pelo treino `bilstm_focus_signer_cv_allclasses_120epochs_seed42` quanto pelo treino `bilstm_augmented_signer_cv_allclasses_120epochs_seed42`, ambos na rodada com Articulador2 reservado. Os mapas são idênticos por SHA-256.

A coluna `origem` registra a regra de seleção, não uma anotação linguística validada manualmente. `treino` conta vídeos dos Articuladores 1 e 3; `teste` conta vídeos do Articulador 2. A classe Solicitar não tem vídeo do Articulador 2.

305 entradas têm origem `verbo` e 93, `essencial`. Termos como Diretoria, Sala, Cozinha, Comida, Remédio, Ajuda, Socorro e Dor não aparecem no CSV original e portanto não foram treinados. Abacaxi aparece no CSV, mas foi excluído.

| ID | Classe/palavra | Origem | Treino | Teste |
|---:|---|---|---:|---:|
| 0 | Abanar | verbo | 2 | 1 |
| 1 | Abandonar | verbo | 2 | 1 |
| 2 | Abençoar | verbo | 2 | 1 |
| 3 | Abrir a janela | verbo | 2 | 1 |
| 4 | Abrir a porta | verbo | 2 | 1 |
| 5 | Acabar | verbo | 2 | 1 |
| 6 | Aceitar | verbo | 2 | 1 |
| 7 | Acenar | verbo | 2 | 1 |
| 8 | Acidente | essencial | 2 | 1 |
| 9 | Acompanhar | verbo | 2 | 1 |
| 10 | Aconselhar | verbo | 2 | 1 |
| 11 | Acontecer | verbo | 2 | 1 |
| 12 | Acordar (alarme) | verbo | 2 | 1 |
| 13 | Acrescentar | verbo | 2 | 1 |
| 14 | Adaptar | verbo | 2 | 1 |
| 15 | Adiar | verbo | 2 | 1 |
| 16 | Admitir | verbo | 2 | 1 |
| 17 | Adquirir | verbo | 2 | 1 |
| 18 | Advertir | verbo | 2 | 1 |
| 19 | Afogar | verbo | 2 | 1 |
| 20 | Agarrar | verbo | 2 | 1 |
| 21 | Agir (fazer) | verbo | 2 | 1 |
| 22 | Agitar | verbo | 2 | 1 |
| 23 | Agora | essencial | 2 | 1 |
| 24 | Ajoelhar | verbo | 2 | 1 |
| 25 | Ajudar | verbo | 2 | 1 |
| 26 | Aluno do Segundo Ano do Ensino Médio | essencial | 2 | 1 |
| 27 | Amanhã | essencial | 2 | 1 |
| 28 | Amigo | essencial | 2 | 1 |
| 29 | Analisar | verbo | 2 | 1 |
| 30 | Antecipar | verbo | 2 | 1 |
| 31 | Anunciar | verbo | 2 | 1 |
| 32 | Apagar | verbo | 2 | 1 |
| 33 | Apagar o quadro | verbo | 2 | 1 |
| 34 | Apaixonar-se | verbo | 2 | 1 |
| 35 | Apontar para | verbo | 2 | 1 |
| 36 | Aprender | verbo | 2 | 1 |
| 37 | Aprovar | verbo | 2 | 1 |
| 38 | Argumentar | verbo | 2 | 1 |
| 39 | Aspirar | verbo | 2 | 1 |
| 40 | Assinatura | essencial | 2 | 1 |
| 41 | Assistir | verbo | 2 | 1 |
| 42 | Assumir | verbo | 2 | 1 |
| 43 | Atividade | essencial | 2 | 1 |
| 44 | Ato de remar | verbo | 2 | 1 |
| 45 | Aumentar | verbo | 2 | 1 |
| 46 | Avaliar | verbo | 2 | 1 |
| 47 | Aviso | essencial | 2 | 1 |
| 48 | Avó | essencial | 2 | 1 |
| 49 | Avô | essencial | 2 | 1 |
| 50 | Banheiro | essencial | 2 | 1 |
| 51 | Barbear | verbo | 2 | 1 |
| 52 | Bater papo | verbo | 2 | 1 |
| 53 | Bebida | essencial | 2 | 1 |
| 54 | Biblioteca | essencial | 2 | 1 |
| 55 | Bisbilhotar | verbo | 2 | 1 |
| 56 | Bloquear | verbo | 2 | 1 |
| 57 | Bocejar | verbo | 2 | 1 |
| 58 | Borracha | essencial | 2 | 1 |
| 59 | Brigar (dar porrada) | verbo | 2 | 1 |
| 60 | Brilhar | verbo | 2 | 1 |
| 61 | Cair | verbo | 2 | 1 |
| 62 | Calculadora | essencial | 2 | 1 |
| 63 | Cale-se | verbo | 2 | 1 |
| 64 | Cama | essencial | 2 | 1 |
| 65 | Campainha | essencial | 2 | 1 |
| 66 | Cancelar | verbo | 2 | 1 |
| 67 | Casa | essencial | 2 | 1 |
| 68 | Certificar (atestar) | verbo | 2 | 1 |
| 69 | Chave | essencial | 2 | 1 |
| 70 | Chegar | verbo | 2 | 1 |
| 71 | Chegar a algum lugar | verbo | 2 | 1 |
| 72 | Chuveiro | essencial | 2 | 1 |
| 73 | Classe | essencial | 2 | 1 |
| 74 | Colaborar | verbo | 2 | 1 |
| 75 | Colar grau | verbo | 2 | 1 |
| 76 | Coletar | verbo | 2 | 1 |
| 77 | Colocar | verbo | 2 | 1 |
| 78 | Colégio | essencial | 2 | 1 |
| 79 | Com licença | essencial | 2 | 1 |
| 80 | Com medo | essencial | 2 | 1 |
| 81 | Com sede | essencial | 2 | 1 |
| 82 | Comemorar (celebrar) | verbo | 2 | 1 |
| 83 | Comer | verbo | 2 | 1 |
| 84 | Comer com faca | verbo | 2 | 1 |
| 85 | Comer à noite | verbo | 2 | 1 |
| 86 | Começar | verbo | 2 | 1 |
| 87 | Como | essencial | 2 | 1 |
| 88 | Comparar | verbo | 2 | 1 |
| 89 | Comprar | verbo | 2 | 1 |
| 90 | Compreender | verbo | 2 | 1 |
| 91 | Computador | essencial | 2 | 1 |
| 92 | Comunicar | verbo | 2 | 1 |
| 93 | Concordar | verbo | 2 | 1 |
| 94 | Conduzir | verbo | 2 | 1 |
| 95 | Confrontar (enfrentar) | verbo | 2 | 1 |
| 96 | Congelar | verbo | 1 | 1 |
| 97 | Conseguindo | verbo | 2 | 1 |
| 98 | Consertar | verbo | 2 | 1 |
| 99 | Construir | verbo | 2 | 1 |
| 100 | Continuar | verbo | 2 | 1 |
| 101 | Conversar consigo mesmo | verbo | 2 | 1 |
| 102 | Convocar | verbo | 2 | 1 |
| 103 | Correr atrás | verbo | 2 | 1 |
| 104 | Cortar | verbo | 2 | 1 |
| 105 | Costurar | verbo | 2 | 1 |
| 106 | Crescer | verbo | 2 | 1 |
| 107 | Criança | essencial | 1 | 1 |
| 108 | Criar | verbo | 2 | 1 |
| 109 | Cuidado | essencial | 2 | 1 |
| 110 | Culpar | verbo | 2 | 1 |
| 111 | Cuspir | verbo | 2 | 1 |
| 112 | Dar | verbo | 2 | 1 |
| 113 | Decidir | verbo | 2 | 1 |
| 114 | Decolar | verbo | 2 | 1 |
| 115 | Decorar (ornamentar) | verbo | 2 | 1 |
| 116 | Defender (proteger) | verbo | 2 | 1 |
| 117 | Deitar-se | verbo | 2 | 1 |
| 118 | Deixar | verbo | 2 | 1 |
| 119 | Demitir | verbo | 2 | 1 |
| 120 | Depender | verbo | 2 | 1 |
| 121 | Deprimir | verbo | 2 | 1 |
| 122 | Derramar | verbo | 2 | 1 |
| 123 | Descobrir | verbo | 2 | 1 |
| 124 | Desconectar | verbo | 2 | 1 |
| 125 | Desculpa | essencial | 2 | 1 |
| 126 | Desenvolver | verbo | 2 | 1 |
| 127 | Desistir | verbo | 2 | 1 |
| 128 | Destruir | verbo | 2 | 1 |
| 129 | Deteriorar | verbo | 2 | 1 |
| 130 | Devemos | verbo | 2 | 1 |
| 131 | Dever de casa | essencial | 2 | 1 |
| 132 | Devolver | verbo | 2 | 1 |
| 133 | Dicionário | essencial | 2 | 1 |
| 134 | Diminuir | verbo | 2 | 1 |
| 135 | Direcionar | verbo | 2 | 1 |
| 136 | Dirigir | verbo | 2 | 1 |
| 137 | Discordar | verbo | 2 | 1 |
| 138 | Discriminar | verbo | 2 | 1 |
| 139 | Discutir (debater) | verbo | 2 | 1 |
| 140 | Disparar flecha | verbo | 2 | 1 |
| 141 | Disputar | verbo | 2 | 1 |
| 142 | Distribuir | verbo | 2 | 1 |
| 143 | Dividir | verbo | 2 | 1 |
| 144 | Dizer | verbo | 2 | 1 |
| 145 | Documento | essencial | 2 | 1 |
| 146 | Doente | essencial | 2 | 1 |
| 147 | Dor de cabeça | essencial | 2 | 1 |
| 148 | Dor de garganta | essencial | 2 | 1 |
| 149 | Dor de ouvido | essencial | 2 | 1 |
| 150 | Dormir | verbo | 2 | 1 |
| 151 | Dormitório | essencial | 2 | 1 |
| 152 | Duplicar | verbo | 2 | 1 |
| 153 | Economizar dinheiro | verbo | 2 | 1 |
| 154 | Embaraçar | verbo | 2 | 1 |
| 155 | Emergência | essencial | 2 | 1 |
| 156 | Emprestar algo | verbo | 2 | 1 |
| 157 | Encerrar | verbo | 2 | 1 |
| 158 | Encontrar | verbo | 2 | 1 |
| 159 | Encorajar | verbo | 2 | 1 |
| 160 | Endereço | essencial | 2 | 1 |
| 161 | Enfatizar | verbo | 2 | 1 |
| 162 | Enfermeira | essencial | 2 | 1 |
| 163 | Enganar | verbo | 2 | 1 |
| 164 | Ensaiar | verbo | 2 | 1 |
| 165 | Ensinar | verbo | 2 | 1 |
| 166 | Enterrar | verbo | 2 | 1 |
| 167 | Entrar | verbo | 2 | 1 |
| 168 | Entrar em algo | verbo | 2 | 1 |
| 169 | Enviando mensagem de texto | verbo | 2 | 1 |
| 170 | Enviar | verbo | 2 | 1 |
| 171 | Enviar e-mail | verbo | 2 | 1 |
| 172 | Enviar mensagem de texto | verbo | 2 | 1 |
| 173 | Escada | essencial | 2 | 1 |
| 174 | Escapar | verbo | 2 | 1 |
| 175 | Escola | essencial | 2 | 1 |
| 176 | Escolher | verbo | 2 | 1 |
| 177 | Escovar os dentes | verbo | 2 | 1 |
| 178 | Escrever | verbo | 2 | 1 |
| 179 | Escurecer | verbo | 2 | 1 |
| 180 | Esfregar | verbo | 2 | 1 |
| 181 | Esperar | verbo | 2 | 1 |
| 182 | Espere | verbo | 2 | 1 |
| 183 | Espirrar | verbo | 2 | 1 |
| 184 | Esquecer | verbo | 2 | 1 |
| 185 | Estimular | verbo | 2 | 1 |
| 186 | Estudo | essencial | 2 | 1 |
| 187 | Eu amo você | verbo | 2 | 1 |
| 188 | Eu vejo | verbo | 2 | 1 |
| 189 | Evitar | verbo | 2 | 1 |
| 190 | Exceder (passar, ir além) | verbo | 2 | 1 |
| 191 | Excluir | verbo | 2 | 1 |
| 192 | Executar | verbo | 2 | 1 |
| 193 | Exercite-se | verbo | 2 | 1 |
| 194 | Expandir | verbo | 2 | 1 |
| 195 | Experimentar | verbo | 2 | 1 |
| 196 | Explicar | verbo | 2 | 1 |
| 197 | Explodir | verbo | 2 | 1 |
| 198 | Expor | verbo | 2 | 1 |
| 199 | Expressar | verbo | 2 | 1 |
| 200 | Faculdade | essencial | 2 | 1 |
| 201 | Falhar | verbo | 2 | 1 |
| 202 | Faminto | essencial | 2 | 1 |
| 203 | Família | essencial | 2 | 1 |
| 204 | Fascinar | verbo | 2 | 1 |
| 205 | Faz | verbo | 2 | 1 |
| 206 | Fechar | verbo | 2 | 1 |
| 207 | Fechar a porta | verbo | 2 | 1 |
| 208 | Ferver | verbo | 2 | 1 |
| 209 | Ficar com vergonha | verbo | 2 | 1 |
| 210 | Ficha de registro | essencial | 2 | 1 |
| 211 | Filha | essencial | 2 | 1 |
| 212 | Filho | essencial | 2 | 1 |
| 213 | Fome | essencial | 2 | 1 |
| 214 | Formulário | essencial | 2 | 1 |
| 215 | Garagem | essencial | 2 | 1 |
| 216 | Gastar | verbo | 2 | 1 |
| 217 | Giz | essencial | 2 | 1 |
| 218 | Gostar | verbo | 2 | 1 |
| 219 | Gritar | verbo | 2 | 1 |
| 220 | Hospital | essencial | 2 | 1 |
| 221 | Identificar | verbo | 2 | 1 |
| 222 | Identificar documento | verbo | 2 | 1 |
| 223 | Ignorar | verbo | 2 | 1 |
| 224 | Imaginar | verbo | 2 | 1 |
| 225 | Implorar (pedir) | verbo | 2 | 1 |
| 226 | Impressionar | verbo | 2 | 1 |
| 227 | Imprimir | verbo | 2 | 1 |
| 228 | Instituição | essencial | 2 | 1 |
| 229 | Interpretar | verbo | 2 | 1 |
| 230 | Investir | verbo | 2 | 1 |
| 231 | Ir | verbo | 2 | 1 |
| 232 | Ir embora (partir) | verbo | 2 | 1 |
| 233 | Ir pra casa | verbo | 2 | 1 |
| 234 | Irmã | essencial | 2 | 1 |
| 235 | Janela | essencial | 2 | 1 |
| 236 | Jogar | verbo | 2 | 1 |
| 237 | Jogar basquetebol | verbo | 2 | 1 |
| 238 | Jogar futebol | verbo | 2 | 1 |
| 239 | Lançar | verbo | 2 | 1 |
| 240 | Lavar | verbo | 2 | 1 |
| 241 | Lavar as mãos | verbo | 2 | 1 |
| 242 | Lavar o rosto | verbo | 2 | 1 |
| 243 | Lavar prato | verbo | 2 | 1 |
| 244 | Lembrar (ter em mente) | verbo | 2 | 1 |
| 245 | Lembre-se | verbo | 2 | 1 |
| 246 | Levar | verbo | 2 | 1 |
| 247 | Ligar | verbo | 2 | 1 |
| 248 | Ligar algo | verbo | 2 | 1 |
| 249 | Livra-se | verbo | 2 | 1 |
| 250 | Livrar-se de | verbo | 2 | 1 |
| 251 | Livro | essencial | 2 | 1 |
| 252 | Lápis | essencial | 2 | 1 |
| 253 | Machucar | verbo | 2 | 1 |
| 254 | Manter (guardar) | verbo | 2 | 1 |
| 255 | Mastigar | verbo | 2 | 1 |
| 256 | Matar | verbo | 2 | 1 |
| 257 | Matemática | essencial | 2 | 1 |
| 258 | Medicamento | essencial | 2 | 1 |
| 259 | Melhorar um pouco mais | verbo | 2 | 1 |
| 260 | Memorizar | verbo | 2 | 1 |
| 261 | Menstruar | verbo | 2 | 1 |
| 262 | Misturar | verbo | 2 | 1 |
| 263 | Mochila | essencial | 2 | 1 |
| 264 | Morar junto | verbo | 2 | 1 |
| 265 | Morrer (falecer) | verbo | 2 | 1 |
| 266 | Motivar | verbo | 2 | 1 |
| 267 | Mudar | verbo | 2 | 1 |
| 268 | Mãe | essencial | 2 | 1 |
| 269 | Médico (doutor) | essencial | 2 | 1 |
| 270 | Nadar | verbo | 2 | 1 |
| 271 | Namorar | verbo | 2 | 1 |
| 272 | Nariz escorrendo | essencial | 2 | 1 |
| 273 | Nascer | verbo | 2 | 1 |
| 274 | Nascer do sol | verbo | 2 | 1 |
| 275 | Negligenciar | verbo | 2 | 1 |
| 276 | Nome | essencial | 2 | 1 |
| 277 | Nomear | verbo | 2 | 1 |
| 278 | Não | essencial | 2 | 1 |
| 279 | Não poder | verbo | 2 | 1 |
| 280 | Número | essencial | 2 | 1 |
| 281 | Obrigado | essencial | 2 | 1 |
| 282 | Observar | verbo | 2 | 1 |
| 283 | Oi | essencial | 2 | 1 |
| 284 | Onde | essencial | 2 | 1 |
| 285 | Orar | verbo | 2 | 1 |
| 286 | Orientar | verbo | 2 | 1 |
| 287 | Ouvir | verbo | 2 | 1 |
| 288 | Pagar | verbo | 2 | 1 |
| 289 | Pai | essencial | 2 | 1 |
| 290 | Papel | essencial | 2 | 1 |
| 291 | Parar | verbo | 2 | 1 |
| 292 | Parecer | verbo | 2 | 1 |
| 293 | Passar | verbo | 2 | 1 |
| 294 | Pedir | verbo | 2 | 1 |
| 295 | Pedir algo emprestado | verbo | 2 | 1 |
| 296 | Pegar | verbo | 2 | 1 |
| 297 | Perceber | verbo | 2 | 1 |
| 298 | Perder a competição | verbo | 2 | 1 |
| 299 | Perder peso | verbo | 2 | 1 |
| 300 | Perigo | essencial | 2 | 1 |
| 301 | Permitir | verbo | 2 | 1 |
| 302 | Pertencer | verbo | 2 | 1 |
| 303 | Pesquisar | verbo | 2 | 1 |
| 304 | Piscar os olhos | verbo | 2 | 1 |
| 305 | Podar árvore | verbo | 2 | 1 |
| 306 | Poder | verbo | 2 | 1 |
| 307 | Policial | essencial | 2 | 1 |
| 308 | Por favor | essencial | 2 | 1 |
| 309 | Predizer | verbo | 2 | 1 |
| 310 | Preencher | verbo | 2 | 1 |
| 311 | Pregar a palavra | verbo | 2 | 1 |
| 312 | Prestar atenção | verbo | 2 | 1 |
| 313 | Procurar por | verbo | 2 | 1 |
| 314 | Proibir (vetar) | verbo | 2 | 1 |
| 315 | Prosseguir | verbo | 2 | 1 |
| 316 | Provar | verbo | 2 | 1 |
| 317 | Pulsar | verbo | 2 | 1 |
| 318 | Pôr o anel | verbo | 2 | 1 |
| 319 | Qual | essencial | 2 | 1 |
| 320 | Quando | essencial | 2 | 1 |
| 321 | Quantos | essencial | 2 | 1 |
| 322 | Quarto de dormir | essencial | 2 | 1 |
| 323 | Que | essencial | 2 | 1 |
| 324 | Quebrar | verbo | 2 | 1 |
| 325 | Quem | essencial | 2 | 1 |
| 326 | Quer | verbo | 2 | 1 |
| 327 | Raspar | verbo | 2 | 1 |
| 328 | Reciclar | verbo | 2 | 1 |
| 329 | Recuperar | verbo | 2 | 1 |
| 330 | Recusar | verbo | 2 | 1 |
| 331 | Regular | verbo | 2 | 1 |
| 332 | Rejeitar | verbo | 2 | 1 |
| 333 | Remover | verbo | 2 | 1 |
| 334 | Respirar | verbo | 2 | 1 |
| 335 | Resumir | verbo | 2 | 1 |
| 336 | Rezar | verbo | 2 | 1 |
| 337 | Rir (gargalhar) | verbo | 2 | 1 |
| 338 | Roubar | verbo | 2 | 1 |
| 339 | Saber | verbo | 2 | 1 |
| 340 | Sacudir | verbo | 2 | 1 |
| 341 | Sair | verbo | 2 | 1 |
| 342 | Salvar | verbo | 2 | 1 |
| 343 | Sangue | essencial | 2 | 1 |
| 344 | Se afaste | verbo | 2 | 1 |
| 345 | Se casar | verbo | 2 | 1 |
| 346 | Se foi (já era) | verbo | 2 | 1 |
| 347 | Se gabar | verbo | 2 | 1 |
| 348 | Secretária | essencial | 2 | 1 |
| 349 | Secretário | essencial | 2 | 1 |
| 350 | Seguir | verbo | 2 | 1 |
| 351 | Sentar | verbo | 2 | 1 |
| 352 | Sentir | verbo | 2 | 1 |
| 353 | Sim | essencial | 2 | 1 |
| 354 | Simplificar | verbo | 2 | 1 |
| 355 | Sofrer | verbo | 2 | 1 |
| 356 | Soletrar | verbo | 2 | 1 |
| 357 | Solicitar | verbo | 2 | 0 |
| 358 | Soltar gases | verbo | 2 | 1 |
| 359 | Substituir | verbo | 2 | 1 |
| 360 | Subtrair | verbo | 2 | 1 |
| 361 | Sugerir | verbo | 2 | 1 |
| 362 | Sumir | verbo | 2 | 1 |
| 363 | Supervisionar | verbo | 2 | 1 |
| 364 | Surgir | verbo | 2 | 1 |
| 365 | Suspender (interrromper) | verbo | 2 | 1 |
| 366 | Telefone | essencial | 2 | 1 |
| 367 | Ter | verbo | 2 | 1 |
| 368 | Terminar | verbo | 2 | 1 |
| 369 | Teste | essencial | 2 | 1 |
| 370 | Tirar foto | verbo | 2 | 1 |
| 371 | Tirar proveito | verbo | 2 | 1 |
| 372 | Tocar o coração | verbo | 2 | 1 |
| 373 | Todos | essencial | 2 | 1 |
| 374 | Tomar conta de | verbo | 2 | 1 |
| 375 | Tomar notas | verbo | 2 | 1 |
| 376 | Tornar-se gordo | verbo | 2 | 1 |
| 377 | Tosse | essencial | 2 | 1 |
| 378 | Traduzir | verbo | 2 | 1 |
| 379 | Transbordar | verbo | 2 | 1 |
| 380 | Trapacear | verbo | 2 | 1 |
| 381 | Trazer | verbo | 2 | 1 |
| 382 | Urgente | essencial | 2 | 1 |
| 383 | Use língua de sinais | verbo | 2 | 1 |
| 384 | Vai | verbo | 2 | 1 |
| 385 | Vamos | verbo | 2 | 1 |
| 386 | Vencer | verbo | 2 | 1 |
| 387 | Vender | verbo | 2 | 1 |
| 388 | Venha | verbo | 2 | 1 |
| 389 | Ver | verbo | 2 | 1 |
| 390 | Verificar | verbo | 2 | 1 |
| 391 | Virar a página | verbo | 2 | 1 |
| 392 | Visualizar | verbo | 2 | 1 |
| 393 | Viver | verbo | 2 | 1 |
| 394 | Voar de avião | verbo | 2 | 1 |
| 395 | Vomitar | verbo | 2 | 1 |
| 396 | Xingar | verbo | 2 | 1 |
| 397 | Água | essencial | 2 | 1 |
