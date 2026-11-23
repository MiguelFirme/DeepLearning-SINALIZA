import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import ts from "typescript";

const source = readFileSync(new URL("../src/utils/predictionStabilizer.ts", import.meta.url), "utf8");
const js = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.ESNext } }).outputText;
const { PredictionStabilizer, stabilizationOptions } = await import(`data:text/javascript;base64,${Buffer.from(js).toString("base64")}`);
const prediction = (sign, status = "recognized") => ({
  sign, status, confidence: 0.8, top_k: [{ sign, class_id: 0, confidence: 0.8 }],
  is_cooldown: false, processing_time_ms: 10,
});

test("confirma duas previsões consecutivas, inclusive a primeira palavra", () => {
  const gate = new PredictionStabilizer();
  assert.equal(gate.update(prediction("Banheiro"), 0), null);
  assert.equal(gate.update(prediction("Escola"), 400), null);
  assert.equal(gate.update(prediction("Banheiro"), 800), null);
  const current = prediction("Banheiro");
  assert.equal(gate.update(current, 1200), current);
});

test("respeita 1500 ms desde a aceitação, sem adiar o relógio por palpites recusados", () => {
  const gate = new PredictionStabilizer();
  gate.update(prediction("Banheiro"), 0);
  assert.ok(gate.update(prediction("Banheiro"), 400));
  assert.equal(gate.update(prediction("Escola"), 800), null);
  assert.equal(gate.update(prediction("Escola"), 1200), null);
  assert.equal(gate.update(prediction("Escola"), 1899), null);
  assert.equal(gate.update(prediction("Escola"), 1900)?.sign, "Escola");
});

test("mantém a última palavra sem repetir no histórico mesmo após o intervalo", () => {
  const gate = new PredictionStabilizer();
  const history = [];
  for (const now of [0, 400, 800, 2000, 4000, 10000]) {
    const accepted = gate.update(prediction("Banheiro"), now);
    if (accepted) history.push(accepted.sign);
  }
  assert.deepEqual(history, ["Banheiro"]);
  gate.update(prediction("Escola"), 10400);
  history.push(gate.update(prediction("Escola"), 10800).sign);
  gate.update(prediction("Banheiro"), 12000);
  history.push(gate.update(prediction("Banheiro"), 12400).sign);
  assert.deepEqual(history, ["Banheiro", "Escola", "Banheiro"]);
});

test("rastreamento ruim, incerteza e nenhum sinal interrompem a confirmação", () => {
  for (const status of ["poor_tracking", "uncertain", "no_sign", "idle"]) {
    const gate = new PredictionStabilizer();
    gate.update(prediction("Banheiro"), 0);
    assert.equal(gate.update(prediction(null, status), 400), null);
    assert.equal(gate.update(prediction("Banheiro"), 800), null);
    assert.equal(gate.update(prediction("Banheiro"), 1200)?.sign, "Banheiro");
  }
});

test("não publica candidato antigo: troca exige nova confirmação após a espera", () => {
  const gate = new PredictionStabilizer();
  gate.update(prediction("Banheiro"), 0);
  gate.update(prediction("Banheiro"), 400);
  gate.update(prediction("Escola"), 800);
  gate.update(prediction("Escola"), 1200);
  assert.equal(gate.update(prediction("Querer"), 1900), null);
  assert.equal(gate.update(prediction("Querer"), 2300)?.sign, "Querer");
});

test("reinício de conexão descarta candidatos e permite reconhecer novamente", () => {
  const gate = new PredictionStabilizer();
  gate.update(prediction("Banheiro"), 0);
  gate.update(prediction("Banheiro"), 400);
  gate.update(prediction("Escola"), 800);
  gate.reset();
  assert.equal(gate.update(prediction("Escola"), 1000), null);
  assert.equal(gate.update(prediction("Escola"), 1400)?.sign, "Escola");
});

test("cooldown sem classificação não confirma nem interrompe uma sequência", () => {
  const gate = new PredictionStabilizer();
  gate.update(prediction("Banheiro"), 0);
  assert.equal(gate.update({ ...prediction(null), is_cooldown: true }, 100), null);
  assert.equal(gate.update(prediction("Banheiro"), 1200)?.sign, "Banheiro");
});

test("permite ajustar parâmetros; configuração inválida usa padrões", () => {
  assert.deepEqual(stabilizationOptions(), { confirmations: 2, minIntervalMs: 1500 });
  assert.deepEqual(stabilizationOptions("-1", "inválido"), stabilizationOptions());
  assert.deepEqual(stabilizationOptions("2.5", "-100"), stabilizationOptions());
  const gate = new PredictionStabilizer(stabilizationOptions("3", "2000"));
  assert.equal(gate.update(prediction("Banheiro"), 0), null);
  assert.equal(gate.update(prediction("Banheiro"), 400), null);
  assert.ok(gate.update(prediction("Banheiro"), 800));
  gate.update(prediction("Escola"), 1200);
  gate.update(prediction("Escola"), 1600);
  assert.equal(gate.update(prediction("Escola"), 2400), null);
  assert.equal(gate.update(prediction("Escola"), 2800)?.sign, "Escola");
});
