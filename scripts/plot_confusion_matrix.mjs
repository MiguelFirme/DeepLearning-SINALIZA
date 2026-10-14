#!/usr/bin/env node
// Visualiza uma matriz de confusao ja registrada por audit_checkpoint_val.py.
import { readFileSync, mkdirSync, writeFileSync, existsSync } from 'node:fs';
import { resolve, dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const defaults = {
  metrics: join(root, 'Modelos salvos/transformer_minds_20_classes_pose_face/records/val_metrics.json'),
  output: join(root, 'reports/transformer_minds_20_classes_pose_face/validation'),
};

function option(flag, fallback) {
  const index = process.argv.indexOf(flag);
  if (index < 0) return fallback;
  if (!process.argv[index + 1]) throw new Error(`Falta valor para ${flag}`);
  return resolve(process.argv[index + 1]);
}

const metricsPath = option('--metrics', defaults.metrics);
const outputDir = option('--output-dir', defaults.output);
const metrics = JSON.parse(readFileSync(metricsPath, 'utf8'));
const entries = Object.entries(metrics.classes).sort((a, b) => a[1] - b[1]);
const names = entries.map(([name], index) => {
  if (entries[index][1] !== index) throw new Error('Indices de classes devem ser consecutivos a partir de zero');
  return name;
});
const matrix = metrics.confusion_matrix;
if (!Array.isArray(matrix) || matrix.length !== names.length ||
    matrix.some(row => !Array.isArray(row) || row.length !== names.length ||
      row.some(value => !Number.isInteger(value) || value < 0))) {
  throw new Error('Matriz de confusao invalida ou incompatível com as classes');
}

const totals = matrix.map(row => row.reduce((sum, value) => sum + value, 0));
const splitLabel = metrics.split === 'val' ? 'Validação' : metrics.split === 'test' ? 'Teste' : metrics.split;
const titleIndex = process.argv.indexOf('--title');
const title = titleIndex < 0
  ? `Transformer MINDS-Libras — ${names.length} classes — ${splitLabel}`
  : process.argv[titleIndex + 1];
if (!title) throw new Error('Falta valor para --title');
const count = totals.reduce((sum, value) => sum + value, 0);
const correct = matrix.reduce((sum, row, index) => sum + row[index], 0);
if (count !== metrics.total_samples || Math.abs(correct / count - metrics.top1) > 1e-8) {
  throw new Error('Soma ou acuracia da matriz nao coincide com o JSON original');
}
// Quando existe o CSV da mesma auditoria, recontar cada video independentemente.
const predictionsPath = metricsPath.replace(/_metrics\.json$/, '_predictions.csv');
if (predictionsPath !== metricsPath && existsSync(predictionsPath)) {
  const lines = readFileSync(predictionsPath, 'utf8').trim().split(/\r?\n/);
  if (lines.shift() !== 'path,true,predicted,confidence') {
    throw new Error('Cabecalho inesperado no CSV de previsoes');
  }
  const fromVideos = names.map(() => names.map(() => 0));
  for (const line of lines) {
    const parts = line.split(',');
    if (parts.length !== 4 || !(parts[1] in metrics.classes) || !(parts[2] in metrics.classes)) {
      throw new Error(`Previsao CSV invalida: ${line}`);
    }
    fromVideos[metrics.classes[parts[1]]][metrics.classes[parts[2]]] += 1;
  }
  if (JSON.stringify(fromVideos) !== JSON.stringify(matrix)) {
    throw new Error('Matriz do JSON difere das previsoes por video');
  }
  console.log(`Conferencia por video: ${lines.length} previsoes coincidem com a matriz.`);
}
const stats = names.map((name, index) => ({
  name, total: totals[index], correct: matrix[index][index],
  errors: totals[index] - matrix[index][index],
  accuracy: totals[index] ? matrix[index][index] / totals[index] : null,
})).sort((a, b) => b.errors - a.errors || a.name.localeCompare(b.name));
const pairs = matrix.flatMap((row, actual) => row.map((value, predicted) => ({
  actual: names[actual], predicted: names[predicted], count: value,
})).filter(item => item.count > 0 && item.actual !== item.predicted))
  .sort((a, b) => b.count - a.count || a.actual.localeCompare(b.actual) ||
    a.predicted.localeCompare(b.predicted));

const escapeXml = value => String(value).replace(/[&<>"']/g, char => ({
  '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&apos;',
})[char]);
const csv = fields => fields.map(value => `"${String(value).replaceAll('"', '""')}"`).join(',');
const max = Math.max(...matrix.flat(), 1);
const cell = names.length <= 10 ? 68 : 34;
const left = 180;
const top = 190;
const size = cell * names.length;
const width = left + size + 95;
const height = top + size + 80;
const color = value => {
  const ratio = value / max;
  return `rgb(${Math.round(248 - 210 * ratio)},${Math.round(250 - 170 * ratio)},${Math.round(255 - 55 * ratio)})`;
};
const svg = [
  `<svg xmlns="http://www.w3.org/2000/svg" width="${width * 2}" height="${height * 2}" viewBox="0 0 ${width} ${height}">`,
  '<rect width="100%" height="100%" fill="white"/>',
  `<text x="${left}" y="35" font-size="21" font-family="Arial" font-weight="bold">${escapeXml(title)}</text>`,
  `<text x="${left}" y="60" font-size="13" font-family="Arial">Matriz de confusão | ${count} vídeos; ${correct} acertos (${(100 * correct / count).toFixed(1)}%)</text>`,
  `<text x="${left + size / 2}" y="${height - 15}" text-anchor="middle" font-size="16" font-family="Arial">Classe prevista</text>`,
  `<text transform="translate(25 ${top + size / 2}) rotate(-90)" text-anchor="middle" font-size="16" font-family="Arial">Classe real</text>`,
];
names.forEach((name, index) => {
  const position = index * cell + cell / 2;
  svg.push(`<text x="${left - 10}" y="${top + position + 5}" text-anchor="end" font-size="12" font-family="Arial">${escapeXml(name)}</text>`);
  svg.push(`<text transform="translate(${left + position + 4} ${top - 10}) rotate(-55)" text-anchor="start" font-size="12" font-family="Arial">${escapeXml(name)}</text>`);
  matrix[index].forEach((value, predicted) => {
    const x = left + predicted * cell;
    const y = top + index * cell;
    svg.push(`<rect x="${x}" y="${y}" width="${cell}" height="${cell}" fill="${color(value)}" stroke="#d8dee8"><title>${escapeXml(names[index])} → ${escapeXml(names[predicted])}: ${value}</title></rect>`);
    if (value) svg.push(`<text x="${x + cell / 2}" y="${y + cell / 2 + 5}" text-anchor="middle" font-size="13" font-family="Arial" fill="${value / max > 0.55 ? 'white' : '#172033'}">${value}</text>`);
  });
});
svg.push('</svg>');

mkdirSync(outputDir, { recursive: true });
writeFileSync(join(outputDir, 'confusion_matrix.svg'), svg.join('\n') + '\n');
writeFileSync(join(outputDir, 'confusion_matrix.csv'), [
  csv(['Real \\ Predito', ...names]),
  ...matrix.map((row, index) => csv([names[index], ...row])),
].join('\n') + '\n');
writeFileSync(join(outputDir, 'per_class.csv'), [
  csv(['classe', 'total', 'acertos', 'erros', 'acuracia']),
  ...stats.map(item => csv([item.name, item.total, item.correct, item.errors,
    item.accuracy === null ? '' : item.accuracy.toFixed(4)])),
].join('\n') + '\n');
const report = [
  `# ${title}`, '',
  `Fonte: ${metricsPath}`, '',
  `Total: **${count} videos**, **${correct} acertos** (${(100 * correct / count).toFixed(1)}%).`,
  'Linhas da matriz = classe real; colunas = classe prevista.', '',
  '## Erros por classe', '',
  '| Classe real | Acertos | Erros | Acuracia |', '|---|---:|---:|---:|',
  ...stats.map(item => `| ${item.name} | ${item.correct}/${item.total} | ${item.errors} | ${item.accuracy === null ? '—' : `${(item.accuracy * 100).toFixed(0)}%`} |`),
  '', '## Pares de confusao mais frequentes', '',
  '| Real | Previsto | Videos |', '|---|---|---:|',
  ...pairs.slice(0, 15).map(item => `| ${item.actual} | ${item.predicted} | ${item.count} |`),
  '', 'Cada classe tem cinco vídeos neste split; uma diferença de um acerto equivale a 20 pontos percentuais. As métricas descrevem este split e este checkpoint.', '',
];
writeFileSync(join(outputDir, 'report.md'), report.join('\n'));
console.log(`Matriz: ${join(outputDir, 'confusion_matrix.svg')}`);
console.log(`Resumo: ${join(outputDir, 'report.md')}`);
console.log(`${correct}/${count} acertos; ${stats.filter(item => item.correct === 0).length} classes com zero acertos.`);
