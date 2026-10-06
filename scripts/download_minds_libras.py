#!/usr/bin/env python3
"""Baixa MP4s públicos do MINDS-Libras com retomada e verificação de tamanho.

O índice de arquivos vem da API pública do Kaggle. Um download interrompido fica
como `.part`; só após confirmar o tamanho informado pela API vira `.mp4`.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.data.minds_libras import parse_video_name

SLUG = "j0aopsantos/minds-libras"
API = f"https://www.kaggle.com/api/v1/datasets/list/{SLUG}"


def curl_text(url: str) -> str:
    return subprocess.check_output(
        ["curl", "--location", "--fail", "--retry", "2", "--silent", "--show-error", url],
        text=True, timeout=90,
    )


def fetch_index() -> list[dict]:
    result: list[dict] = []
    token = ""
    while True:
        url = f"{API}?pageSize=200"
        if token:
            url += "&pageToken=" + quote(token, safe="")
        page = json.loads(curl_text(url))
        files = page.get("datasetFiles", [])
        result.extend({"name": f["name"], "bytes": int(f["totalBytes"])} for f in files)
        token = page.get("nextPageToken", "")
        if not token:
            break
    if not result or len({f["name"] for f in result}) != len(result):
        raise ValueError("Índice Kaggle vazio ou com nomes duplicados")
    for file in result:
        parse_video_name(file["name"])
    return result


def select_files(index: list[dict], labels: set[str] | None = None,
                 max_files: int | None = None) -> list[dict]:
    files = [file for file in index
             if labels is None or parse_video_name(file["name"])["label"] in labels]
    if labels is not None:
        found = {parse_video_name(file["name"])["label"] for file in files}
        if labels - found:
            raise ValueError(f"Classes ausentes do Kaggle: {sorted(labels - found)}")
    files.sort(key=lambda item: item["name"])
    return files[:max_files] if max_files is not None else files


def download_file(file: dict, output: Path) -> None:
    name = file["name"]
    parse_video_name(name)
    target = output / name
    partial = output / f"{name}.part"
    expected = int(file["bytes"])
    if target.exists():
        if target.stat().st_size == expected:
            return
        raise ValueError(f"Vídeo completo com tamanho incorreto: {target}")
    if partial.exists() and partial.stat().st_size > expected:
        raise ValueError(f"Arquivo parcial maior que o esperado: {partial}")
    url = f"https://www.kaggle.com/api/v1/datasets/download/{SLUG}/{quote(name)}"
    command = ["curl", "--location", "--fail", "--silent", "--show-error",
               "--retry", "3", "--retry-all-errors",
               "--connect-timeout", "30", "--continue-at", "-", "--output", str(partial), url]
    subprocess.run(command, check=True)
    actual = partial.stat().st_size
    if actual != expected:
        raise ValueError(f"Download incompleto de {name}: {actual}/{expected} bytes")
    partial.replace(target)


def pending_bytes(files: list[dict], output: Path) -> int:
    pending = 0
    for file in files:
        name, expected = file["name"], int(file["bytes"])
        target = output / name
        partial = output / f"{name}.part"
        if target.exists():
            if target.stat().st_size != expected:
                raise ValueError(f"Vídeo completo com tamanho incorreto: {target}")
            continue
        current = partial.stat().st_size if partial.exists() else 0
        if current > expected:
            raise ValueError(f"Arquivo parcial maior que o esperado: {partial}")
        pending += expected - current
    return pending


def main() -> None:
    parser = argparse.ArgumentParser(description="Baixar MINDS-Libras com retomada")
    parser.add_argument("--output", type=Path, default=ROOT / "data/raw/minds_libras")
    parser.add_argument("--index", type=Path, help="Índice JSON já obtido da API")
    parser.add_argument("--labels", nargs="+", help="Palavras exatas a baixar; padrão: todas")
    parser.add_argument("--max-files", type=int, help="Limite para teste de download")
    parser.add_argument("--jobs", type=int, default=1,
                        help="Downloads simultâneos de arquivos diferentes (padrão: 1)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.max_files is not None and args.max_files < 1:
        parser.error("--max-files precisa ser positivo")
    if args.jobs < 1:
        parser.error("--jobs precisa ser positivo")
    args.output.mkdir(parents=True, exist_ok=True)
    index_path = args.index or args.output / "kaggle_file_index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8-sig"))
    else:
        index = fetch_index()
        index_path.write_text(json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8")
    files = select_files(index, set(args.labels) if args.labels else None, args.max_files)
    pending = pending_bytes(files, args.output)
    print(f"Arquivos selecionados: {len(files)} | pendentes: {pending / 2**30:.2f} GiB")
    if args.dry_run:
        return
    free = shutil.disk_usage(args.output).free
    if pending + 2 * 2**30 > free:
        raise RuntimeError(f"Espaço insuficiente: {free / 2**30:.2f} GiB livres")
    state_path = args.output / "download_state.json"
    if args.jobs == 1:
        for position, file in enumerate(files, 1):
            print(f"[{position}/{len(files)}] {file['name']}", flush=True)
            download_file(file, args.output)
            state_path.write_text(json.dumps({"completed": position, "selected": len(files),
                                              "last_file": file["name"]}, indent=2), encoding="utf-8")
    else:
        completed = 0
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futures = {pool.submit(download_file, file, args.output): file for file in files}
            for future in as_completed(futures):
                future.result()
                completed += 1
                name = futures[future]["name"]
                print(f"[{completed}/{len(files)}] {name}", flush=True)
                state_path.write_text(json.dumps({"completed": completed,
                                                  "selected": len(files),
                                                  "last_file": name}, indent=2), encoding="utf-8")
    print("Download concluído e tamanhos verificados.")


if __name__ == "__main__":
    main()
