#!/usr/bin/env python
"""Prepara a mídia do dicionário a partir da pasta "Dicionário - SINALIZA".

Espelha a estrutura de pastas da origem (uma subpasta por categoria) em
backend/static/dictionary/, mantendo os mesmos nomes de pastas e de arquivos:

  - vídeos (.mp4 etc.): recodificados para um MP4 leve (480 px, 24 fps, H.264, sem
    áudio) + um frame de capa  <nome>.poster.jpg
  - imagens (.png, .jpg ...): copiadas como estão

Os originais têm 1 a 5 MB cada; os clipes saem com poucas centenas de KB, o que permite
tocar vários ao mesmo tempo na tela do dicionário.

Requisito: ffmpeg no PATH (Windows: winget install Gyan.FFmpeg, depois reabra o terminal).

Exemplos (rodar da raiz do projeto):
    python scripts/prepare_dictionary_videos.py --list  "CAMINHO/Dicionário - SINALIZA"
    python scripts/prepare_dictionary_videos.py          "CAMINHO/Dicionário - SINALIZA"
    python scripts/prepare_dictionary_videos.py --only Emoções --limit 3  "CAMINHO/..."   # teste rápido
    python scripts/prepare_dictionary_videos.py --force  "CAMINHO/..."   # refaz tudo
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".webm", ".avi", ".mkv"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".svg"}
POSTER_SUFFIX = ".poster.jpg"


def fold(text: str) -> str:
    ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return ascii_text.lower()


def run_ffmpeg(args: list[str]) -> None:
    result = subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", *args],
        capture_output=True, text=True, errors="replace",
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "ffmpeg falhou")


def convert_video(src: Path, out_dir: Path, width: int, crf: int) -> None:
    video_out = out_dir / f"{src.stem}.mp4"
    poster_out = out_dir / f"{src.stem}{POSTER_SUFFIX}"
    run_ffmpeg([
        "-i", str(src), "-an",
        "-vf", f"scale={width}:-2,fps=24",
        "-c:v", "libx264", "-preset", "slow", "-crf", str(crf),
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(video_out),
    ])
    run_ffmpeg(["-i", str(src), "-vf", f"scale={width}:-2", "-frames:v", "1", "-q:v", "4", str(poster_out)])


def is_up_to_date(src: Path, *outputs: Path) -> bool:
    return all(o.is_file() and o.stat().st_mtime >= src.stat().st_mtime for o in outputs)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("src", type=Path, help='pasta "Dicionário - SINALIZA" (com as subpastas de categorias)')
    parser.add_argument("--out", type=Path, default=ROOT / "backend" / "static" / "dictionary",
                        help="pasta de saída, a que o backend serve")
    parser.add_argument("--only", nargs="+", metavar="CATEGORIA", help="processar só estas categorias")
    parser.add_argument("--limit", type=int, help="processar no máximo N arquivos por categoria (para testar)")
    parser.add_argument("--width", type=int, default=480, help="largura dos clipes em px (padrão: 480)")
    parser.add_argument("--crf", type=int, default=28, help="qualidade H.264; menor = melhor e maior (padrão: 28)")
    parser.add_argument("--force", action="store_true", help="refaz arquivos que já estão atualizados")
    parser.add_argument("--list", action="store_true", help="só lista o que seria processado")
    args = parser.parse_args()

    if not args.src.is_dir():
        print(f"Pasta não encontrada: {args.src}")
        print("Dica: aponte para a pasta que contém Alfabeto, Educação, Emoções etc.")
        return 1

    categories = sorted((d for d in args.src.iterdir() if d.is_dir()), key=lambda d: fold(d.name))
    if args.only:
        wanted = {fold(c) for c in args.only}
        categories = [d for d in categories if fold(d.name) in wanted]
    if not categories:
        print("Nenhuma subpasta de categoria encontrada.")
        return 1

    strays = [f.name for f in args.src.iterdir() if f.is_file()]
    if strays:
        print(f"Aviso: {len(strays)} arquivo(s) soltos na raiz foram ignorados (só subpastas contam).")

    need_ffmpeg = not args.list and any(
        f.suffix.lower() in VIDEO_EXTS for d in categories for f in d.iterdir() if f.is_file()
    )
    if need_ffmpeg and shutil.which("ffmpeg") is None:
        print("ffmpeg não encontrado no PATH. Instale com: winget install Gyan.FFmpeg (e reabra o terminal).")
        return 1

    done = skipped = failed = 0
    for cat_dir in categories:
        files = sorted(
            (f for f in cat_dir.iterdir() if f.is_file() and f.suffix.lower() in VIDEO_EXTS | IMAGE_EXTS),
            key=lambda f: fold(f.name),
        )
        if args.limit:
            files = files[: args.limit]
        print(f"\n[{cat_dir.name}] {len(files)} arquivo(s)")
        if args.list:
            for f in files:
                print(f"  {f.name}  ({f.stat().st_size / 1024:.0f} KB)")
            continue

        out_dir = args.out / cat_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)
        for f in files:
            if f.suffix.lower() in IMAGE_EXTS:
                target = out_dir / f.name
                if not args.force and is_up_to_date(f, target):
                    skipped += 1
                    continue
                shutil.copy2(f, target)
                done += 1
                print(f"  copiado  {f.name}")
                continue

            video_out = out_dir / f"{f.stem}.mp4"
            poster_out = out_dir / f"{f.stem}{POSTER_SUFFIX}"
            if not args.force and is_up_to_date(f, video_out, poster_out):
                skipped += 1
                continue
            try:
                convert_video(f, out_dir, args.width, args.crf)
                done += 1
                print(f"  ok       {f.name}  ->  {video_out.stat().st_size / 1024:.0f} KB")
            except RuntimeError as exc:
                failed += 1
                print(f"  ERRO     {f.name}: {exc}")

    if not args.list:
        print(f"\nConcluído: {done} processado(s), {skipped} já atualizado(s), {failed} com erro.")
        print(f"Saída: {args.out}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
