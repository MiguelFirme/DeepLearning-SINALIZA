import csv

from sinaliza.datasets.alfabeto_libras import descobrir_amostras as descobrir_imagens
from sinaliza.datasets.v_librasil import (
    descobrir_amostras as descobrir_videos,
    gravar_manifesto,
)


def test_vlibrasil_interpreta_rotulo_e_articulador_pelo_nome(tmp_path) -> None:
    diretorio = tmp_path / "videos"
    diretorio.mkdir()
    (diretorio / "Bom_dia_Articulador1.mp4").touch()
    (diretorio / "Bom_dia_Articulador2.mp4").touch()

    amostras = descobrir_videos(tmp_path)

    assert [amostra.rotulo for amostra in amostras] == ["Bom dia", "Bom dia"]
    assert [amostra.articulador_id for amostra in amostras] == [
        "Articulador1",
        "Articulador2",
    ]


def test_manifesto_vlibrasil_usa_campos_padronizados(tmp_path) -> None:
    video = tmp_path / "Obrigado_Articulador1.mp4"
    video.touch()
    destino = tmp_path / "v_librasil.csv"

    gravar_manifesto(descobrir_videos(tmp_path), destino)

    with destino.open("r", encoding="utf-8", newline="") as arquivo:
        linha = next(csv.DictReader(arquivo))
    assert linha["dataset"] == "v_librasil"
    assert linha["rotulo"] == "Obrigado"
    assert linha["articulador_id"] == "Articulador1"


def test_alfabeto_usa_pasta_da_imagem_como_rotulo(tmp_path) -> None:
    diretorio = tmp_path / "A"
    diretorio.mkdir()
    (diretorio / "amostra.jpg").touch()

    amostras = descobrir_imagens(tmp_path)

    assert len(amostras) == 1
    assert amostras[0].rotulo == "A"

