"""Copia o projeto para a pasta local do repositório GitHub, sem o artigo, as apresentações e os rascunhos.
Uso (na pasta do projeto): python scripts_fase3/exporta_repositorio.py C:\\Users\\<usuario>\\Documents\\GitHub\\cpc-puc-rio-2026
A pasta de destino guarda o .git; o script só substitui os arquivos do projeto."""
import shutil, sys
from pathlib import Path

ORIGEM = Path(".").resolve()
DESTINO = Path(sys.argv[1]).resolve()
FORA_PASTAS = {"artigo", "__pycache__", ".ipynb_checkpoints", ".git"}
FORA_ARQUIVOS = {"experimentos_clustering_por_area.ipynb",
                 "PPT_atualizado_visao_geral.png", "prototipo_treino_teste_Civil.png"}
FORA_EXT = {".pptx", ".ppt", ".docx", ".doc", ".pdf", ".tmp"}


def entra(p: Path) -> bool:
    rel = p.relative_to(ORIGEM)
    if any(parte in FORA_PASTAS for parte in rel.parts):
        return False
    return not (p.name in FORA_ARQUIVOS or p.suffix.lower() in FORA_EXT or p.name.startswith("~$"))


DESTINO.mkdir(parents=True, exist_ok=True)
for item in DESTINO.iterdir():                     # limpa a cópia anterior, preservando o .git e os arquivos do repositório
    if item.name in {".git", "README.md", ".gitignore", "requirements.txt", "LICENSE"}:
        continue
    shutil.rmtree(item) if item.is_dir() else item.unlink()
n = mb = 0
for p in ORIGEM.rglob("*"):
    if p.is_file() and entra(p):
        alvo = DESTINO / p.relative_to(ORIGEM)
        alvo.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, alvo); n += 1; mb += p.stat().st_size / 2**20
print(f"{n} arquivos copiados ({mb:.0f} MB) para {DESTINO}")
