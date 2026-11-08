#!/usr/bin/env python3
"""Build one valid, Drive-backed Colab notebook per mapped augmentation technique."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


TECHNIQUES = {
    "selective-polarity": ("Polaridad selectiva", "selective-polarity"),
    "constraint-perturbation": ("Perturbación de restricciones", "constraint"),
    "objective-perturbation": ("Perturbación del objetivo", "objective"),
    "synthetic-conditioned": ("Generación sintética condicionada", "synthetic"),
}


def code_cell(source: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": source.splitlines(keepends=True)}


def markdown_cell(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {},
            "source": source.splitlines(keepends=True)}


def notebook(technique: str, label: str, kind: str) -> dict:
    # Each technique notebook launches one selected problem row. Four distinct
    # technique notebooks can therefore run concurrently in four Colab runtimes.
    setup = r'''from google.colab import drive
drive.mount('/content/drive', force_remount=True)

from pathlib import Path
import json, os, shutil, subprocess, sys

DRIVE_ROOT = Path('/content/drive/MyDrive/PBO-Tesis')
REPO = Path('/content/backbone-pbo')
SOURCE_DIR = DRIVE_ROOT / 'runs/colab-source'
if not (SOURCE_DIR / 'pyproject.toml').is_file():
    raise FileNotFoundError(f'Código fuente en bruto no encontrado: {SOURCE_DIR}')
shutil.copytree(SOURCE_DIR, REPO, dirs_exist_ok=True)

# Our repository owns augmentation, orchestration, and audit logic. The current
# adapter intentionally uses the pinned BackPaS GTR implementation as its trainer.
# MVC labels are already present, so GuroBack is not needed in this notebook.
BACKPAS = REPO / 'external' / 'backpas'
if not (BACKPAS / 'src/3_train.py').is_file():
    BACKPAS_URL = 'https://github.com/bryan-alvarado-ulloa/backpas.git'
    BACKPAS_COMMIT = 'b5c633579fa73d673f3c4e65e177c89bcddea3d7'
    BACKPAS.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(['git', 'clone', BACKPAS_URL, str(BACKPAS)], check=True)
    subprocess.run(['git', '-C', str(BACKPAS), 'checkout', '--detach', BACKPAS_COMMIT], check=True)

DATA = REPO / 'data/backpas'
DATA.mkdir(parents=True, exist_ok=True)
DRIVE_DATA = DRIVE_ROOT / 'datasets'
PROBLEM = input('Benchmark para este runtime (MVC/MIS/CA): ').strip().upper()
if PROBLEM not in {'MVC', 'MIS', 'CA'}:
    raise ValueError('Elige MVC, MIS o CA')
raw_source = DRIVE_DATA / PROBLEM
if not (raw_source / 'instance').is_dir() or not (raw_source / 'backbone').is_dir():
    raise FileNotFoundError(
        f'Faltan carpetas de datos en bruto instance/ y backbone/: {raw_source}')
dest = DATA / PROBLEM
if dest.exists():
    shutil.rmtree(dest)
shutil.copytree(raw_source, dest)
if not (dest / 'partitions.json').is_file():
    # Rebuild a missing split manifest from the imported instance names and
    # their completed labels. Preserve train/valid/test membership by prefix.
    instances = {path.name for path in (dest / 'instance').iterdir() if path.is_file()}
    partitions = {'train': [], 'validation': [], 'test': []}
    for label in sorted((dest / 'backbone').glob('*.backbone')):
        name = label.name[:-len('.backbone')]
        if name not in instances:
            continue
        try:
            if label.read_text(encoding='utf-8').splitlines()[-1].strip() != 'b 0':
                continue
        except (OSError, IndexError):
            continue
        parts = name.split('_', 2) if name.startswith('backpas_') else name.split('_', 1)
        split = parts[1] if name.startswith('backpas_') and len(parts) > 1 else parts[0]
        if split in {'train', 'valid', 'test'}:
            partitions['validation' if split == 'valid' else split].append(name)
    assigned = sum(map(len, partitions.values()))
    if assigned != len(instances):
        raise RuntimeError(f'Particiones inferidas incompletas: {assigned}/{len(instances)} instancias')
    (dest / 'partitions.json').write_text(
        json.dumps({'partitions': partitions}, indent=2) + '\n', encoding='utf-8')
    print('Particiones reconstruidas desde nombres y etiquetas:',
          {name: len(files) for name, files in partitions.items()})
if not (dest / 'source.json').is_file():
    print('Aviso: source.json no está en Drive; la auditoría lo marcará como procedencia incompleta.')
instance_count = sum(1 for path in (dest / 'instance').iterdir() if path.is_file())
instance_names = {path.name for path in (dest / 'instance').iterdir() if path.is_file()}
backbone_files = {path.name for path in (dest / 'backbone').glob('*.backbone') if path.is_file()}
backbone_count = sum(f'{name}.backbone' in backbone_files for name in instance_names)
extra_labels = backbone_files - {f'{name}.backbone' for name in instance_names}
if extra_labels:
    print(f'Ignorando {len(extra_labels)} etiquetas sin instancia correspondiente.')
if (instance_count, backbone_count) != (900, 900) and PROBLEM == 'MVC':
    raise RuntimeError(f'MVC en bruto incompleto: {instance_count} instancias, '
                       f'{backbone_count} etiquetas')
print(f'{PROBLEM} en bruto: {instance_count} instancias, {backbone_count} etiquetas')

subprocess.run([sys.executable, '-m', 'pip', 'install', '-e', '.[ml,backpas,data]'],
               cwd=REPO, check=True)

# The authoritative experiment matrix is generated from the repository code.
subprocess.run([sys.executable, 'scripts/materialize_pending_augmentations.py',
                '--repo', str(REPO)], cwd=REPO, check=True)
'''
    setup += f"\nTECHNIQUE = {technique!r}\nAUGMENTATION_KIND = {kind!r}\n"
    setup += r'''
MATRIX = REPO / 'configs/pending_augmentation_jobs.json'
matrix = json.loads(MATRIX.read_text(encoding='utf-8'))
def job_state(job):
    problem, technique, seed = job.split('/')
    work = DRIVE_ROOT / 'runs' / problem / technique / seed
    if any((work / marker).is_file() for marker in ('metrics.json', 'experiment-status.json')):
        return 'completed'
    status_path = work / 'status.json'
    if status_path.is_file():
        state = json.loads(status_path.read_text(encoding='utf-8')).get('state')
        if state in {'generating', 'preparing', 'training', 'running'}:
            return state
    return 'available'

ALL_JOBS = {row['problem']: f"{row['problem']}/{row['technique']}/seed-{row['seed']}"
            for row in matrix['jobs'] if row['technique'] == TECHNIQUE}
STATES = {problem: job_state(job) for problem, job in ALL_JOBS.items()}
TECHNIQUE_ORDER = ['selective-polarity', 'constraint-perturbation',
                   'objective-perturbation', 'synthetic-conditioned']
TECHNIQUE_INDEX = TECHNIQUE_ORDER.index(TECHNIQUE)
PROBLEM_ORDER = ['MVC', 'MIS', 'CA']
PREFERRED = PROBLEM_ORDER[TECHNIQUE_INDEX % len(PROBLEM_ORDER)]
print('Técnica:', TECHNIQUE)
print('Estado actual de la matriz para esta técnica:')
for problem, state in STATES.items():
    print(f'  {problem}/{TECHNIQUE}/seed-0: {state}')
AVAILABLE = [problem for problem, state in STATES.items() if state == 'available']
if not AVAILABLE:
    raise RuntimeError('No hay trabajos disponibles para esta técnica; los existentes están activos o completados.')
print('Problemas disponibles:', ', '.join(AVAILABLE))
if PREFERRED in AVAILABLE:
    print(f'Asignación sugerida para mantener variedad entre técnicas: {PREFERRED}')
if PROBLEM not in AVAILABLE:
    raise ValueError(f"{PROBLEM!r} no está disponible. Elige uno de: {', '.join(AVAILABLE)}")
JOBS = [ALL_JOBS[PROBLEM]]
print('Trabajo elegido:', JOBS[0])
print('Checkpoint:', DRIVE_ROOT / 'runs' / JOBS[0])
if TECHNIQUE == 'objective-perturbation':
    cached = DRIVE_ROOT / 'runs/generation-cache/CA-objective-perturbation.partial'
    local = REPO / 'data/backpas/.CA-objective-perturbation-v1.partial'
    if cached.is_dir() and not local.exists():
        shutil.copytree(cached, local)
        print('Restaurada generación parcial de CA desde Drive.')
'''
    run = r'''# Esta ventana ejecuta un único trabajo de su técnica; las otras técnicas
# corren en sus propios notebooks y runtimes independientes.
import json, torch

# Keep the local baseline's BackPaS architecture and training hyperparameters.
matrix_path = REPO / 'configs/pending_augmentation_jobs.json'
settings = json.loads(matrix_path.read_text(encoding='utf-8'))
settings['defaults'].update({
    'epochs': 65,
    'threads': 4,
    'batch_size': 64,
    'batch_accumulation_size': 64,
    'cuda': True,
})
matrix_path.write_text(json.dumps(settings, indent=2) + '\n', encoding='utf-8')
# Colab's CUDA scatter-add for BackPaS macro-accuracy has no strict deterministic
# kernel. Keep training deterministic where supported and allow that metric to warn.
runner = REPO / 'scripts/run_backpas_seeded.py'
runner_source = runner.read_text(encoding='utf-8')
runner_source = runner_source.replace(
    'torch.use_deterministic_algorithms(True)',
    'torch.use_deterministic_algorithms(True, warn_only=True)',
)
if 'CUBLAS_WORKSPACE_CONFIG' not in runner_source:
    runner_source = runner_source.replace(
        'os.environ["MKL_NUM_THREADS"] = str(args.threads)',
        'os.environ["MKL_NUM_THREADS"] = str(args.threads)\n'
        '    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")',
    )
runner.write_text(runner_source, encoding='utf-8')
print('BackPaS GTR, 8 capas, 65 épocas, lr=0.001, batch=64, acumulación=64, seed=0.')

for JOB in JOBS:
    command = [sys.executable, 'scripts/cloud_run.py', '--repo', str(REPO),
               '--matrix', 'configs/pending_augmentation_jobs.json',
               '--job', JOB, '--drive-root', str(DRIVE_ROOT)]
    if not torch.cuda.is_available():
        command.append('--cpu')
    print(f"\n=== {JOB} ({'GPU' if torch.cuda.is_available() else 'CPU'}) ===", flush=True)
    subprocess.run(command, cwd=REPO, check=True)
print(f"\nCompletado(s): {', '.join(JOBS)}")
'''
    return {
        "cells": [
            markdown_cell(f"# PBO — {label}\n\nSesión independiente para **{technique}**. "
                          "Permite elegir un trabajo de la matriz (MVC, MIS o CA) para "
                          "este runtime. Ejecuta un único trabajo de esta técnica; "
                          "usa un notebook y runtime independiente por cada técnica "
                          "para entrenarlas simultáneamente. "
                          "La matriz del repositorio excluye baseline, polarity y permutation "
                          "ya entrenados; cada trabajo conserva su checkpoint en Drive."),
            code_cell(setup),
            code_cell(run),
        ],
        "metadata": {
            "accelerator": "GPU",
            "colab": {"name": f"PBO — {label}", "provenance": [], "gpuType": "T4"},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, default=Path('notebooks/colab-techniques'))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for slug, (label, kind) in TECHNIQUES.items():
        path = args.output_dir / f'PBO-{slug}.ipynb'
        path.write_text(json.dumps(notebook(slug, label, kind), ensure_ascii=False,
                                   indent=1) + '\n', encoding='utf-8')
        # Parse back what was written so malformed notebook JSON never ships.
        saved = json.loads(path.read_text(encoding='utf-8'))
        assert saved['nbformat'] == 4 and len(saved['cells']) == 3
        print(path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
