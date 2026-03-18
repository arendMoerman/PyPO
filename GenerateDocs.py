#!/usr/bin/env python

# Written with py39 in mind

"""!
@file
PyPO docs generator.

For this script to work properly, you should have installed the docs prerequisites.
"""

from pathlib import Path
import shutil
import subprocess
import argparse

# Paths/names of command-line tools
DOXYGEN = "doxygen"
JUPYTER = "jupyter"
PANDOC = "pandoc"
#BIBTEX = "bibtex"  # not called directly by this script; called by doxygen

def check_dep(dep: str) -> None:
    try:
        run_cmd(dep, '--version')
    except (subprocess.CalledProcessError, FileNotFoundError) as err:
        raise Exception(
            f"External dependency `{dep}` is missing or broken, "
            "go fix it!"
            ) from err

def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="options for generating docs"
        )

    parser.add_argument(
        "-t", "--tutorials",
        help="generate docs from tutorials",
        action="store_true"
        )
    parser.add_argument(
        "-g", "--guitutorials",
        help="generate GUI tutorial docs",
        action="store_true"
        )
    parser.add_argument(
        "-d", "--demos",
        help="generate docs from demos",
        action="store_true"
        )
    return parser

def generate_docs() -> None:
    base_path = Path(__file__).resolve().parent
    doc_path = base_path / "docs"

    parser = setup_parser()
    args = parser.parse_args()
    
    if doc_path.exists():
        # TODO log
        shutil.rmtree(doc_path)

    if args.tutorials:
        check_dep(JUPYTER)
        convert_ipynb(
            base_path / 'tutorials',
            doc_path / 'tutorials',
            )
       
    if args.demos:
        check_dep(JUPYTER)
        convert_ipynb(
            base_path / 'demos',
            doc_path / 'demos',
            )
        
    if args.guitutorials:
        check_dep(PANDOC)
        convert_md(
            base_path / 'tutorials' / 'Gui',
            doc_path / 'Gui',
            )

    run_doxy(base_path / 'doxy' / 'Doxyfile')
    fix_filelist(doc_path / "files.html")
        
    
def fglob(directory: Path, glob: str) -> list[Path]:
    """Return all (symlinks to) files in dir by glob non-recursively"""
    files = [
        path
        for path in directory.glob(glob)
        if path.is_file()
        # is_file should also return True for symlinks to files
        ]
    assert len(files) > 0, f"No `{glob}` files found in `{directory}`! "
    return files

def convert_ipynb(source_path: Path, dest_path: Path) -> None:

    dest_path.mkdir(exist_ok=True, parents=True)

    for file in fglob(source_path, '*.ipynb'):
        html_path = run_jupyter(file)
        html_path.rename(dest_path / html_path.name)

def convert_md(source_path: Path, dest_path: Path) -> None:
    # Copy resources (e.g. images) to dest but no the md source
    shutil.copytree(
        source_path,
        dest_path,
        ignore=shutil.ignore_patterns("*.md"),
        )

    # Convert md in source to html in dest
    for file in fglob(source_path, '*.md'):
        run_pandoc(file, dest_path / file.with_suffix('.html').name)


def fix_filelist(filelist_path: Path) -> None:
    if not filelist_path.exists():
        # TODO log
        return

    # TODO better way to do this??
    content = filelist_path.read_text()
    content = content.replace('File List', 'Full Software Documentation')
    content = content.replace(
        "Here is a list of all documented files with brief descriptions:",
        "Here is a list containing the full software documentation. The structure of this page reflects the source code hierarchy."
    )
    filelist_path.write_text(content)

def run_doxy(doxyfile_path: Path) -> None:
    run_cmd(
        DOXYGEN,
        str(assert_is_file(doxyfile_path)),
        )

def run_jupyter(source: Path) -> Path:
    run_cmd(
        JUPYTER,
        "nbconvert",
        "--to",
        "html",
        "--template",
        "lab",
        "--theme",
        "dark",
        str(assert_is_file(source)),
    )
    return assert_is_file(source.with_suffix('.html'))

def run_pandoc(source: Path, dest: Path) -> None:
    run_cmd(
        PANDOC,
        str(assert_is_file(source)),
        '-t',
        'html',
        '-o',
        str(dest),
        )

def assert_is_file(path: Path) -> Path:
    assert path.is_file(), f"`{path}`: not a file."
    return path

def run_cmd(*argv: str) -> None:
    print(f"RUNNING: {argv}")
    subprocess.run(argv, shell=False, check=True)

if __name__ == "__main__":
    check_dep(DOXYGEN)
    generate_docs()

