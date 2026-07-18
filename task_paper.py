import shutil
import subprocess
from pytask import Product, mark
from pytask_latex import compilation_steps as cs

from pathlib import Path
from typing import Annotated

TEX = Path("tex_files")
BLD = Path("bld")
OUTPUT = Path("output") / "pdf"


def task_generate_numerical_example(
    code: Path = Path("src") / "num_example.py",
    produces: Path = Path("bld") / "numerical_example.tex",
):
    """Generates LaTeX macros from the numerical example parameters."""
    subprocess.run(["python", code], check=True)


products = {
    "probabilities_across_T_png": BLD / "commitment_probabilities.png",
    "probabilities_across_T_pdf": BLD / "commitment_probabilities.pdf",
    "payoff_and_crash_png": BLD / "commitment_periods_plot.png",
    "payoff_and_crash_pdf": BLD / "commitment_periods_plot.pdf",
}


def task_generate_graphs(
    code: Path = Path("src") / "generate_graph.py",
    parameters: Path = Path("src") / "num_example.py",
    produces: Path = Path("bld") / "commitment_probabilities.png",
):
    """Generates graphs for the paper."""
    subprocess.run(["python", code], check=True)


@mark.latex(
    script=TEX / "figures" / "game_tree.tex",
    document=BLD / "game_tree.pdf",
    compilation_steps=cs.latexmk(
        options=("--pdf", "--interaction=nonstopmode", "--cd",
                 "--output-directory=../../bld"),
    ),
)
def task_compile_game_tree():
    """Compiles the game tree as a vector figure."""


@mark.latex(
    script=TEX / "figures" / "game_types.tex",
    document=BLD / "game_types.pdf",
    compilation_steps=cs.latexmk(
        options=("--pdf", "--interaction=nonstopmode", "--cd",
                 "--output-directory=../../bld"),
    ),
)
def task_compile_game_types():
    """Compiles the game-type regions as a vector figure."""


@mark.latex(
    script=TEX / "paper.tex",
    document=TEX / "out" / "paper.pdf",
    compilation_steps=cs.latexmk(
        options=("--pdf", "--interaction=nonstopmode", "--synctex=1", "--cd",
                 "--output-directory=out"),
    ),
)
def task_compile_paper():
    """Compiles the paper PDF with latexmk."""


def task_publish_paper(
    source: Path = TEX / "out" / "paper.pdf",
    produces: Annotated[Path, Product] = OUTPUT / "paper.pdf",
):
    """Copies the compiled paper to the final output directory."""
    produces.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, produces)
