import subprocess
from pytask import Product, mark
from pytask_latex import compilation_steps as cs

from pathlib import Path
from typing import Annotated

TEX = Path("tex_files")
BLD = Path("bld")


def task_generate_numerical_example(
    code: Path = Path("src") / "num_example.py",
    produces: Path = Path("bld") / "numerical_example.tex",
):
    """Generates LaTeX macros from the numerical example parameters."""
    subprocess.run(["python", code], check=True)


products = {
    "probabilities_across_T_png": BLD/'commitment_probabilities.png',
    "probabilities_across_T_pdf": BLD/'commitment_probabilities.pdf',
    "payoff_and_crash_png": BLD/'commitment_periods_plot.png',
    "payoff_and_crash_pdf": BLD/'commitment_periods_plot.pdf',
}
def task_generate_graphs(
    code: Path = Path("src") / "generate_graph.py",
    parameters: Path = Path("src") / "num_example.py",
    produces: Path = Path("bld") / "commitment_probabilities.png",
):
    """Generates graphs for the paper."""
    subprocess.run(["python", code], check=True)


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
