from pathlib import Path
from typing import Annotated

from pytask import Product, mark
from pytask_latex import compilation_steps as cs

from src.num_example import (
    A_DEFAULT,
    B_DEFAULT,
    C_DEFAULT,
    D_DEFAULT,
    generate_numerical_example,
)

TEX = Path("tex_files")


def task_generate_numerical_example(
    script: Path = Path("src/num_example.py"),
    produces: Annotated[Path, Product] = TEX / "numerical_example.tex",
):
    """Generates LaTeX macros from the numerical example parameters."""
    generate_numerical_example(A_DEFAULT, B_DEFAULT, C_DEFAULT, D_DEFAULT, produces)


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
