# Commitment Races

This repository contains the paper "Commitment Races" by Loren K. Fryxell and Maximilian Reith.

The paper develops a formal model to study commitment races in game theory. We show that in symmetric 2x2 games like the Game of Chicken, the ability to credibly commit can lead both players to be worse off than if neither had this ability.

## Paper

The compiled paper is available as [paper.pdf](paper.pdf).

## Running the Code

The Python code generates numerical examples and figures used in the paper. If you would like to run it locally, we use [pixi](https://pixi.sh) for package management.

1. Install pixi (see https://pixi.sh for instructions)
2. Run:
   ```bash
   pixi install
   ```
3. You're good to go. For example, generate the numerical example, graphs and final pdf with:
   ```bash
   pixi run pytask
   ```
