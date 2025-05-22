import tyro

from .detecbench import evaluate

if __name__ == "__main__":
    tyro.cli(evaluate)
