"""
Script for optimizng hyperparameters of the Word2Vec model
"""
from itertools import islice
from typing import Any, Dict, Tuple

import optuna
import wandb
from gensim.models import Word2Vec

from utils.evaluation import evaluate_word2vec
from utils.streams import chunk, sentence_stream, stream_cleaned_texts
from utils.training import train

# This is all set up for my Ucloud instance, if you have a different setup please change the script
# I just didn't wantr to write a CLI for this as it would have taken way more time
# and it isn't going to be used as excessively as the main script
DATA_PATH = "/work/netarkivet-cleaned/"


def suggest_hyperparameters(trial: optuna.Trial) -> Dict[str, Any]:
    """
    Makes the trial suggest a Word2Vec model with certain hyperparameters.

    Parameters
    -----------
    trial: optuna.Trial
        Trial object to suggest hyperparameters

    Returns
    -----------
    hyperparameters: dict
        Dictionary containing the hyperparameters
    """
    vector_size = trial.suggest_int("vector_size", low=50, high=550, step=50)
    window_size = trial.suggest_int("window_size", low=4, high=26, step=2)
    shrink_windows = trial.suggest_categorical("shrink_windows", [True, False])
    skip_gram = trial.suggest_categorical("skip_gram", [1, 0])
    negative = trial.suggest_int("negative", low=5, high=20, step=5)
    hs = trial.suggest_categorical("hierachical_softmax", [1, 0])
    ns_exponent = trial.suggest_float("ns_exponent", low=-1.0, high=1.0, step=0.25)
    cbow_mean = trial.suggest_categorical("cbow_mean", [1, 0])
    return dict(
        vector_size=vector_size,
        window=window_size,
        shrink_windows=shrink_windows,
        sg=skip_gram,
        negative=negative,
        hs=hs,
        ns_exponent=ns_exponent,
        cbow_mean=cbow_mean,
    )


def training_sequence(model: Word2Vec, n_chunks: int = 10) -> None:
    """
    Trains the model on n batches.

    Parameters
    -----------
    model: Word2Vec
        Model to train
    n_chunks: int, default 10
        Number of chunks to train the model on
    """
    text_chunks = islice(
        chunk(
            stream_cleaned_texts(DATA_PATH, filter_porn=True),
            chunk_size=100_000,
            sample_size=150_100,
        ),
        n_chunks,
    )
    preprocess = lambda texts: sentence_stream(texts, workers=2)
    for i_chunk, loss in enumerate(
        train(
            model=model,
            text_chunks=text_chunks,
            preprocessing=preprocess,
            save_path=None,
        )
    ):
        print(f"Loss after chunk {i_chunk}: {loss}")


def objective_with_table(table: wandb.Table):
    """
    This whole mess is here so that I can pass a table to the objective function and it
    can log to a wandb table.
    I create a closure in which the table is stored and return the objective function with
    table logging included.

    Parameters
    -----------
    table: wandb Table
        Table to log trials to

    Returns
    -----------
    objective: function
        Objective function that optuna can optimize
        with table logging included.
    """

    def objective(trial: optuna.Trial) -> Tuple[float, float, float]:
        """
        Trains and evaluates model and returns objectives to be optimized.

        Parameters
        -----------
        trial: optuna.Trial

        Returns
        -----------
        acc_odd: float
            Accuracy on the odd-one-out test
        dsd_rho: float
            Spearman's rho on the DSD similarity test
        w353_rho: float
            Spearman's rho on the W353 similarity test
        """
        hyperparameters = suggest_hyperparameters(trial)
        model = Word2Vec(**hyperparameters)
        training_sequence(model)
        metrics = evaluate_word2vec(model)
        acc_odd, dsd_rho, w353_rho = (
            metrics["Accuracy - Odd one out"],
            metrics["DSD similarities Spearman's ρ"],
            metrics["W353 similarities Spearman's ρ"],
        )
        # Logging hyperparameters and results to wandb table
        table.add_data(
            hyperparameters["vector_size"],
            hyperparameters["window"],
            hyperparameters["shrink_windows"],
            hyperparameters["sg"],
            hyperparameters["negative"],
            hyperparameters["hs"],
            hyperparameters["ns_exponent"],
            hyperparameters["cbow_mean"],
            acc_odd,
            dsd_rho,
            w353_rho,
        )
        # I'm also logging metrics so we can see if the grid search really improves stuff
        wandb.log(metrics)
        return acc_odd, dsd_rho, w353_rho

    return objective


# AGAIN I was lazy not to make a CLI, please live with it
N_TRIALS = 100


def main() -> None:
    """
    Main function running the optimization
    """
    # Initialise Weights and biases
    wandb.init(
        project="netarkivet-word2vec-hyperparameter-optimization", entity="chcaa"
    )
    columns = [
        "vector_size",
        "window",
        "shrink_windows",
        "sg",
        "negative",
        "hs",
        "ns_exponent",
        "cbow_mean",
        "acc_odd",
        "dsd_rho",
        "w353_rho",
    ]
    results_table = wandb.Table(columns=columns)
    # Create a study that maximizes all three metrics
    study = optuna.create_study(directions=["maximize"] * 3)
    objective = objective_with_table(results_table)
    study.optimize(objective, n_trials=N_TRIALS)


if __name__ == "__main__":
    main()
