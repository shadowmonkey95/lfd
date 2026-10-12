import json
import pathlib
from typing import get_args
from typing import Literal
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyod.models.auto_encoder import AutoEncoder
from pyod.models.vae import VAE
from pyclad.analysis.scenario_heatmap import plot_metric_heatmap
from pyclad.callbacks.evaluation.concept_metric_evaluation import ConceptMetricCallback
from pyclad.callbacks.evaluation.energy_evaluation import EnergyEvaluationCallback
from pyclad.callbacks.evaluation.time_evaluation import TimeEvaluationCallback
from pyclad.data.datasets.nsl_kdd_dataset import NslKddDataset
from pyclad.data.datasets.unsw_dataset import UnswDataset
from pyclad.data.readers.concepts_readers import read_dataset_from_npy
from pyclad.metrics.base.roc_auc import RocAuc
from pyclad.metrics.continual.average_continual import ContinualAverage
from pyclad.metrics.continual.backward_transfer import BackwardTransfer
from pyclad.metrics.continual.forward_transfer import ForwardTransfer
from pyclad.models.adapters.pyod_adapters import (
    COPODAdapter,
    ECODAdapter,
    IsolationForestAdapter,
    LocalOutlierFactorAdapter,
    OneClassSVMAdapter,
    PyODAdapter,
)
from pyclad.output.json_writer import JsonOutputWriter
from pyclad.scenarios.concept_incremental import ConceptIncrementalScenario
from pyclad.strategies.baselines.cumulative import CumulativeStrategy
from pyclad.strategies.baselines.naive import NaiveStrategy
from pyclad.strategies.replay.buffers.adaptive_balanced import AdaptiveBalancedReplayBuffer
from pyclad.strategies.replay.replay import ReplayEnhancedStrategy
from pyclad.strategies.replay.selection.random import RandomSelection
from data.helpers.data_loader import load_csv_dataset, load_npy_dataset
from src.helpers import clustering
from src.helpers.prepare_scenario import prepare_and_save_scenario, prepare_scenario
from src.helpers.scenario_config import ScenarioConfig, ScenarioType
from sklearn.metrics import average_precision_score
from pyclad.metrics.base.base_metric import BaseMetric


# ====== Custom Metrics ======

class PrAuc(BaseMetric):
    def compute(self, anomaly_scores, y_pred, y_true) -> float:
        return average_precision_score(y_true=y_true, y_score=anomaly_scores)

    def name(self) -> str:
        return "PR-AUC"


class NormalizedPrAuc(BaseMetric):
    """PR-AUC normalized by the random-classifier baseline (class prevalence).

    normalized = (PR-AUC - prevalence) / (1 - prevalence)

    Ranges from 0 (random) to 1 (perfect). Can be negative for below-random classifiers.
    Returns NaN when all labels are the same class (prevalence is 0 or 1).
    """

    def compute(self, anomaly_scores, y_pred, y_true) -> float:
        prevalence = np.mean(y_true)
        if prevalence == 0.0 or prevalence == 1.0:
            return float("nan")
        pr_auc = average_precision_score(y_true=y_true, y_score=anomaly_scores)
        return (pr_auc - prevalence) / (1.0 - prevalence)

    def name(self) -> str:
        return "Normalized-PR-AUC"


# ====== Config ======

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent

dataset_name    = '4sparkov_preprocessed_without_time'
target_col_name = 'is_fraud'
time_col_name   = 'unix_time'
concepts_no     = 8
size_per_concept = 230_206

ScenarioType = Literal[
    'random_anomalies',
    'clustered_with_random_assignment',
    'clustered_with_closest_assignment',
    'all_anomalies',
    'time_based',
]

# To run a single scenario, set this to a tuple with one element, e.g.:
#   scenarios_to_run = ("time_based",)
# To run all scenarios, use:
#   scenarios_to_run = get_args(ScenarioType)

# scenarios_to_run = ("time_based",)
scenarios_to_run = get_args(ScenarioType)


if __name__ == '__main__':

    # ====== Load CSV once, shared across all scenarios ======
    file_name = dataset_name + ".csv"
    file_path = BASE_DIR / "data/preprocessed/4_sparkov" / file_name

    normal_data, anomaly_data, column_names = load_csv_dataset(
        file_path,
        target_col=target_col_name,
        exclude_columns=[time_col_name],
        keep_columns=[time_col_name],
    )
    print(f"Columns returned by loader: {column_names}")

    try:
        sort_idx = column_names.index(time_col_name)
    except ValueError:
        sort_idx = None

    # ====== Iterate over every scenario type ======
    for scenario_type in scenarios_to_run:
        print(f"\n{'='*60}")
        print(f"SCENARIO: {scenario_type}")
        print(f"{'='*60}")

        config = ScenarioConfig(
            scenario_type=scenario_type,
            concepts_no=concepts_no,
            size_per_concept=size_per_concept,
        )

        # ====== Generate and save .npy scenario ======
        prepare_and_save_scenario(
            dataset_name,
            normal_data,
            anomaly_data,
            config,
            sort_column=sort_idx,
        )

        # ====== Load .npy scenario (path uses scenario_type) ======
        dataset_path = (
            BASE_DIR
            / "out"
            / f"{dataset_name}_npy"
            / f"{dataset_name}_{scenario_type}_{concepts_no}_concepts.npy"
        )
        dataset = read_dataset_from_npy(dataset_path, dataset_name=dataset_name)

        # ====== Per-scenario output directory ======
        out_dir = (
            pathlib.Path("output")
            / f"{dataset_name}_json"
            / f"{scenario_type}_{concepts_no}_concepts_{size_per_concept}_each"
        )
        out_dir.mkdir(exist_ok=True, parents=True)

        # ====== Models × Strategies ======
        for model in [
            PyODAdapter(
                AutoEncoder(
                    hidden_neuron_list=[32, 24, 16],
                    epoch_num=20,
                ),
                model_name="AE",
            ),
            IsolationForestAdapter(),
            LocalOutlierFactorAdapter(),
            OneClassSVMAdapter(),
            COPODAdapter(),
            ECODAdapter(),
        ]:
            for strategy in [
                NaiveStrategy(model),
                CumulativeStrategy(model),
                ReplayEnhancedStrategy(
                    model,
                    AdaptiveBalancedReplayBuffer(
                        selection_method=RandomSelection(),
                        max_size=1000,
                    ),
                ),
            ]:
                print(f"Running {model.name()} with {strategy.name()} on {dataset.name}")

                callbacks = [
                    EnergyEvaluationCallback(),
                    TimeEvaluationCallback(),
                    # ROC-AUC with BWT and FWT
                    ConceptMetricCallback(
                        base_metric=RocAuc(),
                        metrics=[
                            ContinualAverage(),
                            BackwardTransfer(),
                            ForwardTransfer(),
                        ],
                    ),
                    # Normalized PR-AUC with BWT and FWT
                    ConceptMetricCallback(
                        base_metric=NormalizedPrAuc(),
                        metrics=[
                            ContinualAverage(),
                            BackwardTransfer(),
                            ForwardTransfer(),
                        ],
                    ),
                ]

                scenario = ConceptIncrementalScenario(
                    dataset=dataset,
                    strategy=strategy,
                    callbacks=callbacks,
                )
                scenario.run()

                print(f"Finished {model.name()} with {strategy.name()} on {dataset.name}\n")

                json_output_writer = JsonOutputWriter(
                    out_dir / f"output_{model.name()}_{strategy.name()}_{dataset.name()}.json"
                )
                json_output_writer.write([model, strategy, dataset, *callbacks])

        # ====== Plot heatmaps (ROC-AUC + Normalized-PR-AUC) ======
        output_base_dir = (
            BASE_DIR
            / "heatmaps"
            / dataset_name
            / f"{scenario_type}_{concepts_no}_concepts_{size_per_concept}_each"
        )
        output_base_dir.mkdir(parents=True, exist_ok=True)

        metric_keys = [
            "concept_metric_callback_ROC-AUC",
            "concept_metric_callback_Normalized-PR-AUC",
        ]

        for results_path in out_dir.glob("*.json"):
            print(f"Processing: {results_path.name}")

            with open(results_path) as fp:
                loaded_data = json.load(fp)

            for metric_key in metric_keys:
                if metric_key not in loaded_data:
                    print(f"  Skipping {metric_key} (not found in JSON)")
                    continue

                concepts_order = loaded_data[metric_key]["concepts_order"]
                metric_matrix  = loaded_data[metric_key]["metric_matrix"]
                names_mapping  = {concept: f"C{idx}" for idx, concept in enumerate(concepts_order)}

                metric_suffix   = metric_key.replace("concept_metric_callback_", "")
                output_filename = results_path.stem + f"_{metric_suffix}.pdf"
                output_path     = output_base_dir / output_filename

                plot_metric_heatmap(
                    metric_matrix,
                    concepts_order,
                    names_mapping=names_mapping,
                    annotate=True,
                    output_path=output_path,
                    ignore_upper_diagonal=False,
                )
                plt.close("all")
                print(f"  Heatmap saved to {output_path}")