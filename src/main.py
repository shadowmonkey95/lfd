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

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
dataset_name = 'creditcard_with_time_preprocessed'
target_col = 'Class'

concepts_no = 6
size_per_concept = 47467

ScenarioType = Literal['random_anomalies', 'clustered_with_random_assignment', 'clustered_with_closest_assignment', 'time_based']
scenario_config = "time_based"
if __name__ == '__main__':
    # ====== Split normal, anomaly data + generate npy scenario======
    file_name = dataset_name + ".csv"
    file_path = BASE_DIR / "data/preprocessed" / file_name

    normal_data, anomaly_data = load_csv_dataset(file_path, target_col=target_col, exclude_columns=[])

    for scenario_type in get_args(ScenarioType):
        config = ScenarioConfig(scenario_type=scenario_type, concepts_no=concepts_no, size_per_concept=size_per_concept)

        prepare_and_save_scenario(dataset_name, normal_data, anomaly_data, config)
    
    # ====== Load npy scenario ======
    dataset_path = (
        BASE_DIR
        / "out"
        / f"{dataset_name}_npy"
        / f"{dataset_name}_{scenario_config}_{concepts_no}_concepts_{size_per_concept}_per_cluster.npy"
    )
    dataset = read_dataset_from_npy(dataset_path, dataset_name=dataset_name)
    out_dir = pathlib.Path("output") / f"{dataset_name}_json" / f"{scenario_config}_{concepts_no}_concepts_{size_per_concept}_each"
    out_dir.mkdir(exist_ok=True, parents=True)

    # ====== Models ======
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
        ECODAdapter()
        ]:
        
        # ====== Strategies ======
        for strategy in [
            NaiveStrategy(model),
            CumulativeStrategy(model),
            ReplayEnhancedStrategy(model, AdaptiveBalancedReplayBuffer(selection_method=RandomSelection(), max_size=1000))
            ]:
            print(f"Running {model.name()} with {strategy.name()} on {dataset.name}")

            callbacks = [
                EnergyEvaluationCallback(),
                TimeEvaluationCallback(),
                ConceptMetricCallback(
                    base_metric=RocAuc(),
                    metrics=[
                        ContinualAverage(), 
                        BackwardTransfer(), 
                        ForwardTransfer()
                        ]
                    )
                ]

            scenario = ConceptIncrementalScenario(dataset=dataset, strategy=strategy, callbacks=callbacks)
            scenario.run()
            print(f"Finished {model.name()} with {strategy.name()} on {dataset.name}\n")
            json_output_writer = JsonOutputWriter(out_dir / f"output_{model.name()}_{strategy.name()}_{dataset.name()}.json")
            json_output_writer.write([model, strategy, dataset, *callbacks])
        
        
    # ====== Plot heatmaps ======
    results_dir = BASE_DIR / "output" / dataset_name
    results_dir = out_dir
    
    output_base_dir = BASE_DIR / "heatmaps" / dataset_name / f"{scenario_config}_{concepts_no}_concepts_{size_per_concept}_each"
    output_base_dir.mkdir(parents=True, exist_ok=True)

    for results_path in results_dir.glob("*.json"):
        print(f"Processing: {results_path.name}")
        
        with open(results_path) as fp:
            loaded_data = json.load(fp)
        
        concepts_order = loaded_data["concept_metric_callback_ROC-AUC"]["concepts_order"]
        metric_matrix = loaded_data["concept_metric_callback_ROC-AUC"]["metric_matrix"]
        
        names_mapping = {concept: f"C{idx}" for idx, concept in enumerate(concepts_order)}
        
        output_filename = results_path.stem + ".pdf"
        output_path = output_base_dir / output_filename
        
        plot_metric_heatmap(
            metric_matrix,
            concepts_order,
            names_mapping=names_mapping,
            annotate=True,
            output_path=output_path,
            ignore_upper_diagonal=True,
        )
        plt.close('all')
        print(f"Heatmap saved to {output_path}")