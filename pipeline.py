import pathlib

from pyclad.callbacks.evaluation.concept_metric_evaluation import ConceptMetricCallback
from pyclad.callbacks.evaluation.energy_evaluation import EnergyEvaluationCallback
from pyclad.callbacks.evaluation.time_evaluation import TimeEvaluationCallback
from pyclad.data.datasets.nsl_kdd_dataset import NslKddDataset
from pyclad.data.datasets.unsw_dataset import UnswDataset
from pyclad.metrics.base.roc_auc import RocAuc
from pyclad.metrics.continual.average_continual import ContinualAverage
from pyclad.metrics.continual.backward_transfer import BackwardTransfer
from pyclad.metrics.continual.forward_transfer import ForwardTransfer
from pyclad.models.adapters.pyod_adapters import (
    IsolationForestAdapter,
    LocalOutlierFactorAdapter,
    OneClassSVMAdapter,
)
from pyclad.models.adapters.pyod_adapters import PyODAdapter
from pyclad.output.json_writer import JsonOutputWriter
from pyclad.scenarios.concept_incremental import ConceptIncrementalScenario
from pyclad.strategies.baselines.naive import NaiveStrategy
from pyod.models.vae import VAE

if __name__ == "__main__":
    out_dir = pathlib.Path("output/unsw_random_anomalies")
    out_dir.mkdir(exist_ok=True, parents=True)

    dataset = UnswDataset(dataset_type="random_anomalies")

    for model in [
        # IsolationForestAdapter(), LocalOutlierFactorAdapter(), OneClassSVMAdapter(),
        PyODAdapter(
            VAE(
                encoder_neuron_list=[32, 24, 16],
                decoder_neuron_list=[16, 24, 32],
                latent_dim=8,
                epoch_num=20,
                preprocessing=False,
            ),
            model_name="VAE",
        )]:
        for strategy in [NaiveStrategy(model)]:
            print(f"Running {model.name()} with {strategy.name()} on {dataset.name}")

            callbacks = [EnergyEvaluationCallback(), TimeEvaluationCallback(),
                         ConceptMetricCallback(base_metric=RocAuc(),
                                               metrics=[ContinualAverage(), BackwardTransfer(), ForwardTransfer()])]

            scenario = ConceptIncrementalScenario(dataset=dataset, strategy=strategy, callbacks=callbacks)

            scenario.run()

            print(f"Finished {model.name()} with {strategy.name()} on {dataset.name}\n")
            json_output_writer = JsonOutputWriter(
                out_dir / f"output_{model.name()}_{strategy.name()}_{dataset.name()}.json")
            json_output_writer.write([model, strategy, dataset, *callbacks])
