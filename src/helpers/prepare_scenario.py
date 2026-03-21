import pathlib
from typing import List, Optional, Union

import numpy as np
from numpy.random import shuffle

from src.helpers.clustering import create_concepts, create_concepts_with_order, create_random_anomaly_clusters, create_anomaly_clusters_randomly_assigned, \
    create_anomaly_clusters_closest_to_normal, create_anomaly_clusters_time_based, create_all_anomaly_clusters
from src.helpers.concept import Concept
from src.helpers.csv_writer import save_scenario_as_csv
from src.helpers.scenario_config import ScenarioConfig
import inspect


# REGULAR
# def split_into_train_test(normal_cluster, anomaly_cluster):
#     split_point = int(3 * len(normal_cluster.data) / 5)
#     train_data_normal = np.array(normal_cluster.data[:split_point])
#     test_data_normal = np.array(normal_cluster.data[split_point:])
#     test_data_normal_with_labels = np.append(test_data_normal, np.zeros((len(test_data_normal), 1)), axis=1)
#     test_data_anomaly_with_labels = np.append(anomaly_cluster, np.ones((len(anomaly_cluster), 1)), axis=1)

#     test_data_with_labels = np.concatenate((test_data_normal_with_labels, test_data_anomaly_with_labels))
#     shuffle(test_data_with_labels)

#     test_data, test_labels = test_data_with_labels[:, :-1], test_data_with_labels[:, -1]

#     return train_data_normal, test_data, test_labels

def split_into_train_test(normal_cluster: np.ndarray, anomaly_cluster: np.ndarray, train_ratio: float = 0.6):
    split_point = int(train_ratio * len(normal_cluster))
    
    train_data_normal = normal_cluster[:split_point]
    test_data_normal = normal_cluster[split_point:]
    
    # Create labels separately to avoid appending them to the feature array yet
    labels_normal = np.zeros(len(test_data_normal))
    labels_anomaly = np.ones(len(anomaly_cluster))
    
    # Concatenate the features and labels independently first
    test_data = np.concatenate([test_data_normal, anomaly_cluster], axis=0)
    test_labels = np.concatenate([labels_normal, labels_anomaly], axis=0)
    
    # Shuffle both simultaneously to keep labels aligned (remove if timebase)
    idx = np.arange(len(test_data))
    # np.random.shuffle(idx)
    
    return train_data_normal, test_data[idx], test_labels[idx]

# REGULAR
# def _create_anomaly_clusters(normal_clusters, anomaly_data, config: ScenarioConfig, sort_column: int = None):
#     anomalies_no_per_cluster = min(int(len(anomaly_data) / len(normal_clusters)), int(0.2 * config.size_per_concept / 5))

#     if config.scenario_type == 'random_anomalies':
#         return create_random_anomaly_clusters(anomaly_data, clusters_no=config.concepts_no,
#                                               size_per_cluster=anomalies_no_per_cluster)
#     elif config.scenario_type == 'clustered_with_random_assignment':
#         return create_anomaly_clusters_randomly_assigned(anomaly_data, clusters_no=config.concepts_no,
#                                                          size_per_cluster=anomalies_no_per_cluster)
#     elif config.scenario_type == 'clustered_with_closest_assignment':
#         return create_anomaly_clusters_closest_to_normal(anomaly_data, normal_clusters, clusters_no=config.concepts_no,
#                                                          size_per_cluster=anomalies_no_per_cluster)
#     elif config.scenario_type == 'time_based':
#         # sort_column = getattr(config, "time_column_index", 0)
#         return create_anomaly_clusters_time_based(
#             normal_clusters=normal_clusters,
#             anomaly_data=anomaly_data,
#             sort_column=sort_column,
#             max_anomalies=anomalies_no_per_cluster,
#             config=config
#         )
#     elif config.scenario_type == 'all_anomalies':
#         return create_all_anomaly_clusters(anomaly_data, clusters_no=config.concepts_no)

def _create_anomaly_clusters(normal_clusters, anomaly_data, config: ScenarioConfig, sort_column: int = None):
    # Make anomaly count more flexible to avoid empty concepts
    # Aim for a target ratio (e.g., 5-10% anomalies) 
    avg_anomalies = len(anomaly_data) // len(normal_clusters)
    max_allowed = int(config.size_per_concept * 0.2)
    
    anomalies_no_per_cluster = min(avg_anomalies, max_allowed)

    if config.scenario_type == 'random_anomalies':
        return create_random_anomaly_clusters(anomaly_data, clusters_no=config.concepts_no,
                                              size_per_cluster=anomalies_no_per_cluster)
    elif config.scenario_type == 'clustered_with_random_assignment':
        return create_anomaly_clusters_randomly_assigned(anomaly_data, clusters_no=config.concepts_no,
                                                         size_per_cluster=anomalies_no_per_cluster)
    elif config.scenario_type == 'clustered_with_closest_assignment':
        return create_anomaly_clusters_closest_to_normal(anomaly_data, normal_clusters, clusters_no=config.concepts_no,
                                                         size_per_cluster=anomalies_no_per_cluster)
    elif config.scenario_type == 'time_based':
        return create_anomaly_clusters_time_based(
            normal_clusters=normal_clusters,
            anomaly_data=anomaly_data,
            sort_column=sort_column,
            max_anomalies=anomalies_no_per_cluster,
            config=config
        )
    elif config.scenario_type == 'all_anomalies':
        return create_all_anomaly_clusters(anomaly_data, clusters_no=config.concepts_no)

def prepare_scenario(normal_data: np.ndarray, anomaly_data: np.ndarray, config: ScenarioConfig) -> List[Concept]:
    normal_clusters = create_concepts(normal_data, concepts_no=config.concepts_no,
                                      size_per_concept=config.size_per_concept)

    anomalies_clusters = _create_anomaly_clusters(normal_clusters, anomaly_data, config=config)
    concepts = []
    for i, (normal_cluster, anomaly_cluster) in enumerate(zip(normal_clusters, anomalies_clusters)):
        train_data, test_data, test_labels = split_into_train_test(normal_cluster, anomaly_cluster)
        concepts.append(
            Concept(name=f'Concept_{i}', train_data=train_data, test_data=test_data, test_labels=test_labels))

    return concepts

def prepare_scenario_with_order(
    normal_data: np.ndarray, 
    anomaly_data: np.ndarray, 
    config: ScenarioConfig, 
    sort_column: int
) -> List[Concept]:
    normal_clusters = create_concepts_with_order(
        normal_data, 
        concepts_no=config.concepts_no,
        size_per_concept=config.size_per_concept,
        sort_column=sort_column
    )
    
    anomalies_clusters = _create_anomaly_clusters(
        normal_clusters, 
        anomaly_data, 
        config=config, 
        sort_column=sort_column
    )

    concepts = []
    for i, (n_cluster, a_cluster) in enumerate(zip(normal_clusters, anomalies_clusters)):
        train_data, test_data, test_labels = split_into_train_test(n_cluster, a_cluster) 
        # Remove the metadata columns so the model only sees actual fraud features.
        train_data_cleaned = train_data[:, :sort_column] 
        test_data_cleaned = test_data[:, :sort_column]

        concepts.append(
            Concept(
                name=f'Concept_{i}', 
                train_data=train_data_cleaned, 
                test_data=test_data_cleaned, 
                test_labels=test_labels
            )
        )

    return concepts

# REGULAR
# def prepare_and_save_scenario(scenario_name: str, normal_data: np.ndarray, anomaly_data: np.ndarray,
#                               config: ScenarioConfig, sort_column: int = 0):
#     if sort_column is not None:
#         concepts = prepare_scenario_with_order(normal_data, anomaly_data, config, sort_column)
#     else:
#         concepts = prepare_scenario(normal_data, anomaly_data, config)
#     path = pathlib.Path(f'out/{scenario_name}_npy/')
#     path.mkdir(exist_ok=True, parents=True)
#     output_filename = f'{scenario_name}_{config.scenario_type}_{config.concepts_no}_concepts_{config.size_per_concept}_per_cluster'
#     np.save(str(path / output_filename), np.array(concepts))
#     save_scenario_as_csv(scenario=concepts, output_dir=path)
    
def prepare_and_save_scenario(scenario_name: str, 
                             normal_data: np.ndarray, 
                             anomaly_data: np.ndarray,
                             config: ScenarioConfig, 
                             sort_column: int = 0):
    print(f"--- Preparing Scenario: {scenario_name} ---")
    print(f"--- sort_column: {sort_column} ---")

    # Create the concepts based on time/order
    if sort_column is not None:
        concepts = prepare_scenario_with_order(
            normal_data, 
            anomaly_data, 
            config, 
            sort_column
        )
    else:
        concepts = prepare_scenario(normal_data, anomaly_data, config)

    # Save logic
    path = pathlib.Path(f'out/{scenario_name}_npy/')
    path.mkdir(exist_ok=True, parents=True)
    
    output_filename = f'{scenario_name}_{config.scenario_type}_{config.concepts_no}_concepts'
    
    # Store the list of Concept objects
    np.save(str(path / output_filename), np.array(concepts, dtype=object))
    
    # Helper for CSV inspection
    save_scenario_as_csv(scenario=concepts, output_dir=path)
    
    print(f"Successfully saved {len(concepts)} concepts to {path}")
