import stim
import numpy as np
import torch
from scipy.spatial.distance import cdist

from utils import create_flattened_graph_from_shot, cultivate_edge_weights

def create_dataset_from_shots(directories):
    circuit = stim.Circuit.from_file(directories.get("circuit", "circuit_ideal.stim"))
    dem = stim.DetectorErrorModel.from_file(directories.get("dem", "circuit_detector_error_model.dem"))
    syndromes = stim.read_shot_data_file(path=directories.get("syndromes", "detection_events_new.b8"), format="b8", num_detectors=circuit.num_detectors)
    logical_flips = np.loadtxt(directories.get("logical_flips", "obs_flips_actual_new.01"), dtype=np.uint8).reshape(-1, 1)

    # --- 2. Reconstruct graph structure and node features ---
    coords_dict = dem.get_detector_coordinates()
    coords_array = np.array([coords_dict[i] for i in sorted(coords_dict.keys())])

    # Separate time (t) and spatial (x,y) coordinates
    spatial_coords = coords_array[:, :2] # x, y coordinates
    time_coords = coords_array[:, 2]    # t coordinates

    # Identify unique spatial nodes (representing the node at t=0)
    unique_spatial_nodes, spatial_indices = np.unique(spatial_coords, axis=0, return_index=True)
    num_spatial_nodes = len(unique_spatial_nodes)
    num_rounds = int(time_coords.max()) + 1

    node_info = {"num_spatial_nodes": num_spatial_nodes,
                "num_rounds": num_rounds
                }

    coords_array = np.array([coords_dict[i] for i in sorted(coords_dict.keys())])
    spatial_coords = coords_array[:, :2]
    unique_spatial_nodes, spatial_indices = np.unique(spatial_coords, axis=0, return_index=True)

    # Construct spatial graph edges
    dist_matrix = cdist(unique_spatial_nodes, unique_spatial_nodes)
    rows, cols = np.where(dist_matrix)
    edge_list = [[r, c] for r, c in zip(rows, cols) if r < c]
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)

    num_shots = syndromes.shape[0]
    time_flattened_features = np.zeros((num_shots, num_spatial_nodes, num_rounds), dtype=np.float32)

    # Map original syndrome data to new feature tensor
    for detector_idx, coord in coords_dict.items():
        # Find which spatial node this detector corresponds to
        spatial_coord = coord[:2]
        spatial_node_idx = np.where((unique_spatial_nodes == spatial_coord).all(axis=1))[0][0]
        # Get time round index
        time_round_idx = int(coord[2])
        # Copy corresponding features
        time_flattened_features[:, spatial_node_idx, time_round_idx] = syndromes[:, detector_idx]

    static_edge_weights = cultivate_edge_weights(edge_index)
    dataset = [
        create_flattened_graph_from_shot(time_flattened_features[i], logical_flips[i], edge_index, static_edge_weights)
        for i in range(num_shots)
    ]
    return node_info, dataset