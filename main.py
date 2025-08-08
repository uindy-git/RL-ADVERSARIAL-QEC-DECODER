import os
import matplotlib

from src.utils import evaluate_agent, generate_vulnerability_map
from src.train_rl_agent import train_rl_agent
from src.train_decoder import train_gat_decoder
from src.dataset import create_dataset_from_shots
from src.train_adversarial_decoder import train_adversarial_GAT_decoder

if __name__ == "__main__":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    matplotlib.use("Agg")  # Use Agg backend for matplotlib to avoid GUI issues

    directories = {
        "circuit": "data/surface_code_bX_d3_r01_center_3_5/circuit_ideal.stim",
        "dem": "data/surface_code_bX_d3_r01_center_3_5/circuit_detector_error_model.dem",
        "syndromes": "data/surface_code_bX_d3_r01_center_3_5/detection_events_new.b8",
        "logical_flips": "data/surface_code_bX_d3_r01_center_3_5/obs_flips_actual_new.01",
        "gat_model": "model/gat_decoder.pth",
        "rl_model": "model/rl_agent.pth",
        "robust_gat_model": "model/robust_gat_decoder.pth",
        "figures": "figures",
    }
    rl_vul_map_filename1 = "heatmap1.pdf"
    rl_vul_map_filename2 = "heatmap2.pdf"

    # Create dataset from shots
    node_info, dataset = create_dataset_from_shots(directories)
    num_rounds = node_info["num_rounds"]

    # Train GAT decoder
    gat_model = train_gat_decoder(node_info, dataset, directories)

    # Train RL agent
    test_neg_samples, rl_model = train_rl_agent(gat_model, node_info, dataset, directories)
    evaluate_agent(rl_model, gat_model, test_neg_samples)
    generate_vulnerability_map(rl_model, gat_model, node_info, dataset, directories, rl_vul_map_filename1)

    # Train robust GAT decoder
    robust_gat_model = train_adversarial_GAT_decoder(gat_model, rl_model, num_rounds, dataset, directories)
    print("--- Evaluate Robust GAT Decoder ---")
    evaluate_agent(rl_model, robust_gat_model, test_neg_samples)
    generate_vulnerability_map(rl_model, robust_gat_model, node_info, dataset, directories, rl_vul_map_filename2)
    print("--- Training and Evaluation Completed ---")