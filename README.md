EGKD: Efficient Graph Knowledge Distillation for Key Node Identification
This repository provides a reproducible implementation of EGKD, a teacher-student graph knowledge distillation framework for node influence prediction and key node identification in complex networks.
The current release contains three benchmark datasets and their precomputed SIR influence labels under the 1.25× epidemic threshold setting.
1. Project Structure
   
EGKD/

├── My_EGKD_main.py

├── utils_data.py

├── cora.txt

├── cora_beta_1.25_sir_scores.pkl

├── powergrid.txt

├── powergrid_beta_1.25_sir_scores.pkl

├── Chicago.txt

├── Chicago_beta_1.25_sir_scores.pkl

└── README.md

2. Method Overview
EGKD is designed for label-efficient node influence prediction. The main workflow contains:
Topological feature construction
The input node feature matrix is built from representative graph structural metrics.

Pseudo-label generation
A small set of topology-based indicators is fused to generate pseudo influence labels.

Teacher model training
A GraphSAGE-based teacher model is first pretrained using pseudo-labels and then fine-tuned using a small number of SIR-labeled nodes.

Student model distillation
A lightweight GraphSAGE student model learns from the teacher through global pairwise ranking distillation, Top-K distillation, and sparse SIR label supervision.

Evaluation
The model is evaluated by Kendall tau, Spearman correlation, Jaccard@K, NDCG@K, and Imprecision@K.

3. Datasets
The repository includes three datasets:
Dataset	Graph file	SIR label file
Cora	cora.txt	cora_beta_1.25_sir_scores.pkl
PowerGrid	powergrid.txt	powergrid_beta_1.25_sir_scores.pkl
Chicago	Chicago.txt	Chicago_beta_1.25_sir_scores.pkl

Each graph file is an edge list. Each line contains one undirected edge:
node_id_1 node_id_2
The SIR label file is a Python pickle file containing a dictionary:
{
    node_id: sir_influence_score
}
where sir_influence_score is the average final infection size obtained by repeated SIR simulations with each node used as the single initial infected seed.

4. Environment Requirements
Recommended environment:
Python >= 3.8
PyTorch >= 1.12
NumPy
SciPy
NetworkX
scikit-learn
tqdm
Install dependencies:
pip install torch numpy scipy networkx scikit-learn tqdm
If CUDA is available, the script will use GPU automatically.

5. Reproducing Experiments
Run the main script:
python My_EGKD_main.py

For questions about the implementation or reproduction, please open an issue in this repository.
