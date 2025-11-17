import os
import argparse

def get_parser():
    parser = argparse.ArgumentParser()

    # ---training params---
    parser.add_argument("--log_path", type=str, default="./log")
    parser.add_argument("--dataset", type=str, default="sst2")
    parser.add_argument("--save_path", type=str, default="./trained_model") 
    parser.add_argument("--model_type", type=str, default="bert-base-uncased")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--num_labels", type=float, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_len",type=int,default=128)
    parser.add_argument("--batch_size",type=int,default=64)
    parser.add_argument("--use_cuda", type=bool, default=True)
    parser.add_argument("--num_workers",type=int,default=os.cpu_count())
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--log_steps",type=int,default=10) 
    parser.add_argument("--eval_steps",type=int,default=10)

    # ---CusText params---
    parser.add_argument("--eps", type=float, default=1.0)
    parser.add_argument("--embedding_type", type=str, default="cf-vectors")
    parser.add_argument("--mapping_strategy", type=str, default="conservative")
    parser.add_argument("--privatization_strategy", type=str, default="s1")
    parser.add_argument("--save_stop_words", type=bool, default=False)
    parser.add_argument("--mapping_algorithm", type=str, default="entropy", choices=["entropy"])
    parser.add_argument("--lambda_reg", type=float, default=0.2)
    parser.add_argument("--max_iter", type=int, default=10)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--cluster_prob", type=str, default="uniform", choices=["uniform","softmax"])

    parser.add_argument("--viz_entropy", action="store_true",help="Run visualization after mapping")
    parser.add_argument("--reports_dir", type=str, default="./reports/entropy_kmeans")
    parser.add_argument("--data_dir", type=str, default="./datasets")
    parser.add_argument("--embeddings_dir", type=str, default="./embeddings")
    parser.add_argument("--sim_root", type=str, default="./sim_word_dict")
    parser.add_argument("--viz_max_points", type=int, default=3000)
    parser.add_argument("--annotate_top_n", type=int, default=0)
    parser.add_argument("--cluster_dump_dir", type=str, default="./cluster_members",help="Where to write center→members dumps (CSV/JSONL).")
    parser.add_argument("--text_col", type=str, default=None,help="Name of the text column in TSVs (auto-detect if not set).")
    parser.add_argument("--num_centroids", type=int, default=60)
    parser.add_argument("--softmax_temp", type=float, default=1.0,help="Temperature for replacement softmax (smaller = peakier toward closer tokens).")
    parser.add_argument("--dist_entropy_c", type=float, default=0.5,help="Penalty weight c in score = -(distance + c*H_fixed).")
    return parser
