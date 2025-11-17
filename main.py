import os
# ---- make sure TF/Flax are never pulled in by transformers (fixes your earlier DLL errors) ----
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import json
import datetime
import torch

from args import get_parser
import utils as U                  # all project helpers live here
from logger import get_logger
from torch.utils.data import DataLoader

# (Optional) only import training stack if we actually do training
DO_TRAIN = bool(int(os.getenv("DO_TRAIN", "0")))
if DO_TRAIN:
    # Keep imports lazy to avoid heavy deps when only mapping/viz is needed
    from torch.optim import AdamW        # use torch's AdamW, not transformers'
    from transformers import (
        BertForSequenceClassification,
        get_linear_schedule_with_warmup,
    )

def main():
    # ---------------------- args & logging ----------------------
    parser = get_parser()
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Stable tag used by utils everywhere (matches what utils saves/loads)
    run_tag = U._mapping_tag(args, 16)

    logger = get_logger(
        log_path=args.log_path,
        log_file=f"{args.embedding_type}_{args.mapping_strategy}_{args.privatization_strategy}_{run_tag}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt",
    )
    logger.info(f"[args] {args}")

    # ---------------------- load dataset -----------------------
    train_df, dev_df, test_df = U.load_data(args.dataset)
    logger.info(f"[data] train/dev/test sizes = {len(train_df)}/{len(dev_df)}/{len(test_df)}")
    # quick peek (first 3 rows)
    try:
        logger.info("[data] train.head(3):\n" + train_df.head(3).to_string(index=False))
    except Exception:
        pass

    # ------------------ load-or-build mapping ------------------
    sim_dir = os.path.join("./sim_word_dict", args.embedding_type, args.mapping_strategy)
    p_dir   = os.path.join("./p_dict",       args.embedding_type, args.mapping_strategy)
    os.makedirs(sim_dir, exist_ok=True)
    os.makedirs(p_dir,  exist_ok=True)

    sim_path = os.path.join(sim_dir, f"{run_tag}.txt")
    p_path   = os.path.join(p_dir,   f"{run_tag}.txt")

    if os.path.exists(sim_path) and os.path.exists(p_path):
        with open(sim_path, "r", encoding="utf-8") as f:
            sim_word_dict = json.load(f)
        with open(p_path, "r", encoding="utf-8") as f:
            p_dict = json.load(f)
        logger.info(f"[mapping] loaded artifacts:\n  {sim_path}\n  {p_path}")
    else:
        logger.info("[mapping] artifacts missing; building via entropy-only soft clustering …")
        sim_word_dict, p_dict = U.get_customized_mapping(eps=args.eps)
        # (utils already saves them using the same tag/dirs)

    # ------------------ optional visualization -----------------
    if getattr(args, "viz_entropy", False):
        logger.info("[viz] generating 2D/3D cluster plots + metrics …")
        try:
            U.viz_entropy_clusters(args)  # saves under args.reports_dir
        except Exception as e:
            logger.info(f"[viz] skipped (reason: {e})")

    # ------------------ privatize (S1) if requested ------------
    if args.privatization_strategy == "s1":
        logger.info("[priv] applying S1 on train/test using learned soft clusters …")
        train_df = U.generate_new_sents_s1(
            df=train_df, sim_word_dict=sim_word_dict, p_dict=p_dict,
            save_stop_words=args.save_stop_words
        )
        test_df = U.generate_new_sents_s1(
            df=test_df, sim_word_dict=sim_word_dict, p_dict=p_dict,
            save_stop_words=args.save_stop_words, type="test"
        )

    # ------------------ Torch datasets/loaders ------------------
    train_dataset = U.Bert_dataset(train_df)
    dev_dataset   = U.Bert_dataset(dev_df)
    test_dataset  = U.Bert_dataset(test_df)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    dev_loader   = DataLoader(dev_dataset,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    logger.info("[dataloader] ready.")

    # ------------------ (Optional) Training --------------------
    if not DO_TRAIN:
        logger.info("[training] skipped (set DO_TRAIN=1 to enable). Finished.")
        return


if __name__ == "__main__":
    main()

       