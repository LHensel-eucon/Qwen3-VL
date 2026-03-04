import argparse
import logging
import os
import shlex
import subprocess
import sys
from pathlib import Path

import mlflow


def cli_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--train_frac", type=str)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--context_length", type=int)
    parser.add_argument("--model_out", type=str)
    parser.add_argument("--position_data", type=str)
    parser.add_argument("--image_data", type=str)
    # parser.add_argument("--cache_dir", type=str)

    return parser.parse_args()


def main():
    # Logging setup
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
    logger = logging.getLogger(__name__)
    args = cli_args()

    # -------- Confirm parameters
    logger.info(f"Train fraction: {args.train_frac}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Context length: {args.context_length}")

    # -------- Confiure
    OUTPUT_DIR = Path(args.model_out)
    # CACHE_DIR = Path(args.cache_dir)
    DATASETS = f"xml_dataset%{args.train_frac}"  # Use N% of the dataset for testing

    PER_DEVICE_TRAIN_BATCH_SIZE = args.batch_size  # lower to save memory
    GRAD_ACCUM_STEPS = 2  # lower to save memory

    # Higher LR works better with small LoRA ranks
    LEARNING_RATE = 2e-4
    MM_PROJECTOR_LR = 1e-4
    VISION_TOWER_LR = 1e-6

    NUM_EPOCHS = 1
    MODEL_MAX_LENGTH = args.context_length

    # Pixel limits recommended by Qwen fine-tuning docs
    MIN_PIXELS = str(16 * 28 * 28)
    MAX_PIXELS = str(64 * 28 * 28)

    # LoRA
    LORA_ENABLE = "True"
    LORA_R = 4  # very small rank; increase for better performance
    LORA_ALPHA = 16
    LORA_DROPOUT = 0.05

    # Logging & checkpoints
    LOG_STEPS = 10
    SAVE_STEPS = 500
    SAVE_TOTAL_LIMIT = 3

    # Single GPU
    # MASTER_ADDR = "127.0.0.1"
    # MASTER_PORT = str(random.randint(20000, 29999))
    # NPROC = 1

    # No DeepSpeed for single GPU
    # DEEPSPEED_CFG = ""

    DATA_FLATTEN = "False"  # erhöht effizienz wenn an, aber momentan für debugging aus
    DATA_PACKING = "False"

    # -------- setup paths and code locations
    MODEL_PATH = Path(args.model)

    SCRIPT_DIR = Path(__file__).resolve().parent
    REPO_ROOT = str(SCRIPT_DIR)
    TRAINER = os.path.join(REPO_ROOT, "qwen-vl-finetune", "qwenvl", "train", "train_qwen.py")

    if not os.path.exists(TRAINER):
        sys.stderr.write(f"ERROR: Could not find train_qwen.py at {TRAINER}\n")
        sys.exit(1)

    # --------  Training arguments as per Qwen fine-tune documentation
    args_model = [
        # TRAINER,
        "--model_name_or_path",
        str(MODEL_PATH),
        "--tune_mm_llm",
        "True",
        "--tune_mm_vision",
        "False",
        "--tune_mm_mlp",
        "False",
        "--dataset_use",
        DATASETS,
        "--output_dir",
        str(OUTPUT_DIR),
        # "--cache_dir",
        # CACHE_DIR,
        # Runtime precision
        "--bf16",
        "--per_device_train_batch_size",
        str(PER_DEVICE_TRAIN_BATCH_SIZE),
        "--gradient_accumulation_steps",
        str(GRAD_ACCUM_STEPS),
        # Learning rates
        "--learning_rate",
        str(LEARNING_RATE),
        "--mm_projector_lr",
        str(MM_PROJECTOR_LR),
        "--vision_tower_lr",
        str(VISION_TOWER_LR),
        "--optim",
        "adamw_torch",
        # Sequence settings
        "--model_max_length",
        str(MODEL_MAX_LENGTH),
        "--data_flatten",
        DATA_FLATTEN,
        "--data_packing",
        DATA_PACKING,
        # Image constraints
        "--max_pixels",
        MAX_PIXELS,
        "--min_pixels",
        MIN_PIXELS,
        # Scheduling
        "--num_train_epochs",
        str(NUM_EPOCHS),
        "--warmup_steps",
        "2",
        "--lr_scheduler_type",
        "cosine",
        "--weight_decay",
        "0.01",
        # Logging
        "--logging_steps",
        str(LOG_STEPS),
        "--save_steps",
        str(SAVE_STEPS),
        "--save_total_limit",
        str(SAVE_TOTAL_LIMIT),
        # LoRA
        "--lora_enable",
        LORA_ENABLE,
        "--lora_r",
        str(LORA_R),
        "--lora_alpha",
        str(LORA_ALPHA),
        "--lora_dropout",
        str(LORA_DROPOUT),
    ]

    # torchrun_cmd = [
    #   sys.executable, "-m", "torch.distributed.run",
    #  f"--nproc_per_node={NPROC}",
    #   f"--master_addr={MASTER_ADDR}",
    #   f"--master_port={MASTER_PORT}",
    # ] + args_model

    # after parsing args
    POSITION_PATH = Path(args.position_data) if args.position_data else None
    IMAGE_PATH = Path(args.image_data) if args.image_data else None

    env = os.environ.copy()
    if POSITION_PATH is not None:
        env["AZUREML_INPUT_position_data"] = str(POSITION_PATH)
    if IMAGE_PATH is not None:
        env["AZUREML_INPUT_image_data"] = str(IMAGE_PATH)

    torchrun_cmd = [sys.executable, TRAINER] + args_model  # Single GPU → do NOT use torch.distributed.run

    logger.info("\n[Launching training]")
    logger.info(" ".join(shlex.quote(str(t)) for t in torchrun_cmd))

    env.setdefault("PYTHONWARNINGS", "ignore")

    proc = subprocess.Popen(torchrun_cmd, env=env)
    proc.wait()

    if proc.returncode == 0:
        logger.info("[Logging model artifacts to MLflow]")
        mlflow.log_artifacts(str(OUTPUT_DIR), artifact_path="model_output")
        mlflow.log_params(
            {
                "lora_r": LORA_R,
                "lora_alpha": LORA_ALPHA,
                "lora_dropout": LORA_DROPOUT,
                "learning_rate": LEARNING_RATE,
                "num_epochs": NUM_EPOCHS,
                "model_max_length": MODEL_MAX_LENGTH,
                "batch_size": PER_DEVICE_TRAIN_BATCH_SIZE,
                "train_frac": args.train_frac,
            }
        )
        logger.info("Model artifacts logged successfully.")
    else:
        logger.error(f"Training failed with return code {proc.returncode}")

    sys.exit(proc.returncode)


if __name__ == "__main__":
    os.environ.setdefault("MLFLOW_REGISTRY_URI", "")
    with mlflow.start_run():
        main()
