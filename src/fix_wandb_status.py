import wandb

run = wandb.init(
    entity="raoufkessouar-sorbonne-universit-",
    project="Article5-DVSGC-SResNet",
    id="5oniqllg",
    resume="must",
)

run.summary["epoch"] = 100
run.summary["lr"] = 1e-4
run.summary["train/loss"] = 0.0237
run.summary["train/accuracy"] = 0.9972
run.summary["val/loss"] = 0.0408
run.summary["val/accuracy"] = 0.9864
run.summary["best_val_accuracy"] = 0.9957
run.summary["elapsed_hours"] = 147.17

wandb.finish(exit_code=0)
