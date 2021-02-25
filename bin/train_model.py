import argparse
import json
import logging
import os
import pkg_resources
from pathlib import Path

from assistive_gym.learn import train

from riddbot.gym import make_env


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_ZOO_DIR = Path(pkg_resources.resource_filename("riddbot", "model_zoo"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_path", type=Path, required=True)
    return parser.parse_args()


def read_config(config_path: Path) -> dict:
    logger.info(f"Reading config from {config_path}")
    with open(config_path, "r") as f:
        config = json.load(f)
    return config


def create_new_model_dir() -> Path:
    num_existing_models = len(list(MODEL_ZOO_DIR.glob("version_*")))
    new_model_dir = MODEL_ZOO_DIR / f"version_{num_existing_models:03}"
    new_model_dir = new_model_dir.absolute()
    os.mkdir(new_model_dir)

    logger.info(f"Saving model to {new_model_dir}")
    return new_model_dir


def main():
    args = parse_args()

    config = read_config(args.config_path.absolute())
    model_dir = create_new_model_dir()
    with open(model_dir / "config.json", "w") as f:
        json.dump(config, f, indent=4)

    env_name, env = make_env(reward_weights=config["reward_weights"])

    train(
        env_name,
        algo=config["algo"],
        timesteps_total=config["train_timesteps"],
        save_dir=str(model_dir),
        load_policy_path=str(model_dir),
        seed=config["seed"],
        coop=False,
        extra_configs=dict(env_config=dict(reward_weights=config["reward_weights"])),
    )


if __name__ == "__main__":
    main()
