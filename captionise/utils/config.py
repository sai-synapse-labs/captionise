import os
import argparse
import bittensor as bt

# If you have a custom logger module:
from captionise.utils.logger import setup_file_logging, add_events_level



def check_config(cls, config: "bt.Config"):
    """
    Checks/validates the config namespace object, ensures relevant directories exist,
    sets up event logging, etc.
    """

    # Build a path for storing local data/logs, e.g. "~/.bittensor/miners/wallet/hotkey/netuid/neuron_name"
    full_path = os.path.expanduser(
        f"{os.path.join('~/.bittensor/miners', config.wallet.name, config.wallet.hotkey, 'netuid' + str(config.netuid), config.neuron.name)}"
    )
    config.neuron.full_path = os.path.expanduser(full_path)

    if not os.path.exists(config.neuron.full_path):
        os.makedirs(config.neuron.full_path, exist_ok=True)

    # Optionally set up event logging if you want local logs
    if not config.neuron.dont_save_events:
        add_events_level()
        setup_file_logging(
            os.path.join(config.neuron.full_path, "events.log"),
            config.neuron.events_retention_size,
        )
        pass


def add_args(cls, parser):
    """
    Adds general arguments for the Bittensor wallet, subtensor, axon, etc.
    You can keep or remove any that you no longer need in a captioning context.
    """

    parser.add_argument("--netuid", type=int, help="Subnet netuid", default=25)

    parser.add_argument(
        "--neuron.device",
        type=str,
        help="Device to run on (e.g. cpu, cuda:0).",
        default="cpu",
    )
    parser.add_argument(
        "--neuron.metagraph_resync_length",
        type=int,
        help="Number of blocks until metagraph is resynced.",
        default=100,
    )
    parser.add_argument(
        "--neuron.epoch_length",
        type=int,
        help="How often (in 12s blocks) we set weights or do major sync steps.",
        default=150,
    )

    # If you use a mock environment for testing:
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Mock neuron and network components for local testing.",
        default=False,
    )
    parser.add_argument(
        "--neuron.mock",
        action="store_true",
        help="Alias for mock mode.",
        default=False,
    )

    # Additional arguments for basic logging
    parser.add_argument(
        "--neuron.events_retention_size",
        type=str,
        help="Events log retention size, e.g. '25 MB'.",
        default="25 MB",
    )
    parser.add_argument(
        "--neuron.dont_save_events",
        action="store_true",
        help="If set, we don’t save events to a log file.",
        default=False,
    )

    # Minimal example of wandb toggles if you use it:
    parser.add_argument(
        "--wandb.off",
        action="store_true",
        help="Turn off wandb logging.",
        default=False,
    )
    parser.add_argument(
        "--wandb.offline",
        action="store_true",
        help="Runs wandb in offline mode.",
        default=False,
    )
    parser.add_argument(
        "--wandb.notes",
        type=str,
        help="Notes for wandb run.",
        default="",
    )

    # Add more arguments as needed for your “Captionise” tasks (e.g. concurrency, text, STT-related flags, etc.)


def add_miner_args(cls, parser):
    """
    Adds miner-specific arguments to the parser.
    """
    parser.add_argument(
        "--neuron.name",
        type=str,
        help="Unique name for this miner’s local data folder.",
        default="miner",
    )

    parser.add_argument(
        "--neuron.suppress_cmd_output",
        action="store_true",
        help="Suppress text output of certain commands to reduce clutter.",
        default=True,
    )

    parser.add_argument(
        "--neuron.max_workers",
        type=int,
        help="Number of concurrent tasks or processes this miner can run.",
        default=8,
    )

    parser.add_argument(
        "--blacklist.force_validator_permit",
        action="store_true",
        help="If set, only allow requests from entities with a validator permit.",
        default=False,
    )

    parser.add_argument(
        "--blacklist.allow_non_registered",
        action="store_true",
        help="If set, miners accept queries from non-registered hotkeys. (Potentially risky!)",
        default=False,
    )

    parser.add_argument(
        "--wandb.project_name",
        type=str,
        default="captionise-miners",
        help="Wandb project name if you use wandb for logging.",
    )
    parser.add_argument(
        "--wandb.entity",
        type=str,
        default="your-wandb-entity",
        help="Wandb entity/org name.",
    )


def add_validator_args(cls, parser):
    """
    Adds validator-specific arguments to the parser.
    """
    parser.add_argument(
        "--neuron.name",
        type=str,
        help="Unique name for this validator’s local data folder.",
        default="validator",
    )
    parser.add_argument(
        "--neuron.timeout",
        type=float,
        help="Timeout for forward calls (in seconds).",
        default=45,
    )
    parser.add_argument(
        "--neuron.ping_timeout",
        type=float,
        help="Timeout for ping calls (in seconds).",
        default=45,
    )
    parser.add_argument(
        "--neuron.update_interval",
        type=float,
        help="Interval (in seconds) between validator queries to miners.",
        default=60,
    )
    parser.add_argument(
        "--neuron.queue_size",
        type=int,
        help="Number of jobs to keep in the validator queue.",
        default=10,
    )
    parser.add_argument(
        "--neuron.sample_size",
        type=int,
        help="Number of miners to query in a single step.",
        default=10,
    )
    parser.add_argument(
        "--neuron.disable_set_weights",
        action="store_true",
        help="Disable setting weights on-chain for this validator.",
        default=False,
    )
    parser.add_argument(
        "--neuron.moving_average_alpha",
        type=float,
        help="Moving average alpha for scores or metrics.",
        default=0.1,
    )
    parser.add_argument(
        "--neuron.axon_off",
        action="store_true",
        help="If set, the validator won’t attempt to serve an Axon.",
        default=False,
    )
    parser.add_argument(
        "--neuron.vpermit_tao_limit",
        type=int,
        help="Max TAO for queries with a vpermit.",
        default=4096,
    )
    parser.add_argument(
        "--wandb.project_name",
        type=str,
        help="Wandb project name for validators.",
        default="captionise-validators",
    )
    parser.add_argument(
        "--wandb.entity",
        type=str,
        help="Wandb entity/org for validators.",
        default="your-wandb-entity",
    )


def config(cls):
    """
    Returns a Bittensor config object for the neuron (miner or validator)
    after adding relevant arguments from the base and from Bittensor modules.
    """
    parser = argparse.ArgumentParser()
    bt.wallet.add_args(parser)
    bt.subtensor.add_args(parser)
    # If you want to add axon or dendrite CLI arguments:
    bt.axon.add_args(parser)
    # bt.dendrite.add_args(parser)

    # Add base arguments
    cls.add_args(parser)
    return bt.config(parser)
