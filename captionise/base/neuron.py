import copy
from abc import ABC, abstractmethod
import os

import bittensor as bt
import tenacity
from dotenv import load_dotenv

from captionise import __version__ as version  # if you track your version here
from captionise import __spec_version__ as spec_version
from captionise.mock import MockSubtensor, MockMetagraph

from captionise.utils.config import add_args, check_config, config  # adapt if you have a global config
from captionise.utils.logger import logger 
from captionise.utils.misc import ttl_get_block


load_dotenv()

class BaseNeuron(ABC):
    """
    Base class for Bittensor miners. This class is abstract and should be inherited by a subclass. It contains the core logic for all neurons; validators and miners.

    In addition to creating a wallet, subtensor, and metagraph, this class also handles the synchronization of the network state via a basic checkpointing mechanism based on epoch length.
    """

    @classmethod
    def check_config(cls, config: "bt.Config"):
        check_config(cls, config)

    @classmethod
    def add_args(cls, parser):
        add_args(cls, parser)

    @classmethod
    def config(cls):
        return config(cls)

    subtensor: "bt.subtensor"
    wallet: "bt.wallet"
    metagraph: "bt.metagraph"
    spec_version: int = spec_version

    @property
    @tenacity.retry(
        stop=tenacity.stop_after_attempt(3),
        wait=tenacity.wait_exponential(multiplier=1, min=4, max=15),
    )
    def block(self):
        """
        Return the current chain block, with retry on failure.
        """
        return ttl_get_block(self)

    def __init__(self, config=None):
        """
        The constructor handles:
         - Merging config 
         - Setting up device, wallet, subtensor, metagraph
         - Checking registration 
         - Setting self.uid if the wallet is a registered hotkey 
        """
        # Merge provided config with default
        base_config = copy.deepcopy(config or BaseNeuron.config())
        self.config = self.config()
        self.config.merge(base_config)
        self.check_config(self.config)

        self.device = self.config.get("neuron.device", "cpu")

        logger.info(f"Using device: {self.device}")


        logger.info("Setting up Bittensor wallet/subtensor/metagraph.")
        if self.config.mock:
            self.wallet = bt.MockWallet(config=self.config)
            self.subtensor = MockSubtensor(self.config.netuid, wallet=self.wallet)
            self.metagraph = MockMetagraph(self.config.netuid, subtensor=self.subtensor)
        else:
            self.wallet = bt.wallet(config=self.config)
            self.subtensor = bt.subtensor(config=self.config)
            self.metagraph = self.subtensor.metagraph(self.config.netuid, lite=False)

        logger.info(f"Wallet: {self.wallet}")
        logger.info(f"Subtensor: {self.subtensor}")
        logger.info(f"Metagraph: {self.metagraph}")

        # Enforce registration check
        self.check_registered()

        # Each neuron has a unique UID in the metagraph
        self.uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
        logger.info(
            f"Running neuron on subnet: {self.config.netuid} with uid {self.uid} "
            f"on endpoint: {self.subtensor.chain_endpoint}"
        )

        self.step = 0  # track steps or epochs if desired

        # Get the path of the project folder
        self.project_path: str = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        
        logger.info(f"Spec version: {self.spec_version} | code version: {version}")

    @abstractmethod
    async def forward(self, synapse: bt.Synapse) -> bt.Synapse:
        """
        Subclasses must implement their main forward logic for tasks.
        """
        ...

    def sync(self):
        """
        Synchronize the network state for a miner or validator:
          - check registration
          - (optionally) resync metagraph
          - (optionally) set weights
          - save state
        """
        self.check_registered()

        if self.should_sync_metagraph():
            self.resync_metagraph()

        if self.should_set_weights():
            self.weight_setter()

        self.save_state()

    def weight_setter(self):
        """Tries setting weights on the chain, with possible retries."""
        try:
            weights_are_set = self.set_weights()
            if weights_are_set:
                logger.success("Weight setting successful!")
        except tenacity.RetryError as e:
            logger.error(
                f"Failed to set weights after retry attempts. Skipping for {self.config.neuron.epoch_length} blocks."
            )

    def check_registered(self):
        """
        Checks if the wallet hotkey is registered on the chain.
        Exits if not.
        """
        if not self.subtensor.is_hotkey_registered(
            netuid=self.config.netuid,
            hotkey_ss58=self.wallet.hotkey.ss58_address,
        ):
            logger.error(
                f"Wallet: {self.wallet.hotkey.ss58_address} not registered on netuid {self.config.netuid}. "
                "Please register the hotkey first."
            )
            exit()

    def should_sync_metagraph(self) -> bool:
        """
        Decide if enough blocks have elapsed to justify a metagraph resync.
        For example, if self.block - self.metagraph.last_update[self.uid] > X
        """
        return (
            self.block - self.metagraph.last_update[self.uid]
        ) > self.config.neuron.metagraph_resync_length

    def should_set_weights(self) -> bool:
        """
        Decide if enough blocks have elapsed to set new weights, or if disabled.
        E.g. only set weights if not self.config.neuron.disable_set_weights, 
             and the node is a validator, etc.
        """
        # e.g. skip on the first step
        if self.step == 0:
            return False

        # Check if enough epoch blocks have elapsed since the last epoch.
        if self.config.neuron.disable_set_weights:
            return False

        # Do not allow weight setting if the neuron is not a validator.
        if not self.metagraph.validator_permit[self.uid]:
            return False

        # Define appropriate logic for when set weights.
        return (
            self.block - self.metagraph.last_update[self.uid]
        ) > self.config.neuron.epoch_length

    def save_state(self):
        """
        Save any checkpoint or data needed by your miner/validator.
        E.g. model weights, progress. Not implemented here.
        """
        pass

    def load_state(self):
        """
        Load any prior state from disk or from a distributed store.
        Not implemented by default.
        """
        logger.warning("load_state() not implemented in BaseNeuron.")


