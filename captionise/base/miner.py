import asyncio
import threading
import argparse
import traceback

import bittensor as bt

from captionise.base.neuron import BaseNeuron
from captionise.protocol import PingSynapse, ParticipationSynapse
from captionise.utils.logger import logger
from captionise.utils.config import add_miner_args


class BaseMinerNeuron(BaseNeuron):  # Or rename to something like CaptioniseBaseMiner
    """
    Base class for Bittensor miners in the Captionise project.
    """

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        """
        Optionally add command-line args for your miner.
        For example:
            parser.add_argument('--max_workers', type=int, default=1, help='max concurrency')
        """
        super().add_args(parser)
        add_miner_args(cls, parser)

    def __init__(self, config=None):
        """
        Initialize your base miner neuron. Typically:
          - Load config
          - Create wallet
          - Create and configure axon
          - Possibly attach forward functions
        """ 
        super().__init__(config=config)

         # Warn if allowing incoming requests from anyone.
        if not self.config.blacklist.force_validator_permit:
            logger.warning(
                "You are allowing non-validators to send requests to your miner. This is a security risk."
            )
        if self.config.blacklist.allow_non_registered:
            logger.warning(
                "You are allowing non-registered entities to send requests to your miner. This is a security risk."
            )

        # Create the axon
        self.axon = bt.axon(
            wallet=self.wallet,
            config=self.config,
        )

        # Attach your forward functions
        logger.info("Attaching forward function(s) to miner axon.")
        self.axon.attach(
            forward_fn=self.forward,
            blacklist_fn=self.blacklist,
            priority_fn=self.priority,
        ).attach(
            forward_fn=self.ping_forward,
        ).attach(
            forward_fn=self.participation_forward,
        )
        logger.info(f"Axon created: {self.axon}")

        # Runners
        self.should_exit: bool = False
        self.is_running: bool = False
        self.thread: threading.Thread = None
        self.lock = asyncio.Lock()

    def ping_forward(self, synapse: PingSynapse):
        """
        Respond to a 'ping' request from a validator or node, letting them
        know if you are able to serve new tasks.

        Suppose synapse has attributes like:
          - synapse.available_compute
          - synapse.can_serve
        Then adapt as needed.
        """
        logger.info(f"Received ping request from {synapse.dendrite.hotkey[:8]}")

        synapse.available_compute = self.max_workers - len(self.simulations)

         # TODO: add more conditions.
        if synapse.available_compute > 0:
            synapse.can_serve = True
            logger.success("Telling validator we can serve ✅")
        return synapse

    def participation_forward(self, synapse: ParticipationSynapse):
        """
        A stub for handling requests about 'participation' in a specific job/task.
        Typically, you'd update synapse fields to indicate if you're joining or not.
        """
        logger.debug("Received participation request.")
        pass

    def run(self):
        pass

    def run_in_background_thread(self):
        """
        Starts the miner in a separate background thread for non-blocking operation.
        """
        if not self.is_running:
            logger.debug("Starting miner in background thread.")
            self.should_exit = False
            self.thread = threading.Thread(target=self.run, daemon=True)
            self.thread.start()
            self.is_running = True
            logger.debug("Miner started in background.")

    def stop_run_thread(self):
        """
        Gracefully stop the background thread.
        """
        if self.is_running:
            logger.debug("Stopping miner in background thread.")
            self.should_exit = True
            self.thread.join(5)
            self.is_running = False
            logger.debug("Miner stopped in background.")

    def __enter__(self):
        """
        Allows 'with BaseMinerNeuron() as miner:' usage,
        automatically starting the background thread.
        """
        self.run_in_background_thread()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        On exiting the 'with' context, stop the background thread.
        Args:
            exc_type: The type of the exception that caused the context to be exited.
                      None if the context was exited without an exception.
            exc_value: The instance of the exception that caused the context to be exited.
                       None if the context was exited without an exception.
            traceback: A traceback object encoding the stack trace.
                       None if the context was exited without an exception.
        """
        self.stop_run_thread()

    def resync_metagraph(self):
        """
        If your miner needs to resync the metagraph for updated staking data, etc.
        """
        logger.info("resync_metagraph()")
        if self.metagraph and self.subtensor:
            self.metagraph.sync(subtensor=self.subtensor)
        else:
            logger.warning("No metagraph/subtensor found. Skipping sync.")

    def set_weights(self):
        pass
