# The MIT License (MIT)
# Copyright © 2024 ...
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software
# and associated documentation files (the “Software”), to deal in the Software without
# restriction, including without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom
# the Software is furnished to do so, subject to the following conditions:
#
# [ Full MIT license text here ]
#
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import os
import json
import copy
import torch
import asyncio
import argparse
import threading
import numpy as np
import bittensor as bt
from pathlib import Path
from typing import List, Optional

# If you have a custom logger:
from captionise.utils.logger import logger


# If you have a mock environment:
from captionise.mock import MockDendrite
from captionise.base.neuron import BaseNeuron
from captionise.utils.config import add_validator_args
# from captionise.base.miner_registry import MinerRegistry # if you use a registry approach
import tenacity

ROOT_DIR = Path(__file__).resolve().parents[2]


class BaseValidatorNeuron(BaseNeuron):  # or if you have your own BaseNeuron class
    """
    Base class for a Bittensor validator node in the Captionise subnet.
    Manages scoring of miner outputs, setting on-chain weights, and optional data synchronization.
    """

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        super().add_args(parser)
        add_validator_args(cls, parser)

    def __init__(self, config=None):
        super().__init__(config=config)

        # Copy the hotkeys from the metagraph for local usage
        self.hotkeys = copy.deepcopy(self.metagraph.hotkeys)

        # If you’re in mock mode, we might set up a MockDendrite
        if getattr(self.config, "mock", False):
            # from captionise.mock import MockDendrite
            self.dendrite = MockDendrite(wallet=self.wallet)
        else:
            self.dendrite = bt.dendrite(wallet=self.wallet)
        logger.info(f"Dendrite: {self.dendrite}")

        # Score array for each miner UID (exponential moving average, etc.)
        self.device = self.config.get("neuron.device", "cpu")
        # Set up initial scoring weights for validation
        logger.info("Building validation weights.")
        self.scores = torch.zeros(self.metagraph.n, dtype=torch.float32, device=self.device)

        # Serve axon to enable external connections.
        if not self.config.neuron.axon_off:
            self.axon = bt.axon(wallet=self.wallet, config=self.config)
            self._serve_axon()  # start serving
        else:
            logger.warning("Axon off, not serving IP to chain.")

        # Create event loop for async tasks
        self.loop = asyncio.get_event_loop()

        # Additional runner attributes
        self.should_exit: bool = False
        self.is_running: bool = False
        self.thread: threading.Thread = None
        self.lock = asyncio.Lock()

        # If we have a registry for known miners, set it up
        self.miner_registry = None
        if self.config.neuron.miner_registry:
            # from captionise.base.miner_registry import MinerRegistry
            # self.miner_registry = MinerRegistry(...)
            logger.info("Miner registry is enabled (placeholder).")

        # Load any existing local state or config merges
        self.load_and_merge_configs()
        self.load_state()  # if you have a saved checkpoint

    @tenacity.retry(
        stop=tenacity.stop_after_attempt(3),
        wait=tenacity.wait_exponential(multiplier=1, min=4, max=15),
    )
    def _serve_axon(self):
        """Serve Axon to the chain so peers can discover and query us."""
        validator_uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
        logger.info(f"Serving validator IP of UID {validator_uid} to chain...")

        self.axon.serve(netuid=self.config.netuid, subtensor=self.subtensor).start()

    @tenacity.retry(
        stop=tenacity.stop_after_attempt(3),
        wait=tenacity.wait_fixed(1),
        retry=tenacity.retry_if_result(lambda result: result is False),
    )
    def set_weights(self):
        """
        Sets validator weights on-chain based on self.scores.
        This is how the validator incentivizes the best miners.
        """
        if self.config.neuron.disable_set_weights:
            logger.warning("Weight-setting is disabled. Skipping set_weights call.")
            return False

        logger.info("Attempting to set weights...")

        # If the scores contain NaN, we warn and zero them out
        if torch.isnan(self.scores).any():
            logger.warning("Scores contain NaN values. Replacing with zeros.")
            self.scores = torch.nan_to_num(self.scores, nan=0.0)

        # Normalize with PyTorch and convert to CPU numpy
        raw_weights = torch.nn.functional.normalize(self.scores, p=1, dim=0).cpu().numpy()
        logger.debug(f"Raw weights: {raw_weights}")

        # Bittensor utility to process & clamp the weights
        processed_uids, processed_weights = bt.utils.weight_utils.process_weights_for_netuid(
            uids=self.metagraph.uids,
            weights=raw_weights,
            netuid=self.config.netuid,
            subtensor=self.subtensor,
            metagraph=self.metagraph,
        )

        # Convert to uint16 for final on-chain submission
        uint_uids, uint_weights = bt.utils.weight_utils.convert_weights_and_uids_for_emit(
            uids=processed_uids,
            weights=processed_weights
        )

        logger.debug(f"Processed weights: {processed_weights}")
        logger.debug(f"Processed uids: {processed_uids}")

        # Submit transaction
        result = self.subtensor.set_weights(
            wallet=self.wallet,
            netuid=self.config.netuid,
            uids=uint_uids,
            weights=uint_weights,
            wait_for_finalization=False,
            wait_for_inclusion=False,
            version_key=self.spec_version,
        )

        logger.debug(f"Set weights result: {result}")
        return result[0]  # typically a bool

    def resync_metagraph(self):
        """
        Re-sync the metagraph from chain to detect changes in hotkeys, new UIDs, etc.
        Zero out or reset scores for replaced or new hotkeys as needed.
        """
        logger.info("Resyncing metagraph.")
        old_metagraph = copy.deepcopy(self.metagraph)
        self.metagraph.sync(subtensor=self.subtensor)

        # Compare axons or hotkeys to see if changed
        if old_metagraph.axons == self.metagraph.axons:
            return

        logger.info("Metagraph updated, re-syncing hotkeys, dendrite pool and moving averages")
        for uid, old_hotkey in enumerate(self.hotkeys):
            if uid >= len(self.metagraph.hotkeys):
                # The network shrunk, or we have fewer hotkeys now
                break

            if old_hotkey != self.metagraph.hotkeys[uid]:
                # This hotkey is replaced
                self.scores[uid] = 0
                # If you track a miner registry or other state, reset it
                # self.miner_registry.reset(uid) # if you have a miner registry

        # If the net grew, we need bigger arrays
        if len(self.hotkeys) < len(self.metagraph.hotkeys):
            new_size = self.metagraph.n
            expanded_scores = torch.zeros(new_size, dtype=torch.float32, device=self.device)
            min_len = min(len(self.scores), len(self.hotkeys))
            expanded_scores[:min_len] = self.scores[:min_len]
            self.scores = expanded_scores

        self.hotkeys = copy.deepcopy(self.metagraph.hotkeys)

    async def update_scores(self, rewards: torch.FloatTensor, uids: List[int]):
        """
        Updates self.scores via exponential moving average using `rewards`.
        For example, if alpha=0.1, new_score = alpha*rewards + (1-alpha)*old_score.
        """
        if torch.isnan(rewards).any():
            logger.warning(f"NaN in rewards: {rewards}")
            rewards = torch.nan_to_num(rewards, nan=0.0)

        if not isinstance(uids, torch.Tensor):
            uids_tensor = torch.tensor(uids, dtype=torch.long, device=self.device)
        else:
            uids_tensor = uids.to(self.device)

        # Scatter the reward values into the self.scores
        updated_scores = self.scores.clone()
        updated_scores[uids_tensor] = rewards

        # Exponential moving average
        alpha = self.config.neuron.moving_average_alpha
        self.scores = alpha * updated_scores + (1.0 - alpha) * self.scores

    def save_state(self):
        """Persist validator’s current step, scores, hotkeys, or other data to disk."""
        logger.info("Saving validator state.")
        state_file = os.path.join(self.config.neuron.full_path, "state.pt")

        torch.save(
            {
                "scores": self.scores,
                "hotkeys": self.hotkeys,
                # "step": self.step, # if you track a step
            },
            state_file
        )
        logger.info(f"State saved at: {state_file}")

        # If you have a miner registry:
        # self.miner_registry.save_registry(os.path.join(...))

    def load_state(self):
        """Load validator state from disk if it exists."""
        state_file = os.path.join(self.config.neuron.full_path, "state.pt")
        try:
            loaded_state = torch.load(state_file, map_location=self.device)
            self.scores = loaded_state["scores"]
            self.hotkeys = loaded_state["hotkeys"]
            # self.step = loaded_state.get("step", 0)
            logger.info("Validator state loaded from file.")
        except FileNotFoundError:
            logger.info("No saved state found; starting fresh.")
            # Optionally init self.scores from chain or set everything to zero
            # self.scores = self.get_chain_weights()

        # If you have a miner registry:
        # try:
        #     self.miner_registry = MinerRegistry.load_registry(...path...)
        # except FileNotFoundError:
        #     logger.info("No miner registry found, starting a new one.")

    def get_chain_weights(self) -> torch.Tensor:
        """
        Example method to initialize local scores from chain data.
        Weighted average of all validator weights. 
        """
        valid_indices = np.where(self.metagraph.validator_permit)[0]
        if len(valid_indices) == 0:
            logger.warning("No validators with permit found in metagraph. Returning zeros.")
            return torch.zeros(self.metagraph.n, dtype=torch.float32, device=self.device)

        valid_weights = self.metagraph.weights[valid_indices]
        valid_stakes = self.metagraph.S[valid_indices]
        stakes_sum = np.sum(valid_stakes)
        if stakes_sum == 0:
            # fallback
            logger.warning("No stake found among validators, returning zeros.")
            return torch.zeros(self.metagraph.n, dtype=torch.float32, device=self.device)

        normalized_stakes = valid_stakes / stakes_sum
        # Weighted average across valid validators
        chain_init = np.dot(normalized_stakes, valid_weights)
        return torch.tensor(chain_init, dtype=torch.float32).to(self.device)

    def load_config_json(self):
        """
        If you want to merge additional config from a local JSON file. 
        Otherwise, omit this or adapt it as needed for your project.
        """
        config_json_path = os.path.join(str(ROOT_DIR), "captionise/utils/config_input.json")
        if not os.path.isfile(config_json_path):
            logger.warning("No config_input.json found. Skipping additional merges.")
            return {}
        with open(config_json_path, "r") as file:
            conf = json.load(file)
        return conf

    def load_and_merge_configs(self):
        """
        Optionally merges config from a local JSON file into self.config.
        """
        json_conf = self.load_config_json()
        if not json_conf:
            return
        logger.info("Merging config from config_input.json.")
        for k, v in json_conf.items():
            # Example: if self.config.neuron is a dictionary
            if k not in self.config.neuron:
                self.config.neuron[k] = v
