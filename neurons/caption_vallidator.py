import time
import random
import jiwer
from typing import List
from base.validator import BaseValidator
from captionise.protocol import CaptionSynapse, CaptionSegment
import bittensor as bt

class CaptionValidator(BaseValidator):
    def __init__(self, wallet: bt.wallet, config: bt.Config):
        super().__init__(wallet, config)
        self.alpha_time = 0.2  # weighting factor for time-based reward
        self.alpha_wer = 0.8   # weighting factor for WER-based reward
        bt.logging.success("CaptionValidator initialized.")

    def evaluate_caption(self, reference_text: str, segments: List[CaptionSegment], elapsed: float) -> float:
        """Compute a final [0..1] score for a single miner's transcript."""
        predicted_text = " ".join([seg.text.strip() for seg in segments])
        # WER (0 is perfect, 1 is all wrong). We want a reward = 1 - WER
        wer_value = jiwer.wer(reference_text, predicted_text)
        caption_reward = max(0, 1 - wer_value)

        # Time reward: if they respond quickly (say under 5s), we boost the score
        max_time = 5.0
        time_reward = max(0, 1 - (elapsed / max_time))

        # Weighted combo
        final_score = self.alpha_wer * caption_reward + self.alpha_time * time_reward
        bt.logging.debug(f"Caption Score => WER: {wer_value:.3f}, cReward: {caption_reward:.3f}, timeR: {time_reward:.3f}, final: {final_score:.3f}")
        return final_score

    def run_validation_round(self, base64_audio: str, reference_text: str, lang="en"):
        """
        Example flow:
        1. Build a CaptionSynapse
        2. Send to random miners
        3. Evaluate responses
        4. Set weights
        """
        syn = CaptionSynapse(base64_audio=base64_audio, language=lang)
        results = self.query_miners(synapse=syn, uids=None, timeout=10)

        weight_map = {}
        for uid, (resp, elapsed) in results.items():
            score = 0.0
            if resp and isinstance(resp, CaptionSynapse) and resp.segments:
                score = self.evaluate_caption(reference_text, resp.segments, elapsed)
            weight_map[uid] = score

        # Normalizes and sets on-chain
        self.set_weights_on_chain(weight_map)
