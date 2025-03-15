import time
import random
import jiwer
import base64
import io
import asyncio
import bittensor as bt
import torch
from typing import List, Dict, Any
from captionise.base.validator import BaseValidatorNeuron
from captionise.protocol import CaptionSynapse, CaptionSegment
from captionise.utils.logger import logger
from captionise.utils.database import db_manager
from captionise.utils.dataset import DatasetManager

class CaptionValidator(BaseValidatorNeuron):
    def __init__(self, config=None):
        super().__init__(config=config)
        
        # Initialize database
        db_manager.initialize_schema()
        
        # Initialize dataset manager (defaults to English)
        languages = getattr(config.neuron, "languages", ["en"]) if config else ["en"]
        self.dataset_manager = DatasetManager(languages=languages)
        
        # Scoring parameters
        self.alpha_time = getattr(config.neuron, "alpha_time", 0.2) if config else 0.2
        self.alpha_wer = getattr(config.neuron, "alpha_wer", 0.8) if config else 0.8
        self.max_time = getattr(config.neuron, "max_time", 30.0) if config else 30.0
        
        # Job batching parameters
        self.batch_size = getattr(config.neuron, "batch_size", 5) if config else 5
        self.miner_sample_size = getattr(config.neuron, "sample_size", 10) if config else 10
        
        # Counter for dataset refresh
        self.dataset_refresh_interval = 100  # Refresh dataset every 100 steps
        self.dataset_counter = 0
        
        logger.success("CaptionValidator initialized")
    
    async def forward(self, synapse: bt.Synapse) -> bt.Synapse:
        """Handle incoming requests (not typically used by validators)."""
        return synapse
    
    def evaluate_caption(self, reference_text: str, transcript: str, elapsed: float) -> Dict[str, float]:
        """
        Compute scores for a miner's transcript.
        
        Args:
            reference_text: The ground truth transcript
            transcript: The miner's submitted transcript
            elapsed: Processing time in seconds
            
        Returns:
            Dictionary with various scores
        """
        # Calculate WER (lower is better)
        try:
            wer_value = jiwer.wer(reference_text, transcript)
            # Bound WER for reasonable scoring (1.0 = all wrong)
            wer_value = min(1.0, wer_value)
            # WER reward (1.0 = perfect match, 0.0 = completely wrong)
            wer_reward = 1.0 - wer_value
        except Exception as e:
            logger.error(f"Error calculating WER: {e}")
            wer_value = 1.0
            wer_reward = 0.0
        
        # Time reward: if they respond quickly we boost the score
        time_reward = max(0, 1.0 - (elapsed / self.max_time))
        
        # Weighted combination
        final_score = self.alpha_wer * wer_reward + self.alpha_time * time_reward
        
        logger.debug(f"Caption Score => WER: {wer_value:.3f}, WER reward: {wer_reward:.3f}, Time reward: {time_reward:.3f}, Final: {final_score:.3f}")
        
        return {
            "score": final_score,
            "wer": wer_value,
            "wer_reward": wer_reward,
            "time_reward": time_reward
        }
    
    async def run_validation_round(self):
        """
        Run a single validation round:
        1. Get unprocessed jobs from database
        2. Send to random miners
        3. Evaluate responses
        4. Update scores and database
        """
        # Check if we need to refresh our dataset
        if self.dataset_counter % self.dataset_refresh_interval == 0:
            logger.info("Refreshing validation dataset")
            success = self.dataset_manager.load_voxpopuli(split="validation", sample_size=50)
            if success:
                self.dataset_manager.store_dataset_in_db()
        
        self.dataset_counter += 1
        
        # Get unprocessed jobs
        jobs = self.dataset_manager.get_unprocessed_jobs(limit=self.batch_size)
        if not jobs:
            logger.warning("No unprocessed jobs found. Skipping validation round.")
            return
            
        logger.info(f"Running validation on {len(jobs)} jobs")
        
        # Select a random subset of miners to query
        available_uids = self.metagraph.uids.tolist()
        if len(available_uids) > self.miner_sample_size:
            selected_uids = random.sample(available_uids, self.miner_sample_size)
        else:
            selected_uids = available_uids
            
        # Send each job to the selected miners
        all_scores = torch.zeros(self.metagraph.n, device=self.device)
        responses = []
        
        for job in jobs:
            # Convert binary audio to base64
            base64_audio = base64.b64encode(job['audio_segment']).decode('utf-8')
            
            # Create synapse
            synapse = CaptionSynapse(
                job_id=str(job['job_id']),
                base64_audio=base64_audio,
                language=job['language']
            )
            
            # Query miners
            logger.info(f"Querying {len(selected_uids)} miners for job {job['job_id']}")
            results = await self.dendrite.forward(
                axons=[self.metagraph.axons[uid] for uid in selected_uids],
                synapse=synapse,
                timeout=self.config.neuron.timeout
            )
            
            # Process results
            for uid, result in zip(selected_uids, results):
                if result and isinstance(result, CaptionSynapse) and result.transcript:
                    # Calculate scores
                    scores = self.evaluate_caption(
                        reference_text=job['transcript_source'],
                        transcript=result.transcript,
                        elapsed=result.processing_time or self.max_time
                    )
                    
                    # Store the response for database update
                    responses.append({
                        'job_id': job['job_id'],
                        'miner_uid': uid,
                        'miner_hotkey': self.metagraph.hotkeys[uid],
                        'transcript': result.transcript,
                        'scores': scores
                    })
                    
                    # Update running score for this miner
                    all_scores[uid] += scores['score']
        
        # Normalize scores based on number of jobs
        if len(jobs) > 0:
            all_scores = all_scores / len(jobs)
        
        # Update database with results
        conn = db_manager.get_connection()
        try:
            with conn.cursor() as cursor:
                for resp in responses:
                    cursor.execute("""
                        INSERT INTO job_results
                        (job_id, miner_hotkey, score, wer, processing_time)
                        VALUES (%s, %s, %s, %s, %s)
                    """, (
                        resp['job_id'],
                        resp['miner_hotkey'],
                        resp['scores']['score'],
                        resp['scores']['wer'],
                        resp.get('processing_time', 0.0)
                    ))
                    
                    # Mark job as processed
                    cursor.execute("""
                        UPDATE jobs
                        SET processed = TRUE, transcript_submitted = %s, updated_at = CURRENT_TIMESTAMP
                        WHERE job_id = %s
                    """, (
                        resp['transcript'],
                        resp['job_id']
                    ))
                
                conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Database error saving validation results: {e}")
        finally:
            db_manager.return_connection(conn)
        
        # Update validator's score tracking
        await self.update_scores(all_scores, selected_uids)
        
        # Potentially set weights if enough blocks have passed
        if self.should_set_weights():
            success = self.set_weights()
            if success:
                logger.success("Set weights successfully")
        
        logger.info(f"Validation round complete: processed {len(jobs)} jobs")
    
    async def run_async(self):
        """Main async loop for the validator."""
        while not self.should_exit:
            try:
                # Run a validation round
                await self.run_validation_round()
                
                # Sync with the network
                self.resync_metagraph()
                
                # Save state periodically
                self.save_state()
                
                # Sleep between validation rounds
                await asyncio.sleep(self.config.neuron.update_interval)
            except Exception as e:
                logger.error(f"Error in validator main loop: {e}")
                await asyncio.sleep(30)  # Sleep longer on error
    
    def run(self):
        """Entry point for the validator's main loop."""
        self.loop.run_until_complete(self.run_async())


if __name__ == "__main__":
    # Parse config and run validator
    config = CaptionValidator.config()
    validator = CaptionValidator(config=config)
    
    with validator:
        while True:
            logger.info(
                f"Validator running:: network: {validator.subtensor.network} | netuid: {validator.config.netuid} | uid: {validator.uid}"
            )
            time.sleep(60)
