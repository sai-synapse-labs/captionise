# import bittensor as bt
# import whisper as ow
# import os
# import uuid
# from io import BytesIO
# from pydub import AudioSegment
# from captionise.miners.captionise_miner import CaptioniseMiner as BaseMiner
# from captionise.protocol import CaptionSynapse, CaptionSegment

# class CaptionMiner(BaseMiner):
#     def __init__(self, wallet: bt.wallet, config: bt.Config):
#         super().__init__(wallet, config)
#         # Load a whisper model
#         model_name = "base"  # or "medium", "large", etc.
#         self.model = ow.load_model(model_name)
#         bt.logging.success(f"Loaded Whisper model: {model_name}")

#     async def forward(self, synapse: bt.Synapse) -> bt.Synapse:
#         """Handle incoming CaptionSynapse requests."""
#         if isinstance(synapse, CaptionSynapse):
#             try:
#                 audio_seg = self.decode_audio(synapse.base64_audio)
#                 # Convert to WAV for whisper
#                 wav_path = f"/tmp/{uuid.uuid4()}.wav"
#                 audio_seg.export(wav_path, format="wav")

#                 result = self.model.transcribe(wav_path, language=synapse.language)

#                 # Build segment objects
#                 segments = []
#                 for seg in result.get("segments", []):
#                     segments.append(CaptionSegment(
#                         start_time=seg["start"],
#                         end_time=seg["end"],
#                         text=seg["text"]
#                     ))
#                 # Clean up
#                 os.remove(wav_path)

#                 synapse.segments = segments
#             except Exception as e:
#                 bt.logging.warning(f"Error in STT: {e}")
#         return synapse

import os
import time
import base64
import uuid
import io
import torch
import whisper
import bittensor as bt
from io import BytesIO
from pydub import AudioSegment
import numpy as np
from captionise.base.miner import BaseMinerNeuron
from captionise.protocol import CaptionSynapse, CaptionSegment
from captionise.utils.logger import logger
from captionise.utils.database import db_manager


class CaptionMiner(BaseMinerNeuron):
    def __init__(self, config=None):
        super().__init__(config=config)
        
        # Initialize database
        db_manager.initialize_schema()
        
        # Load a whisper model
        model_name = getattr(config.neuron, "whisper_model", "base") if config else "base"
        self.device = torch.device(self.config.neuron.device if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading Whisper model '{model_name}' on device {self.device}")
        
        try:
            self.model = whisper.load_model(model_name, device=self.device)
            logger.success(f"Loaded Whisper model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise
        
        # Start the miner in background thread
        logger.info("Starting CaptionMiner background operations")
        self.run_in_background_thread()

    def decode_audio(self, base64_audio: str) -> AudioSegment:
        """Decode base64 audio data to AudioSegment object."""
        try:
            audio_bytes = base64.b64decode(base64_audio)
            return AudioSegment.from_file(BytesIO(audio_bytes))
        except Exception as e:
            logger.error(f"Error decoding audio: {e}")
            raise

    async def forward(self, synapse: bt.Synapse) -> bt.Synapse:
        """Handle incoming CaptionSynapse requests."""
        if not isinstance(synapse, CaptionSynapse):
            return synapse
            
        try:
            # Start timing the processing
            start_time = time.time()
            
            # Decode the audio and save as a temporary WAV file
            audio_seg = self.decode_audio(synapse.base64_audio)
            temp_filepath = f"/tmp/{uuid.uuid4()}.wav"
            audio_seg.export(temp_filepath, format="wav")
            
            # Save to database if job_id is provided
            if synapse.job_id:
                conn = db_manager.get_connection()
                try:
                    with conn.cursor() as cursor:
                        # Check if this job already exists
                        cursor.execute(
                            "SELECT job_id FROM jobs WHERE job_id = %s", 
                            (synapse.job_id,)
                        )
                        
                        if cursor.fetchone() is None:
                            # Insert the new job
                            cursor.execute("""
                                INSERT INTO jobs (job_id, audio_segment, language)
                                VALUES (%s, %s, %s)
                            """, (
                                synapse.job_id,
                                audio_seg.raw_data,
                                synapse.language or "en"
                            ))
                            conn.commit()
                except Exception as e:
                    conn.rollback()
                    logger.error(f"Database error saving job: {e}")
                finally:
                    db_manager.return_connection(conn)
            
            # Transcribe with Whisper
            result = self.model.transcribe(
                temp_filepath, 
                language=synapse.language,
                task="transcribe"
            )
            
            # Clean up temporary file
            os.remove(temp_filepath)
            
            # Record processing time
            processing_time = time.time() - start_time
            synapse.processing_time = processing_time
            
            # Build segment objects
            segments = []
            for seg in result.get("segments", []):
                segments.append(CaptionSegment(
                    start_time=seg["start"],
                    end_time=seg["end"],
                    text=seg["text"].strip()
                ))
            
            # Set the transcript segments and full text
            synapse.segments = segments
            synapse.transcript = result["text"].strip()
            
            # Update database with transcript if job_id is provided
            if synapse.job_id:
                conn = db_manager.get_connection()
                try:
                    with conn.cursor() as cursor:
                        cursor.execute("""
                            UPDATE jobs 
                            SET transcript_submitted = %s, processed = TRUE, updated_at = CURRENT_TIMESTAMP
                            WHERE job_id = %s
                        """, (
                            synapse.transcript,
                            synapse.job_id
                        ))
                        conn.commit()
                except Exception as e:
                    conn.rollback()
                    logger.error(f"Database error updating job: {e}")
                finally:
                    db_manager.return_connection(conn)
            
            logger.success(f"Successfully transcribed audio (time: {processing_time:.2f}s)")
            return synapse
            
        except Exception as e:
            logger.error(f"Error in transcription: {e}")
            return synapse

    def run(self):
        """
        The main loop for background tasks.
        Periodically syncs with the network and handles any maintenance.
        """
        logger.info("CaptionMiner main loop starting.")
        
        while not self.should_exit:
            try:
                # Sync with the network periodically
                self.resync_metagraph()
                
                # Sleep to avoid excessive CPU usage
                time.sleep(10)
            except Exception as e:
                logger.error(f"Error in miner background task: {e}")
                time.sleep(30)  # Longer sleep on error


# This is the main function, which runs the miner.
if __name__ == "__main__":
    with CaptionMiner() as m:
        while True:
            logger.warning(
                f"Miner running:: network: {m.subtensor.network} | step: {m.step} | uid: {m.uid} | trust: {m.metagraph.trust[m.uid]:.3f} | emission {m.metagraph.emission[m.uid]:.3f}"
            )
            time.sleep(30)