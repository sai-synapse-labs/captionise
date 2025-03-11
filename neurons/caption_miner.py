import bittensor as bt
import whisper as ow
import os
import uuid
from io import BytesIO
from pydub import AudioSegment
from captionise.miners.captionise_miner import CaptioniseMiner as BaseMiner
from captionise.protocol import CaptionSynapse, CaptionSegment

class CaptionMiner(BaseMiner):
    def __init__(self, wallet: bt.wallet, config: bt.Config):
        super().__init__(wallet, config)
        # Load a whisper model
        model_name = "base"  # or "medium", "large", etc.
        self.model = ow.load_model(model_name)
        bt.logging.success(f"Loaded Whisper model: {model_name}")

    async def forward(self, synapse: bt.Synapse) -> bt.Synapse:
        """Handle incoming CaptionSynapse requests."""
        if isinstance(synapse, CaptionSynapse):
            try:
                audio_seg = self.decode_audio(synapse.base64_audio)
                # Convert to WAV for whisper
                wav_path = f"/tmp/{uuid.uuid4()}.wav"
                audio_seg.export(wav_path, format="wav")

                result = self.model.transcribe(wav_path, language=synapse.language)

                # Build segment objects
                segments = []
                for seg in result.get("segments", []):
                    segments.append(CaptionSegment(
                        start_time=seg["start"],
                        end_time=seg["end"],
                        text=seg["text"]
                    ))
                # Clean up
                os.remove(wav_path)

                synapse.segments = segments
            except Exception as e:
                bt.logging.warning(f"Error in STT: {e}")
        return synapse
