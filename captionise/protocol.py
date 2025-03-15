# The MIT License (MIT)
# Copyright © 2024 ...
# (Include full MIT license text here as appropriate.)

import typing
import base64
import bittensor as bt
from captionise.utils.logger import logger  # Or loguru



class PingSynapse(bt.Synapse):
    """
    Used by validators to check if a Captionise miner
    can accept new tasks or has available capacity.
    """
    can_serve: bool = False
    available_compute: typing.Optional[int] = None  # e.g., number of threads / GPU capacity


class ParticipationSynapse(bt.Synapse):
    """
    Used to check if a miner is participating in a particular 'job' or session.
    """
    job_id: str
    is_participating: bool = False


class CaptionSegment(bt.Synapse):
    """
    Represents a single segment/chunk of a transcribed audio file.
    
    Attributes:
        start_time: Start time of this segment in seconds
        end_time: End time of this segment in seconds  
        text: The transcribed text for this segment
    """
    start_time: float
    end_time: float 
    text: str


class CaptionSynapse(bt.Synapse):
    """
    A core synapse for the Captionise subnet.
    This synapse helps pass audio (base64) or text data between miner & validator.
    
    Attributes:
      - job_id: Unique ID representing the specific captioning task.
      - base64_audio: The raw audio data encoded in base64.
      - language: The language code for STT (e.g., 'en', 'fr').
      - segments: An optional list of transcribed segments (start_time, end_time, text).
      - miner_state: A field for the miner to store any additional status info.
      - transcript: The complete generated transcript as a string.
      - processing_time: Time taken to process the audio in seconds.
    """
    job_id: typing.Optional[str] = None
    base64_audio: typing.Optional[str] = None
    language: typing.Optional[str] = "en"
    segments: typing.Optional[typing.List[CaptionSegment]] = None
    transcript: typing.Optional[str] = None
    processing_time: typing.Optional[float] = None
    miner_state: typing.Optional[str] = None

    def deserialize(self) -> "CaptionSynapse":
        """
        Optional deserialization step. If your segments are stored
        in a specialized format or if you have to decode anything else,
        you can do it here.
        """
        logger.info(f"Deserializing CaptionSynapse for job_id: {self.job_id}")
        if self.segments is not None:
            logger.debug(f"Segments found: {len(self.segments)}")
        return self


class JobSubmissionSynapse(bt.Synapse):
    """
    Example of a more generic job-submission synapse if your subnet
    needs to handle chunk-based tasks or multi-step workflows.
    
    Attributes:
      - job_id: An identifier for the job.
      - request_data: Arbitrary data relevant to the job (e.g. text, metadata).
      - response_data: The miner's output.
      - miner_seed: An optional integer seed for randomization or partial tasks.
    """
    job_id: str
    request_data: typing.Optional[dict] = None
    response_data: typing.Optional[dict] = None
    miner_seed: typing.Optional[int] = None

    def deserialize(self) -> "JobSubmissionSynapse":
        """
        Convert base64 or other encoded data in response_data if needed.
        """
        logger.info(f"Deserializing JobSubmissionSynapse for job_id: {self.job_id}")
        if self.response_data is not None:
            decoded_output = {}
            for k, v in self.response_data.items():
                try:
                    decoded_output[k] = base64.b64decode(v)
                except Exception as e:
                    logger.error(f"Error decoding {k} from response_data: {e}")
                    decoded_output[k] = None
            self.response_data = decoded_output

        return self
