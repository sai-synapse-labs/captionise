import os
import time
import glob
import copy
import json
import base64
import random
import hashlib
import traceback
import concurrent.futures
import asyncio
from collections import defaultdict
from typing import Dict, List, Tuple, Any

from captionise.base.miner import BaseMinerNeuron  # if you have a base class that sets up Bittensor wallet, axon, etc.
from captionise.protocol import JobSubmissionSynapse  # or any specialized synapse for receiving job details
from captionise.store import CSVJobStore, Job        # CSV job store references
from captionise.utils.logger import logger


############################################
# Utility: attach files to a synapse object
############################################
def attach_files(files_to_attach: List[str], synapse: JobSubmissionSynapse) -> JobSubmissionSynapse:
    """
    Attaches local files (by path) to the synapse as base64 data.
    This can be used to return partial or final results to the validator.
    """
    logger.info(f"Attaching files: {files_to_attach}")
    if not hasattr(synapse, "md_output"):
        synapse.md_output = {}

    for file_path in files_to_attach:
        try:
            with open(file_path, "rb") as f:
                filename = os.path.basename(file_path)
                synapse.md_output[filename] = base64.b64encode(f.read())
        except Exception as e:
            logger.error(f"Failed to read file {file_path!r}, error: {e}")
            # log traceback, skip attaching
    return synapse

def attach_files_to_synapse(synapse:JobSubmissionSynapse, data_directory: str, state: str, seed: int) -> JobSubmissionSynapse:
    """
    Gathers local data from 'data_directory' for the given 'state',
    attaching it to synapse as 'md_output'.
    """
    synapse.md_output = {}
    try:
        pattern = os.path.join(data_directory, f"{state}*")
        state_files = glob.glob(pattern)
        if not state_files:
            logger.warning(f"No files found for state '{state}' in {data_directory}.")
        synapse = attach_files(files_to_attach=state_files, synapse=synapse)
        synapse.miner_seed = seed
        synapse.miner_state = state
    except Exception as e:
        logger.error(f"Failed to attach files for job {getattr(synapse, 'job_id', '')}, error: {e}")
        synapse.md_output = {}
    return synapse

############################################
# Actual Miner class
############################################
class CaptioniseMiner(BaseMinerNeuron):
    """
    Adapts the HPC concurrency logic to a captioning context, reading tasks from CSV,
    spawning concurrency jobs (STT / text generation), and returning partial or final outputs.
    """

    def __init__(self, config=None): 
        super().__init__(config=config)
        
        # A subfolder for storing local data (partial transcripts, etc.)
        self.miner_data_path = os.path.join(self.project_path, "miner-data")
        self.base_data_path = os.path.join(self.miner_data_path, self.wallet_hotkey[:8])

        # concurrency
        self.max_workers = getattr(config.neuron, "max_workers", 4) if config else 4
        logger.info(f"Starting CaptioniseMiner with {self.max_workers} workers...")

        self.executor = concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers)
        self.simulations = self._create_nested_dict()

        # The HPC code used HPC states; we define simplified states for caption tasks
        self.STATES = ["initial", "transcribing", "finished"]
        # If you want real-time partial results, you might define more states.
        # e.g., chunk_1_done, chunk_2_done, etc.

        # concurrency loop
        self.loop = asyncio.get_event_loop()
        self.should_exit = False
        self.is_running = False

        # We can define or load a CSVJobStore here for reading tasks
        self.job_store = None
        self.csv_path = getattr(config, "csv_path", "assets/jobs.csv") if config else "assets/jobs.csv"
        # init the job store
        self._init_job_store()

        # random seed generator
        self.generate_random_seed = lambda: random.randint(1000, 9999)

        logger.info("CaptioniseMiner initialized")

    def _create_nested_dict(self):
        def nested_dict():
            return defaultdict(lambda: None)
        return defaultdict(nested_dict)

    def _init_job_store(self):
        """
        Initialize the CSVJobStore to read pending caption tasks.
        """
        try:
            self.job_store = CSVJobStore(csv_path=self.csv_path)
            logger.info(f"Loaded CSVJobStore from {self.csv_path}")
        except ImportError:
            logger.warning("CSVJobStore not found or store.py missing. Running without job queue features.")
            self.job_store = None

    ########################################
    # The main loop
    ########################################
    def run(self):
        """
        The main loop, analogous to HPC code's run().
        Periodically loads tasks from CSV, spawns concurrency jobs, and checks for completed tasks.
        """
        logger.info("CaptioniseMiner main loop starting.")
        last_job_check = time.time()
        job_check_interval = 10  # seconds to poll new jobs from CSV

        try:
            while not self.should_exit:
                # HPC code might do chain sync or set weights. We omit that here or call super().sync()

                # check for concurrency completions
                self._check_finished_tasks()

                # Check if we have capacity to add new tasks from CSV
                if self.job_store and (time.time() - last_job_check > job_check_interval):
                    self.add_active_jobs_from_csv()
                    last_job_check = time.time()

                time.sleep(3)
        except KeyboardInterrupt:
            logger.warning("Miner interrupted via keyboard.")
        except Exception as e:
            logger.error(f"Miner run encountered error: {e}")
        finally:
            logger.info("Exiting miner run loop.")

    ########################################
    # CSV job store integration
    ########################################
    def add_active_jobs_from_csv(self):
        """
        Check the CSV for 'active' jobs (where 'active' = 1),
        and if there's capacity, spawn concurrency tasks for them.
        """
        # how many concurrency tasks are we currently running?
        running_count = len(self.simulations)
        if running_count >= self.max_workers:
            logger.info("No available concurrency slots for new jobs.")
            return

        # load from CSV
        active_jobs_queue = self.job_store.get_queue(ready_only=True)
        if active_jobs_queue.empty():
            logger.info("No active jobs found in CSV store.")
            return

        logger.info(f"{active_jobs_queue.qsize()} active job(s) found in CSV.")
        available_slots = self.max_workers - running_count

        # For each job in the queue, see if we've worked on it or are currently working on it
        added = 0
        while not active_jobs_queue.empty() and added < available_slots:
            csv_job = active_jobs_queue.get()
            job_id = csv_job.job_id

            has_worked_on_job, condition, event = self.check_if_job_was_worked_on(job_id)
            if has_worked_on_job:
                logger.info(f"Job {job_id} is already known: {condition}. Skipping.")
                continue

            # otherwise, start a new concurrency pipeline
            logger.success(f"Starting concurrency for new CSV job {job_id}.")
            self.start_new_job_csv(csv_job)
            added += 1

        logger.info(f"Added {added} new concurrency job(s) from CSV store.")

    def start_new_job_csv(self, csv_job: Job):
        """
        Like HPC's start_new_job, but using CSV Job objects directly.
        """
        job_id = csv_job.job_id
        job_hash = self._make_task_hash(job_id, csv_job.text)
        output_dir = os.path.join(self.base_data_path, f"{job_id}_{job_hash}")
        os.makedirs(output_dir, exist_ok=True)

        seed = self.generate_random_seed()
        with open(os.path.join(output_dir, "seed.txt"), "w", encoding="utf-8") as f:
            f.write(str(seed))

        future = self.executor.submit(self.caption_pipeline, job_id, csv_job.text, output_dir, seed)
        self.simulations[job_hash]["job_id"] = job_id
        self.simulations[job_hash]["future"] = future
        self.simulations[job_hash]["output_dir"] = output_dir
        self.simulations[job_hash]["seed"] = seed
        self.simulations[job_hash]["current_state"] = "initial"

    ########################################
    # concurrency pipeline
    ########################################
    def caption_pipeline(self, job_id: str, text: str, output_dir: str, seed: int) -> tuple[str, Any]:
        """
        The concurrency function that simulates STT / text generation or partial chunk-based tasks.
        HPC code used HPC states; we do "initial", "transcribing", "finished."
        """
        try:
            logger.info(f"Begin 'caption' pipeline for job_id={job_id}, text='{text[:30]}...', seed={seed}")
            states = ["initial", "transcribing", "finished"]
            wait_time = 2.0  # each step takes 2 seconds

            # A placeholder for actual STT or text generation
            # In a real system, you'd call your ML model or library here
            for i, state in enumerate(states):
                self._write_state_file(output_dir, state)
                self._simulate_stt_progress(output_dir, text, step_idx=i)  # placeholder
                time.sleep(wait_time)
                self._update_simulation_task(job_id, state)

            final_state = "finished"
            self._write_state_file(output_dir, final_state)
            logger.success(f"Completed pipeline for job {job_id}.")
            return final_state, None
        except Exception as e:
            error_info = {
                "type": "UnexpectedException",
                "message": str(e),
                "traceback": traceback.format_exc(),
            }
            final_state = "failed"
            self._write_state_file(output_dir, final_state)
            logger.error(f"Caption pipeline for job {job_id} failed: {error_info}")
            return final_state, error_info

    def _simulate_stt_progress(self, output_dir: str, text: str, step_idx: int):
        """
        A placeholder method that simulates partial STT or text generation.
        Writes partial results to a file named transcribe_step{step_idx}.txt
        """
        partial_file = os.path.join(output_dir, f"transcribe_step{step_idx}.txt")
        partial_text = text[: max(10 * (step_idx + 1), len(text))]
        with open(partial_file, "w", encoding="utf-8") as f:
            f.write(partial_text + "\n")

    def _update_simulation_task(self, job_id: str, state: str):
        """
        HPC code updated HPC states, we do similar for text states
        so the miner can attach partial results for the validator.
        """
        for job_hash, sim in self.simulations.items():
            if sim.get("job_id") == job_id:
                sim["current_state"] = state
                break

    def _write_state_file(self, output_dir: str, state: str):
        """
        HPC code wrote a 'state_file_name' with the HPC simulation state.
        For text tasks, do the same with e.g. 'transcription_state.txt'
        """
        state_file = os.path.join(output_dir, "transcription_state.txt")
        with open(state_file, "w", encoding="utf-8") as f:
            f.write(state + "\n")

    ########################################
    # HPC-likes checks for concurrency tasks
    ########################################
    def _check_finished_tasks(self):
        """
        Periodically checks concurrency tasks, removing them if final or failed.
        """
        finished_keys = []
        for job_hash, sim_data in self.simulations.items():
            future = sim_data.get("future")
            if future and future.done():
                final_state, error_info = future.result()
                job_id = sim_data.get("job_id")
                if final_state == "finished":
                    logger.info(f"Job {job_id} completed. Removing from concurrency list.")
                else:
                    logger.error(f"Job {job_id} failed with error: {error_info}")
                finished_keys.append(job_hash)

        for job_hash in finished_keys:
            del self.simulations[job_hash]
        if finished_keys:
            logger.info(f"Removed {len(finished_keys)} completed tasks from concurrency list.")

    ########################################
    # HPC-like checks for known or new tasks
    ########################################
    def check_if_job_was_worked_on(self, job_id: str) -> tuple[bool, str, Dict]:
        """
        Returns whether the job is known or new, along with a 'condition' and an event dict.
        HPC code references HPC states, we do text concurrency states.
        """
        event = {}
        if not job_id:
            return False, "job_not_worked_on", event

        job_hash = self._make_task_hash(job_id, salt="some-constant")
        event["job_hash"] = job_hash
        output_dir = os.path.join(self.base_data_path, f"{job_id}_{job_hash}")
        event["output_dir"] = output_dir

        # If concurrency is active
        if job_hash in self.simulations:
            return True, "running_simulation", event

        if os.path.exists(output_dir):
            # HPC code returned 'found_existing_data' if local files exist
            return True, "found_existing_data", event

        return False, "job_not_worked_on", event

    def _make_task_hash(self, job_id: str, text: str = "", salt: str = "") -> str:
        """
        HPC code hashed (pdb_id + system config). For text tasks, let's hash job_id + partial text + salt.
        """
        combined = job_id + text + salt
        hash_object = hashlib.sha256(combined.encode("utf-8"))
        return hash_object.hexdigest()[:6]
