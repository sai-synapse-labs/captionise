import os
import random
import pandas as pd
from queue import Queue
from datetime import datetime
from captionise.utils.logger import logger


class CSVJobStore:
    """
    A CSV-based job store for the Captionise project.
    Each row in the CSV is a 'job' with fields:
      - job_id
      - text
      - active (0 or 1)
      - created_at (timestamp)
      - updated_at (timestamp)
    """

    def __init__(self, csv_path: str = "assets/jobs.csv"):
        """
        Args:
            csv_path (str): Path to the CSV file containing job data.
        """
        self.csv_path = csv_path
        if not os.path.isfile(self.csv_path):
            logger.warning(f"{self.csv_path} not found. Creating an empty CSV with default columns.")
            df = pd.DataFrame(columns=["job_id","text","active","created_at","updated_at"])
            df.to_csv(self.csv_path, index=False)
        logger.info(f"Using CSV job store at: {self.csv_path}")

    def load_jobs(self) -> pd.DataFrame:
        """Load all jobs from CSV into a DataFrame."""
        return pd.read_csv(self.csv_path, parse_dates=["created_at","updated_at"])

    def save_jobs(self, df: pd.DataFrame):
        """Save the provided DataFrame back to CSV."""
        df.to_csv(self.csv_path, index=False)

    def row_to_job(self, row: pd.Series) -> "Job":
        """Convert a DataFrame row into a Job object."""
        return Job(
            job_id=row["job_id"],
            text=row["text"],
            # convert numeric type 0/1 to bool
            active=bool(row["active"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def get_queue(self, ready_only: bool=True) -> Queue:
        """
        Get a queue of 'active' jobs or all jobs from the CSV.
        
        Args:
            ready_only (bool): if True, only include rows where active=1.
        
        Returns:
            Queue: a queue of Job objects.
        """
        df = self.load_jobs()
        if ready_only:
            df = df[df["active"] == 1]  # numeric type in CSV
        job_queue = Queue()

        for _, row in df.iterrows():
            job_queue.put(self.row_to_job(row))

        return job_queue

    def get_inactive_queue(self, since: str=None) -> Queue:
        """
        Get a queue of inactive jobs (active=0). 
        If `since` is given (e.g., '2024-01-20 10:00:00'), only those updated after that timestamp.
        """
        df = self.load_jobs()
        df = df[df["active"] == 0]
        if since:
            df = df[df["updated_at"] >= since]

        job_queue = Queue()
        for _, row in df.iterrows():
            job_queue.put(self.row_to_job(row))

        return job_queue

    def deactivate_job(self, job_id: str):
        """Mark a job as inactive in the CSV."""
        df = self.load_jobs()
        idx = df[df["job_id"] == job_id].index
        if not idx.empty:
            df.loc[idx, "active"] = 0
            df.loc[idx, "updated_at"] = pd.Timestamp.now()
            self.save_jobs(df)
            logger.info(f"Deactivated job {job_id}")
        else:
            logger.warning(f"No job found with job_id={job_id}")

    def add_job(self, text: str) -> str:
        """
        Add a new job with given text snippet, auto-generating a job_id.
        
        Returns:
            str: The assigned job_id
        """
        df = self.load_jobs()
        new_id = f"job-{random.randint(1000,9999)}"
        now = pd.Timestamp.now()

        new_row = {
            "job_id": new_id,
            "text": text,
            "active": 1,
            "created_at": now,
            "updated_at": now,
        }
        df = df.append(new_row, ignore_index=True)
        self.save_jobs(df)

        logger.info(f"Added new job {new_id} with text: {text[:30]}...")
        return new_id

    def __repr__(self):
        df = self.load_jobs()
        return df.__repr__()

class Job:
    """A minimal job representation for the Captionise project."""
    def __init__(
        self,
        job_id: str,
        text: str,
        active: bool,
        created_at: pd.Timestamp,
        updated_at: pd.Timestamp,
    ):
        self.job_id = job_id
        self.text = text
        self.active = active
        self.created_at = created_at
        self.updated_at = updated_at

    def update(self, new_text: str=None, deactivate: bool=False):
        """
        Local method to modify text or deactivate the job.
        Typically you'd call CSVJobStore to persist this change to CSV.
        """
        if new_text:
            self.text = new_text
        if deactivate:
            self.active = False
        self.updated_at = pd.Timestamp.now()

    def __repr__(self):
        return f"Job({self.job_id}, active={self.active}, text='{self.text[:30]}...')"
