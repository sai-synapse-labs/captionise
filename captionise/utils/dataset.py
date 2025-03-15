import hashlib
import io
from typing import List, Dict, Tuple
import bittensor as bt
from datasets import load_dataset
from captionise.utils.logger import logger
from captionise.utils.database import db_manager

class DatasetManager:
    """Manages dataset acquisition and storage for the Caption Subnet."""
    
    def __init__(self, languages: List[str] = None):
        """
        Initialize the dataset manager.
        
        Args:
            languages: List of language codes to load from VoxPopuli
        """
        self.languages = languages or ["en"]
        self.dataset = None
        
    def load_voxpopuli(self, split: str = "train", sample_size: int = 100):
        """
        Load the VoxPopuli dataset with specified languages.
        
        Args:
            split: Dataset split to load ("train", "validation", or "test")
            sample_size: Number of samples to load per language
        """
        try:
            logger.info(f"Loading VoxPopuli dataset for languages: {self.languages}")
            
            all_samples = []
            for lang in self.languages:
                # Load dataset for specific language
                dataset = load_dataset("facebook/voxpopuli", lang, split=split, streaming=True)
                # Take sample_size samples
                samples = list(dataset.take(sample_size))
                all_samples.extend(samples)
                
                logger.info(f"Loaded {len(samples)} samples for language {lang}")
            
            self.dataset = all_samples
            logger.info(f"Total dataset size: {len(self.dataset)} samples")
            return True
        except Exception as e:
            logger.error(f"Error loading VoxPopuli dataset: {e}")
            return False
            
    def store_dataset_in_db(self):
        """Store the loaded dataset in the database."""
        if not self.dataset:
            logger.error("No dataset loaded. Call load_voxpopuli() first.")
            return False
            
        conn = db_manager.get_connection()
        try:
            stored_count = 0
            with conn.cursor() as cursor:
                for sample in self.dataset:
                    # Generate hash of audio content to avoid duplicates
                    audio_hash = hashlib.sha256(sample['audio']['array']).hexdigest()
                    
                    # Convert audio array to binary
                    audio_bytes = io.BytesIO()
                    sample['audio']['array'].tofile(audio_bytes)
                    audio_binary = audio_bytes.getvalue()
                    
                    # Check if this audio already exists
                    cursor.execute(
                        "SELECT job_id FROM jobs WHERE audio_hash = %s", 
                        (audio_hash,)
                    )
                    
                    if cursor.fetchone() is None:
                        # Insert new job
                        cursor.execute("""
                            INSERT INTO jobs (audio_segment, audio_hash, transcript_source, language)
                            VALUES (%s, %s, %s, %s)
                            RETURNING job_id
                        """, (
                            audio_binary,
                            audio_hash,
                            sample['normalized_text'],
                            sample.get('language', 'en')
                        ))
                        stored_count += 1
                
                conn.commit()
                logger.success(f"Stored {stored_count} new audio samples in the database")
                return True
        except Exception as e:
            conn.rollback()
            logger.error(f"Error storing dataset in database: {e}")
            return False
        finally:
            db_manager.return_connection(conn)
            
    def get_unprocessed_jobs(self, limit: int = 10) -> List[Dict]:
        """
        Retrieve unprocessed jobs from the database.
        
        Args:
            limit: Maximum number of jobs to retrieve
            
        Returns:
            List of job dictionaries
        """
        conn = db_manager.get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT job_id, audio_segment, transcript_source, language
                    FROM jobs
                    WHERE processed = FALSE
                    LIMIT %s
                """, (limit,))
                
                jobs = []
                for row in cursor.fetchall():
                    jobs.append({
                        'job_id': row[0],
                        'audio_segment': row[1],
                        'transcript_source': row[2],
                        'language': row[3]
                    })
                
                return jobs
        except Exception as e:
            logger.error(f"Error retrieving unprocessed jobs: {e}")
            return []
        finally:
            db_manager.return_connection(conn) 