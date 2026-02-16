"""
Shared utilities for the support clustering pipeline.
Provides logging, validation, and error handling across all steps.
"""

import logging
import sys
import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional
import pandas as pd

# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging(step_name: str, log_dir: str = "logs") -> logging.Logger:
    """
    Set up logging for a pipeline step.
    Logs to both console and file with timestamps.

    Args:
        step_name: Name of the step (e.g., "disaggregate", "embed", "cluster")
        log_dir: Directory to store log files

    Returns:
        Configured logger
    """
    # Create logs directory if it doesn't exist
    Path(log_dir).mkdir(exist_ok=True)

    # Create logger
    logger = logging.getLogger(step_name)
    logger.setLevel(logging.DEBUG)

    # Clear any existing handlers
    logger.handlers = []

    # Console handler - INFO and above
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console_fmt = logging.Formatter('[%(levelname)s] %(message)s')
    console.setFormatter(console_fmt)
    logger.addHandler(console)

    # File handler - DEBUG and above (more detail)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(log_dir) / f"{step_name}_{timestamp}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_fmt = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    file_handler.setFormatter(file_fmt)
    logger.addHandler(file_handler)

    logger.info(f"Logging to: {log_file}")
    return logger


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

class ValidationError(Exception):
    """Raised when input validation fails."""
    pass


def validate_input_files(paths: List[str], logger: logging.Logger) -> List[str]:
    """
    Validate that input files exist and are readable.

    Args:
        paths: List of file paths (can include glob patterns)
        logger: Logger instance

    Returns:
        List of valid file paths

    Raises:
        ValidationError if no valid files found
    """
    from glob import glob

    valid_files = []
    for path in paths:
        # Handle glob patterns
        if '*' in path:
            matches = glob(path)
            if not matches:
                logger.warning(f"No files match pattern: {path}")
            else:
                valid_files.extend(matches)
        else:
            if os.path.exists(path):
                valid_files.append(path)
            else:
                logger.warning(f"File not found: {path}")

    if not valid_files:
        raise ValidationError(f"No valid input files found. Checked: {paths}")

    logger.info(f"Found {len(valid_files)} input file(s)")
    for f in valid_files:
        size_mb = os.path.getsize(f) / (1024 * 1024)
        logger.debug(f"  - {f} ({size_mb:.1f} MB)")

    return valid_files


def validate_columns(df: pd.DataFrame, required: List[str], step_name: str, logger: logging.Logger):
    """
    Validate that a DataFrame has required columns.

    Args:
        df: DataFrame to validate
        required: List of required column names
        step_name: Name of the current step (for error messages)
        logger: Logger instance

    Raises:
        ValidationError if columns are missing
    """
    missing = [col for col in required if col not in df.columns]
    if missing:
        logger.error(f"Missing required columns: {missing}")
        logger.error(f"Available columns: {list(df.columns)}")
        raise ValidationError(
            f"[{step_name}] Missing required columns: {missing}. "
            f"Check that the previous step completed successfully."
        )
    logger.debug(f"Validated columns: {required}")


def validate_output(output_path: str, min_rows: int, logger: logging.Logger):
    """
    Validate that output file was created and has data.

    Args:
        output_path: Path to output file
        min_rows: Minimum expected rows
        logger: Logger instance

    Raises:
        ValidationError if output is invalid
    """
    if not os.path.exists(output_path):
        raise ValidationError(f"Output file was not created: {output_path}")

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    logger.info(f"Output file created: {output_path} ({size_mb:.1f} MB)")

    # Quick row count check
    if output_path.endswith('.parquet'):
        df = pd.read_parquet(output_path)
    else:
        df = pd.read_csv(output_path, nrows=min_rows + 1)

    if len(df) < min_rows:
        raise ValidationError(
            f"Output has only {len(df)} rows, expected at least {min_rows}. "
            f"Check for errors in processing."
        )

    logger.info(f"Output validated: {len(df)} rows")


# =============================================================================
# DATA I/O
# =============================================================================

def load_data(paths: List[str], logger: logging.Logger) -> pd.DataFrame:
    """
    Load data from CSV or Parquet files, combining multiple files if needed.

    Args:
        paths: List of file paths
        logger: Logger instance

    Returns:
        Combined DataFrame
    """
    dfs = []
    for path in paths:
        logger.info(f"Loading: {path}")
        try:
            if path.endswith('.parquet'):
                df = pd.read_parquet(path)
            else:
                df = pd.read_csv(path)
            logger.debug(f"  Loaded {len(df)} rows, {len(df.columns)} columns")
            dfs.append(df)
        except Exception as e:
            logger.error(f"Failed to load {path}: {e}")
            raise ValidationError(f"Cannot load file: {path}. Error: {e}")

    if len(dfs) == 1:
        return dfs[0]

    # Combine multiple DataFrames
    logger.info(f"Combining {len(dfs)} files...")
    combined = pd.concat(dfs, ignore_index=True)
    logger.info(f"Combined total: {len(combined)} rows")
    return combined


def save_data(df: pd.DataFrame, path: str, logger: logging.Logger):
    """
    Save DataFrame to Parquet or CSV based on file extension.

    Args:
        df: DataFrame to save
        path: Output file path
        logger: Logger instance
    """
    logger.info(f"Saving {len(df)} rows to {path}...")
    try:
        if path.endswith('.parquet'):
            df.to_parquet(path, index=False)
        else:
            df.to_csv(path, index=False)

        size_mb = os.path.getsize(path) / (1024 * 1024)
        logger.info(f"Saved successfully ({size_mb:.1f} MB)")
    except Exception as e:
        logger.error(f"Failed to save: {e}")
        raise


# =============================================================================
# STEP TRACKING
# =============================================================================

class StepTracker:
    """Track progress through pipeline steps with checkpoints."""

    def __init__(self, step_name: str, logger: logging.Logger):
        self.step_name = step_name
        self.logger = logger
        self.start_time = None
        self.checkpoints = []

    def start(self, description: str):
        """Mark the start of the step."""
        import time
        self.start_time = time.time()
        self.logger.info("=" * 60)
        self.logger.info(f"STARTING: {self.step_name}")
        self.logger.info(f"  {description}")
        self.logger.info("=" * 60)

    def checkpoint(self, name: str, count: Optional[int] = None):
        """Record a checkpoint within the step."""
        import time
        elapsed = time.time() - self.start_time if self.start_time else 0
        self.checkpoints.append((name, elapsed, count))
        msg = f"[CHECKPOINT] {name}"
        if count is not None:
            msg += f" ({count} items)"
        msg += f" - {elapsed:.1f}s elapsed"
        self.logger.info(msg)

    def complete(self, output_path: str, row_count: int):
        """Mark successful completion of the step."""
        import time
        elapsed = time.time() - self.start_time if self.start_time else 0
        self.logger.info("=" * 60)
        self.logger.info(f"COMPLETED: {self.step_name}")
        self.logger.info(f"  Output: {output_path}")
        self.logger.info(f"  Rows: {row_count}")
        self.logger.info(f"  Time: {elapsed:.1f}s")
        self.logger.info("=" * 60)

    def fail(self, error: Exception):
        """Mark step failure with error details."""
        import time
        import traceback
        elapsed = time.time() - self.start_time if self.start_time else 0
        self.logger.error("=" * 60)
        self.logger.error(f"FAILED: {self.step_name}")
        self.logger.error(f"  Error: {error}")
        self.logger.error(f"  Time: {elapsed:.1f}s")
        self.logger.error("=" * 60)
        self.logger.debug(f"Traceback:\n{traceback.format_exc()}")

        # Print checkpoints to help debug where it failed
        if self.checkpoints:
            self.logger.error("Last successful checkpoints:")
            for name, t, count in self.checkpoints[-3:]:
                self.logger.error(f"  - {name} at {t:.1f}s")
