# src/rlhf_ui/data/storage.py
"""
Data storage and retrieval for RLHF preference collection.
Handles persistence, versioning, and backup of preference data.
"""

import logging
import json
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class PreferenceDataStorage:
    """
    Storage manager for preference data.
    Handles saving, loading, and versioning of preference data.
    """
    
    DEFAULT_FILENAME = "preferences.csv"
    BACKUP_PATTERN = "preferences_backup_{timestamp}.csv"
    
    def __init__(
        self, 
        output_folder: Union[str, Path],
        auto_backup: bool = True,
        backup_interval_minutes: int = 10
    ):
        """
        Initialize the preference data storage.
        
        Args:
            output_folder: Folder to store preference data
            auto_backup: Whether to automatically create backups
            backup_interval_minutes: How often to create backups (minutes)
        """
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(exist_ok=True, parents=True)
        
        self.preference_file = self.output_folder / self.DEFAULT_FILENAME
        self.auto_backup = auto_backup
        self.backup_interval_minutes = backup_interval_minutes
        
        self.last_backup_time = datetime.min
        self.data_modified = False
        
        # Initialize or load data
        self._initialize_data()
    
    def _initialize_data(self) -> None:
        """Initialize or load preference data."""
        if self.preference_file.exists():
            try:
                self.preferences_df = pd.read_csv(self.preference_file)
                logger.info(f"Loaded {len(self.preferences_df)} preference records from {self.preference_file}")
            except Exception as e:
                logger.error(f"Error loading preference data: {e}")
                self._create_empty_dataframe()
                self._recover_from_backup()
        else:
            self._create_empty_dataframe()
    
    def _create_empty_dataframe(self) -> None:
        """Create an empty preference dataframe with the correct structure."""
        self.preferences_df = pd.DataFrame({
            'image1': [],
            'image2': [],
            'preferred': [],  # 1 for image1, 2 for image2, 0 for tie/skip
            'prompt': [],
            'timestamp': [],
            'rater_id': [],
            'response_time_ms': []
        })
    
    def _recover_from_backup(self) -> bool:
        """
        Try to recover data from the most recent backup.
        
        Returns:
            bool: True if recovery was successful, False otherwise
        """
        backup_files = list(self.output_folder.glob("preferences_backup_*.csv"))
        if not backup_files:
            logger.warning("No backup files found for recovery")
            return False
        
        # Sort by modification time (most recent first)
        backup_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        for backup_file in backup_files:
            try:
                self.preferences_df = pd.read_csv(backup_file)
                logger.info(f"Recovered {len(self.preferences_df)} preference records from {backup_file}")
                self.save()  # Save recovered data to main file
                return True
            except Exception as e:
                logger.error(f"Failed to recover from backup {backup_file}: {e}")
        
        return False
    
    def save(self) -> None:
        """Save preference data to CSV file."""
        try:
            # Use atomic write to prevent data corruption
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as temp_file:
                self.preferences_df.to_csv(temp_file.name, index=False)
                temp_path = temp_file.name
            
            # Move the temp file to the final location (atomic operation)
            shutil.move(temp_path, self.preference_file)
            
            self.data_modified = True
            logger.info(f"Saved {len(self.preferences_df)} preference records to {self.preference_file}")
            
            # Check if we should create a backup
            self._check_create_backup()
        except Exception as e:
            logger.error(f"Error saving preference data: {e}")
            raise
    
    def _check_create_backup(self) -> None:
        """Check if we should create a backup and do so if needed."""
        if not self.auto_backup or not self.data_modified:
            return
            
        now = datetime.now()
        elapsed_minutes = (now - self.last_backup_time).total_seconds() / 60
        
        if elapsed_minutes >= self.backup_interval_minutes:
            self.create_backup()
            self.last_backup_time = now
            self.data_modified = False
    
    def create_backup(self, custom_suffix: Optional[str] = None) -> Path:
        """
        Create a backup of the current preference data.
        
        Args:
            custom_suffix: Optional custom suffix for the backup file
            
        Returns:
            Path: Path to the created backup file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if custom_suffix:
            backup_filename = f"preferences_backup_{timestamp}_{custom_suffix}.csv"
        else:
            backup_filename = self.BACKUP_PATTERN.format(timestamp=timestamp)
        
        backup_path = self.output_folder / backup_filename
        
        try:
            # Copy the current file if it exists
            if self.preference_file.exists():
                shutil.copy2(self.preference_file, backup_path)
            else:
                # If the file doesn't exist, save the current dataframe
                self.preferences_df.to_csv(backup_path, index=False)
                
            logger.info(f"Created backup at {backup_path}")
            return backup_path
        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
            raise
    
    def add_preference(
        self, 
        image1: Union[str, Path], 
        image2: Union[str, Path], 
        preferred: int, 
        prompt: str = "",
        rater_id: str = "default",
        response_time_ms: Optional[float] = None
    ) -> None:
        """
        Add a new preference record.
        
        Args:
            image1: Path to first image
            image2: Path to second image
            preferred: 1 if image1 is preferred, 2 if image2, 0 if tie/skip
            prompt: Optional context prompt
            rater_id: ID of the human rater
            response_time_ms: Response time in milliseconds
        """
        new_row = pd.DataFrame({
            'image1': [str(image1)],
            'image2': [str(image2)],
            'preferred': [preferred],
            'prompt': [prompt],
            'timestamp': [datetime.now().isoformat()],
            'rater_id': [rater_id],
            'response_time_ms': [response_time_ms]
        })
        
        self.preferences_df = pd.concat([self.preferences_df, new_row], ignore_index=True)
        self.save()
    
    def get_preferences(self, exclude_ties: bool = False) -> pd.DataFrame:
        """
        Get the preference dataframe.
        
        Args:
            exclude_ties: Whether to exclude ties/skips (preferred=0)
            
        Returns:
            pd.DataFrame: Preference data
        """
        if exclude_ties:
            return self.preferences_df[self.preferences_df['preferred'] > 0].reset_index(drop=True)
        
        return self.preferences_df
    
    def get_preference_count(self, by_rater: bool = False) -> Union[int, Dict[str, int]]:
        """
        Get the number of preference records.
        
        Args:
            by_rater: Whether to break down counts by rater ID
            
        Returns:
            Union[int, Dict[str, int]]: Count of preferences (total or by rater)
        """
        if not by_rater:
            return len(self.preferences_df)
        
        # Group by rater and count
        return self.preferences_df['rater_id'].value_counts().to_dict()
    
    def get_image_coverage(self) -> Dict[str, Any]:
        """
        Calculate image coverage statistics.
        
        Returns:
            Dict[str, Any]: Statistics about image coverage
        """
        # Get all unique images
        all_images = set()
        for img in self.preferences_df['image1']:
            all_images.add(img)
        for img in self.preferences_df['image2']:
            all_images.add(img)
        
        # Count preferences per image
        image_counts = {}
        for img in all_images:
            count1 = self.preferences_df[self.preferences_df['image1'] == img].shape[0]
            count2 = self.preferences_df[self.preferences_df['image2'] == img].shape[0]
            image_counts[img] = count1 + count2
        
        # Calculate statistics
        counts = list(image_counts.values())
        return {
            'total_images': len(all_images),
            'min_comparisons': min(counts) if counts else 0,
            'max_comparisons': max(counts) if counts else 0,
            'mean_comparisons': np.mean(counts) if counts else 0,
            'median_comparisons': np.median(counts) if counts else 0,
            'uncovered_images': 0  # This would require knowing all available images
        }
    
    def get_comparison_matrix(self) -> Dict[str, set]:
        """
        Get a matrix of which images have been compared.
        
        Returns:
            Dict[str, set]: For each image, a set of images it has been compared with
        """
        comparison_matrix: Dict[str, set] = {}
        
        for _, row in self.preferences_df.iterrows():
            img1, img2 = row['image1'], row['image2']
            
            if img1 not in comparison_matrix:
                comparison_matrix[img1] = set()
            if img2 not in comparison_matrix:
                comparison_matrix[img2] = set()
                
            comparison_matrix[img1].add(img2)
            comparison_matrix[img2].add(img1)
        
        return comparison_matrix
    
    def clean_data(self) -> int:
        """
        Clean preference data by removing invalid records.
        - Remove duplicates
        - Remove records with missing image paths
        
        Returns:
            int: Number of records removed
        """
        initial_count = len(self.preferences_df)
        
        # Remove duplicates
        self.preferences_df = self.preferences_df.drop_duplicates()
        
        # Remove records with missing image paths
        self.preferences_df = self.preferences_df[
            self.preferences_df['image1'].notna() & self.preferences_df['image2'].notna()
        ]
        
        # Check for and fix any inconsistent data types
        if 'preferred' in self.preferences_df.columns:
            self.preferences_df['preferred'] = self.preferences_df['preferred'].astype(int)
        
        # Save cleaned data
        removed_count = initial_count - len(self.preferences_df)
        if removed_count > 0:
            self.save()
            logger.info(f"Cleaned {removed_count} invalid records from preference data")
        
        return removed_count
    
    def export_data(self, format: str = "csv", output_path: Optional[Path] = None) -> Path:
        """
        Export preference data in various formats.
        
        Args:
            format: Export format ('csv', 'json', 'excel')
            output_path: Path to save exported data (default: auto-generated)
            
        Returns:
            Path: Path to the exported file
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if format == "json":
                output_path = self.output_folder / f"preferences_export_{timestamp}.json"
            elif format == "excel":
                output_path = self.output_folder / f"preferences_export_{timestamp}.xlsx"
            else:  # Default to CSV
                output_path = self.output_folder / f"preferences_export_{timestamp}.csv"
        
        # Export based on format
        if format == "json":
            self.preferences_df.to_json(output_path, orient="records", indent=2)
        elif format == "excel":
            self.preferences_df.to_excel(output_path, index=False)
        else:  # Default to CSV
            self.preferences_df.to_csv(output_path, index=False)
        
        logger.info(f"Exported preference data to {output_path}")
        return output_path
    
    def import_data(
        self, 
        import_path: Union[str, Path], 
        merge_strategy: str = "append"
    ) -> int:
        """
        Import preference data from external file.
        
        Args:
            import_path: Path to the file to import
            merge_strategy: How to merge data ('append', 'replace', 'merge')
            
        Returns:
            int: Number of records imported
        """
        import_path = Path(import_path)
        
        # Load import data based on file extension
        if import_path.suffix.lower() == '.json':
            import_df = pd.read_json(import_path)
        elif import_path.suffix.lower() in ['.xlsx', '.xls']:
            import_df = pd.read_excel(import_path)
        else:  # Default to CSV
            import_df = pd.read_csv(import_path)
        
        # Process based on merge strategy
        if merge_strategy == "replace":
            old_df = self.preferences_df.copy()
            self.preferences_df = import_df
        elif merge_strategy == "merge":
            # Merge and remove duplicates
            combined_df = pd.concat([self.preferences_df, import_df], ignore_index=True)
            self.preferences_df = combined_df.drop_duplicates()
        else:  # Default to append
            self.preferences_df = pd.concat([self.preferences_df, import_df], ignore_index=True)
        
        # Save imported data
        record_count = len(self.preferences_df)
        self.save()
        
        if merge_strategy == "replace":
            # Create backup of replaced data
            backup_path = self.output_folder / f"preferences_replaced_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            old_df.to_csv(backup_path, index=False)
            
        logger.info(f"Imported {record_count} records from {import_path}")
        return record_count
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the preference data.
        
        Returns:
            Dict[str, Any]: Statistics about the preference data
        """
        stats: Dict[str, Any] = {}
        
        # Basic counts
        stats['total_records'] = len(self.preferences_df)
        
        # Count by preference
        if not self.preferences_df.empty:
            pref_counts = self.preferences_df['preferred'].value_counts().to_dict()
            stats['preferred_image1'] = pref_counts.get(1, 0)
            stats['preferred_image2'] = pref_counts.get(2, 0)
            stats['preferred_tie'] = pref_counts.get(0, 0)
        else:
            stats['preferred_image1'] = 0
            stats['preferred_image2'] = 0
            stats['preferred_tie'] = 0
        
        # Count by rater
        if not self.preferences_df.empty:
            stats['rater_counts'] = self.preferences_df['rater_id'].value_counts().to_dict()
        else:
            stats['rater_counts'] = {}
        
        # Response time statistics
        if 'response_time_ms' in self.preferences_df.columns and not self.preferences_df.empty:
            valid_times = self.preferences_df['response_time_ms'].dropna()
            
            if len(valid_times) > 0:
                stats['response_time'] = {
                    'min': valid_times.min(),
                    'max': valid_times.max(),
                    'mean': valid_times.mean(),
                    'median': valid_times.median()
                }
            else:
                stats['response_time'] = {'min': 0, 'max': 0, 'mean': 0, 'median': 0}
        else:
            stats['response_time'] = {'min': 0, 'max': 0, 'mean': 0, 'median': 0}
        
        # Image coverage statistics
        stats['image_coverage'] = self.get_image_coverage()
        
        # Time range
        if 'timestamp' in self.preferences_df.columns and not self.preferences_df.empty:
            try:
                # Convert ISO timestamps to datetime objects
                timestamps = pd.to_datetime(self.preferences_df['timestamp'])
                stats['time_range'] = {
                    'start': timestamps.min().isoformat(),
                    'end': timestamps.max().isoformat(),
                    'duration_hours': (timestamps.max() - timestamps.min()).total_seconds() / 3600
                }
            except (ValueError, TypeError):
                stats['time_range'] = {'start': None, 'end': None, 'duration_hours': 0}
        else:
            stats['time_range'] = {'start': None, 'end': None, 'duration_hours': 0}
        
        # Prompt usage statistics
        if 'prompt' in self.preferences_df.columns:
            empty_prompts = self.preferences_df['prompt'].isna() | (self.preferences_df['prompt'] == '')
            stats['prompt_usage'] = {
                'with_prompt': len(self.preferences_df) - empty_prompts.sum(),
                'without_prompt': empty_prompts.sum()
            }
        else:
            stats['prompt_usage'] = {'with_prompt': 0, 'without_prompt': len(self.preferences_df)}
        
        return stats
    
    def manage_backups(self, max_backups: int = 10) -> int:
        """
        Manage backup files by removing oldest backups if there are too many.
        
        Args:
            max_backups: Maximum number of backup files to keep
            
        Returns:
            int: Number of backup files removed
        """
        backup_files = list(self.output_folder.glob("preferences_backup_*.csv"))
        
        if len(backup_files) <= max_backups:
            return 0
        
        # Sort by modification time (oldest first)
        backup_files.sort(key=lambda x: x.stat().st_mtime)
        
        # Remove oldest backups
        to_remove = backup_files[:(len(backup_files) - max_backups)]
        
        for file_path in to_remove:
            try:
                file_path.unlink()
                logger.info(f"Removed old backup: {file_path}")
            except Exception as e:
                logger.error(f"Failed to remove backup {file_path}: {e}")
        
        return len(to_remove)
    
    def get_data_summary(self) -> str:
        """
        Get a human-readable summary of the preference data.
        
        Returns:
            str: Summary text
        """
        stats = self.get_stats()
        
        summary = [
            "Preference Data Summary",
            "-----------------------",
            f"Total records: {stats['total_records']}",
            f"  - Preferred image 1: {stats['preferred_image1']}",
            f"  - Preferred image 2: {stats['preferred_image2']}",
            f"  - Ties/skips: {stats['preferred_tie']}",
            "",
            "Image coverage:",
            f"  - Total unique images: {stats['image_coverage']['total_images']}",
            f"  - Average comparisons per image: {stats['image_coverage']['mean_comparisons']:.1f}",
            "",
        ]
        
        if stats['time_range']['start']:
            summary.extend([
                "Collection period:",
                f"  - Start: {stats['time_range']['start'].split('T')[0]}",
                f"  - End: {stats['time_range']['end'].split('T')[0]}",
                f"  - Duration: {stats['time_range']['duration_hours']:.1f} hours",
                ""
            ])
        
        if stats['rater_counts']:
            summary.extend([
                f"Raters: {len(stats['rater_counts'])}",
                f"  - {', '.join(f'{k}: {v}' for k, v in list(stats['rater_counts'].items())[:5])}"
            ])
            
            if len(stats['rater_counts']) > 5:
                summary.append(f"  - ...and {len(stats['rater_counts']) - 5} more")
        
        return "\n".join(summary)


class ModelCheckpointManager:
    """
    Manager for model checkpoints and metadata.
    Handles saving, loading, and metadata for model checkpoints.
    """
    
    def __init__(self, model_output_dir: Union[str, Path]):
        """
        Initialize the checkpoint manager.
        
        Args:
            model_output_dir: Directory for model checkpoints
        """
        self.model_output_dir = Path(model_output_dir)
        self.model_output_dir.mkdir(exist_ok=True, parents=True)
    
    def get_checkpoints(self) -> List[Dict[str, Any]]:
        """
        Get list of available checkpoints with metadata.
        
        Returns:
            List[Dict[str, Any]]: List of checkpoint information
        """
        checkpoints = []
        
        # Find all checkpoint files
        checkpoint_files = list(self.model_output_dir.glob("reward_model_epoch_*.safetensors"))
        
        for ckpt_file in checkpoint_files:
            # Try to load metadata
            metadata_file = Path(str(ckpt_file).replace('.safetensors', '_metadata.json'))
            
            checkpoint_info = {
                'path': ckpt_file,
                'filename': ckpt_file.name,
                'size_mb': ckpt_file.stat().st_size / (1024 * 1024),
                'modified': datetime.fromtimestamp(ckpt_file.stat().st_mtime).isoformat()
            }
            
            if metadata_file.exists():
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    
                    # Add metadata to checkpoint info
                    checkpoint_info.update(metadata)
                except Exception as e:
                    logger.error(f"Error loading checkpoint metadata from {metadata_file}: {e}")
            
            checkpoints.append(checkpoint_info)
        
        # Sort by epoch
        checkpoints.sort(key=lambda x: x.get('epoch', 0)) # type: ignore
        
        return checkpoints
    
    def get_best_checkpoint(self, metric: str = 'loss') -> Optional[Dict[str, Any]]:
        """
        Get the best checkpoint based on a metric.
        
        Args:
            metric: Metric to use for ranking ('loss' or 'accuracy')
            
        Returns:
            Optional[Dict[str, Any]]: Best checkpoint info or None if no checkpoints
        """
        checkpoints = self.get_checkpoints()
        
        if not checkpoints:
            return None
        
        # Filter out checkpoints without the metric
        valid_checkpoints = [c for c in checkpoints if metric in c]
        
        if not valid_checkpoints:
            return checkpoints[-1]  # Return latest if no metrics available
        
        # Sort by metric (ascending for loss, descending for accuracy)
        if metric == 'accuracy':
            sorted_checkpoints = sorted(valid_checkpoints, key=lambda x: x[metric], reverse=True)
        else:
            sorted_checkpoints = sorted(valid_checkpoints, key=lambda x: x[metric])
        
        return sorted_checkpoints[0]
    
    def clean_checkpoints(
        self, 
        keep_best: bool = True, 
        keep_latest: bool = True, 
        keep_first: bool = False,
        max_to_keep: int = 5
    ) -> int:
        """
        Clean checkpoint directory by removing unnecessary checkpoints.
        
        Args:
            keep_best: Whether to keep the best checkpoint
            keep_latest: Whether to keep the latest checkpoint
            keep_first: Whether to keep the first checkpoint
            max_to_keep: Maximum number of checkpoints to keep
            
        Returns:
            int: Number of checkpoints removed
        """
        checkpoints = self.get_checkpoints()
        
        if len(checkpoints) <= max_to_keep:
            return 0
        
        # Determine which checkpoints to keep
        to_keep = []
        
        if keep_best:
            best = self.get_best_checkpoint()
            if best:
                to_keep.append(str(best['path']))
        
        if keep_latest and checkpoints:
            latest = max(checkpoints, key=lambda x: x.get('epoch', 0))
            to_keep.append(str(latest['path']))
        
        if keep_first and checkpoints:
            first = min(checkpoints, key=lambda x: x.get('epoch', 0))
            to_keep.append(str(first['path']))
        
        # If we still need more, keep evenly spaced checkpoints
        if len(checkpoints) - len(to_keep) > max_to_keep:
            # Sort by epoch
            sorted_ckpts = sorted(checkpoints, key=lambda x: x.get('epoch', 0))
            
            # Calculate spacing
            remaining_to_keep = max_to_keep - len(to_keep)
            if remaining_to_keep > 0 and len(sorted_ckpts) > remaining_to_keep:
                indices = np.linspace(0, len(sorted_ckpts) - 1, remaining_to_keep).astype(int)
                for i in indices:
                    to_keep.append(str(sorted_ckpts[i]['path']))
        
        # Remove duplicates
        to_keep = list(set(to_keep))
        
        # Remove checkpoints not in the keep list
        removed = 0
        for checkpoint in checkpoints:
            ckpt_path = checkpoint['path']
            if str(ckpt_path) not in to_keep:
                try:
                    # Remove checkpoint file
                    ckpt_path.unlink()
                    
                    # Remove metadata file
                    metadata_file = Path(str(ckpt_path).replace('.safetensors', '_metadata.json'))
                    if metadata_file.exists():
                        metadata_file.unlink()
                    
                    removed += 1
                    logger.info(f"Removed checkpoint: {ckpt_path}")
                except Exception as e:
                    logger.error(f"Failed to remove checkpoint {ckpt_path}: {e}")
        
        return removed