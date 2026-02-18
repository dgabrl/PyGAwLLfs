"""
Experiment Performance Logger.
Handles SQLite persistence for timing and hardware usage statistics.
"""

import sqlite3
import os
from datetime import datetime
from typing import Dict, Any

class ExperimentLogger:
    def __init__(self, db_name: str = 'time_statistics.db'):
        self.results_dir = os.path.join(os.getcwd(), 'results')
        os.makedirs(self.results_dir, exist_ok=True)
        self.db_path = os.path.join(self.results_dir, db_name)
        self.conn = sqlite3.connect(self.db_path)
        self._create_schema()

    def _create_schema(self) -> None:
        cursor = self.conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS experiment_groups(
                id_group INTEGER PRIMARY KEY AUTOINCREMENT,
                dataset TEXT,
                model TEXT,
                n_runs_planned INTEGER,
                timestamp DATETIME
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS run_times(
                id_run INTEGER PRIMARY KEY AUTOINCREMENT,
                id_group INTEGER,
                run_index INTEGER,
                time_gawll REAL,
                time_permutation REAL,
                time_intrinsic REAL,
                peak_ram_gb REAL,
                FOREIGN KEY (id_group) REFERENCES experiment_groups (id_group)
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS final_summaries(
                id_group INTEGER PRIMARY KEY,
                avg_time_gawll REAL,
                avg_time_permutation REAL,
                avg_time_intrinsic REAL,
                avg_peak_ram_gb REAL,
                FOREIGN KEY (id_group) REFERENCES experiment_groups (id_group)
            )
        ''')
        self.conn.commit()

    def register_experiment(self, dataset: str, model: str, n_runs: int) -> int:
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO experiment_groups (dataset, model, n_runs_planned, timestamp)
            VALUES (?, ?, ?, ?)
        ''', (dataset, model, n_runs, datetime.now()))
        self.conn.commit()
        return int(cursor.lastrowid)

    def log_run(self, id_group: int, run_data: Dict[str, Any]) -> None:
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO run_times (id_group, run_index, time_gawll, time_permutation, time_intrinsic, peak_ram_gb)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (id_group, run_data['run'], run_data['time_gawll'],
              run_data['time_perm'], run_data['time_intr'], run_data['peak_ram']))
        self.conn.commit()

    def finalize_summary(self, id_group: int) -> None:
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO final_summaries (
                id_group, avg_time_gawll, avg_time_permutation, avg_time_intrinsic, avg_peak_ram_gb
            )
            SELECT 
                id_group, AVG(time_gawll), AVG(time_permutation), AVG(time_intrinsic), AVG(peak_ram_gb)
            FROM run_times
            WHERE id_group = ?
        ''', (id_group,))
        self.conn.commit()

    def close(self) -> None:
        if self.conn: self.conn.close()