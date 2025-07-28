"""
Database Migration Script - NEAT Options System
Production-ready database setup and migration management.
"""

import sqlite3
import os
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import json

class DatabaseMigration:
    """Handle database migrations for NEAT Options System."""
    
    def __init__(self, db_path: str = "data/neat_agents.db"):
        """Initialize database migration manager."""
        self.db_path = db_path
        self.schema_path = "src/database/agent_schema.sql"
        
        # Ensure data directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Migration history
        self.migrations = [
            {
                'version': 1,
                'name': 'initial_schema',
                'description': 'Create initial NEAT agent database schema',
                'sql_file': 'src/database/agent_schema.sql'
            },
            {
                'version': 2,
                'name': 'performance_indexes',
                'description': 'Add performance optimization indexes',
                'sql_commands': [
                    "CREATE INDEX IF NOT EXISTS idx_agent_performance_symbol_date ON agent_performance(symbol, date);",
                    "CREATE INDEX IF NOT EXISTS idx_training_sessions_symbol_status ON training_sessions(symbol, status);"
                ]
            },
            {
                'version': 3,
                'name': 'market_data_tables',
                'description': 'Add market data and regime tables',
                'sql_commands': [
                    """
                    CREATE TABLE IF NOT EXISTS market_data (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        date DATE NOT NULL,
                        open REAL,
                        high REAL,
                        low REAL,
                        close REAL,
                        volume INTEGER,
                        adj_close REAL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(symbol, date)
                    );
                    """,
                    """
                    CREATE TABLE IF NOT EXISTS market_indicators (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        date DATE NOT NULL,
                        vix REAL,
                        spy_close REAL,
                        treasury_10y REAL,
                        dxy REAL,
                        market_regime TEXT,
                        regime_confidence REAL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(date)
                    );
                    """,
                    "CREATE INDEX IF NOT EXISTS idx_market_data_symbol_date ON market_data(symbol, date);",
                    "CREATE INDEX IF NOT EXISTS idx_market_indicators_date ON market_indicators(date);"
                ]
            }
        ]
    
    def run_migrations(self) -> bool:
        """Run all pending migrations."""
        
        self.logger.info("Starting database migrations...")
        
        try:
            # Create connection
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create migration tracking table
            self._create_migration_table(cursor)
            
            # Get current migration version
            current_version = self._get_current_version(cursor)
            self.logger.info(f"Current database version: {current_version}")
            
            # Run pending migrations
            migrations_run = 0
            
            for migration in self.migrations:
                if migration['version'] > current_version:
                    self.logger.info(f"Running migration {migration['version']}: {migration['name']}")
                    
                    success = self._run_migration(cursor, migration)
                    
                    if success:
                        self._record_migration(cursor, migration)
                        migrations_run += 1
                        self.logger.info(f"Migration {migration['version']} completed successfully")
                    else:
                        self.logger.error(f"Migration {migration['version']} failed")
                        conn.rollback()
                        return False
            
            # Commit all changes
            conn.commit()
            
            if migrations_run > 0:
                self.logger.info(f"Successfully ran {migrations_run} migrations")
            else:
                self.logger.info("Database is up to date")
            
            # Verify database integrity
            self._verify_database(cursor)
            
            conn.close()
            return True
            
        except Exception as e:
            self.logger.error(f"Migration failed: {str(e)}")
            return False
    
    def _create_migration_table(self, cursor: sqlite3.Cursor):
        """Create migration tracking table."""
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                checksum TEXT
            )
        """)
    
    def _get_current_version(self, cursor: sqlite3.Cursor) -> int:
        """Get current database version."""
        
        try:
            cursor.execute("SELECT MAX(version) FROM schema_migrations")
            result = cursor.fetchone()
            return result[0] if result[0] is not None else 0
        except sqlite3.OperationalError:
            # Table doesn't exist yet
            return 0
    
    def _run_migration(self, cursor: sqlite3.Cursor, migration: Dict) -> bool:
        """Run a single migration."""
        
        try:
            if 'sql_file' in migration:
                # Run SQL from file
                if os.path.exists(migration['sql_file']):
                    with open(migration['sql_file'], 'r') as f:
                        sql_content = f.read()
                    
                    # Execute SQL commands
                    cursor.executescript(sql_content)
                else:
                    self.logger.error(f"SQL file not found: {migration['sql_file']}")
                    return False
            
            elif 'sql_commands' in migration:
                # Run SQL commands directly
                for sql_command in migration['sql_commands']:
                    cursor.execute(sql_command)
            
            else:
                self.logger.error(f"No SQL content found for migration {migration['version']}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error running migration {migration['version']}: {str(e)}")
            return False
    
    def _record_migration(self, cursor: sqlite3.Cursor, migration: Dict):
        """Record successful migration."""
        
        cursor.execute("""
            INSERT INTO schema_migrations (version, name, description, applied_at)
            VALUES (?, ?, ?, ?)
        """, (
            migration['version'],
            migration['name'],
            migration['description'],
            datetime.now()
        ))
    
    def _verify_database(self, cursor: sqlite3.Cursor):
        """Verify database integrity and structure."""
        
        self.logger.info("Verifying database integrity...")
        
        # Check integrity
        cursor.execute("PRAGMA integrity_check")
        integrity_result = cursor.fetchone()
        
        if integrity_result[0] != "ok":
            raise Exception(f"Database integrity check failed: {integrity_result[0]}")
        
        # Verify required tables exist
        required_tables = [
            'agents', 'agent_performance', 'historical_agents', 
            'market_regimes', 'training_sessions', 'predictions',
            'monitoring_metrics', 'stock_metadata'
        ]
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        existing_tables = [row[0] for row in cursor.fetchall()]
        
        missing_tables = [table for table in required_tables if table not in existing_tables]
        
        if missing_tables:
            self.logger.warning(f"Missing tables: {missing_tables}")
        else:
            self.logger.info("All required tables present")
        
        # Check indexes
        cursor.execute("SELECT name FROM sqlite_master WHERE type='index'")
        indexes = [row[0] for row in cursor.fetchall()]
        
        self.logger.info(f"Database has {len(indexes)} indexes")
        
        self.logger.info("Database verification completed successfully")
    
    def backup_database(self, backup_path: Optional[str] = None) -> str:
        """Create database backup."""
        
        if backup_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"data/backups/neat_agents_backup_{timestamp}.db"
        
        # Ensure backup directory exists
        os.makedirs(os.path.dirname(backup_path), exist_ok=True)
        
        try:
            # Copy database file
            import shutil
            shutil.copy2(self.db_path, backup_path)
            
            self.logger.info(f"Database backed up to: {backup_path}")
            return backup_path
            
        except Exception as e:
            self.logger.error(f"Backup failed: {str(e)}")
            raise
    
    def restore_database(self, backup_path: str) -> bool:
        """Restore database from backup."""
        
        if not os.path.exists(backup_path):
            self.logger.error(f"Backup file not found: {backup_path}")
            return False
        
        try:
            # Create current backup first
            current_backup = self.backup_database()
            self.logger.info(f"Current database backed up to: {current_backup}")
            
            # Restore from backup
            import shutil
            shutil.copy2(backup_path, self.db_path)
            
            self.logger.info(f"Database restored from: {backup_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Restore failed: {str(e)}")
            return False
    
    def get_migration_status(self) -> Dict:
        """Get current migration status."""
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create migration table if it doesn't exist
            self._create_migration_table(cursor)
            
            # Get applied migrations
            try:
                cursor.execute("""
                    SELECT version, name, description, applied_at 
                    FROM schema_migrations 
                    ORDER BY version
                """)
                
                applied_migrations = []
                for row in cursor.fetchall():
                    applied_migrations.append({
                        'version': row[0],
                        'name': row[1],
                        'description': row[2],
                        'applied_at': row[3]
                    })
            except sqlite3.OperationalError:
                applied_migrations = []
            
            # Get pending migrations
            current_version = self._get_current_version(cursor)
            pending_migrations = [
                m for m in self.migrations if m['version'] > current_version
            ]
            
            conn.close()
            
            return {
                'current_version': current_version,
                'applied_migrations': applied_migrations,
                'pending_migrations': pending_migrations,
                'database_exists': os.path.exists(self.db_path),
                'database_size_mb': os.path.getsize(self.db_path) / (1024 * 1024) if os.path.exists(self.db_path) else 0
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get migration status: {str(e)}")
            return {
                'current_version': 0,
                'applied_migrations': [],
                'pending_migrations': self.migrations,
                'database_exists': os.path.exists(self.db_path),
                'database_size_mb': os.path.getsize(self.db_path) / (1024 * 1024) if os.path.exists(self.db_path) else 0,
                'error': str(e)
            }
    
    def reset_database(self) -> bool:
        """Reset database (delete and recreate)."""
        
        self.logger.warning("Resetting database - all data will be lost!")
        
        try:
            # Backup current database if it exists
            if os.path.exists(self.db_path):
                backup_path = self.backup_database()
                self.logger.info(f"Current database backed up to: {backup_path}")
                
                # Remove current database
                os.remove(self.db_path)
            
            # Run migrations to recreate
            return self.run_migrations()
            
        except Exception as e:
            self.logger.error(f"Database reset failed: {str(e)}")
            return False


def main():
    """Run database migrations."""
    
    print("🗄️ NEAT Options System - Database Migration")
    print("=" * 50)
    
    migration = DatabaseMigration()
    
    # Get current status
    status = migration.get_migration_status()
    
    print(f"Database exists: {status.get('database_exists', False)}")
    print(f"Current version: {status.get('current_version', 0)}")
    print(f"Database size: {status.get('database_size_mb', 0):.1f} MB")
    
    if status.get('pending_migrations'):
        print(f"Pending migrations: {len(status['pending_migrations'])}")
        
        # Run migrations
        print("\nRunning migrations...")
        success = migration.run_migrations()
        
        if success:
            print("✅ Migrations completed successfully!")
        else:
            print("❌ Migration failed!")
            return False
    else:
        print("✅ Database is up to date!")
    
    # Final status
    final_status = migration.get_migration_status()
    print(f"\nFinal database version: {final_status.get('current_version', 0)}")
    print(f"Final database size: {final_status.get('database_size_mb', 0):.1f} MB")
    
    return True


if __name__ == "__main__":
    main()
