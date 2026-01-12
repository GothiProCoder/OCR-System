#!/usr/bin/env python3
"""
Database Initialization Script
==============================
One-command database setup for development.

Usage:
    python scripts/init_db.py              # Create all tables
    python scripts/init_db.py --reset      # Drop and recreate all tables
    python scripts/init_db.py --check      # Check connection only
    
Requirements:
    1. PostgreSQL running (docker-compose up -d postgres OR local)
    2. .env file configured with database credentials
"""

import sys
import argparse
from pathlib import Path

# Add backend to path for imports
backend_path = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_path))

from config import settings


def check_env_file():
    """Verify .env file exists"""
    env_path = Path(__file__).parent.parent / ".env"
    env_example = Path(__file__).parent.parent / ".env.example"
    
    if not env_path.exists():
        print("❌ .env file not found!")
        print(f"   Copy {env_example} to {env_path} and configure it.")
        print("\n   Quick command:")
        print(f"   cp {env_example} {env_path}")
        return False
    return True


def check_connection():
    """Test database connectivity"""
    print("🔌 Testing database connection...")
    print(f"   Host: {settings.DB_HOST}:{settings.DB_PORT}")
    print(f"   Database: {settings.DB_NAME}")
    print(f"   User: {settings.DB_USER}")
    
    try:
        from sqlalchemy import create_engine, text
        
        # Try connecting
        engine = create_engine(settings.database_url)
        with engine.connect() as conn:
            result = conn.execute(text("SELECT version();"))
            version = result.scalar()
            print(f"\n✅ Connected successfully!")
            print(f"   PostgreSQL: {version}")
            return True
            
    except Exception as e:
        print(f"\n❌ Connection failed: {e}")
        print("\n💡 Troubleshooting:")
        print("   1. Is PostgreSQL running?")
        print("      docker-compose up -d postgres")
        print("   2. Check your .env credentials")
        print("   3. Verify database 'ocr_system' exists")
        return False


def create_database_if_not_exists():
    """Create the database if it doesn't exist"""
    print("\n📦 Checking if database exists...")
    
    try:
        from sqlalchemy import create_engine, text
        from sqlalchemy.exc import ProgrammingError
        
        # Connect to default 'postgres' database to create our database
        postgres_url = f"postgresql://{settings.DB_USER}:{settings.DB_PASSWORD}@{settings.DB_HOST}:{settings.DB_PORT}/postgres"
        engine = create_engine(postgres_url, isolation_level="AUTOCOMMIT")
        
        with engine.connect() as conn:
            # Check if database exists
            result = conn.execute(
                text(f"SELECT 1 FROM pg_database WHERE datname = '{settings.DB_NAME}'")
            )
            exists = result.scalar()
            
            if exists:
                print(f"   Database '{settings.DB_NAME}' already exists ✓")
            else:
                print(f"   Creating database '{settings.DB_NAME}'...")
                conn.execute(text(f'CREATE DATABASE "{settings.DB_NAME}"'))
                print(f"   Database created ✓")
        
        engine.dispose()
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def create_tables():
    """Create all database tables"""
    print("\n🏗️  Creating database tables...")
    
    try:
        from database.connection import sync_engine, Base
        from database import models  # Import to register all models
        
        # Create all tables
        Base.metadata.create_all(bind=sync_engine)
        
        # List created tables
        print("\n   Tables created:")
        for table_name in Base.metadata.tables.keys():
            print(f"   ✓ {table_name}")
        
        print("\n✅ All tables created successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error creating tables: {e}")
        import traceback
        traceback.print_exc()
        return False


def drop_tables():
    """Drop all database tables"""
    print("\n⚠️  Dropping all tables...")
    
    try:
        from database.connection import sync_engine, Base
        from database import models
        
        Base.metadata.drop_all(bind=sync_engine)
        print("   All tables dropped ✓")
        return True
        
    except Exception as e:
        print(f"\n❌ Error dropping tables: {e}")
        return False


def create_initial_data():
    """Create initial/seed data (optional)"""
    print("\n🌱 Creating initial data...")
    
    try:
        from database.connection import get_db
        from database.models import FormTemplate
        
        with get_db() as db:
            # Check if templates already exist
            existing = db.query(FormTemplate).first()
            if existing:
                print("   Initial data already exists, skipping...")
                return True
            
            # Create default form templates
            templates = [
                FormTemplate(
                    name="General Form",
                    description="Generic form extraction template",
                    expected_fields=[
                        {"key": "Name", "type": "name"},
                        {"key": "Date", "type": "date"},
                        {"key": "Email", "type": "email"},
                        {"key": "Phone", "type": "phone"},
                    ]
                ),
                FormTemplate(
                    name="Invoice",
                    description="Invoice and billing document template",
                    expected_fields=[
                        {"key": "Invoice Number", "type": "text"},
                        {"key": "Date", "type": "date"},
                        {"key": "Total Amount", "type": "currency"},
                        {"key": "Vendor Name", "type": "name"},
                    ]
                ),
                FormTemplate(
                    name="Application Form",
                    description="Job or general application template",
                    expected_fields=[
                        {"key": "Full Name", "type": "name"},
                        {"key": "Email", "type": "email"},
                        {"key": "Phone", "type": "phone"},
                        {"key": "Address", "type": "address"},
                        {"key": "Date of Birth", "type": "date"},
                    ]
                ),
            ]
            
            for template in templates:
                db.add(template)
            
            db.commit()
            print(f"   Created {len(templates)} form templates ✓")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error creating initial data: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Database initialization script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/init_db.py              # Full setup
    python scripts/init_db.py --check      # Test connection only
    python scripts/init_db.py --reset      # Drop and recreate tables
    python scripts/init_db.py --no-seed    # Skip initial data
        """
    )
    parser.add_argument(
        "--check", 
        action="store_true",
        help="Only check database connection"
    )
    parser.add_argument(
        "--reset", 
        action="store_true",
        help="Drop existing tables and recreate (DESTRUCTIVE!)"
    )
    parser.add_argument(
        "--no-seed", 
        action="store_true",
        help="Skip creating initial seed data"
    )
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("🚀 OCR System - Database Initialization")
    print("=" * 50)
    
    # Check .env exists
    if not check_env_file():
        sys.exit(1)
    
    # Check only mode
    if args.check:
        success = check_connection()
        sys.exit(0 if success else 1)
    
    # Create database if needed
    if not create_database_if_not_exists():
        sys.exit(1)
    
    # Check connection
    if not check_connection():
        sys.exit(1)
    
    # Reset mode - drop tables first
    if args.reset:
        confirm = input("\n⚠️  This will DELETE all data. Continue? (yes/no): ")
        if confirm.lower() != "yes":
            print("Aborted.")
            sys.exit(0)
        if not drop_tables():
            sys.exit(1)
    
    # Create tables
    if not create_tables():
        sys.exit(1)
    
    # Create seed data
    if not args.no_seed:
        if not create_initial_data():
            sys.exit(1)
    
    print("\n" + "=" * 50)
    print("✅ Database initialization complete!")
    print("=" * 50)
    print("\nNext steps:")
    print("  1. Start the backend: cd backend && uvicorn main:app --reload")
    print("  2. Start the frontend: cd frontend && streamlit run app.py")


if __name__ == "__main__":
    main()
