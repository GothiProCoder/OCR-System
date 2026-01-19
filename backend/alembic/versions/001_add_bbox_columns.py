"""Add bounding box columns to extraction table

Revision ID: 001_add_bbox_columns
Revises: 
Create Date: 2026-01-16

This migration adds columns for bounding box highlighting feature:
- layout_data: JSONB storing all OCR bounding boxes (words, lines, tables)
- processed_image_paths: JSONB mapping page numbers to processed image file paths
- page_dimensions: JSONB storing page dimensions for coordinate transformation
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB


# revision identifiers, used by Alembic
revision = '001_add_bbox_columns'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add bounding box related columns to extractions table."""
    
    # Add layout_data column
    op.add_column(
        'extractions',
        sa.Column(
            'layout_data',
            JSONB,
            nullable=True,
            server_default='[]',
            comment='All bounding boxes from OCR (words, lines, tables)'
        )
    )
    
    # Add processed_image_paths column
    op.add_column(
        'extractions',
        sa.Column(
            'processed_image_paths',
            JSONB,
            nullable=True,
            server_default='{}',
            comment='Paths to processed images by page: {"1": "path/to/page1.jpg"}'
        )
    )
    
    # Add page_dimensions column
    op.add_column(
        'extractions',
        sa.Column(
            'page_dimensions',
            JSONB,
            nullable=True,
            server_default='{}',
            comment='Page dimensions: {page_num: {width, height, unit}}'
        )
    )
    
    print("✅ Added layout_data, processed_image_paths, page_dimensions columns to extractions table")


def downgrade() -> None:
    """Remove bounding box related columns from extractions table."""
    
    op.drop_column('extractions', 'page_dimensions')
    op.drop_column('extractions', 'processed_image_paths')
    op.drop_column('extractions', 'layout_data')
    
    print("✅ Removed bounding box columns from extractions table")
