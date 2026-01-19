"""add key_value_bbox

Revision ID: 002_add_key_value_bbox
Revises: 001_initial
Create Date: 2026-01-19 16:30:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

# revision identifiers, used by Alembic.
revision = '002_add_key_value_bbox'
down_revision = '001_add_bbox_columns'  # Correct revision ID from 001_add_bbox_columns.py
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add new columns for enhanced bounding box support
    op.add_column('extracted_fields', sa.Column('key_bbox', JSONB, nullable=True, comment='Key bounding box: {polygon, matched_text, confidence, page}'))
    op.add_column('extracted_fields', sa.Column('value_bbox', JSONB, nullable=True, comment='Value bounding box: {polygon, matched_text, confidence, page}'))
    op.add_column('extracted_fields', sa.Column('original_ocr_text', sa.Text, nullable=True, comment='Original OCR text matched to value'))
    
    # Drop the old single bounding_box column
    op.drop_column('extracted_fields', 'bounding_box')


def downgrade() -> None:
    # Revert changes
    op.add_column('extracted_fields', sa.Column('bounding_box', JSONB, nullable=True, comment='Position: {x, y, width, height, page}'))
    op.drop_column('extracted_fields', 'original_ocr_text')
    op.drop_column('extracted_fields', 'value_bbox')
    op.drop_column('extracted_fields', 'key_bbox')
