"""set embedding dimension to 768

Revision ID: 20260619_2328_set_emb_dim_to_768
Revises: b8fcfbfe81c0
Create Date: 2026-06-19 23:28:00.000000+00:00
"""
from typing import Sequence, Union

from alembic import op
from pgvector.sqlalchemy import Vector

# revision identifiers, used by Alembic.
revision: str = '20260619_2328_set_emb_dim_to_768'
down_revision: Union[str, None] = 'b8fcfbfe81c0'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("TRUNCATE TABLE episodic_memories, semantic_memories, procedural_memories CASCADE")
    op.alter_column('episodic_memories', 'embedding', type_=Vector(768), existing_type=Vector(384))
    op.alter_column('semantic_memories', 'embedding', type_=Vector(768), existing_type=Vector(384))
    op.alter_column('procedural_memories', 'embedding', type_=Vector(768), existing_type=Vector(384))


def downgrade() -> None:
    op.alter_column('episodic_memories', 'embedding', type_=Vector(384), existing_type=Vector(768))
    op.alter_column('semantic_memories', 'embedding', type_=Vector(384), existing_type=Vector(768))
    op.alter_column('procedural_memories', 'embedding', type_=Vector(384), existing_type=Vector(768))


_ = (revision, down_revision, branch_labels, depends_on)
