from typing import Dict, List, Optional
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from app.infrastructure.connectors.postgres.database import db_session


class PostgresService:
    def __init__(self):
        self.async_session = db_session.get_session()

    async def execute_raw_query(self, query: str, params: Optional[Dict] = None) -> Optional[List[Dict]]:
        """
        Execute a raw SQL query.
        Returns rows as a list of dicts for SELECT, or None for DML.
        """
        try:
            async with self.async_session() as session:
                async with session.begin():
                    result = await session.execute(text(query), params)
                    if result.returns_rows:
                        return [row for row in result.mappings()]
                    return None
        except SQLAlchemyError as e:
            raise e
