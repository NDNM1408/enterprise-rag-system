from typing import List, Dict, Optional
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError


class PostgresService:
    """
    Thin raw-SQL executor used by tasks that don't fit neatly into a
    typed repository (currently: dlq_tasks writes to dlq_log).

    Accepts an async_sessionmaker from the caller.  When none is provided
    it falls back to the container's shared NullPool session_factory,
    which is always safe to use from any asyncio.run() context.
    """

    def __init__(self, session_factory=None):
        if session_factory is not None:
            self.async_session = session_factory
        else:
            from app.container import container
            self.async_session = container.session_factory

    async def execute_raw_query(
        self, query: str, params: Optional[Dict] = None
    ) -> Optional[List[Dict]]:
        """
        Execute a raw SQL query.

        Returns rows as a list of dicts for SELECT statements, or None for
        DML (INSERT / UPDATE / DELETE).
        """
        try:
            async with self.async_session() as session:
                async with session.begin():
                    result = await session.execute(text(query), params)
                    if result.returns_rows:
                        return [dict(row) for row in result.mappings()]
                    return None
        except SQLAlchemyError as e:
            raise e
