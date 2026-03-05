from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.ext.asyncio import async_sessionmaker
from app.configurations.configurations import settings



class DatabaseSession:
    _instance = None 
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DatabaseSession, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    def __init__(self):
        if not self._initialized:
            self.engine = create_async_engine(
                settings.DATABASE_URL,
                pool_size=20,
                max_overflow=10,
                pool_timeout=30,
                pool_pre_ping=True,  # Reconnect on stale connections
            )
            self.async_session = async_sessionmaker(self.engine, expire_on_commit=False)
            self._initialized = True

    def get_session(self) -> AsyncSession:
        return self.async_session
    def get_engine(self):
        return self.engine

db_session = DatabaseSession()
