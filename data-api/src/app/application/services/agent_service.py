"""Service for agent management operations."""

import logging
from typing import List, Optional, Dict, Any

from app.infrastructure.connectors.postgres.repositories.agent_repository import AgentRepository
from app.infrastructure.connectors.postgres.schema import Agent
from app.exceptions import ResourceNotFoundError, ValidationError


logger = logging.getLogger(__name__)


def serialize_agent(agent: Agent) -> Dict[str, Any]:
    """Convert an Agent ORM row into the dict shape the frontend expects.

    The columns ``llm_temperature`` and ``is_active`` are stored as strings;
    coerce them back to ``float`` / ``bool`` so consumers don't have to.
    """
    data = agent.to_dict()
    raw_temp = data.get("llm_temperature")
    if raw_temp is not None:
        try:
            data["llm_temperature"] = float(raw_temp)
        except (TypeError, ValueError):
            pass
    raw_active = data.get("is_active")
    if isinstance(raw_active, str):
        data["is_active"] = raw_active.lower() == "true"
    return data


class AgentService:
    """Service for agent CRUD operations."""

    def __init__(self):
        self.repository = AgentRepository()

    async def create_agent(
        self,
        name: str,
        llm_model: str,
        description: str = None,
        system_prompt: str = None,
        llm_temperature: float = 0.7,
        tenant_id: str = None,
        created_by: str = None,
    ) -> Agent:
        """Create a new agent."""
        logger.info(f"Creating agent: {name}")
        return await self.repository.create(
            name=name,
            llm_model=llm_model,
            description=description,
            system_prompt=system_prompt,
            llm_temperature=str(llm_temperature),
            is_active="true",
            tenant_id=tenant_id,
            created_by=created_by,
        )

    async def get_agent(self, agent_id: str) -> Agent:
        """Get an agent by ID."""
        return await self.repository.get(id=agent_id)

    async def update_agent(self, agent_id: str, **kwargs) -> Agent:
        """Update an agent."""
        data = {k: v for k, v in kwargs.items() if v is not None}
        if not data:
            raise ValidationError("No fields to update")
        # Coerce types to match String columns in schema
        if "llm_temperature" in data:
            data["llm_temperature"] = str(data["llm_temperature"])
        if "is_active" in data:
            data["is_active"] = str(data["is_active"]).lower()

        logger.info(f"Updating agent: {agent_id}")
        return await self.repository.update(data=data, where={"id": agent_id})

    async def delete_agent(self, agent_id: str) -> None:
        """Delete an agent."""
        logger.info(f"Deleting agent: {agent_id}")
        await self.repository.delete(id=agent_id)

    async def list_agents(
        self,
        skip: int = 0,
        limit: int = 10,
        tenant_id: str = None,
        is_active: bool = None,
    ) -> List[Agent]:
        """List agents with optional filters."""
        where = {}
        if tenant_id:
            where["tenant_id"] = tenant_id
        if is_active is not None:
            # Schema stores is_active as String ("true"/"false")
            where["is_active"] = str(is_active).lower()

        return await self.repository.paging(
            skip=skip,
            limit=limit,
            where=where if where else None,
            order_by={"create_time": "desc"},
        )

    async def get_agent_with_kb_ids(self, agent_id: str) -> Dict[str, Any]:
        """Get an agent with its linked knowledge bases (id + name pairs)."""
        agent = await self.repository.get(id=agent_id)
        kbs = await self.repository.get_linked_kbs(agent_id)

        result = serialize_agent(agent)
        result["kb_ids"] = [kb["id"] for kb in kbs]
        result["knowledge_bases"] = kbs
        return result

    async def link_knowledge_base(self, agent_id: str, kb_id: str) -> Dict[str, str]:
        """Link a knowledge base to an agent."""
        await self.repository.get(id=agent_id)

        logger.info(f"Linking KB {kb_id} to agent {agent_id}")
        link = await self.repository.link_knowledge_base(agent_id, kb_id)
        return {"agent_id": agent_id, "kb_id": kb_id, "link_id": link.id}

    async def unlink_knowledge_base(self, agent_id: str, kb_id: str) -> None:
        """Unlink a knowledge base from an agent."""
        logger.info(f"Unlinking KB {kb_id} from agent {agent_id}")
        await self.repository.unlink_knowledge_base(agent_id, kb_id)


def get_agent_service() -> AgentService:
    """Factory function to get AgentService instance."""
    return AgentService()
