"""Service for agent management operations."""

import logging
from typing import List, Optional, Dict, Any

from app.infrastructure.repositories import AgentRepository
from app.infrastructure.connectors.postgres.schema import Agent
from app.exceptions import ResourceNotFoundError, ValidationError


logger = logging.getLogger(__name__)


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
        """
        Create a new agent.

        Args:
            name: Agent name
            llm_model: LLM model to use
            description: Agent description
            system_prompt: System prompt
            llm_temperature: LLM temperature
            tenant_id: Tenant ID
            created_by: Creator user ID

        Returns:
            Created agent
        """
        logger.info(f"Creating agent: {name}")
        return await self.repository.create(
            name=name,
            llm_model=llm_model,
            description=description,
            system_prompt=system_prompt,
            llm_temperature=llm_temperature,
            is_active=True,
            tenant_id=tenant_id,
            created_by=created_by,
        )

    async def get_agent(self, agent_id: str) -> Agent:
        """
        Get an agent by ID.

        Args:
            agent_id: Agent ID

        Returns:
            Agent instance

        Raises:
            ResourceNotFoundError: If agent not found
        """
        return await self.repository.get(id=agent_id)

    async def update_agent(
        self,
        agent_id: str,
        **kwargs,
    ) -> Agent:
        """
        Update an agent.

        Args:
            agent_id: Agent ID
            **kwargs: Fields to update

        Returns:
            Updated agent

        Raises:
            ResourceNotFoundError: If agent not found
        """
        # Filter out None values
        data = {k: v for k, v in kwargs.items() if v is not None}
        if not data:
            raise ValidationError("No fields to update")

        logger.info(f"Updating agent: {agent_id}")
        return await self.repository.update(data=data, where={"id": agent_id})

    async def delete_agent(self, agent_id: str) -> None:
        """
        Delete an agent.

        Args:
            agent_id: Agent ID

        Raises:
            ResourceNotFoundError: If agent not found
        """
        logger.info(f"Deleting agent: {agent_id}")
        await self.repository.delete(id=agent_id)

    async def list_agents(
        self,
        skip: int = 0,
        limit: int = 10,
        tenant_id: str = None,
        is_active: bool = None,
    ) -> List[Agent]:
        """
        List agents with optional filters.

        Args:
            skip: Number of records to skip
            limit: Maximum number of records to return
            tenant_id: Filter by tenant ID
            is_active: Filter by active status

        Returns:
            List of agents
        """
        where = {}
        if tenant_id:
            where["tenant_id"] = tenant_id
        if is_active is not None:
            where["is_active"] = is_active

        return await self.repository.paging(
            skip=skip,
            limit=limit,
            where=where if where else None,
            order_by={"create_time": "desc"},
        )

    async def get_agent_with_kb_ids(self, agent_id: str) -> Dict[str, Any]:
        """
        Get an agent with its linked knowledge base IDs.

        Args:
            agent_id: Agent ID

        Returns:
            Agent dict with kb_ids field
        """
        agent = await self.repository.get(id=agent_id)
        kb_ids = await self.repository.get_linked_kb_ids(agent_id)

        result = agent.to_dict()
        result["kb_ids"] = kb_ids
        return result

    async def link_knowledge_base(self, agent_id: str, kb_id: str) -> Dict[str, str]:
        """
        Link a knowledge base to an agent.

        Args:
            agent_id: Agent ID
            kb_id: Knowledge base ID

        Returns:
            Link info dict
        """
        # Verify agent exists
        await self.repository.get(id=agent_id)

        logger.info(f"Linking KB {kb_id} to agent {agent_id}")
        link = await self.repository.link_knowledge_base(agent_id, kb_id)
        return {"agent_id": agent_id, "kb_id": kb_id, "link_id": link.id}

    async def unlink_knowledge_base(self, agent_id: str, kb_id: str) -> None:
        """
        Unlink a knowledge base from an agent.

        Args:
            agent_id: Agent ID
            kb_id: Knowledge base ID
        """
        logger.info(f"Unlinking KB {kb_id} from agent {agent_id}")
        await self.repository.unlink_knowledge_base(agent_id, kb_id)


def get_agent_service() -> AgentService:
    """Factory function to get AgentService instance."""
    return AgentService()
