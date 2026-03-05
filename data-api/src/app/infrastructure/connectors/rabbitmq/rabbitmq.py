import logging
import asyncio
from aio_pika import connect_robust, Message, ExchangeType
from aio_pika.abc import AbstractChannel, AbstractConnection
import json
from app.configurations.configurations import settings
from datetime import datetime

def json_serializable_default(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()  # Convert datetime to ISO 8601 string
    raise TypeError(f"Type {type(obj)} not serializable")

class RabbitMQService:
    def __init__(self):
        self.connection: AbstractConnection = None
        self.channel: AbstractChannel = None
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)

    async def connect(self, retry_interval: int = 5000):
        """
        Establish a connection to RabbitMQ and create a channel.
        Retries if the connection fails.
        :param retry_interval: Interval between retries in milliseconds.
        """
        while not self.connection:
            try:
                self.logger.info("Connecting to RabbitMQ...")
                self.connection = await connect_robust(settings.RABBITMQ_URL)
                self.channel = await self.connection.channel()
                self.logger.info("Connected to RabbitMQ.")
                return
            except Exception as e:
                self.logger.error("Failed to connect to RabbitMQ", exc_info=e)
                self.logger.info(f"Retrying connection in {retry_interval}ms...")
                await asyncio.sleep(retry_interval / 1000)

    async def _retry_connect(self, retry_interval: int):
        """
        Retry connecting to RabbitMQ with a delay.
        :param retry_interval: Interval between retries in milliseconds.
        """
        await asyncio.sleep(retry_interval / 1000)
        await self.connect(retry_interval)

    def _on_connection_close(self, reason):
        """
        Handle connection close.
        :param reason: Reason for connection closure.
        """
        self.logger.error(f"RabbitMQ connection closed: {reason}")
        self.connection = None
        self.channel = None

    async def disconnect(self):
        """
        Close the RabbitMQ connection and clean up resources.
        """
        try:
            if self.connection:
                self.logger.info("Disconnecting from RabbitMQ...")
                await self.connection.close()
                self.connection = None
                self.channel = None
                self.logger.info("Disconnected from RabbitMQ.")
        except Exception as e:
            self.logger.error("Failed to disconnect from RabbitMQ", exc_info=e)

    async def publish(self, exchange_name: str, routing_key: str, payload: dict):
        """
        Publish a message to the specified exchange with a routing key.
        
        :param exchange_name: Name of the exchange.
        :param routing_key: Routing key for the message.
        :param payload: Payload to publish.
        """
        try:
            if not self.channel:
                self.logger.warning("No RabbitMQ channel available. Reconnecting...")
                await self.connect()

            exchange = await self.channel.declare_exchange(exchange_name, ExchangeType.TOPIC, durable=True)
            self.logger.info(f"Publishing message to exchange '{exchange_name}' with routing key '{routing_key}'")
            # Use the custom default function for JSON serialization
            message = Message(
                body=json.dumps(payload, default=json_serializable_default).encode(),
                delivery_mode=2  # Persistent delivery
            )
            await exchange.publish(message, routing_key=routing_key)
            self.logger.info(f"Message published to exchange '{exchange_name}' with routing key '{routing_key}'")
        except Exception as e:
            self.logger.error("Failed to publish message", exc_info=e)
            raise e
