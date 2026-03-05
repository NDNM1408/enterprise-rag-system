import uuid

class RabbitMQEvent:
    def __init__(self, body: dict):
        self.id = str(uuid.uuid4())
        self.body = body