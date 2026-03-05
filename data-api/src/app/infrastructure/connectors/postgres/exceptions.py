class InvalidActionError(Exception):
    def __init__(self, action: str):
        self.action = action
        self.message = f"Invalid action: {action}"
        super().__init__(self.message)

class InvalidStatusError(Exception):
    def __init__(self, status: str):
        self.status = status
        self.message = f"Invalid status: {status}"
        super().__init__(self.message)
