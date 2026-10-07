class Skip:
    """Marks an output a model did not produce; ``reason`` says why, when known."""

    def __init__(self, reason=None):
        self.reason = reason
