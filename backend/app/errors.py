"""Domain errors. Messages are written to be safe to show to end users."""


class AppError(Exception):
    status_code = 400

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class ParseError(AppError):
    status_code = 422


class PayloadTooLargeError(AppError):
    status_code = 413


class NotFoundError(AppError):
    status_code = 404


class LimitError(AppError):
    status_code = 409


class GuardrailViolation(AppError):
    """Raised when an input guardrail blocks a request."""

    status_code = 400

    def __init__(self, rule: str, message: str) -> None:
        super().__init__(message)
        self.rule = rule


class LLMUnavailableError(AppError):
    status_code = 503
