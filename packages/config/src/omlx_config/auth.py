"""Pure validation of configured authentication credentials."""


def validate_api_key(api_key: str) -> tuple[bool, str]:
    if len(api_key) < 4:
        return False, "API key must be at least 4 characters"
    if any(char.isspace() for char in api_key):
        return False, "API key must not contain whitespace"
    if not api_key.isprintable():
        return False, "API key must contain only printable characters"
    if not api_key.isascii():
        return False, "API key must contain only ASCII characters"
    return True, ""
