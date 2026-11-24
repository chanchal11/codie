"""Split a restaurant bill, including tip, across a group."""


def split_bill(total: float, people: int, tip_percent: float = 10) -> float:
    """Return one person's share of ``total``, including tip."""
    if people <= 0:
        raise ValueError("people must be positive")
    tip = total * tip_percent
    return (total + tip) / people
