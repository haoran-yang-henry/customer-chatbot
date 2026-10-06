from datetime import date


class SystemClock:
    def today(self) -> date:
        return date.today()


class FixedClock:
    """For tests and evaluation: the dataset's expectations assume a fixed 'today'."""

    def __init__(self, today: date) -> None:
        self._today = today

    def today(self) -> date:
        return self._today
