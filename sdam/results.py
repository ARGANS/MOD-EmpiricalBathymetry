from typing import Any, Dict, Optional


class ResultsHandler:
    """
    A class to handle the results of the calibration process.

    Attributes:
        metrics (Dict[str, Any]): A dictionary containing calibration metrics.
        kwargs (Dict[str, Any]): Additional attributes passed during initialization.
    """

    def __init__(
        self, obj: Any, metrics: Optional[Dict[str, Any]] = None, **kwargs: Any
    ) -> None:
        """
        Initialize the ResultsHandler class.

        Args:
            obj (Any): The parent object (typically the SDAM instance).
            metrics (Optional[Dict[str, Any]]): A dictionary containing calibration metrics. Defaults to None.
            **kwargs (Any): Additional attributes to set on the instance.
        """
        self._parent = obj
        self.metrics: Dict[str, Any] = metrics if metrics is not None else {}

        for key, value in kwargs.items():
            setattr(self, key, value)
