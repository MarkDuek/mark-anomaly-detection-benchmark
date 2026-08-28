import sys
from abc import ABC, abstractmethod
from enum import IntEnum
from pathlib import Path

import yaml
from tqdm import tqdm


class LogLevel(IntEnum):
    INFO = 10
    WARNING = 20
    ERROR = 30

    @classmethod
    def from_name(cls, name: str) -> "LogLevel":
        return cls[name.upper()]


class LogOutput(ABC):
    @abstractmethod
    def log(self, message: str) -> None:
        pass


class ConsoleOutput(LogOutput):
    def log(self, message: str) -> None:
        if sys.stderr.isatty():
            tqdm.write(message, file=sys.stderr)
            return
        print(message)


class FileOutput(LogOutput):
    def __init__(self, path: Path) -> None:
        self.path = path

    def log(self, message: str) -> None:
        with self.path.open("a", encoding="utf-8") as file:
            file.write(f"{message}\n")


class ComponentLogger:
    def __init__(self, logger: "Logger", name: str) -> None:
        self.logger = logger
        self.name = name

    def info(self, message: str) -> None:
        self.log(LogLevel.INFO, message)

    def warning(self, message: str) -> None:
        self.log(LogLevel.WARNING, message)

    def error(self, message: str) -> None:
        self.log(LogLevel.ERROR, message)

    def log(self, level: LogLevel, message: str) -> None:
        self.logger.log(level, message, component=self.name)

    def for_component(self, name: str) -> "ComponentLogger":
        """Return another named component logger sharing the same configuration."""
        return self.logger.for_component(name)


class Logger:
    def __init__(
        self,
        *outputs: LogOutput,
        default_level: LogLevel = LogLevel.INFO,
        component_levels: dict[str, LogLevel] | None = None,
    ) -> None:
        self.outputs = list(outputs)
        self.default_level = default_level
        self.component_levels = component_levels or {}

    @classmethod
    def from_config(cls, path: Path, *outputs: LogOutput) -> "Logger":
        config = yaml.safe_load(path.read_text())
        component_levels = {
            name: LogLevel.from_name(level)
            for name, level in config.get("component_levels", {}).items()
        }
        return cls(
            *outputs,
            default_level=LogLevel.from_name(config["default_level"]),
            component_levels=component_levels,
        )

    def add_output(self, output: LogOutput) -> None:
        self.outputs.append(output)

    def for_component(self, name: str) -> ComponentLogger:
        return ComponentLogger(self, name)

    def info(self, message: str) -> None:
        self.log(LogLevel.INFO, message)

    def warning(self, message: str) -> None:
        self.log(LogLevel.WARNING, message)

    def error(self, message: str) -> None:
        self.log(LogLevel.ERROR, message)

    def log(self, level: LogLevel, message: str, component: str | None = None) -> None:
        if level < self.level_for(component):
            return
        formatted_message = f"[{level.name}] {message}"
        for output in self.outputs:
            output.log(formatted_message)

    def level_for(self, component: str | None) -> LogLevel:
        while component:
            if component in self.component_levels:
                return self.component_levels[component]
            component = component.rpartition(".")[0]
        return self.default_level
