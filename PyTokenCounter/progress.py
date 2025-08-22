from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Column

_progressInstance = Progress(
    TextColumn(
        "[bold blue]{task.description}",
        justify="left",
        table_column=Column(width=50),
    ),
    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    BarColumn(bar_width=None),
    MofNCompleteColumn(),
    TextColumn("•"),
    TimeElapsedColumn(),
    TextColumn("•"),
    TimeRemainingColumn(),
    expand=True,
)
_tasks: dict[str, int] = {}


def _InitializeTask(taskName: str, total: int, quiet: bool = False) -> int | None:
    """Internal helper to initialize a progress task."""
    if quiet:
        return None

    if not _progressInstance.live.is_started:
        _progressInstance.start()

    if taskName in _tasks:
        return _tasks[taskName]

    taskId = _progressInstance.add_task(taskName, total=total)
    _tasks[taskName] = taskId

    return taskId


def _UpdateTask(
    taskName: str,
    advance: int,
    description: str | None = None,
    appendDescription: str | None = None,
    quiet: bool = False,
) -> None:
    """Internal helper to update a progress task."""
    if quiet:
        return

    # If the task was cleared (e.g., due to nested operations finishing a
    # different task and stopping the progress), treat this update as a no-op
    # to avoid crashing callers that still hold the original task name.
    if taskName not in _tasks:
        return

    currentTask = _progressInstance.tasks[_tasks[taskName]]
    currentDescription = currentTask.description if currentTask.description else ""

    if appendDescription is not None:
        description = f"{currentDescription} {appendDescription}".strip()
    elif description is None:
        description = currentDescription

    _progressInstance.update(_tasks[taskName], advance=advance, description=description)

    if all(task.finished for task in _progressInstance.tasks):
        _progressInstance.stop()
        _tasks.clear()
