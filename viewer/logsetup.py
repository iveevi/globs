import logging

RESET = "\033[0m"
DIM = "\033[2m"
LEVEL_COLORS = {
    logging.DEBUG: "\033[36m",
    logging.INFO: "\033[32m",
    logging.WARNING: "\033[33m",
    logging.ERROR: "\033[31m",
    logging.CRITICAL: "\033[1;37;41m",
}


class ColorFormatter(logging.Formatter):
    def __init__(self, color):
        super().__init__(datefmt="%H:%M:%S")
        self.color = color

    def format(self, record):
        ts = self.formatTime(record, self.datefmt)
        level = record.levelname[:4].ljust(4)
        message = record.getMessage()
        if record.exc_info:
            message += "\n" + self.formatException(record.exc_info)
        if not self.color:
            return f"{ts} {level} {record.name}: {message}"
        c = LEVEL_COLORS.get(record.levelno, "")
        return f"{DIM}{ts}{RESET} {c}{level}{RESET} {DIM}{record.name}{RESET}: {message}"


def setup(level=logging.INFO):
    handler = logging.StreamHandler()
    handler.setFormatter(ColorFormatter(color=handler.stream.isatty()))
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)
