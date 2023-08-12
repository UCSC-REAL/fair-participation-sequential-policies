import logging
import sys

log = logging
log.basicConfig(
    format="%(asctime)s[%(filename)s.%(levelname)s]: %(message)s",
    stream=sys.stdout,
    level=logging.CRITICAL,
    datefmt="%H:%M:%S",
)

logger = log.getLogger(__name__)
logger.setLevel(logging.INFO)
