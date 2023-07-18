import logging
import sys

log = logging
log.basicConfig(
    format=f"%(asctime)s[%(filename)s.%(levelname)s]: %(message)s",
    stream=sys.stdout,
    level=logging.INFO,
    datefmt="%H:%M:%S",
)

logger = log.getLogger(__name__)
