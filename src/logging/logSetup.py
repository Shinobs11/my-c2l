import logging
from logging.handlers import RotatingFileHandler
import os
import typing
def logSetup(
  logger_name: str,
  logger_path: str,
  level: str |int,
  format: str | None,
   ):
  if format == None:
    format = "[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s"
  
  if not os.path.exists(os.path.split(logger_path)[0]):
    os.makedirs(os.path.split(logger_path)[0])
  log = logging.getLogger(logger_name)
  handler = RotatingFileHandler(logger_path, maxBytes=int(1e6))
  formatter = logging.Formatter(format)
  handler.setFormatter(formatter)
  handler.setLevel(level)
  log.addHandler(handler)
  log.setLevel(level)
  
  return log

