import logging
import datetime
import os
from logging.handlers import TimedRotatingFileHandler

# Set up logging
logging.basicConfig(level=logging.INFO)

# Create two separate loggers with unique names if intended for different purposes
logger = logging.getLogger("my_logger")
sum_logger = logging.getLogger("sum_logger")

# Create a formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# create a folder called logs if not already created
folder_name = "logs"
current_directory = os.path.dirname(os.path.abspath(__file__))
logs_folder_path = os.path.join(current_directory, folder_name)
if not os.path.exists(logs_folder_path):
    os.mkdir(logs_folder_path)

# Create file handlers with separate log files for logger and sum_logger
current_day = datetime.datetime.now().strftime('%Y-%m-%d')

log_filename = f"{current_day}.log"
log_filename = os.path.join(logs_folder_path, log_filename)
f_handler = TimedRotatingFileHandler(log_filename, when='D', interval=1)
f_handler.setLevel(logging.INFO)
f_handler.setFormatter(formatter)

sum_log_filename = f"sum_{current_day}.log"
sum_log_filename = os.path.join(logs_folder_path, sum_log_filename)
sum_f_handler = TimedRotatingFileHandler(sum_log_filename, when='D', interval=1)
sum_f_handler.setLevel(logging.INFO)
sum_f_handler.setFormatter(formatter)

# Create a console handler and set level and formatter
c_handler = logging.StreamHandler()
c_handler.setLevel(logging.INFO)
c_handler.setFormatter(formatter)

# Add handlers to the loggers
logger.addHandler(c_handler)
logger.addHandler(f_handler)

sum_logger.addHandler(c_handler)  # Assuming you still want console output for sum operations
sum_logger.addHandler(sum_f_handler)

# Example usage
# logger.info("==================== logger launched ====================")
# sum_logger.info("==================== sum_logger launched ====================")