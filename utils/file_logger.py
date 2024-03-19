import logging

class FileWriter:
    def __init__(self, log_file_path):
        # Create a logger
        self.logger = logging.getLogger('file_writer')
        self.logger.setLevel(logging.DEBUG)

        # Create a file handler
        self.file_handler = logging.FileHandler(log_file_path)
        self.file_handler.setLevel(logging.DEBUG)

        # Create a formatter and set it on the file handler
        # formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        # self.file_handler.setFormatter(formatter)

        # Add the file handler to the logger
        self.logger.addHandler(self.file_handler)

    def write(self, message):
        # Write message to the log file
        self.logger.info(message)

    def close(self):
        # Close the file handler
        self.file_handler.close()