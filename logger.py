# this is discover-paths-rl/logger.py
# this file contains the Logger class
import sys

class Logger(object):
    """
    This class redirects all print() statements to a log file.
    """
    def __init__(self, filename="simulation_log.txt"):
        # We don't need self.terminal anymore
        # self.terminal = sys.stdout 
        self.log_file = open(filename, 'w', encoding='utf-8')

    def write(self, message):
        """
        Write *only* to the log file, not the console.
        """
        # self.terminal.write(message) # <-- Removed as requested
        self.log_file.write(message)

    def flush(self):
        """Flush the log file stream."""
        # self.terminal.flush() # <-- Removed as requested
        self.log_file.flush()
        
    def close(self):
        """Close the log file."""
        self.log_file.close()