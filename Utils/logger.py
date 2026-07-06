import logging
import os

class LoggerManager:

    @staticmethod
    def get_logs(statements: str) -> logging.Logger:
        # 1. Ensure logs directory exists
        os.makedirs('logs', exist_ok=True)

        logger = logging.getLogger(statements)

        # 2. Prevent log propagation duplication
        logger.propagate = False

        if not logger.handlers:
            # 3. Set the central logger level
            logger.setLevel(logging.INFO)

            # Correctly formatted log structure
            formatter = logging.Formatter(
                '%(asctime)s | %(name)s | %(levelname)s | %(message)s'
            )

            # File Handler
            file_handler = logging.FileHandler('logs/history.log')
            file_handler.setLevel(logging.INFO)  # Explicitly set level
            file_handler.setFormatter(formatter)

            # Console Handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)  # Explicitly set level
            console_handler.setFormatter(formatter)

            # Attach handlers
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)

        return logger