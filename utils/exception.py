import sys
import traceback
from utils.logger import logger

class CustomException(Exception):
    def __init__(self, error, error_detail: sys):
        super().__init__(error)
        self.error_message = CustomException.get_detailed_error_msg(error, error_detail)
        logger.error(self.error_message)

    @staticmethod
    def get_detailed_error_msg(error, error_detail: sys):
        _, _, exc_tb = error_detail.exc_info()
        file_name = exc_tb.tb_frame.f_code.co_filename
        line = exc_tb.tb_lineno

        error_msg = (
            f"\nERROR OCCURRED:\n"
            f"File: {file_name}\n"
            f"Line: {line}\n"
            f"Message: {str(error)}\n"
        )
        return error_msg

    def __str__(self):
        return self.error_message