
import sys
import os
def get_error_msg_detail(error_msg,error_detail:sys):
    _,_,exc_tb=error_detail.exc_info()
    line_no=exc_tb.tb_lineno
    file_name=exc_tb.tb_frame.f_code.co_filename
    error_message=f"The error has occured in line No:{line_no} with the file name :{file_name} and the message is {str(error_msg)}"

    return error_message

class CustomException(Exception):
    def __init__(self,error_msg,error_detail:sys):
        super().__init__(error_msg)
        self.error_msg=get_error_msg_detail(error_msg,error_detail=error_detail)
    def __str__(self):
        return self.error_msg



