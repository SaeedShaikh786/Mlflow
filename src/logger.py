from datetime import datetime
import sys
import os
import logging

file_name=f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
log_path=os.path.join(os.getcwd(),"logs",file_name)

os.makedirs(log_path,exist_ok=True)

log_file_path=os.path.join(log_path,file_name)

logging.basicConfig(filename=log_file_path,level=logging.INFO)