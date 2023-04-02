import logging
import os
import time
def get_logger(file_path='',log_type=logging.INFO): 
    logger = logging.getLogger()
    logger.setLevel(log_type) 
    formatter = logging.Formatter('%(asctime)s %(filename)s %(funcName)s [line:%(lineno)d] %(levelname)s %(message)s')
    
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    file_path=os.path.join(os.getcwd(),file_path,str(int(time.time()))+'.log')
    fh = logging.FileHandler(file_path, encoding='utf8')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    return logger