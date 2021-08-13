import logging

def Logger(_name, level = logging.DEBUG):
    logger = logging.getLogger(_name)
    logger.setLevel(logging.DEBUG)
    
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(pathname)s - (Line: %(lineno)d) - %(message)s")
    console = logging.StreamHandler()
    
    console.setLevel(logging.DEBUG)
    console.setFormatter(formatter)
    
    logger.addHandler(console)
    
    return logger
        
if __name__ == '__main__':
    # mylogger = logging.getLogger("my")
    # mylogger.setLevel(logging.INFO)

    # stream_handler = logging.StreamHandler()
    # mylogger.addHandler(stream_handler)     # 콘솔창에 출력하겠다.
    # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # stream_handler.setFormatter(formatter)
    # file_handler = logging.FileHandler('my.log')
    # file_handler.setFormatter(formatter)
    # mylogger.addHandler(file_handler)       # 파일에 출력하겠다.

    # mylogger.info("server start!!")
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - (줄 번호: %(lineno)d) - %(pathname)s - %(message)s")
    console = logging.StreamHandler()
    file_handler = logging.FileHandler("my.log")
    
    console.setLevel(logging.INFO)
    file_handler.setLevel(logging.DEBUG)
    
    console.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    logger.addHandler(console)
    logger.addHandler(file_handler)
    
    logger.info("Message")
