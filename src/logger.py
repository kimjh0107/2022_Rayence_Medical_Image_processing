from pathlib import Path
import shutil
import os

class print_logger :
    def __init__(self, log_path : Path) : 
        self.log_path = log_path            
        self.check_exists()
        if log_path.suffix == '.txt' :
            log_path.parent.mkdir(parents = True, exist_ok=True)
        else :         
            log_path.mkdir(parents=True, exist_ok=True)      
        

    def __call__(self, log, end = '\n') : 
        print(log, end = end)
        with open(self.log_path, 'a') as f :
            f.write(log + end)

    def check_exists(self) : 
        try :  
            if self.log_path.exists() : 
                shutil.rmtree(self.log_path)
        except : 
                os.remove(self.log_path)    