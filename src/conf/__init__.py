import os
current_folder = os.path.dirname(os.path.realpath(__file__))
all_files_under_conf = [str(f.split('.')[0]) for f in os.listdir(current_folder)]
__all__ = all_files_under_conf
