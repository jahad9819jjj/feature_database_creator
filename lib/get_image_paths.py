import pathlib
import os

def get_paths_image()->list[pathlib.Path]:
    cwd = pathlib.Path(os.getcwd())
    datap = pathlib.Path('data')
    data_fullp = cwd / datap

    directories = []
    for _, path in enumerate(data_fullp.iterdir()):
        if path.is_dir():
            directories.append(path.stem)

    print('Directories:', directories)
    condition = input('Select which directory?\n')

    if condition not in directories:
        raise ValueError('No such directory.')

    selected_path = data_fullp / condition
    jpg_files:list[pathlib.Path] = list(selected_path.rglob('*.jpg'))

    if not jpg_files:
        raise ValueError('No JPG files found in the selected directory.')

    return jpg_files
