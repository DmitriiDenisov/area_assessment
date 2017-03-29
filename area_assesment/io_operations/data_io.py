import os


def filenames_in_dir(dir_, endswith_):
    """
    Returns paths to files with given extension endswith_ in the given directory dir_.

    filenames_in_dir('../../data/mass_buildings/valid/sat/', endswith_='.tiff')

    :param dir_: path to directory with files
    :param endswith_: file extension, for example '.tiff'
    :return: list with full paths to files with given extension endswith_ in the given directory dir_
    """
    return sorted([os.path.join(dir_, filename) for filename in os.listdir(dir_) if filename.endswith(endswith_)])

