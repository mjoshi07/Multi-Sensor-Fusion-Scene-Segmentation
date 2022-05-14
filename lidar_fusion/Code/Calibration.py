import numpy as np


class CalibrationData:
    def __init__(self, calib_file):
        calibs = read_calib_data(calib_file)
        P = calibs["P2"]
        self.P = np.reshape(P, (3, 4))

        L2C = calibs["Tr_velo_to_cam"]
        self.L2C = np.reshape(L2C, (3, 4))

        R0 = calibs["R0_rect"]
        self.R0 = np.reshape(R0, (3, 3))


def read_calib_data(filepath):
    """ Read in a calibration file and parse into a dictionary.
    Ref: https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
    """
    data = {}
    with open(filepath, "r") as f:
        for line in f.readlines():
            line = line.rstrip()
            if len(line) == 0:
                continue
            key, value = line.split(":", 1)
            # The only non-float values in these files are dates, which
            # we don't care about anyway
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass
    return data
