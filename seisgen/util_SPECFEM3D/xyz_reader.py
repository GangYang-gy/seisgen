# -------------------------------------------------------------------
# XYZ reader.
# *x.bin, *y.bin, *z.bin
#
# Author: Liang Ding
# Email: myliang.ding@mail.utoronto.ca
# -------------------------------------------------------------------

import os.path
from seisgen.util_SPECFEM3D import get_proc_name, CONSTANT_INDEX_27_GLL, get_index_8_anchors
from seisgen.util_SPECFEM3D.ibool_reader import read_ibool_by_scipy
from scipy.io import FortranFile
import numpy as np

def read_xyz_bin_by_scipy(file_path):
    '''
    * read XYZ bin file by using the SciPy package.

    :param file_path: The path of the XYZ bin file.
    :param file_type: The data type of the XYZ bin.

    * x,y,z -- float32

    :return: The whole data.
    '''

    data_type = 'float32'

    try:
        f = FortranFile(file_path, 'r')
        dat = f.read_reals(dtype=data_type)
        f.close()
    except:
        print("Unable to open file: ", str(file_path))
        return None
    return dat

def DEnquire_XYZ_GLLs_Element(data_dir, idx_processor, idx_element, NSPEC):
    '''
    return the x, y, z of the 27 GLL points where the SGT been stored in the selected element.

    :param data_dir:        The dir of the *.bin files.
    :param idx_proc:        The index of the processor. INT
    :param idx_element:     The index of the element in the processor. INT
    :param NSPEC_PER_SLICE:    The number of the element in the processor. INT
    :return:                The x, y, and z array of the GLL points.
    '''

    proc_name = get_proc_name(idx_processor)
    ibool_file = os.path.join(str(data_dir),  str(proc_name) + "_ibool.bin")
    x_file = os.path.join(str(data_dir),  str(proc_name) + "_x.bin")
    y_file = os.path.join(str(data_dir), str(proc_name) + "_y.bin")
    z_file = os.path.join(str(data_dir), str(proc_name) + "_z.bin")

    ibool = read_ibool_by_scipy(ibool_file, NSPEC)
    
    x = read_xyz_bin_by_scipy(x_file)
    y = read_xyz_bin_by_scipy(y_file)
    z = read_xyz_bin_by_scipy(z_file)
    glls_idx = ibool[idx_element][CONSTANT_INDEX_27_GLL]

    return x[glls_idx], y[glls_idx], z[glls_idx]

def DEnquire_XYZ_anchors_Element(data_dir, idx_processor, idx_element, NSPEC):
    '''
    return the geometric coordinates x, y, z of the 8 anchors (control nodes in the corners of one element).

    :param data_dir:        The dir of the *.bin files.
    :param idx_proc:        The index of the processor. INT
    :param idx_element:     The index of the element in the processor. INT
    :param NSPEC_PER_SLICE:    The number of the element in the processor. INT
    :return:                The x, y, and z array of the GLL points.
    '''

    CONSTANT_INDEX_8_anchors = get_index_8_anchors()
    proc_name = get_proc_name(idx_processor)
    ibool_file = os.path.join(str(data_dir),  str(proc_name) + "_ibool.bin")
    x_file = os.path.join(str(data_dir),  str(proc_name) + "_x.bin")
    y_file = os.path.join(str(data_dir), str(proc_name) + "_y.bin")
    z_file = os.path.join(str(data_dir), str(proc_name) + "_z.bin")

    ibool = read_ibool_by_scipy(ibool_file, NSPEC)
    
    x = read_xyz_bin_by_scipy(x_file)
    y = read_xyz_bin_by_scipy(y_file)
    z = read_xyz_bin_by_scipy(z_file)
    anchors_idx = ibool[idx_element][CONSTANT_INDEX_8_anchors]

    x_alpha = np.zeros(8)
    y_alpha = np.zeros(8)
    z_alpha = np.zeros(8)
    
    for i in range(len(anchors_idx)):
        x_alpha[i] = x[anchors_idx[i]]
        y_alpha[i] = y[anchors_idx[i]]
        z_alpha[i] = z[anchors_idx[i]]

    return x_alpha, y_alpha, z_alpha
