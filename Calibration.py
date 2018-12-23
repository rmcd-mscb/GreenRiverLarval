from __future__ import print_function  # Only Python 2.x
import numpy as np
import matplotlib.pyplot as plt
from shutil import copyfile
import h5py
import vtk
import subprocess
import os
from itertools import count
import configparser
import sys


def getCellValue(vtkSGrid2D, newPoint2D, cellID, valarray):
    pcoords = [0.0, 0.0, 0.0]
    weights = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    clspoint = [0., 0., 0.]
    tmpid = vtk.mutable(0)
    vtkid2 = vtk.mutable(0)
    vtkcell2D = vtk.vtkQuad()
    vtkcell2D = vtkSGrid2D.GetCell(cellID)
    tmpres = vtkcell2D.EvaluatePosition(newPoint2D, clspoint, tmpid, pcoords, vtkid2, weights)
    print(newPoint2D, clspoint, tmpid, pcoords, vtkid2, weights)
    idlist1 = vtk.vtkIdList()
    numpts = vtkcell2D.GetNumberOfPoints()
    idlist1 = vtkcell2D.GetPointIds()
    tmpVal = 0.0
    for x in range(0, numpts):
        tmpVal = tmpVal + weights[x] * valarray.GetTuple(idlist1.GetId(x))[0]
    return tmpVal


def isCellWet(vtkSGrid2D, newPoint2D, cellID, IBC_2D):
    pcoords = [0.0, 0.0, 0.0]
    weights = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    clspoint = [0., 0., 0.]
    tmpid = vtk.mutable(0)
    vtkid2 = vtk.mutable(0)
    vtkcell2D = vtk.vtkQuad()
    vtkcell2D = vtkSGrid2D.GetCell(cellID)
    tmpres = vtkcell2D.EvaluatePosition(newPoint2D, clspoint, tmpid, pcoords, vtkid2, weights)
    idlist1 = vtk.vtkIdList()
    numpts = vtkcell2D.GetNumberOfPoints()
    idlist1 = vtkcell2D.GetPointIds()
    tmpIBC = 0.0
    for x in range(0, numpts):
        tmpIBC = tmpIBC + weights[x] * abs(IBC_2D.GetTuple(idlist1.GetId(x))[0])
        # print(tmpIBC,abs(IBC_2D.GetTuple(idlist1.GetId(x))[0]))
    if tmpIBC >= .9999999:
        return True
    else:
        return False


def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


def execute(cmd):
    popen = subprocess.Popen(cmd, stdout=subprocess.PIPE, universal_newlines=True)
    for stdout_line in iter(popen.stdout.readline, ""):
        yield stdout_line
    popen.stdout.close()
    return_code = popen.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, cmd)


def gen_filenames(prefix, suffix, places=3):
    """Generate sequential filenames with the format <prefix><index><suffix>

       The index field is padded with leading zeroes to the specified number of places

       http://stackoverflow.com/questions/5068461/how-do-you-increment-file-name-in-python
    """
    pattern = "{}{{:0{}d}}{}".format(prefix, places, suffix)
    for i in count(1):
        yield pattern.format(i)


def fastmech_change_cd(hdf_file, newCd):
    # hdf5_file_name = r'F:\Kootenai Project\USACE\Braided\Case11_tmp.cgn'
    # r+ adds read/write permisions to file
    file = h5py.File(hdf_file, 'r+')
    group = file['/iRIC/CalculationConditions/FM_HydAttCD/Value']
    dset = group[u' data']
    # print dset[0]
    dset[0] = newCd
    # print dset[0]
    file.close()


def fastmech_BCs(hdf_file, Q, H_DS, H_US, iniType, OneDCD):
    file = h5py.File(hdf_file, 'r+')
    group = file['/iRIC/CalculationConditions/FM_HydAttQ/Value']
    dset = group[u' data']
    dset[0] = Q
    group2 = file['/iRIC/CalculationConditions/FM_HydAttWS2/Value']
    dset2 = group2[u' data']
    dset2[0] = H_US
    group3 = file['/iRIC/CalculationConditions/FM_HydAttWS/Value']
    dset3 = group3[u' data']
    dset3[0] = H_DS
    group4 = file['/iRIC/CalculationConditions/FM_HydAttWSType/Value']
    dset4 = group4[u' data']
    dset4[0] = iniType
    # group5 = file['/iRIC/CalculationConditions/FM_HydAttWS1DStage/Value']
    # dset5 = group5[u' data']
    # dset5[0] = OneDStage
    # group6 = file['/iRIC/CalculationConditions/FM_HydAttWS1DDisch/Value']
    # dset6 = group6[u' data']
    # dset6[0] = OneDQ
    group7 = file['/iRIC/CalculationConditions/FM_HydAttWS1DCD/Value']
    dset7 = group7[u' data']
    dset7[0] = OneDCD
    file.close()


def fastmech_params(hdf_file, Itermax, endLEV):
    file = h5py.File(hdf_file, 'r+')
    group = file['/iRIC/CalculationConditions/FM_EndLEV/Value']
    dset = group[u' data']
    dset[0] = endLEV
    group = file['/iRIC/CalculationConditions/FM_SolAttItm/Value']
    dset = group[u' data']
    dset[0] = Itermax
    file.close()


def fastmech_change_var_cd(hdf_file, newCd_0, newCd_1):
    # hdf5_file_name = r'F:\Kootenai Project\USACE\Braided\Case11_tmp.cgn'
    # r+ adds read/write permisions to file
    file = h5py.File(hdf_file, 'r+')
    group = file['/iRIC/iRICZone/GridConditions/sanddepth/Value']
    dset = group[u' data']
    group2 = file['/iRIC/iRICZone/GridConditions/roughness/Value']
    dset2 = group2[u' data']
    for index, val in enumerate(dset):
        if val == 0.0:
            dset2[index] = newCd_0
        # else:
        # dset2[index] = newCd_1 #keep values in original project, change only values with 0
    # print dset[0]
    # print dset[0]
    file.close()


def add_fastmech_solver_to_path(solverPath):
    print(os.environ['PATH'])
    os.environ['PATH'] += solverPath
    print("\n")
    print('new path')
    print(os.environ['PATH'])


def create_vtk_structured_grid(sgrid, hdf5_file_name, xoffset, yoffset):
    # type: (object) -> object
    file = h5py.File(hdf5_file_name, 'r')
    xcoord_grp = file['/iRIC/iRICZone/GridCoordinates/CoordinateX']
    print(xcoord_grp.keys())
    ycoord_grp = file['/iRIC/iRICZone/GridCoordinates/CoordinateY']
    print(ycoord_grp.keys())
    wse_grp = file['iRIC/iRICZone/FlowSolution1/WaterSurfaceElevation']
    print(wse_grp.keys())
    topo_grp = file['iRIC/iRICZone/FlowSolution1/Elevation']
    print(topo_grp.keys())
    ibc_grp = file['iRIC/iRICZone/FlowSolution1/IBC']
    velx_grp = file['iRIC/iRICZone/FlowSolution1/VelocityX']
    vely_grp = file['iRIC/iRICZone/FlowSolution1/VelocityY']

    xcoord_data = xcoord_grp[u' data']
    ycoord_data = ycoord_grp[u' data']
    wse_data = wse_grp[u' data']
    topo_data = topo_grp[u' data']
    ibc_data = ibc_grp[u' data']
    velx_data = velx_grp[u' data']
    vely_data = vely_grp[u' data']

    # SGrid = vtk.vtkStructuredGrid()
    ny, nx, = xcoord_data.shape
    print(ny, nx)
    sgrid.SetDimensions(nx, ny, 1)
    points = vtk.vtkPoints()
    wseVal = vtk.vtkFloatArray()
    wseVal.SetNumberOfComponents(1)
    ibcVal = vtk.vtkIntArray()
    ibcVal.SetNumberOfComponents(1)
    velVal = vtk.vtkFloatArray()
    velVal.SetNumberOfComponents(1)
    for j in range(ny):
        for i in range(nx):
            points.InsertNextPoint(xcoord_data[j, i] - xoffset, ycoord_data[j, i] - yoffset, 0.0)
            wseVal.InsertNextValue(wse_data[j, i])
            ibcVal.InsertNextValue(ibc_data[j, i])
            velVal.InsertNextValue(np.sqrt(np.power(velx_data[j, i], 2) + np.power(vely_data[j, i], 2)))
        sgrid.SetPoints(points)

        sgrid.GetPointData().AddArray(wseVal)
        sgrid.GetPointData().AddArray(ibcVal)
        sgrid.GetPointData().AddArray(velVal)
    wseVal.SetName("WSE")
    ibcVal.SetName("IBC")
    velVal.SetName("Velocity")


print('Number of arguments:', len(sys.argv), 'arguments.')
print('Argument List:', str(sys.argv))
setfile = sys.argv[1]
config = configparser.ConfigParser()
config.read(setfile)

# meas_wse = np.genfromtxt(r'D:\USACE\MeanderCalibration\2011\m20110718_457pt3cms\m20110718_457pt3cms.csv', delimiter=',', skip_header=1)
meas_wse = np.genfromtxt(config.get('Params', 'meas_WSE_File'), delimiter=',', skip_header=1)
nummeas = meas_wse.shape[0]

# meas_vel = np.genfromtxt(r'E:\Kootenai Project\USACE\VelocityComp_stg_534_04\KR_2016_Task_2bi_Myrtle_0615_Magnitude.csv', delimiter=',', skip_header=1)
# nummeas_v = meas_vel.shape[0]
#
# add_fastmech_solver_to_path(r';C:\Users\rmcd\iRIC\guis\prepost')
# add_fastmech_solver_to_path(r';C:\Users\rmcd\iRIC\solvers\fastmech')
add_fastmech_solver_to_path(config.get('Params', 'lib_path'))
add_fastmech_solver_to_path(config.get('Params', 'solver_path'))
# add_fastmech_solver_to_path(r';C:\Users\rmcd\iRIC\guis\prepost')

# os.chdir(r'D:\USACE\MeanderCalibration\2011\m20110718_457pt3cms')
os.chdir(config.get('Params', 'working_dir'))
g = gen_filenames("FM_Calib_Flow_", ".cgns")
# cdmin = 0.001
# cdmax = 0.004
# cdinc = 0.0001
# xoffset = 0.0
# yoffset = 0.0
# Q = 457.3
# H_DS = 534.88
# H_US = 540.81 + .75
# iniType = 2
# OneDCD = 0.01
cdmin = config.getfloat('Params', 'cdmin')
cdmax = config.getfloat('Params', 'cdmax')
cdinc = config.getfloat('Params', 'cdinc')
xoffset = config.getfloat('Params', 'xoffset')
yoffset = config.getfloat('Params', 'yoffset')
Q = config.getfloat('Params', 'Q')
H_DS = config.getfloat('Params', 'H_DS')
H_US = config.getfloat('Params', 'H_US') + 0.75
iniType = config.getfloat('Params', 'iniType')
OneDCD = config.getfloat('Params', 'OneDCD')
numcds = np.arange(cdmin, cdmax, cdinc).shape[0]

rmse_data = np.zeros(numcds)
cd_val = np.zeros(numcds)
meas_and_sim_wse = np.zeros(shape=(nummeas, numcds + 1))

# rmse_data_vel = np.zeros(numcds)
# meas_and_sim_vel = np.zeros(shape=(numcds+1, nummeas_v))

for cdind, Cd in enumerate(np.arange(cdmin, cdmax, cdinc)):
    hdf5_file_name = next(g)
    # copyfile(r'D:\USACE\MeanderCalibration\2011\Meander_Base_2011_5m\Case1.cgn',
    #          hdf5_file_name)
    copyfile(config.get('Params', 'base_file'),
             hdf5_file_name)

    # fastmech_change_var_cd(hdf5_file_name, Cd, 0.0032)
    fastmech_change_cd(hdf5_file_name, Cd)
    fastmech_BCs(hdf5_file_name, Q, H_DS, H_US, iniType, OneDCD)

    # fastmech_params(hdf5_file_name, 10000, 0.036)

    for path in execute(["Fastmech.exe", hdf5_file_name]):
        print(path, end="")

    SGrid = vtk.vtkStructuredGrid()
    print('before create grid')
    create_vtk_structured_grid(SGrid, hdf5_file_name, xoffset, yoffset)
    print('after create grid')
    cellLocator2D = vtk.vtkCellLocator()
    cellLocator2D.SetDataSet(SGrid)
    # cellLocator2D.SetNumberOfCellsPerBucket(10);
    cellLocator2D.BuildLocator()

    WSE_2D = SGrid.GetPointData().GetScalars('WSE')
    IBC_2D = SGrid.GetPointData().GetScalars('IBC')
    Velocity_2D = SGrid.GetPointData().GetScalars('Velocity')

    # gw = vtk.vtkXMLStructuredGridWriter()
    # gw.SetFileName(r'F:\Kootenai Project\USACE\Braided\testvtk.vtu')
    # gw.SetInputData(SGrid)
    # gw.SetDataModeToAscii()
    # gw.Write()

    simwse = np.zeros(meas_wse.shape[0])
    measwse = np.zeros(meas_wse.shape[0])
    for counter, line in enumerate(meas_wse):
        point2D = [line[0] - xoffset, line[1] - yoffset, 0.0]
        pt1 = [line[0] - xoffset, line[1] - yoffset, 10.0]
        pt2 = [line[0] - xoffset, line[1] - yoffset, -10]
        idlist1 = vtk.vtkIdList()
        cellLocator2D.FindCellsAlongLine(pt1, pt2, 0.0, idlist1)
        cellid = idlist1.GetId(0)
        # cellid = cellLocator2D.FindCell(point2D)
        # print (isCellWet(SGrid, point2D, cellid, IBC_2D))
        tmpwse = getCellValue(SGrid, point2D, cellid, WSE_2D)
        if Cd == cdmin:
            meas_and_sim_wse[counter, 0] = line[2]
        #     print counter
        simwse[counter] = tmpwse
        measwse[counter] = line[2]
        print(cellid, line[2], tmpwse)
    meas_and_sim_wse[:, cdind + 1] = simwse
    rmse_data[cdind] = rmse(simwse, measwse)
    cd_val[cdind] = Cd
    print(rmse_data[cdind])
    print(cd_val)
    print(rmse_data)
    trmse = np.column_stack((cd_val.flatten(), rmse_data.flatten()))
    print(trmse)
    np.savetxt(config.get('Params', 'rmse_file'), trmse, delimiter=',')
    np.savetxt(config.get('Params', 'meas_vs_sim_file'), meas_and_sim_wse, delimiter=',')

    # simvel = np.zeros(meas_vel.shape[0])
    # measvel = np.zeros(meas_vel.shape[0])
    # for counter, line in enumerate(meas_vel):
    #     point2D = [line[0] - xoffset, line[1] - yoffset, 0.0]
    #     cellid = cellLocator2D.FindCell(point2D)
    #     print(isCellWet(SGrid, point2D, cellid, IBC_2D))
    #     tmpvel = getCellValue(SGrid, point2D, cellid, Velocity_2D)
    #     if Cd == cdmin:
    #         meas_and_sim_vel[0, counter] = line[2]
    #         #     print counter
    #     simvel[counter] = tmpvel
    #     measvel[counter] = line[2]
    #     print(cellid, line[2], tmpvel)
    # meas_and_sim_vel[cdind+1, :] = simvel
    # rmse_data_vel[cdind] = rmse(simvel, measvel)
    # trmse = np.column_stack((cd_val.flatten(), rmse_data_vel.flatten()))
    # np.savetxt('Meander_VelComp_10Kcfs_6_15_16b_vel_rmse.txt', trmse, delimiter=',')
    # np.savetxt('Meander_VelComp_10Kcfs_6_15_16b_vel_meas_v_sim.txt', rmse_data_vel, delimiter=',')
#     print WSE_2D.GetRange()
#     print max(WSE_2D)
print(rmse_data)
print(meas_and_sim_wse)
