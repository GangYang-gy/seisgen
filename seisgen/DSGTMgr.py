# -------------------------------------------------------------------
# Strain Green's Tensor (SGT) database Manager
#
# Author: Liang Ding
# Email: myliang.ding@mail.utoronto.ca
# -------------------------------------------------------------------

import os.path
from seisgen.DPointMgr import DPointCloud
from seisgen.seismic.DSyn import DSyn, RTP_to_DENZ
from seisgen.util_SPECFEM3D import get_proc_name
from seisgen.MTTools.DMomentTensors import DMT_enz
from seisgen.util_SPECFEM3D.ibool_reader import DEnquire_Element
from seisgen.util_SPECFEM3D.xyz_reader import DEnquire_XYZ_GLLs_Element
from seisgen.greens_function.sgt_reader import DEnquire_SGT, read_header_info
from seisgen.math.interp_tools import DCreate_anchors_xi_eta_gamma, DLagrange_interp_sgt, DLagrange_any3D

from obspy.core.util.attribdict import AttribDict
from obspy.core import Stream, Trace
from obspy.clients.iris import Client
from obspy.taup import TauPyModel

import numpy as np
import pandas as pd

import time
import utm as utm_lib


class DSGTMgr(DPointCloud):
    '''Strain Green's Tensor (SGT) database Manager'''

    def __init__(self, sgt_database_folder, model3D_folder, point_cloud_file, DLite=False):
        '''
        :param sgt_database_folder:     The directory to the SGT database.
        :param model3D_folder:          The directory to the 3D background model.
        :param point_cloud_file:        The hdf5 file storing the information of user-selected points in the 3D model.
        '''

        self.sgt_database_folder = sgt_database_folder
        self.model3D_folder      = model3D_folder
        self.idx_element         = -1
        self.dt                  = 0
        self.NSPEC               = 0
        
        self._sgt_cache_ndarray = {}    # key = (station, proc_name)
        self._lagrange_cache = {}       # key = (xi, eta, gamma)

        # initial parameters of the SGT database.
        if DLite:
            pass
        else:
            self.__initial_paras()
        super().__init__(point_cloud_file)

    def __initial_paras(self):
        '''Function to initial parameters.'''
        hdf5_files = []
        for (dirpath, dirnames, filenames) in os.walk(self.sgt_database_folder):
            for file in filenames:
                if str(file).lower().endswith('.hdf5') or str(file).lower().endswith('.h5'):
                    hdf5_files.append(os.path.join(dirpath, file))

        # Read the Header file to get the information of SGT database.
        for file in hdf5_files:
            try:
                self.dt, self.NSPEC = read_header_info(file)
                break
            except:
                continue

        if 0 == self.dt or 0 == self.NSPEC:
            raise Exception

    def _initial_element_frame(self):
        ''' return the gll information (index, location) at one selected element. '''
        ibool_file = os.path.join(str(self.model3D_folder), str(self.proc_name)+str("_ibool.bin"))
        self.idx_glls = DEnquire_Element(ibool_file, self.idx_element, self.NSPEC)

        x_glls, y_glls, z_glls = DEnquire_XYZ_GLLs_Element(self.model3D_folder,
                                                           self.idx_processor,
                                                           self.idx_element,
                                                           self.NSPEC)
                                                           
        self.xyz_glls = np.transpose(np.vstack([x_glls, y_glls, z_glls]))

    def _initial_SGTs_N_station(self):
        '''Return the SGT between origin and station. '''
        '''
        dir_string = os.path.join(str(self.sgt_database_folder),
                                     str(self.station.network),
                                     str(self.station.station),
                                     str(self.proc_name))
        '''
        #edit by Gang
        key = (self.station.station, self.proc_name)
        if key in self._sgt_cache_ndarray:
            self.sgts = self._sgt_cache_ndarray[key]
        else:
            dir_string = os.path.join(self.sgt_database_folder, self.station.station, self.proc_name)
            sgt_data_path = dir_string + "_sgt_data.bin"
            sgt_hder_path = dir_string + "_header.hdf5"
            self.sgts = DEnquire_SGT(sgt_data_path, sgt_hder_path, self.idx_glls)
            self._sgt_cache_ndarray[key] = self.sgts  # 缓存数据

    def interp_sgt_Lagrange(self, xi, eta, gamma):
        '''Using lagrange method to interpolate the SGT data.'''
        key = (xi, eta, gamma)
        if key in self._lagrange_cache:
            h_xi_arr, h_eta_arr, h_gamma_arr = self._lagrange_cache[key]
        else:
            xi_gll, eta_gll, gamma_gll = DCreate_anchors_xi_eta_gamma(3)
            h_xi_arr, h_eta_arr, h_gamma_arr = DLagrange_any3D(xi, eta, gamma, xi_gll, eta_gll, gamma_gll)
            self._lagrange_cache[key] = (h_xi_arr, h_eta_arr, h_gamma_arr)
    
        self.sgt_interp = DLagrange_interp_sgt(
            h_xi_arr, h_eta_arr, h_gamma_arr, self.sgts,
            ngll_x=3, ngll_y=3, ngll_z=3
        )

    def set_dt(self, dt):
        '''
        Set time interval.
        Function observed for SeisClient.
        '''
        self.dt = dt
        return self


    def get_sgt(self, station, origin, b_new_origin=True, b_verbose=False, use_utm=False):
        '''
        Get the interpolated SGT between the station-origin pair.
        Unit: m/N.m
        :param station: An instance of the obspy AttribDict class. For example:
                        station = AttribDict({ 'latitude': 34.0210,
                                                'longitude': -118.287,
                                                'network': 'CI',
                                                'station': 'USC',
                                                'location': '',
                                                'id': 'USC'})

        :param origin:  An instance of the obspy AttribDict class. For example:
                        origin = Origin({'time': '2019-07-04T18:39:44.0000Z',
                                              'latitude': 35.601333,
                                              'longitude': -117.597,
                                              'depth_in_m': 2810.0,
                                              'id': 'evt11056825'})
        :param b_new_origin: True, If acquiring SGF at multiple stations for a same origin to save time.
        '''

        if b_verbose:
            t0 = time.time()

        self.station = station
        print(self.station)

        if not self.b_pointcloud_initial:
            raise Exception
        
        try:
            z = origin.depth_in_m
            b_depth = True
        except:
            z = origin.elevation_in_m
            b_depth = False

        if b_new_origin:
            if not use_utm:
                _, _, _, _, \
                _, _, _, \
                self.idx_processor, self.element_index, \
                self.xi, self.eta, self.gamma = self.find(x=origin.latitude, y=origin.longitude, z=z, n=1, b_depth=b_depth)
                #print(self.idx_processor, self.element_index, self.xi, self.eta, self.gamma)
          
            # edit by Gang
            else:
                utm_x, utm_y,_,_ = utm_lib.from_latlon(origin.latitude, origin.longitude)
                _, _, _, _, \
                _, _, _, \
                self.idx_processor, self.element_index, \
                self.xi, self.eta, self.gamma = self.find(x=utm_x, y=utm_y, z=z, n=1, mode='UTM',b_depth=b_depth)
                #print(self.idx_processor, self.element_index, self.xi, self.eta, self.gamma)
            
            #return self.idx_processor

            # MUST subtract 1, the element_index in the point_cloud_file starts from 1.
            self.idx_element = self.element_index - 1
            self.proc_name = get_proc_name(self.idx_processor)
            self._initial_element_frame()

        self._initial_SGTs_N_station()
        # The Lagrange interpolation to get interpolated SGT.
        self.interp_sgt_Lagrange(self.xi, self.eta, self.gamma)
        if b_verbose:
            print("Station={}, time cost={} s.".format(station.id, time.time()-t0))


        return self.sgt_interp


    def get_greens_function(self, station, origin, b_new_origin=True):
        '''
        Get Greens Function between the station-origin pair.
        Unit: m/N.m

        :param station: An instance of the obspy AttribDict class. For example:
                        station = AttribDict({ 'latitude': 34.0210,
                                                'longitude': -118.287,
                                                'network': 'CI',
                                                'station': 'USC',
                                                'location': '',
                                                'id': 'USC'})

        :param origin:  An instance of the obspy AttribDict class. For example:
                        origin = Origin({'time': '2019-07-04T18:39:44.0000Z',
                                              'latitude': 35.601333,
                                              'longitude': -117.597,
                                              'depth_in_m': 2810.0,
                                              'id': 'evt11056825'})

        :param b_new_origin: Accelerating the extraction of SGT data for multiple stations with the same origin.
        '''

        sgt = self.get_sgt(station, origin, b_new_origin=b_new_origin)
        #print(station)
        #return self.get_greens_function_next(sgt, station, origin, b_new_origin)
        #edit by Gang
        return self.get_greens_function_next(sgt, station, origin)


    def get_greens_function_next(self, sgt, station, origin):
        '''The next step of get_greens_function()'''
        client = Client()
        res = client.distaz(stalat=station.latitude, stalon=station.longitude,
                            evtlat=origin.latitude, evtlon=origin.longitude)
        back_azimuth = res['backazimuth']
        azimuth = res['azimuth']
        distance_deg = res['distance']
        distance_m = res['distancemeters']
        stream = self._SGT2GF(sgt, azimuth, back_azimuth)
        stream.id = station.id
        return stream

    def get_waveform(self, station, origin, mt_RTP, b_RTZ=False, b_new_origin=True):
        '''
        Return the synthetic 3-C waveform in RTZ.
        Unit: m

        :param station: An instance of the obspy AttribDict class. For example:
                        station = AttribDict({ 'latitude': 34.0210,
                                                'longitude': -118.287,
                                                'network': 'CI',
                                                'station': 'USC',
                                                'location': '',
                                                'id': 'USC'})

        :param origin:  An instance of the obspy AttribDict class. For example:
                        origin = Origin({'time': '2019-07-04T18:39:44.0000Z',
                                              'latitude': 35.601333,
                                              'longitude': -117.597,
                                              'depth_in_m': 2810.0,
                                              'id': 'evt11056825'})

        :param mt: the moment tensor in RTP [Mrr, Mtt, Mpp, Mrt, Mrp, Mtp]
        :param b_new_origin: Accelerating the extraction of SGT data for multiple stations with the same origin.
        '''

        sgt = self.get_sgt(station, origin, b_new_origin=b_new_origin)
        return self.get_waveform_next(sgt, station, origin, mt_RTP, b_RTZ)


    def get_waveform_next(self, sgt, station, origin, mt_RTP, b_RTZ=False):
        '''The next step of get_waveform()'''

        client = Client()
        res = client.distaz(stalat=station.latitude, stalon=station.longitude,
                            evtlat=origin.latitude, evtlon=origin.longitude)
        back_azimuth = res['backazimuth']
        distance_deg = res['distance']
        distance_m = res['distancemeters']

        # synthetic
        element='SY'
        mt_enz = RTP_to_DENZ(mt_RTP)
        _st = DSyn(mt_enz, sgt, element)
        for _tr in _st:
            _tr.stats.delta = self.dt
            _tr.stats.sampling_rate = int(1.0 / self.dt)

        # waveform in ENZ or RTZ convention
        if b_RTZ:
            _st.rotate(method='NE->RT', back_azimuth=back_azimuth)
        return _st

    def _SGT2GF(self, sgt, azi, ba, b_USE=True):
        '''
        Get 3D MT Green's functions
        ( Up-South-East convention (USE) by default, which is compatible with MTUQ,
        otherwise North-East-Down convention)
        '''

        mt_rtp = pd.DataFrame({
            "Mrr": np.array([1.0,  0,    0,    0,   0,    0]),
            "Mtt": np.array([0,    1.0,  0,    0,   0,    0]),
            "Mpp": np.array([0,    0,   1.0,   0,   0,    0]),
            "Mrt": np.array([0,    0,   0,     1.0, 0,    0]),
            "Mrp": np.array([0,    0,   0,     0,   1.0,  0]),
            "Mtp": np.array([0,    0,   0,     0,   0,    1.0]),
        })

        stream = Stream()
        mt_orders = ['Mrr', 'Mtt', 'Mpp', 'Mrt', 'Mrp', 'Mtp',]

        for i_mt in mt_orders:
            mt_enz = RTP_to_DENZ(mt_rtp[i_mt].values)
            _st = DSyn(mt_enz, sgt, element='SYN')
            _st.rotate(method='NE->RT', back_azimuth=ba)
            for _tr in _st:
                ch = _tr.stats.channel
                _tr.stats.channel = '%s.%s' % (ch[-1], i_mt)
                _tr.stats._component = ch[-1]
                _tr.stats.delta = self.dt
                _tr.stats.sampling_rate = int(1.0 / self.dt)

            stream += _st
        return stream
    

    def get_sgt_HMC(self, station_names, b_verbose=False):
        """
        Get the interpolated SGT for HMC inversion.
        Returns ndarray compatible with numba-accelerated interpolation.
        """
        if b_verbose:
            import time
            t0 = time.time()

        self.station_names = station_names

        if not self.b_pointcloud_initial:
            raise Exception

        # setup element frame
        self.idx_element = self.element_index - 1
        self.proc_name = get_proc_name(self.idx_processor)
        self._initial_element_frame()

        # load SGT as ndarray
        self.sgts = DEnquire_SGT(
            data_path=os.path.join(self.sgt_database_folder, str(self.station_names), f"{self.proc_name}_sgt_data.bin"),
            header_path=os.path.join(self.sgt_database_folder, str(self.station_names), f"{self.proc_name}_header.hdf5"),
            g_indx_GLL_points=self.idx_glls
        )

        # interpolate with Lagrange (Numba accelerated)
        self.interp_sgt_Lagrange(self.xi, self.eta, self.gamma)

        return self.sgt_interp

        # pad zeros if needed (310 samples example)
        sgt_new = np.zeros((310, 3, 6), dtype=np.float32)
        for i in range(3):
            for j in range(6):
                if len(self.sgt_interp[:, i, j]) == 290:
                    sgt_new[:, i, j] = np.hstack([np.zeros(20), self.sgt_interp[:, i, j]])
                else:
                    sgt_new[:, i, j] = self.sgt_interp[:, i, j]
        self.sgt_interp = sgt_new.copy()

        if b_verbose:
            print(f"SGT_HMC for {station_names} loaded, time cost={time.time()-t0:.2f}s")
        
        return self.sgt_interp

    