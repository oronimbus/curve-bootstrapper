from collections import OrderedDict
from .dateutils import *
from scipy.interpolate import interp1d
from scipy.optimize import least_squares, OptimizeResult

import numpy as np
import pandas as pd

    
def zc_to_df(t, r, f='cont'):
    _r = np.array(r)
    _t = np.array(t)

    # filter out t == 0
    idx = np.where(_t != 0)[0]
    r = _r[idx]
    t = _t[idx]

    if f == 'cont':
        v = np.exp(-r * t)
    elif f == 'simple':
        v =  1 / (1 + r * t)
    elif isinstance(f, int):
        v =  1 / np.power(1 + r / f, f * t)
    else:
        raise ValueError('Frequency not supported.')
    return v

def df_to_zc(t, df, f='cont'):
    _df = np.array(df)
    _t = np.array(t)

    # filter out t == 0
    idx = np.where(_t != 0)[0]
    df = _df[idx]
    t = _t[idx]

    if f == 'cont':
        r =  -np.log(df) / t
    elif f == 'simple':
        r =  (1 / df - 1) / t
    elif isinstance(f, int):
        r = (np.power(1 / df, 1 / (f * t)) - 1) * f 
    else:
        raise ValueError('Frequency not supported.')    
    return r 

class Interpolator:
    def __init__(self, x, y):
        self.x = np.array(x)
        self.y = np.array(y)
        
    def linear(self, x_i):
        return interp1d(self.x, self.y, kind='linear', fill_value='extrapolate')(x_i)

    def log_linear(self, x_i):
        log_y = np.log(self.y)
        return np.exp(interp1d(self.x, log_y, kind='linear', fill_value='extrapolate')(x_i))
    
    def convex_monotone(self, x_i):
        pass
    
    def log_cubic(self, x_i):
        log_y = np.log(self.y)
        return np.exp(interp1d(self.x, log_y, kind='cubic', fill_value='extrapolate')(x_i))

    
class CurveInterpolator(Interpolator):
    def __init__(self, x, y, on):
        Interpolator.__init__(self, x, y)
        assert (on == 'df') or (on == 'zc'), 'Parameter `on` needs to be either `df` or `zc`.'
        self.on = on
    
    def interpolate(self, t_hat, how, **kwargs):
        assert not ((self.on == 'zc') and ((how == 'log-linear') or (how == 'cubic'))), "Log interpolation not supported on `zc`."
        interp_methods = {'log-linear': self.log_linear,
                          'linear': self.linear,
                          'convex-monotone':self.convex_monotone,
                          'cubic': self.log_cubic}
        
        if self.on == 'zc':
            self.y = df_to_zc(self.x, self.y, **kwargs)
            self.x = self.x[np.where(self.x != 0)[0]]
        
        y_hat = interp_methods[how](t_hat)
        
        if self.on == 'zc':
            return np.append([1], zc_to_df(t_hat, y_hat))
        else:
            return y_hat
        
class SwapCurve:
    def __init__(self, settle_date, interpolation, interp_on='df'):
        self.settle_date = settle_date
        
        # dictionary with all curve instruments
        self._par_curve = OrderedDict()
        
        # knots
        self._knots = [] 
        self._knots_taus = []
        self._knots_dfs = []
        self._knots_zcs = []
        self._knots_par_rates = []
        
        # interpolation 
        self.interpolation = interpolation
        self.interp_on = interp_on
        self._interpolator = CurveInterpolator

        
    @property
    def knots(self):
        return self._knots
        
    @property
    def knots_taus(self):
        return self._knots_taus
    
    @property
    def knots_dfs(self):
        return self._knots_dfs
    
    @property
    def knots_zcs(self):
        return self._knots_zcs
    
    @knots_dfs.setter
    def knots_dfs(self, dfs):
        self._knots_dfs = dfs
        
    @knots_zcs.setter
    def knots_zcs(self, zcs):
        self._knots_zcs = zcs
    
    @property
    def knots_par_rates(self):
        return self._knots_par_rates
    
    @property
    def par_curve(self):
        return self._par_curve
    
    def add_inst(self, inst):
        if inst not in self._par_curve.values():
            i = len(self._par_curve) + 1
            self._par_curve[i] = inst       
        
    def remove_inst(self, inst):
        if inst in self._par_curve.values():
            del self._par_curve[inst]
    
    def add_knots(self, align_with_insts=True, **kwargs):
        if align_with_insts == True:
            self._knots = sorted(self.__build_dates())
            self.__initalise_curve()
            
        elif align_with_insts == False : 
            raise NotImplementedError('Custom knots not supported yet.')
            
        else:
            raise ValueError('align_with_insts needs to be a bool.')
            
    def __build_dates(self):
        try:  
            return [self.settle_date] + [inst.end_date for inst in self.par_curve.values()]
            
        except Exception as error:
            print("Dates could not be generated. Error message: {}".format(error))            
    
    def __initalise_curve(self):
        self._knots_taus =  convert_dates_to_dcf(self.settle_date, self._knots, 'Actual_365', '')
        self._knots_par_rates = np.array([np.nan] + [inst.rate if str(inst) != 'Future' else inst.adj_rate for inst in self.par_curve.values()])
        self._knots_zcs = np.copy(self._knots_par_rates)
        self._knots_dfs = np.exp(-self._knots_par_rates * self._knots_taus)
        self._knots_dfs[0] = 1
        
    def __convert_times(self, t):
        _dates = [dt_i for dt_i in dts if type(dt_i) == datetime]
        _t = [t_i for t_i in dts if ((type(t_i) == float) or (type(t_i) == int))]
        _ts = convert_dates_to_dcf(s23.settle_date, _dates, '30_360', 'FD')
        ts = sorted(_t + _ts)
    
    def get_dfs(self, t_i, **kwargs):
        assert len(self.par_curve) > 0, "No instruments found."
        assert len(self.knots) > 0, "Add knots first."
        assert len(self.knots_dfs) > 0, "Strip curve first."
        
        df_int = self._interpolator(self.knots_taus, self.knots_dfs, self.interp_on).interpolate(t_i, self.interpolation, **kwargs)
        return df_int
    
    def get_zcs(self, t_i, f='cont', day_count='Actual_365', **kwargs):
        dfs = self.get_dfs(t_i, **kwargs)
        zcs = df_to_zc(t_i, dfs, f)
        return zcs
    
    def get_fwds(self, t_i, t_j, **kwargs):
        assert (t_j > t_i).all(), "t_j must be strictly greater than t_i."
        df_i = self.get_dfs(t_i, **kwargs)
        df_j = self.get_dfs(t_j, **kwargs)
        t = t_j - t_i
        fwds = -(np.log(df_j) - np.log(df_i)) / t
        return fwds
    
    def df(self):
        assert len(self.par_curve) > 0, "No instruments found."

        instruments = [pd.DataFrame({'Type' : [str(inst)], 
                                     'Start Date' : [inst.start_date], 
                                     'End Date' : [inst.end_date], 
                                     'Par Rate' : [inst.rate * 100]}) for inst in self.par_curve.values()]
        
        instruments_df = pd.concat(instruments)
        instruments_df.index = np.arange(len(self.par_curve))

        stripped_df = pd.DataFrame({'DF' : self.knots_dfs[1:],
                                    'ZC' : self.knots_zcs[1:] * 100})
        
        concat_df = pd.concat([instruments_df, stripped_df], axis=1)
        concat_df.index += 1
        return concat_df

    
    def __repr__(self):
        return "Swap Curve Object | Number of instruments: {}".format(str(len(self.par_curve)))


class CurveStripper:
    def __init__(self):
        """
        Some default args might follow.
        """
        pass
    
    def __calculate_inst_residual(self, inst, curvemap):
        dfs = [df for t, df in curvemap.items() if t in inst.tau]
        target_rate = inst.rate if str(inst) != 'Future' else inst.adj_rate
        par_rate = inst.par_rate(dfs)
        return (target_rate - par_rate)

    def calculate_residuals(self, dfs, t, instruments, how, on):
        # define residuals for LS optimizer
        residuals = []

        # interpolate all dofs
        interpolator = CurveInterpolator(t, dfs, on)
        dofs = self.get_dofs(instruments)
        df_dofs = interpolator.interpolate(dofs, how)
        curvemap = dict(zip(dofs, df_dofs))

        # calculate residuals for each instrument
        for inst in instruments:
            residual = self.__calculate_inst_residual(inst, curvemap)
            residuals.append(residual) 
        return residuals

    def get_dofs(self, instruments):
        dofs = []
        for inst in instruments:
            dofs += list(inst.tau)
        return sorted(list(set(dofs)))

    def strip_curve(self, curve, interpolation='log-linear', interp_on='df'):
        assert len(curve.par_curve) > 0, "Add instruments before stripping."
        assert len(curve.knots) > 0, "Add knots before stripping."              
        
        # bounds
        k = len(curve.knots) - 1
        bnds = (np.zeros(k), np.inf * np.ones(k))
        
        # initial inputs
        dfs_0 = curve.knots_dfs[1:]
        t = curve.knots_taus[1:]
        instruments = curve.par_curve.values()

        # least squares solver
        print('Stripping Curve... Number of knots: {}'.format(k))
        result = least_squares(self.calculate_residuals, dfs_0, args=(t, instruments, interpolation, interp_on), bounds=bnds)
        
        assert isinstance(result, OptimizeResult)
        print('Stripping successful! Residual error: {:.3e}'.format(np.sum(result.fun)))
        curve.knots_dfs[1:] = result.x
        curve.knots_zcs[1:] = -np.log(result.x) / t
        
        return curve