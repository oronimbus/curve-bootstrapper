"""Main module for curve stripping and objective minimization."""

import logging
from collections import OrderedDict
from datetime import datetime
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.optimize import OptimizeResult, least_squares

from bootstrapper.dateutils import convert_dates_to_dcf
from bootstrapper.products import Rate

logger = logging.getLogger(__name__)


def zc_to_df(t: Iterable, r: Iterable, f: str = "cont") -> np.array:
    """Convert zero coupon rates to discount factors."""
    _r = np.array(r)
    _t = np.array(t)

    # filter out t == 0
    idx = np.where(_t != 0)[0]
    r = _r[idx]
    t = _t[idx]

    if f == "cont":
        v = np.exp(-r * t)
    elif f == "simple":
        v = 1 / (1 + r * t)
    elif isinstance(f, int):
        v = 1 / np.power(1 + r / f, f * t)
    else:
        raise ValueError("Frequency not supported.")
    return v


def df_to_zc(t: Iterable, df: Iterable, f: str = "cont") -> np.array:
    """Convert discount factors to zero coupon rates."""
    _df = np.array(df)
    _t = np.array(t)

    # filter out t == 0
    idx = np.where(_t != 0)[0]
    df = _df[idx]
    t = _t[idx]

    if f == "cont":
        r = -np.log(df) / t
    elif f == "simple":
        r = (1 / df - 1) / t
    elif isinstance(f, int):
        r = (np.power(1 / df, 1 / (f * t)) - 1) * f
    else:
        raise ValueError("Frequency not supported.")
    return r


class Interpolator:
    """Base interpolator class supporting common schemes."""

    def __init__(self, x: Iterable, y: Iterable):
        self.x = np.array(x)
        self.y = np.array(y)

    def linear(self, x_i: float) -> float:
        """Linear interpolation using scipy."""
        return interp1d(self.x, self.y, kind="linear", fill_value="extrapolate")(x_i)

    def log_linear(self, x_i: float) -> float:
        """Linear interpolation in log space using scipy."""
        log_y = np.log(self.y)
        return np.exp(
            interp1d(self.x, log_y, kind="linear", fill_value="extrapolate")(x_i)
        )

    def convex_monotone(self, x_i: float) -> float:
        """Hagan & West implementation for convex-monotone splines."""
        # TODO: add Hagan & West implementation for convex monotone splines
        pass

    def log_cubic(self, x_i: float) -> float:
        """Log cubic interpolation using scipy."""
        log_y = np.log(self.y)
        return np.exp(
            interp1d(self.x, log_y, kind="cubic", fill_value="extrapolate")(x_i)
        )


class CurveInterpolator(Interpolator):
    """Interpolator for curves."""

    def __init__(self, x: Iterable, y: Iterable, on: str):
        Interpolator.__init__(self, x, y)
        self.on = on

    def interpolate(self, t_hat: Iterable, how: str, **kwargs: dict) -> np.array:
        """Interpolate along time axis given interpolation scheme."""
        interp_methods = {
            "log-linear": self.log_linear,
            "linear": self.linear,
            "convex-monotone": self.convex_monotone,
            "cubic": self.log_cubic,
        }

        if self.on == "zc":
            self.y = df_to_zc(self.x, self.y, **kwargs)
            self.x = self.x[np.where(self.x != 0)[0]]

        y_hat = interp_methods[how](t_hat)

        if self.on == "zc":
            return np.append([1], zc_to_df(t_hat, y_hat))
        else:
            return y_hat


class SwapCurve:
    """Swap curve class storing all data."""

    def __init__(
        self, settle_date: datetime, interpolation: str, interp_on: str = "df"
    ):
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

    def add_inst(self, inst: Rate):
        """Add instrument to curve set."""
        if inst not in self._par_curve.values():
            i = len(self._par_curve) + 1
            self._par_curve[i] = inst

    def remove_inst(self, inst: Rate):
        """Remove instrument from curve set."""
        if inst in self._par_curve.values():
            del self._par_curve[inst]

    def add_knots(self, align_with_insts: bool = True, **kwargs: dict):
        """Add instruments from curve set to curve knots."""
        if align_with_insts:
            self._knots = sorted(self.__build_dates())
            self.__initalise_curve()

        else:
            raise NotImplementedError("Custom knots not supported yet.")

    def __build_dates(self):
        """Build dates for curve knots."""
        try:
            return [self.settle_date] + [
                inst.end_date for inst in self.par_curve.values()
            ]

        except Exception as error:
            logger.warning(
                "Dates could not be generated. Error message: {}".format(error)
            )

    def __initalise_curve(self):
        """Stage instruments and load curve knots."""
        self._knots_taus = convert_dates_to_dcf(
            self.settle_date, self._knots, "Actual_365", ""
        )
        self._knots_par_rates = np.array(
            [np.nan]
            + [
                inst.rate if str(inst) != "Future" else inst.adj_rate
                for inst in self.par_curve.values()
            ]
        )
        self._knots_zcs = np.copy(self._knots_par_rates)
        self._knots_dfs = np.exp(-self._knots_par_rates * self._knots_taus)
        self._knots_dfs[0] = 1

    def get_dfs(self, t_i: Iterable, **kwargs: dict) -> np.array:
        """Get interpolated discount factors from calibrated curve."""
        assert len(self.par_curve) > 0, "No instruments found."
        assert len(self.knots) > 0, "Add knots first."
        assert len(self.knots_dfs) > 0, "Strip curve first."

        df_int = self._interpolator(
            self.knots_taus, self.knots_dfs, self.interp_on
        ).interpolate(t_i, self.interpolation, **kwargs)
        return df_int

    def get_zcs(
        self,
        t_i: Iterable,
        f: str = "cont",
        day_count: str = "Actual_365",
        **kwargs: dict
    ) -> np.array:
        """Get interpolated zero coupon rates from calibrated curve."""
        dfs = self.get_dfs(t_i, **kwargs)
        zcs = df_to_zc(t_i, dfs, f)
        return zcs

    def get_fwds(self, t_i: Iterable, t_j: Iterable, **kwargs: dict) -> np.array:
        """Get interpolated forwards from calibrated curve."""
        assert (t_j > t_i).all(), "t_j must be strictly greater than t_i."
        df_i = self.get_dfs(t_i, **kwargs)
        df_j = self.get_dfs(t_j, **kwargs)
        t = t_j - t_i
        fwds = -(np.log(df_j) - np.log(df_i)) / t
        return fwds

    def df(self) -> pd.DataFrame:
        """Convert curve set into a pandas DataFrame."""
        assert len(self.par_curve) > 0, "No instruments found."

        instruments = [
            pd.DataFrame(
                {
                    "Type": [str(inst)],
                    "Start Date": [inst.start_date],
                    "End Date": [inst.end_date],
                    "Par Rate": [inst.rate * 100],
                }
            )
            for inst in self.par_curve.values()
        ]

        instruments_df = pd.concat(instruments)
        instruments_df.index = np.arange(len(self.par_curve))

        stripped_df = pd.DataFrame(
            {"DF": self.knots_dfs[1:], "ZC": self.knots_zcs[1:] * 100}
        )

        concat_df = pd.concat([instruments_df, stripped_df], axis=1)
        concat_df.index += 1
        return concat_df

    def __repr__(self):
        return "Swap Curve Object | Number of instruments: {}".format(
            str(len(self.par_curve))
        )


class CurveStripper:
    """Strip swap curve using multidimensional solver."""

    def __init__(self):
        pass

    def __calculate_inst_residual(self, inst: Rate, curvemap: dict) -> float:
        """Calculate residuals from calibrated rate to curve instrument."""
        dfs = [df for t, df in curvemap.items() if t in inst.tau]
        target_rate = inst.rate if str(inst) != "Future" else inst.adj_rate
        par_rate = inst.par_rate(dfs)
        return target_rate - par_rate

    def calculate_residuals(
        self, dfs: Iterable, t: Iterable, instruments: Iterable, how: str, on: str
    ) -> list:
        """Calculate residuals of calibrated curve to market rates."""
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

    def get_dofs(self, instruments: Iterable):
        """Get degrees of freedom from curve instruments."""
        dofs = []
        for inst in instruments:
            dofs += list(inst.tau)
        return sorted(list(set(dofs)))

    def strip_curve(
        self, curve: SwapCurve, interpolation: str = "log-linear", interp_on: str = "df"
    ) -> SwapCurve:
        """Strip curve using interpolation scheme."""
        # bounds
        k = len(curve.knots) - 1
        bnds = (np.zeros(k), np.inf * np.ones(k))

        # initial inputs
        dfs_0 = curve.knots_dfs[1:]
        t = curve.knots_taus[1:]
        instruments = curve.par_curve.values()

        # least squares solver
        logger.info("Stripping Curve... Number of knots: {}".format(k))
        result = least_squares(
            self.calculate_residuals,
            dfs_0,
            args=(t, instruments, interpolation, interp_on),
            bounds=bnds,
        )

        assert isinstance(result, OptimizeResult)
        logger.info(
            "Stripping successful! Residual error: {:.3e}".format(np.sum(result.fun))
        )
        curve.knots_dfs[1:] = result.x
        curve.knots_zcs[1:] = -np.log(result.x) / t

        return curve
