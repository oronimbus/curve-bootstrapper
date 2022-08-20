"""Utility functions for date creation and day count calculations."""

import calendar as cd
from datetime import datetime, timedelta

import numpy as np

from bootstrapper.calendars import (TargetTradingCalendar, UKTradingCalendar,
                                    USTradingCalendar)


def get_trading_holidays(start: datetime, end: datetime, cal: str) -> list:
    """Create list of holidays according to specified trading calendar."""
    calendars = {
        "FD": USTradingCalendar(),
        "TE": TargetTradingCalendar(),
        "LN": UKTradingCalendar(),
    }

    assert cal in calendars.keys(), "Calendar not supported yet."
    inst = calendars[cal]
    return inst.holidays(start, end)


def shift_date(date: datetime, holidays: list) -> datetime:
    """Move date to next available business day."""
    while (date in holidays) or (date.weekday() == 5) or (date.weekday() == 6):
        date += timedelta(1)
    return date


def actual_360(t_i: datetime, t_j: datetime) -> timedelta:
    """Calculate year fraction between two dates using ACT/360."""
    return (t_j - t_i) / timedelta(360)


def thirty_360(t_i: datetime, t_j: datetime) -> timedelta:
    """Calculate year fraction between two dates using 30/360."""
    d1 = min(30, t_i.day)
    d2 = min(30, t_j.day)
    days_between = (
        360 * (t_j.year - t_i.year) + 30 * (t_j.month - t_i.month) + (d2 - d1)
    )
    return days_between / 360


def actual_365(t_i: datetime, t_j: datetime) -> timedelta:
    """Calculate year fraction between two dates using ACT/365."""
    return (t_j - t_i) / timedelta(365)


def actual_actual(t_i: datetime, t_j: datetime) -> timedelta:
    """Calculate year fraction between two dates using ACT/ACT."""
    return (t_j - t_i) / timedelta(365.25)


def add_months(start_date: datetime, months: int) -> datetime:
    """Add specified number of months to date.."""
    month = start_date.month - 1 + months
    year = start_date.year + month // 12
    month = month % 12 + 1
    day = min(start_date.day, cd.monthrange(year, month)[1])
    return datetime(year, month, day)


def year_frac(t_i: datetime, t_j: datetime, day_count: str, hols: list) -> timedelta:
    """Calculate year fraction acording to bus. day adjustment."""
    calc_yf = {
        "Actual_360": actual_360,
        "30_360": thirty_360,
        "Actual_365": actual_365,
        "Actual_Actual": actual_actual,
    }

    t_i, t_j = shift_date(t_i, hols), shift_date(t_j, hols)
    return calc_yf[day_count](t_i, t_j)


def create_maturity(ref_date: datetime, tenor: str) -> datetime:
    """Create maturity date according to specified tenor."""
    time_unit = tenor[-1].upper()
    assert (time_unit == "M") or (
        time_unit == "Y"
    ), "Tenor must be either months or years."

    period = int(tenor[:-1])
    num_months = period if time_unit == "M" else 12 * period
    return add_months(ref_date, num_months)


def convert_dates_to_dcf(
    start: datetime, dates: list, day_count: str, cal: str
) -> list:
    """Convert list of dates to list of day count fractions."""
    end = np.max(dates) + timedelta(days=7)
    hols = [] if cal == "" else get_trading_holidays(start, end, cal)
    taus = np.array([year_frac(start, date, day_count, hols) for date in dates])
    return taus
