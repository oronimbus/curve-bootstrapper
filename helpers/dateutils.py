from datetime import datetime, timedelta
from .calendars import *
import calendar as cd


def get_trading_holidays(start, end, cal):
    calendars = {'FD' : USTradingCalendar(),
                 'TE' : TargetTradingCalendar()}
    inst = calendars[cal]
    return inst.holidays(start, end)

def shift_date(date, holidays):
    while (date in holidays) or (date.weekday() == 5) or (date.weekday() == 6):
        date += timedelta(1)
    return date

def actual_360(t_i, t_j):
    return (t_j - t_i) / timedelta(360)

def thirty_360(t_i, t_j):
    d1 = min(30, t_i.day)
    d2 = min(30, t_j.day) 
    days_between =  360 * (t_j.year - t_i.year) + 30 * (t_j.month - t_i.month) + (d2 - d1)
    return days_between / 360

def actual_365(t_i, t_j): 
    return (t_j - t_i) / timedelta(365)

def actual_actual(t_i, t_j):
    return (t_j - t_i) / timedelta(365.25)

def add_months(start_date, months):
    month = start_date.month - 1 + months
    year = start_date.year + month // 12
    month = month % 12 + 1
    day = min(start_date.day, cd.monthrange(year,month)[1])
    return datetime(year, month, day)

def year_frac(t_i, t_j, day_count, hols):
    calc_yf = {'Actual_360' : actual_360,
               '30_360' : thirty_360,
               'Actual_365' : actual_365,
               'Actual_Actual' : actual_actual}
        
    t_i, t_j = shift_date(t_i, hols), shift_date(t_j, hols)
    return calc_yf[day_count](t_i, t_j)


def create_maturity(ref_date, tenor):
    assert isinstance(tenor, str), 'Tenor not a string.'
    assert isinstance(ref_date, datetime), 'Reference date not of type `datetime`.'
    time_unit = tenor[-1].upper()
    assert ((time_unit == 'M') or (time_unit == 'Y')), 'Tenor must be either months or years.'
    
    period = int(tenor[:-1])
    num_months = period if time_unit == 'M' else 12 * period
    return add_months(ref_date, num_months)