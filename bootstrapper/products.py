from .dateutils import *
from collections import OrderedDict
from datetime import datetime, timedelta

import calendar as cd
import numpy as np
import math

class Rate:
    def __init__(self, rate, settle_date, start_date, end_date, frequency, day_count, calendar):
        self.rate = rate
        self.frequency = frequency
        self.day_count = day_count
        self.calendar = calendar
        self.hols = get_trading_holidays(start_date, end_date, self.calendar)
        self.settle_date = shift_date(settle_date, self.hols)
        self.start_date = shift_date(start_date, self.hols)
        self.end_date = shift_date(end_date, self.hols)
        
    def __str__(self):
        return self.__class__.__name__
    
    def __repr__(self):
        
        freq_dict = {0: 'Zero',
                     1: 'Annual',
                     2: 'Semi-Annual',
                     4: 'Quarterly',
                     12: 'Monthly'}
        
        print_rate = round(self.rate * 100,4)
        print_start = datetime.strftime(self.start_date, '%Y-%m-%d')
        print_end = datetime.strftime(self.end_date, '%Y-%m-%d')
        print_freq = freq_dict[self.frequency]
        
        print_str =  """{} | Interest Rate: {}% | \
                        Effective: {} | Maturity: {} | Frequency: {} | \
                        Day Count Conv.: {}""".format(self.__class__.__name__, print_rate, print_start, 
                                                      print_end, print_freq, self.day_count)
        return print_str

class CashRate(Rate):
    def __init__(self, rate, settle_date, start_date, end_date, day_count, calendar):
        self.frequency = 0 
        Rate.__init__(self, rate, settle_date, start_date, end_date, self.frequency, day_count, calendar)
        
        # dates
        self.t1 = year_frac(self.start_date, self.end_date, self.day_count, self.hols)
        self.tau = np.array([self.t1])
        
    def par_rate(self, dfs):
        return self.rate

class Future(Rate):
    def __init__(self, price, settle_date, start_date, end_date, frequency, day_count, calendar, **kwargs):
        self.price = price
        self.rate = (100 - self.price) / 100
        Rate.__init__(self, self.rate, settle_date, start_date, end_date, frequency, day_count, calendar)        
        
        # dates
        self.t1 = year_frac(self.settle_date, self.start_date, self.day_count, self.hols)
        self.t2 = year_frac(self.settle_date, self.end_date, self.day_count, self.hols)
        self.tau = np.array([self.t1, self.t2])
        
        # adjusted rate
        self.adj_rate = self.rate - self.convexity_adjustment(**kwargs)


    def convexity_adjustment(self, alpha=0.03, market_nvol=0.005):
        b_t1_t2 = (1 - (math.exp(-alpha * (self.t2 - self.t1)))) / alpha
        b_sqrt = math.pow((1 - math.exp(-alpha * self.t1)) / alpha, 2)
        x_t1_t2 = math.pow(market_nvol,2) / (2 * alpha) * b_t1_t2 * (b_t1_t2 * (1 - math.exp(-(2 * alpha * self.t1))) + alpha * b_sqrt)
        
        conv_adj = round((1 - math.exp(-x_t1_t2)) * (self.rate  + 1 / (self.t2 - self.t1)) * 100, 5)
        return conv_adj / 100
    
    def par_rate(self, dfs, **kwargs):
        r_mid = -(np.log(dfs[1]) - np.log(dfs[0])) / (self.t2 - self.t1)
        convexity_adj = self.convexity_adjustment(**kwargs)
        return r_mid + convexity_adj
    
class FRA(Rate):
    def __init__(self, rate, settle_date, start_date, end_date, frequency, day_count, calendar):
        Rate.__init__(self, rate, settle_date, start_date, end_date, frequency, day_count, calendar)

        # dates
        self.t1 = year_frac(self.settle_date, self.start_date, self.day_count, self.hols)
        self.t2 = year_frac(self.settle_date, self.end_date, self.day_count, self.hols)
        self.tau = np.array([self.t1, self.t2])
        
    def cashflow(self, forward):
        t = year_frac(self.start_date, self.end_date, self.day_count, self.hols)
        return (t * (forward - self.rate)) / (1 + t * forward)
    
    def par_rate(self, dfs, **kwargs):
        r_mid = -(np.log(dfs[1]) - np.log(dfs[0])) / (self.t2 - self.t1)
        return r_mid 
    
    
class Swap(Rate):
    def __init__(self, rate, settle_date, start_date, end_date, frequency, day_count, calendar):
        Rate.__init__(self, rate, settle_date, start_date, end_date, frequency, day_count, calendar)
        
        # schedules
        self._fixed_schedule = OrderedDict()
        self._floating_schedule = OrderedDict()
        
        # dcf[0, t_i] for coupons
        self._fixed_year_fracs = []
        self._floating_year_fracs = []
        self.tau = self.fixed_year_fracs
        
    @property
    def fixed_schedule(self):
        self._fixed_schedule = self.__create_fixed_schedule()
        return self._fixed_schedule
    
    @property
    def floating_schedule(self):
        self._floating_schedule = self.__create_floating_schedule()
        return self._floating_schedule
    
    @property
    def fixed_year_fracs(self):
        self._fixed_year_fracs = [year_frac(self.start_date, date, self.day_count, self.hols) for date in self.fixed_schedule]
        return self._fixed_year_fracs
    
    @property
    def floating_year_fracs(self):
        self._floating_year_fracs = [year_frac(self.start_date, date, self.day_count, self.hols) for date in self.floating_schedule]
        return self._floating_year_fracs

    def __create_fixed_schedule(self):
        pay_date = self.start_date
        schedule = []
        cashflows = OrderedDict()
        
        while pay_date < self.end_date:
            try:
                period = int(12 / self.frequency)
                tmp_date = add_months(pay_date, period)
                pay_date = shift_date(tmp_date, self.hols)
                schedule.append(pay_date)
            except:
                raise "Zero Coupon not supported yet."
        
        schdl = [self.start_date] + schedule
        taus = [year_frac(i, j, self.day_count, self.hols) for i,j in zip(schdl[:-1], schdl[1:])]
        cashflows = dict(zip(schedule, taus))
        
        return cashflows
    
    def __create_floating_schedule(self):
        raise NotImplementedError('Floating schedule not supported yet.')
    
    def par_rate(self, dfs):
        float_leg = 1.0
        t = np.array(list(self.fixed_schedule.values()))
        fixed_leg = np.sum(dfs * t)
        r_mid = (float_leg - dfs[-1]) / fixed_leg
        return r_mid
        