from pandas.tseries.holiday import AbstractHolidayCalendar, Holiday, nearest_workday, \
    USMartinLutherKingJr, USPresidentsDay, GoodFriday, USMemorialDay, \
    USLaborDay, USThanksgivingDay, EasterMonday


class USTradingCalendar(AbstractHolidayCalendar):
    rules = [
        Holiday('NewYearsDay', month=1, day=1, observance=nearest_workday),
        USMartinLutherKingJr,
        USPresidentsDay,
        GoodFriday,
        USMemorialDay,
        Holiday('USIndependenceDay', month=7, day=4, observance=nearest_workday),
        USLaborDay,
        USThanksgivingDay,
        Holiday('Christmas', month=12, day=25, observance=nearest_workday)
    ]
    
class TargetTradingCalendar(AbstractHolidayCalendar):
    """
    TARGET: https://www.ecb.europa.eu/press/pr/date/2000/html/pr001214_4.en.html
    """
    rules = [
        Holiday('NewYearsDay', month=1, day=1, observance=nearest_workday),
        GoodFriday,
        EasterMonday,
        Holiday('LabourDay', month=5, day=1, observance=nearest_workday),
        Holiday('Christmas', month=12, day=25, observance=nearest_workday),
        Holiday('BoxingDay', month=12, day=26, observance=nearest_workday)
        
    ]