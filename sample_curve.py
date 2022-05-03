"""Sample 3m$L swap curve provided from market as of 2020-12-03."""

from datetime import datetime, timedelta

import numpy as np

from bootstrapper.dateutils import create_maturity, shift_date
from bootstrapper.products import CashRate, Future, Swap

settle = datetime(2020, 12, 3)
start_date = shift_date(settle + timedelta(days=2), [])

US0003M = CashRate(
    0.0022538, settle, settle, create_maturity(settle, "3M"), "Actual_360", "FD"
)
EDZ0 = Future(
    99.7625,
    settle,
    datetime(2020, 12, 16),
    datetime(2021, 3, 17),
    0,
    "Actual_360",
    "FD",
)
EDH1 = Future(
    99.8, settle, datetime(2021, 3, 17), datetime(2021, 6, 16), 0, "Actual_360", "FD"
)
EDM1 = Future(
    99.805, settle, datetime(2021, 6, 16), datetime(2021, 9, 15), 0, "Actual_360", "FD"
)
EDU1 = Future(
    99.795, settle, datetime(2021, 9, 15), datetime(2021, 12, 15), 0, "Actual_360", "FD"
)
EDZ1 = Future(
    99.75, settle, datetime(2021, 12, 15), datetime(2022, 3, 16), 0, "Actual_360", "FD"
)
EDH2 = Future(
    99.755, settle, datetime(2022, 3, 16), datetime(2022, 6, 15), 0, "Actual_360", "FD"
)

USSW2 = Swap(
    0.0023272,
    settle,
    start_date,
    create_maturity(start_date, "2Y"),
    2,
    4,
    "30_360",
    "FD",
)
USSW3 = Swap(
    0.002773,
    settle,
    start_date,
    create_maturity(start_date, "3Y"),
    2,
    4,
    "30_360",
    "FD",
)
USSW4 = Swap(
    0.0033555,
    settle,
    start_date,
    create_maturity(start_date, "4Y"),
    2,
    4,
    "30_360",
    "FD",
)
USSW5 = Swap(
    0.0045214,
    settle,
    start_date,
    create_maturity(start_date, "5Y"),
    2,
    4,
    "30_360",
    "FD",
)
USSW6 = Swap(
    0.005555,
    settle,
    start_date,
    create_maturity(start_date, "6Y"),
    2,
    4,
    "30_360",
    "FD",
)
USSW7 = Swap(
    0.006542,
    settle,
    start_date,
    create_maturity(start_date, "7Y"),
    2,
    4,
    "30_360",
    "FD",
)
USSW8 = Swap(
    0.007446,
    settle,
    start_date,
    create_maturity(start_date, "8Y"),
    2,
    4,
    "30_360",
    "FD",
)
USSW9 = Swap(
    0.008268,
    settle,
    start_date,
    create_maturity(start_date, "9Y"),
    2,
    4,
    "30_360",
    "FD",
)
USSW10 = Swap(
    0.0090095,
    settle,
    start_date,
    create_maturity(start_date, "10Y"),
    2,
    4,
    "30_360",
    "FD",
)
USSW11 = Swap(
    0.009664,
    settle,
    start_date,
    create_maturity(start_date, "11Y"),
    2,
    4,
    "30_360",
    "FD",
)
USSW12 = Swap(
    0.0102445,
    settle,
    start_date,
    create_maturity(start_date, "12Y"),
    2,
    4,
    "30_360",
    "FD",
)
USSW15 = Swap(
    0.0114945,
    settle,
    start_date,
    create_maturity(start_date, "15Y"),
    2,
    4,
    "30_360",
    "FD",
)
USSW20 = Swap(
    0.0127146,
    settle,
    start_date,
    create_maturity(start_date, "20Y"),
    2,
    4,
    "30_360",
    "FD",
)
USSW25 = Swap(
    0.013246,
    settle,
    start_date,
    create_maturity(start_date, "25Y"),
    2,
    4,
    "30_360",
    "FD",
)
USSW30 = Swap(
    0.0135019,
    settle,
    start_date,
    create_maturity(start_date, "30Y"),
    2,
    4,
    "30_360",
    "FD",
)
USSW40 = Swap(
    0.0131268,
    settle,
    start_date,
    create_maturity(start_date, "40Y"),
    2,
    4,
    "30_360",
    "FD",
)
USSW50 = Swap(
    0.0123827,
    settle,
    start_date,
    create_maturity(start_date, "50Y"),
    2,
    4,
    "30_360",
    "FD",
)

s23_instruments = [
    US0003M,
    EDZ0,
    EDH1,
    EDM1,
    EDU1,
    EDZ1,
    EDH2,
    USSW2,
    USSW3,
    USSW4,
    USSW5,
    USSW6,
    USSW7,
    USSW8,
    USSW9,
    USSW10,
    USSW11,
    USSW12,
    USSW15,
    USSW20,
    USSW25,
    USSW30,
    USSW40,
    USSW50,
]

provided_df = np.array(
    [
        0.999430479,
        0.999310692,
        0.998793927,
        0.998291218,
        0.997765162,
        0.997128046,
        0.996506357,
        0.995282539,
        0.991558586,
        0.985563922,
        0.977044421,
        0.966183531,
        0.953703114,
        0.940029736,
        0.925568237,
        0.91051641,
        0.895279022,
        0.879972175,
        0.835763597,
        0.767290455,
        0.708432274,
        0.656392895,
        0.582955753,
        0.533849825,
    ]
)

provided_zc = np.array(
    [
        0.23363,
        0.24435,
        0.22705,
        0.21903,
        0.21719,
        0.22479,
        0.22893,
        0.23611,
        0.28257,
        0.36353,
        0.46446,
        0.57336,
        0.67718,
        0.77252,
        0.85915,
        0.93718,
        1.00538,
        1.06506,
        1.19563,
        1.32372,
        1.33790,
        1.40229,
        1.34828,
        1.25453,
    ]
)
