# Swap Curve Bootstrapper

This is a simple curve bootstrapper that was built a couple of years ago to illustrate the mechanics of curve building. It can replicate discount factors from market data providers quite closely but should not be used for financial decision making since it's purely for illustration purposes.

It was not developed with OIS stripping in mind but could be easily extended to allow for a multi-curve framework. Please refer to the notebook for usage.

## Install
The latest version is available from [PyPi](https://pypi.org/project/curve-bootstrapper/0.1.0/) and can be easily installed by running:
```
pip install curve-bootstrapper
```

## Workflow
1. Refer to the sample notebook please
2. Create a list of instruments from `bootstrapper.products`, ideally a mix of cash, futures (or FRA's) and swaps
3. Initalise a curve such as `SwapCurve(settle, 'log-linear')`
4. Add the instruments via e.g. list comprehension `for inst in instruments: curve.add_inst(inst)`
5. Stage the curve using `curve.add_knots()`
6. Strip the curve via `CurveStripper().strip_curve(curve, **kwargs)`
7. Access the discount factor, zero rates or forward rates as attributes from the stripped curve
8. All done!