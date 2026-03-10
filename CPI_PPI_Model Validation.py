# =========================================================
# PRE-MODEL DIAGNOSTIC ENGINE FOR CPI–PPI TIME SERIES
# =========================================================

import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss
from arch.unitroot import PhillipsPerron, DFGLS, ZivotAndrews
from statsmodels.tsa.vector_ar.vecm import select_order, select_coint_rank
import ruptures as rpt

# =========================================================
# 1. LOAD DATA
# =========================================================

file_path = "CPI_PPI_INFLATION.xlsx"

df = pd.read_excel(file_path)

df['Date'] = pd.to_datetime(df['Date'], format='%b-%y')
df = df.set_index('Date')

df = df.rename(columns={
    'Inflation Rate': 'CPI',
    'Producer Price Change': 'PPI'
})

df = df.sort_index()
df = df.dropna()

print("\nDATA SUMMARY")
print(df.describe())

# =========================================================
# 2. STRUCTURAL BREAK TESTS
# =========================================================

print("\n================ STRUCTURAL BREAK TESTS ================")

for col in df.columns:

    print(f"\n--- Zivot-Andrews Structural Break Test for {col} ---")
    
    series = df[col].astype(float).dropna()
    
    za = ZivotAndrews(series)
    print(za.summary())

    print(f"\n--- Bai-Perron Style Break Detection for {col} ---")
    
    signal = series.values
    
    algo = rpt.Pelt(model="rbf").fit(signal)
    breaks = algo.predict(pen=10)

    print("Break indices:", breaks)
    
    # Optional: Convert breakpoints to actual dates
    break_dates = [series.index[b-1] for b in breaks if b < len(series)]
    print("Break dates:", break_dates)


def unit_root_suite(series, name):
    results = {}

    # ADF
    adf_res = adfuller(series, autolag='AIC')
    adf_stat = adf_res[0]
    adf_p = adf_res[1]
    adf_crit = adf_res[4]
    adf_stationary = adf_stat < adf_crit['5%']
    results['ADF'] = {'statistic': adf_stat, 'p_value': adf_p, 'crit_5%': adf_crit['5%'], 'stationary': adf_stationary}

    # PP
    pp_res = PhillipsPerron(series)
    pp_stat = pp_res.stat
    pp_p = pp_res.pvalue
    pp_stationary = pp_p < 0.05
    results['PP'] = {'statistic': pp_stat, 'p_value': pp_p, 'stationary': pp_stationary}

    # KPSS
    kpss_stat, kpss_p, _, kpss_crit = kpss(series, regression='c', nlags="auto")
    kpss_stationary = kpss_stat < kpss_crit['5%']
    results['KPSS'] = {'statistic': kpss_stat, 'p_value': kpss_p, 'crit_5%': kpss_crit['5%'], 'stationary': kpss_stationary}

    # DF-GLS
    dfgls_res = DFGLS(series)
    dfgls_stat = dfgls_res.stat
    dfgls_p = dfgls_res.pvalue
    dfgls_stationary = dfgls_p < 0.05
    results['DFGLS'] = {'statistic': dfgls_stat, 'p_value': dfgls_p, 'stationary': dfgls_stationary}

    # Count votes: 1 if stationary, 0 if not
    stationary_votes = sum([adf_stationary, pp_stationary, kpss_stationary, dfgls_stationary])

    return results, stationary_votes


# =========================================================
# 4. INTEGRATION ORDER ASSESSMENT
# =========================================================

print("\n================ INTEGRATION ORDER TESTING ================")

integration_results = {}

for col in df.columns:
    # Level
    level_results, level_votes = unit_root_suite(df[col], f"{col} LEVEL")
    
    # First difference
    diff_results, diff_votes = unit_root_suite(df[col].diff().dropna(), f"{col} DIFF")
    
    # Decide integration order based on majority of tests
    if level_votes >= 3:
        integration_results[col] = 'I(0)'
    elif diff_votes >= 3:
        integration_results[col] = 'I(1)'
    else:
        integration_results[col] = 'Unclear'
    
    # Optional: print table-ready summary
    print(f"\n{col} Integration Summary:")
    print("Test | Statistic | p-value | 5% Crit | Stationary")
    for test, res in level_results.items():
        crit_val = res.get('crit_5%', '-')
        print(f"{test} | {res['statistic']:.4f} | {res['p_value']:.4f} | {crit_val} | {res['stationary']}")

# =========================================================
# 5. MODEL DECISION LOGIC
# =========================================================

orders = list(integration_results.values())

if all(x == "I(0)" for x in orders):
    print("\nMODEL RECOMMENDATION: VAR in LEVELS")

elif all(x == "I(1)" for x in orders):

    print("\nVariables appear I(1). Proceeding to cointegration testing.")

    lag_order = select_order(df, maxlags=12)
    p = lag_order.aic
    print("\nSelected lag (AIC):", p)

    print("\nPantula Principle Testing")

    models = [
        ("No deterministic", -1),
        ("Restricted intercept", 0),
        ("Unrestricted intercept", 1)
    ]

    coint_found = False

    for name, det in models:
        print(f"\nTesting model: {name}")
        coint = select_coint_rank(df, det_order=det, k_ar_diff=p)
        print(coint.summary())

        if coint.rank > 0:
            coint_found = True

    if coint_found:
        print("\nMODEL RECOMMENDATION: VECM")
    else:
        print("\nMODEL RECOMMENDATION: VAR in FIRST DIFFERENCES")

else:
    print("\nMODEL RECOMMENDATION: ARDL or TODA-YAMAMOTO VAR")

# =========================================================
# 2. STRUCTURAL BREAK TESTS
# =========================================================

print("\n================ STRUCTURAL BREAK TESTS ================")

cpi_break_dates = []
ppi_break_dates = []

for col in df.columns:

    print(f"\n--- Zivot-Andrews Structural Break Test for {col} ---")
    
    series = df[col].astype(float).dropna()
    
    za = ZivotAndrews(series)
    print(za.summary())

    print(f"\n--- Bai-Perron Style Break Detection for {col} ---")
    
    signal = series.values
    algo = rpt.Pelt(model="rbf").fit(signal)
    breaks = algo.predict(pen=10)

    print("Break indices:", breaks)
    
    # Convert breakpoints to actual dates
    break_dates = [series.index[b-1] for b in breaks if b < len(series)]
    print("Break dates:", break_dates)
    
    # Save the break dates separately for plotting
    if col == 'CPI':
        cpi_break_dates = break_dates
    elif col == 'PPI':
        ppi_break_dates = break_dates

# =========================================================
# Plotting CPI and PPI with their structural breaks
# =========================================================
import matplotlib.pyplot as plt
plt.figure(figsize=(12,6))
plt.plot(df.index, df['CPI'], label='CPI (Inflation Rate)', color='blue', linewidth=2)
plt.plot(df.index, df['PPI'], label='PPI (Producer Price Change)', color='orange', linewidth=2)

# Highlight CPI structural breaks
for bdate in cpi_break_dates:
    plt.axvline(pd.to_datetime(bdate), color='red', linestyle='--', alpha=0.7, label='CPI Structural Break' if bdate==cpi_break_dates[0] else "")

# Highlight PPI structural breaks
for bdate in ppi_break_dates:
    plt.axvline(pd.to_datetime(bdate), color='green', linestyle='-.', alpha=0.7, label='PPI Structural Break' if bdate==ppi_break_dates[0] else "")

plt.title("CPI and PPI Trends in Ghana(2014-2025)", fontsize=14)
plt.xlabel("Date", fontsize=12)
plt.ylabel("Percentage Change", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()


