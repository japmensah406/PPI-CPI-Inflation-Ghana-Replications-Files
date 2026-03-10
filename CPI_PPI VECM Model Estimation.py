# =====================================================
# VECM ESTIMATION: CPI–PPI IN GHANA SCRIPT
# =====================================================

import pandas as pd
import numpy as np
from statsmodels.tsa.vector_ar.vecm import VECM, select_order, select_coint_rank
from statsmodels.stats.diagnostic import het_arch
from scipy import stats
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import ccf

# =====================================================
# 1. LOAD DATA
# =====================================================

file_path = "CPI_PPI_INFLATION.xlsx"
df = pd.read_excel(file_path)

# Convert Date and set as monthly frequency
df['Date'] = pd.to_datetime(df['Date'], format='%b-%y')
df = df.set_index('Date')
df = df.asfreq('MS')  # Monthly Start frequency to fix warnings

# Rename columns to standard CPI / PPI
df = df.rename(columns={
    'Inflation Rate': 'CPI',
    'Producer Price Change': 'PPI'
})

df = df.dropna()
df = df.astype(float)

# =====================================================
# 2. CREATE STRUCTURAL BREAK DUMMIES
# =====================================================

break_dates = ['2017-04-30', '2021-11-01']  # Bai-Perron detected breaks
for i, bdate in enumerate(break_dates, start=1):
    df[f'DB{i}'] = (df.index >= pd.to_datetime(bdate)).astype(int)

print(df.head())

# =====================================================
# 3. LAG ORDER SELECTION
# =====================================================

maxlags = 12 
lag_order = select_order(df[['CPI', 'PPI']], maxlags=maxlags) 
p = lag_order.aic # AIC selected print("\nSelected lag (AIC):", p)

# =====================================================
# 4. JOHANSEN COINTEGRATION TEST
# =====================================================
coint_test = select_coint_rank(df[['CPI','PPI']], det_order=0, k_ar_diff=p)

print("\nJohansen Cointegration Test Summary:")
print(coint_test.summary())

# Extract integer rank AFTER printing
coint_rank = coint_test.rank
print("\nSelected Cointegration Rank:", coint_rank)

# =====================================================
# 5. VECM ESTIMATION
# =====================================================

deterministic = 'co'  # constant in cointegration relation

vecm_model = VECM(
    endog=df[['CPI', 'PPI']],
    k_ar_diff=p,
    coint_rank=coint_rank,
    deterministic=deterministic,
    exog=df[['DB1', 'DB2']]  # structural break dummies
)

vecm_res = vecm_model.fit()
print("\nVECM Estimation Summary:")
print(vecm_res.summary())

# =====================================================
# 6. DIAGNOSTICS
# =====================================================

from statsmodels.stats.diagnostic import het_arch, acorr_lm
from scipy import stats

# Align residuals index
resid_index = df.index[-vecm_res.resid.shape[0]:]
resid = pd.DataFrame(vecm_res.resid, columns=['CPI_resid', 'PPI_resid'], index=resid_index)

# Prepare results list
diagnostics = []

for col in resid.columns:
    # 1. Normality test (Jarque-Bera)
    jb_stat, jb_p = stats.jarque_bera(resid[col])
    
    # 2. Heteroskedasticity test (ARCH)
    arch_stat, arch_p, _, _ = het_arch(resid[col])
    
    # 3. Autocorrelation (LM test)
    lm_stat, lm_pvalue, _, _ = acorr_lm(resid[col], nlags=vecm_res.k_ar)
    
    diagnostics.append({
        "Residual": col,
        "JB Stat": round(jb_stat, 3),
        "JB p-value": round(jb_p, 3),
        "ARCH Stat": round(arch_stat, 3),
        "ARCH p-value": round(arch_p, 3),
        "LM Stat": round(lm_stat, 3),
        "LM p-value": round(lm_pvalue, 3)
    })

# Convert to DataFrame for a clean table
diag_table = pd.DataFrame(diagnostics)
print(diag_table)

# 6c. Stability: Check eigenvalues
# =====================================================
# VECM stability check using companion matrix
# =====================================================
var_rep = vecm_res.var_rep  # Companion matrix
eigvals = np.linalg.eigvals(var_rep)
moduli = np.abs(eigvals)

print("\nEigenvalues of VECM (companion matrix):")
print(eigvals)
print("\nModulus of eigenvalues:")
print(moduli)

# Check stability
n_endog = df[['CPI', 'PPI']].shape[1]
c_rank = vecm_res.k_ar  # number of cointegration relations?
n_unit_roots_expected = n_endog - vecm_res.coint_rank
n_unit_roots_actual = np.sum(np.isclose(moduli, 1.0))

print(f"\nExpected number of unit roots: {n_unit_roots_expected}")
print(f"Actual number of unit roots: {n_unit_roots_actual}")

if np.all(moduli <= 1) and n_unit_roots_actual == n_unit_roots_expected:
    print("System is stable")
else:
    print("Warning: System may be unstable")

# =====================================================
# 7. IMPULSE RESPONSE FUNCTIONS (IRFs)
# =====================================================

irf_periods = 12  # months
irf = vecm_res.irf(irf_periods)

# Plot IRFs
irf.plot(orth=False)
plt.suptitle("VECM Impulse Response Functions")
plt.show()

# =====================================================
# 8. CROSS-CORRELATION FUNCTION (Optional)
# =====================================================

ccf_vals = ccf(df['PPI'].diff().dropna(), df['CPI'].diff().dropna())
plt.stem(range(len(ccf_vals)), ccf_vals)
plt.xlabel("Lag")
plt.ylabel("CCF")
plt.title("Cross-Correlation Function (PPI → CPI)")
plt.show()

# =====================================================
# 9. LONG-RUN COEFFICIENTS
# =====================================================

print("\nLong-run adjustment coefficients (alpha):")
print(vecm_res.alpha)

print("\nCointegration vector(s) (beta):")
print(vecm_res.beta)
from statsmodels.tsa.vector_ar.vecm import VECM

# Fit the VECM if not already
# vecm_res = vecm_model.fit()

# Granger causality tests (short-run + error correction)
print("\nGranger causality tests (short-run + error correction):")
gc_cpi = vecm_res.test_granger_causality(causing='PPI', caused='CPI')
gc_ppi = vecm_res.test_granger_causality(causing='CPI', caused='PPI')

print("PPI → CPI:")
print(gc_cpi.summary())
print("\nCPI → PPI:")
print(gc_ppi.summary())

# Impulse Response Functions (IRFs)
irf_periods = 12  # months
irf = vecm_res.irf(irf_periods)

# Orthogonalized IRFs (optional)
irf.plot(orth=True)
plt.suptitle("Orthogonalized IRFs")
plt.show()

# Forecast Error Variance Decomposition (FEVD) via VAR representation
from statsmodels.tsa.vector_ar.var_model import VAR

# Compute IRFs
irf_periods = 12
irf = vecm_res.irf(irf_periods)

# You can compute FEVD manually from IRFs
# IRFs shape: (periods, n_endog, n_shocks)
irf_array = irf.irfs
n_endog = irf_array.shape[1]
fevd_array = np.zeros((irf_periods, n_endog, n_endog))

for h in range(irf_periods):
    for i in range(n_endog):
        fevd_array[h, i, :] = (irf_array[:h+1, i, :] ** 2).sum(axis=0)
        fevd_array[h, i, :] /= fevd_array[h, i, :].sum()  # normalize to % contribution

print("FEVD array (horizon x variables x shocks):\n", fevd_array)

# Bootstrap IRFs for robust inference
irf = vecm_res.irf(irf_periods)
irf.plot(orth=True)  # orthogonalized
plt.suptitle("Orthogonalized IRFs (point estimates)")
plt.show()
# =====================================================
# TIME SERIES TREND PLOT (CPI & PPI)
# =====================================================

plt.figure(figsize=(12,6))
plt.plot(df.index, df['CPI'], label='CPI (Inflation Rate)', color='blue', linewidth=2)
plt.plot(df.index, df['PPI'], label='PPI (Producer Price Change)', color='orange', linewidth=2)

# Highlight structural breaks
for bdate in break_dates:
    plt.axvline(pd.to_datetime(bdate), color='red', linestyle='--', alpha=0.7, label='Structural Break' if bdate==break_dates[0] else "")

plt.title("CPI and PPI Trends over Time (Ghana)", fontsize=14)
plt.xlabel("Date", fontsize=12)
plt.ylabel("Percentage Change", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()

# =====================================================
# Stability Check: Alternative Deterministic Structures
# =====================================================

from statsmodels.tsa.vector_ar.vecm import select_order, VECM
import numpy as np
import pandas as pd

det_options = ['co', 'ci', 'ct']

print("\n=== Stability Check Across Deterministic Specifications ===")

for det in det_options:
    print(f"\nTesting deterministic specification: {det}")
    
    # Re-select lag for this deterministic option
    lag_order = select_order(df[['CPI','PPI']], maxlags=12, deterministic=det)
    selected_lag = lag_order.aic  # or choose lag_order.bic / lag_order.hqic if desired
    
    test_model = VECM(
        endog=df[['CPI','PPI']],
        k_ar_diff=selected_lag,
        coint_rank=coint_rank,   # integer rank from previous step
        deterministic=det,
        exog=df[['DB1','DB2']]
    )
    
    test_res = test_model.fit()
    
    eig = np.linalg.eigvals(test_res.var_rep)
    max_mod = np.max(np.abs(eig))
    
    print(f"Selected Lag (AIC): {selected_lag}")
    print("Max modulus:", max_mod)
    if max_mod <= 1:
        print("System is stable")
    else:
        print("Warning: System may be unstable")


# =====================================================
# Cointegration Rank Sensitivity Analysis
# =====================================================

candidate_ranks = [coint_rank-1, coint_rank, coint_rank+1]
candidate_ranks = [r for r in candidate_ranks if r > 0]  # only positive ranks

rank_results = {}

for r in candidate_ranks:
    print(f"\nEstimating VECM with Cointegration Rank = {r}")
    
    # Re-select lag (AIC) for consistency
    lag_order = select_order(df[['CPI','PPI']], maxlags=12, deterministic="ci")
    selected_lag = lag_order.aic
    
    vecm_temp = VECM(
        endog=df[['CPI','PPI']],
        k_ar_diff=selected_lag,
        coint_rank=r,
        deterministic="ci"
    ).fit()
    
    rank_results[r] = {
        "LogLik": vecm_temp.llf,
        "AIC": lag_order.aic,
        "BIC": lag_order.bic,
        "HQIC": lag_order.hqic
    }

print("\nRank Sensitivity Comparison:")
pd.DataFrame(rank_results).T


# =====================================================
# Lag Sensitivity Stability Test
# =====================================================

test_lags = [p-2, p, p+2]
test_lags = [lag for lag in test_lags if lag > 0]

for lag in test_lags:
    print(f"\nTesting Stability with Lag = {lag}")
    
    vecm_test = VECM(
        df[['CPI','PPI']],
        k_ar_diff=lag,
        coint_rank=coint_rank,
        deterministic="ci"
    ).fit()
    
    var_rep = vecm_test.var_rep
    eigvals = np.linalg.eigvals(var_rep)
    moduli = np.abs(eigvals)
    
    print("Max Eigenvalue Modulus:", np.max(moduli))
    if np.max(moduli) <= 1:
        print("Stable")
    else:
        print("Warning: Unstable")


# =====================================================
# Dynamic Stability Check (No Structural Adjustments)
# =====================================================

var_rep = vecm_res.var_rep  # your main fitted VECM
eigvals = np.linalg.eigvals(var_rep)
moduli = np.abs(eigvals)

print("\nEigenvalue Moduli:")
print(moduli)

if np.all(moduli <= 1):
    print("\nModel is dynamically stable")
else:
    print("\nWarning: Dynamic instability detected")

print("\nRoots Greater Than One:")
print(moduli[moduli > 1])

from statsmodels.tsa.vector_ar.vecm import VECM

# Fit VECM with coint_rank=1
vecm_model = VECM(df[['CPI','PPI']], k_ar_diff=p, coint_rank=1, deterministic="ci")
vecm_res = vecm_model.fit()

# Get cointegration residuals
beta = vecm_res.beta  # cointegration vector
resid = df[['CPI','PPI']].values @ beta  # long-run equilibrium residuals
df['EC_resid'] = resid[:,0]  # use the first cointegration relation

import statsmodels.api as sm

# Create lagged differences
df['dCPI'] = df['CPI'].diff()
df['dPPI'] = df['PPI'].diff()
df['EC_lag'] = df['EC_resid'].shift(1)

# Drop NaNs
df_ecm = df.dropna()

# Define regression
X = sm.add_constant(df_ecm[['dPPI', 'EC_lag']])
y = df_ecm['dCPI']

ecm_model = sm.OLS(y, X).fit()
print(ecm_model.summary())

from statsmodels.tsa.ardl import ARDL

# Ensure exog is a DataFrame
exog = df[['PPI']]

# Dependent variable lags = 4, exog lags = 4
ardl_model = ARDL(df['CPI'], lags=4, exog=exog, order=4)
ardl_res = ardl_model.fit()
print(ardl_res.summary())

import pandas as pd



