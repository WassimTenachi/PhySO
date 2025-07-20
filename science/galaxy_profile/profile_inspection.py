import time

# Local imports
from science.galaxy_profile import analytic_properties as ap


profiles = {
    'NFW': "rho0 / ((r / Rs) * (1 + r / Rs)**2)",
    'pISO': "rho0 / (1 + (r / Rs)**2)",
    'pISO1': "1 / (1 + (r / 1)**2)",
    'Lucky13'      : "rho0 / (1 + (r / Rs))**3",
    'Burkert'      : "rho0 / ((1 + (r / Rs))*(1 + (r / Rs)**2))",
    'superNFW'     : "rho0 / ((r / Rs) * (1 + r / Rs)**Rational(5, 2))",
    'Exponential'  : "rho0*exp(-r/(Rs))",
    'Exponential1' : "9.6*exp(-r/(1.4))",
    'Exponential2' : "rho0*exp(-r/(Rs_1+Rs_2))",
}

free_consts_vals = {
    'rho0': 1.0,  # Example value for rho0
    'Rs'  : 1.0,  # Example value for Rs
    'Rs_1': 1.0,  # Example value for Rs_1
    'Rs_2': 1.0,  # Example value for Rs_2
}

free_consts_names = ["Rs", "rho0", "Rs_1", "Rs_2"]

t0 = time.perf_counter()
for name, density_profile_str in profiles.items():
    print(name, density_profile_str)
    results_df = ap.check_analytical_properties(density_profile_str = density_profile_str,
                                                free_consts_names   = free_consts_names,
                                                free_consts_vals    = free_consts_vals,
                                                num_check = False,
                                                verbose   = False,
                                 )
    print(results_df[['Property', "symb_condition", "num_condition"]])
t1 = time.perf_counter()
print(f"\nAnalytical properties check completed in {t1 - t0:.2f} seconds.\n")

