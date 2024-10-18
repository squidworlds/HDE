# standard setup
import sympy as sym
sym.init_printing()
from IPython.display import display_latex
import sympy.plotting as sym_plot
from sympy import *

# Define symbolically
t = sym.symbols('t')
y = sym.Function('y')

# Define ODE in sympy
eq = sym.Eq(y(t).diff(t), 4*t-3*y(t))

# Solve using dsolve and display in latex
eqsol = sym.dsolve(eq, y(t), ics={y(0):1})
print('The solution to the differential equation:')
display_latex(sym.simplify(eqsol))

# Find and display y(0.5)
y1 = eqsol.subs(t, 0.5)
print('The solution at y(0.5) is:')
display_latex(sym.simplify(y1))

# Find minimum of y
ymin = (minimum(eqsol.rhs, t, Interval(0,5))).evalf()
print('The minimum value of y is:')
display_latex(sym.simplify(ymin))

# Find corresponding tmin
new_func = eqsol.rhs - ymin
tmin = (solve(new_func, t)[0]).evalf()
print('at t =')
display_latex(sym.simplify(tmin))

# Create dataframe with numerical solutions for y(0.5)
df1 = df.filter(items=[0.5], axis=0)

# Redefine to evaluate ymin numerically
y_euler, times = ode_Euler(dy_dt, 0, 0.393, 10, 1)
y_AB2, times = ode_AB2(dy_dt, 0, 0.393, 10, 1)
y_AB3, times = ode_AB3(dy_dt, 0, 0.393, 10, 1)

# Define the three dataframes of solutions using methods
df_e = DataFrame(data = y_euler, index = times, columns = ["Euler"])
df_AB2 =  DataFrame(data = y_AB2, index = times, columns = ["AB2"])
df_AB3 = DataFrame(data = y_AB3, index = times, columns = ["AB3"])

# Join the three dataframes together
df2 = df_e.join(df_AB2, how='left')
df2 = df2.join(df_AB3, how='left')
df2 = df2.filter(items=[0.393], axis=0)
pd.concat([df1, df2], axis=0)

df_y = DataFrame(data = [y1.rhs, ymin], index = [0.5, tmin], columns = ["y"])
df1.join(df_y, how='left')
