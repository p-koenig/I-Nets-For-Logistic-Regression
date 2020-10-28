# ----------- Imports -----------

from sympy.polys.polytools import Poly
from sympy.polys import monomials

from functools import reduce
import pandas as pd

# ------- Static variables ------

# ----- Function declaration ----

def build_Poly(coefficients, monomials_, polys_symbols):
    
    if(isinstance(monomials_[0], monomials.Monomial)):
        # transform to tuples
        monomials_ = [monomial_to_power_tuple(mon, polys_symbols) for mon in monomials_]
        
    if(isinstance(monomials_[0], tuple)):
        return Poly.from_dict(dict(zip(monomials_, coefficients)), polys_symbols)     
    else:
        raise Exception("Unknown input format of monomials.")
        
def monomial_to_power_tuple(polys_monomial, polys_symbols):
    
    mon_dic = polys_monomial.as_powers_dict()
    tmp = []
    for sym in polys_symbols:
        if sym in mon_dic:
            tmp.append(mon_dic[sym])
        else:
            tmp.append(0)
    return tuple(tmp)


# - deprecated

def calcualate_function_with_data(coefficient_list, variable_values):
    
    result = 0    
    for coefficient_value, coefficient_multipliers in zip(coefficient_list, list_of_monomial_identifiers):
        partial_results = [variable_value**int(coefficient_multiplier) for coefficient_multiplier, variable_value in zip(coefficient_multipliers, variable_values)]
        
        result += coefficient_value * reduce(lambda x, y: x*y, partial_results)

    return result, variable_values
 
def calculate_function_values_from_polynomial(true_value_test, evaluation_dataset):

    #print('method_call')

    if isinstance(true_value_test, pd.DataFrame):
        true_value_test = true_value_test.values
        
    true_value_fv = []
    true_value_coeff = []
    
    #print('start_loop')
    
    for evaluation in evaluation_dataset:
        true_function_value, true_coeff = calcualate_function_with_data(true_value_test, evaluation)
       
        true_value_fv.append(true_function_value) 
        true_value_coeff.append(true_coeff)


    #print('end_loop')
        
    return [true_value_test, pd.DataFrame(np.array(true_value_coeff))], [true_value_test, pd.DataFrame(np.array(true_value_fv))]
