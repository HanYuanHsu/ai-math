
try:
    from sympy import *

    from sympy import factorial
    
    def seating_arrangements():
        # Total number of ways to arrange the 7 children without any restrictions
        total = factorial(7)
    
        # Number of ways to arrange the 7 children such that no two boys are next to each other
        no_boys_together = factorial(4) / factorial(2) * factorial(3)
    
        # Number of ways to arrange the 7 children such that at least two boys are next to each other
        at_least_two_boys_together = total - no_boys_together
    
        return at_least_two_boys_together
    
    result = seating_arrangements()
    print(result)
except Exception as e:
    print(e)
    print('FAIL')
