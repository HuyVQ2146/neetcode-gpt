class Solution:
    def get_minimizer(self, iterations: int, learning_rate: float, init: int) -> float:
        # Objective function: f(x) = x^2
        # Derivative:         f'(x) = 2x
        # Update rule:        x = x - learning_rate * f'(x)
        # Round final answer to 5 decimal places
        
        # Fx = init^2
        for i in range(iterations):
            fx = init*2
            init = init - learning_rate * fx
        return round(init,5) 

