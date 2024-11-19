import logging

class SimulationError(Exception):
    def __init__(self, message, error_code):
        self.message = message
        self.error_code = error_code
        super().__init__(self.message)

def handle_simulation_error(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.error(f"Simulation error: {str(e)}")
            raise SimulationError(str(e), 500)
    return wrapper 