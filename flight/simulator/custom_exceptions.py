# Add the flight directory to the Python path (when running from FlightSwordLite)
import sys, os
sys.path.append(os.getcwd())

class InvalidStateException(Exception):
    pass

class OutOfWorldException(InvalidStateException):
    pass

# For optimisation
class RatesCompException(Exception):
    pass