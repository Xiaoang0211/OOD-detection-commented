"""
stateful and stateless detectors
stateful for VAEs to avoid consistent growth of martingale
stateless for SVDD
"""

class StatefulDetector(object):
    def __init__(self, sigma, tau):
        self.S = 0
        self.sigma = sigma # has to do with false alarm rate
        self.tau = tau # threshold for both VAEs and SVDD
    
    def __call__(self, M):

        #for small p-values, M should show a growing tendence
        #if there were no self.sigmal, the increasement of S might be 
        #too fast and the detection will be more sensitive, which probably 
        #leads to lots of probably false positive detections
        
        self.S = max(0.0, self.S+M-self.sigma) # accumulation of martingale larger than self.sigma
        if self.S > self.tau:
            temp = self.S
            # after the detection of one anomaly, the value of S is again
            # initialized to 0 to be ready for the detection of next 
            # anomaly
            self.S = 0
            return temp, True # anomaly is detected, the S value can be plotted in the GUI
        else:
            return self.S, False # anomaly is not detected, the S value can be plotted in the GUI

# for SVDD, it is much simpler
# if M grows over a threshold value 
# then we assume M indicates an anomaly
class StatelessDetector(object):
    def __init__(self, tau):
        self.tau = tau
    
    def __call__(self, M):
        if M > self.tau:
            return True # anomaly detected
        else:
            return False # anomaly not detected