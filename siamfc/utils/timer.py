import time

class Timer:
    #Timer class to measure elapsed time
    def __init__(self, convert=False):
        self.start_time = None
        self.end_time = None
        self.elapsed = None
        self.convert = convert
        
    def reset(self):
        #Reset the timer
        self.start_time = time.time()
        self.end_time = None
        self.elapsed = None
        
    def stop(self):
        #Stop the timer and calculate total elapsed time
        self.end_time = time.time()
        self.elapsed = self.end_time - self.start_time
        
        if self.convert:
            self.elapsed = self._format_time(self.elapsed)
            
        return self.elapsed
    
    def _format_time(self, seconds):
        #format to Hours:Minutes:Seconds
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        
        if h > 0:
            return f"{h:.0f}h {m:.0f}m {s:.1f}s"
        elif m > 0:
            return f"{m:.0f}m {s:.1f}s"
        else:
            return f"{s:.3f}s"