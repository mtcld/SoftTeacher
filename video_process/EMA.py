import pandas as pd

class EMA:
    def __init__(self,values=[],span=2):
        self.df = pd.DataFrame({'values':values})
        self.default_span = span
        self.current_span = span
        self.activate_new_span = False
        self.gap = 90
        self.ema_list = self.df.ewm(span=self.current_span).mean()['values'].tolist()

    def add(self,value):
        value = int(value)
        if len(self.df) == 0:
            self.df.loc[len(self.df)] = [value]
            self.ema_list.append(value)
            return value

        if abs(value - self.df.loc[len(self.df)-1]['values']) > self.gap:
            self.current_span = 0
            self.activate_new_span = True

        self.df.loc[len(self.df)] = [value]

        if self.current_span < self.default_span :
            if self.activate_new_span is True and self.current_span == 1 :
                self.activate_new_span = False
            else:
                self.current_span += 1

        ema_value = self.df.ewm(span=self.current_span).mean()['values'].tolist()[-1]
        self.ema_list.append(ema_value)
        
        return ema_value 
    
    def get_origin_values(self):
        return self.df['values'].tolist()
    
    def get_ema_values(self):
        return self.ema_list
