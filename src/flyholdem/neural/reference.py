"""Independent pure-Python scalar reference for the declared .1ms LIF model.

Analytic subthreshold solution: tau_v=20ms, tau_g=5ms, reset/rest=-52,
strict threshold >-45, delay 1.8ms, refractory 2.2ms. At each tick decrement
refractory, integrate/threshold, deliver delayed synapses, then reset new spikes.
Both v and g are frozen and synaptic arrivals discarded while refractory.
"""
import math


class ReferenceBrain:
    def __init__(self,ptr,post,weights):
        self.ptr=list(map(int,ptr));self.post=list(map(int,post));self.weights=list(map(float,weights))
        self.n=len(ptr)-1
        self.v=[-52.0]*self.n;self.g=[0.0]*self.n;self.refractory=[0]*self.n
        self.queue=[[] for _ in range(19)];self.cursor=0

    def advance(self,drive,duration_ms):
        ticks=round(duration_ms*10)
        if ticks<1 or not math.isclose(ticks/10,duration_ms,abs_tol=1e-9):raise ValueError('Duration must be positive .1ms ticks')
        if len(drive)!=self.n or not all(math.isfinite(x) for x in drive):raise ValueError('Invalid drive')
        av=math.exp(-.1/20);ag=math.exp(-.1/5);coupling=(av-ag)/3
        counts=[0]*self.n
        for _ in range(ticks):
            slot=self.cursor%19;future=(self.cursor+18)%19
            fired=[]
            for i in range(self.n):
                if self.refractory[i]>0:self.refractory[i]-=1
                if self.refractory[i]==0:
                    self.v[i]=-52+(self.v[i]+52)*av+float(drive[i])*(1-av)+self.g[i]*coupling
                    self.g[i]*=ag
                    if self.v[i]>-45:
                        counts[i]+=1;fired.append(i)
            for i in self.queue[slot]:
                for edge in range(self.ptr[i],self.ptr[i+1]):
                    j=self.post[edge]
                    if self.refractory[j]==0:self.g[j]+=self.weights[edge]
            self.queue[slot]=[]
            self.queue[future].extend(fired)
            for i in fired:self.v[i]=-52.0;self.g[i]=0.0;self.refractory[i]=22
            self.cursor+=1
        return counts
