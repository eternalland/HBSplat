import math

def linear_ramp(n, m, p, q, current_iter):
    if current_iter < n:
        return p
    elif current_iter <= m:
        return p + (q - p) * (current_iter - n) / (m - n)
    else:
        return q



def exponential_ramp(n, m, p, q, current_iter, k=5):
    if current_iter < n:
        return p
    elif current_iter <= m:
        t = (current_iter - n) / (m - n)  # 归一化到 [0, 1]
        return p + (q - p) * (1 - math.exp(-k * t))
    else:
        return q