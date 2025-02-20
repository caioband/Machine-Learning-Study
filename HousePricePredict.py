import numpy as np
import matplotlib.pyplot as plt


def compute_linear_model_output(x,w,b):
    m = x.shape[0]
    f_wb = np.zeros(m)
    print(f_wb)
    for i in range(m):
        f_wb[i] = w * x[i] + b
        
    return f_wb


x_train = np.array([ 89,  77,  92, 110,  75,  75, 111,  95,  70,  90,  70,  70,  84,
         41,  45,  68,  59,  86,  61,  51, 109,  75,  81,  51,  69,  82,
         56,  87,  67,  74,  67, 117,  79,  58,  96,  55,  84,  40,  53,
         83,  94,  83,  77,  73,  50,  65,  70, 101,  86,  44,  86,  72,
         66,  92, 100,  98,  63,  73,  86,  99,  70,  76,  57,  56,  96,
        107,  78, 100,  87,  67])
y_train = np.array([ 826,  668,  856,  932,  642,  621,  785,  695,  496,  801,  556,
         596,  816,  317,  370,  630,  453,  621,  480,  381, 1067,  706,
         720,  490,  649,  619,  541,  749,  631,  716,  532,  857,  607,
         480,  907,  527,  589,  341,  437,  636,  691,  665,  756,  581,
         427,  592,  566, 1001,  850,  341,  730,  568,  518,  654,  882,
         833,  450,  572,  836,  764,  520,  643,  567,  432,  865,  993,
         601,  918,  704,  596])


tmp_f_wb = compute_linear_model_output(x_train, 8,8)
plt.plot(x_train, tmp_f_wb, c='b',label='Our Prediction')
plt.scatter(x_train, y_train, marker='x', c='r',label='Actual Values')
plt.title("Housing Prices")
plt.ylabel('Price in 1000 BRLs')
plt.xlabel('Size in m^2')
plt.legend()
plt.show()