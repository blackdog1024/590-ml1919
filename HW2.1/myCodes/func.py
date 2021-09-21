import numpy as np
from sklearn.model_selection import train_test_split
def normalize(x):
    xmean = np.mean(x)
    xstd = np.std(x, ddof=1)
    x = (x - xmean) / xstd
    return x

def model(x, p,model_type):
    if (model_type == "linear"): return p[0] * x + p[1]
    if (model_type == "logistic"): return p[0] + p[1] * (1.0 / (1.0 + np.exp(-(x - p[2]) / (p[3] + 0.00001))))

#def minimize(self,loss,p,algo,LR,method):

def optimizer(loss,p, algo, LR, method,xt,xv,yt,yv):
    # PARAM
    NDIM = 4
    pi = p
    pj = p
    if (NDIM == 2): pi = np.array([-2, -2])
    dp = 0.001  # STEP SIZE FOR FINITE DIFFERENCE
    LR = 0.001  # LEARNING RATE
    EDF = 0.000001  # EXPONENTIAL DECAY FACTO
    t = 0  # INITIAL ITERATION COUNTER
    tmax = 100000  # MAX NUMBER OF ITERATION
    tol = 10 ** -30  # EXIT AFTER CHANGE IN F IS LESS THAN THIS


    if ( method =='batch'):

        while (t <= tmax):
            t = t + 1

            # NUMERICALLY COMPUTE GRADIENT
            df_dp = np.zeros(NDIM)
            for i in range(0, NDIM):
                dP = np.zeros(NDIM);
                dP[i] = dp;
                pm1 = pi - dP;  # print(xi,xm1,dX,dX.shape,xi.shape)
                df_dp[i] = (loss(pi,xt,xv,yt,yv) - loss(pm1,xt,xv,yt,yv)) / dp
            # print(xi.shape,df_dx.shape)
            if(algo == 'GD'):xip1 = pi - LR * df_dp  # STEP
            if (algo == 'mom'):
                if (t == 1):
                    xip1 = pi - LR * df_dp
                else:
                    xip1 = pi - LR * df_dp - EDF * pj  # STEP
            if (t % 10 == 0):
                df = np.mean(np.absolute(loss(xip1,xt,xv,yt,yv) - loss(pi,xt,xv,yt,yv)))
                print(t, "	", pi, "	", "	", loss(pi,xt,xv,yt,yv), df)

                if (df < tol):
                    print("STOPPING CRITERION MET (STOPPING TRAINING)")
                    break

            # UPDATE FOR NEXT ITERATION OF LOOP
            if (algo == 'mom'): pj = pi
            pi = xip1

        return pi




    if( method == 'mini-batch'):
        xt1, xt2, yt1, yt2= train_test_split(xt, yt, test_size=0.5)
        xv1, xv2, yv1, yv2= train_test_split(xv, yv, test_size=0.5)
        while (t <= tmax):
            t = t + 1

            # NUMERICALLY COMPUTE GRADIENT
            df_dp = np.zeros(NDIM)
            for i in range(0, NDIM):
                dP = np.zeros(NDIM);
                dP[i] = dp;
                pm1 = pi - dP;  # print(xi,xm1,dX,dX.shape,xi.shape)
                if(t%2 == 0):df_dp[i] = (loss(pi,xt1,xv1,yt1,yv1) - loss(pm1,xt1,xv1,yt1,yv1)) / dp
                else:df_dp[i] = (loss(pi,xt2,xv2,yt2,yv2) - loss(pm1,xt2,xv2,yt2,yv2)) / dp
            # print(xi.shape,df_dx.shape)
            if(algo == 'GD'):xip1 = pi - LR * df_dp  # STEP
            if(algo =='mom'):
                if (t == 1):
                    xip1 = pi - LR * df_dp
                else:
                    xip1 = pi - LR * df_dp - EDF * pj  # STEP
            if (t % 10 == 0):
                df = np.mean(np.absolute(loss(xip1,xt1,xv1,yt1,yv1) - loss(pi,xt1,xv1,yt1,yv1)))
                print(t, "	", pi, "	", "	", loss(pi,xt1,xv1,yt1,yv1), df)

                if (df < tol):
                    print("STOPPING CRITERION MET (STOPPING TRAINING)")
                    break

            # UPDATE FOR NEXT ITERATION OF LOOP
            if(algo =='mom'):pj = pi
            pi = xip1

        return pi



def sto_optimizer(losst,lossv, p, algo, LR, method, xt, xv, yt, yv):
    # PARAM
    NDIM = 4
    pi = p
    pj = p
    if (NDIM == 2): pi = np.array([-2, -2])
    dp = 0.001  # STEP SIZE FOR FINITE DIFFERENCE
    LR = 0.001  # LEARNING RATE
    EDF = 0.000001  # EXPONENTIAL DECAY FACTO
    t = 0  # INITIAL ITERATION COUNTER
    tmax = 1000000  # MAX NUMBER OF ITERATION
    # tmax = 30000  # MAX NUMBER OF ITERATION
    tol = 10 ** -30  # EXIT AFTER CHANGE IN F IS LESS THAN THIS

    if (method == 'stochastic'):
        n = 0
        while (t <= tmax):
            t = t + 1
            if(n < len(xt)-1):n = n + 1
            else:n = 0
            # NUMERICALLY COMPUTE GRADIENT
            df_dp = np.zeros(NDIM)
            for i in range(0, NDIM):
                dP = np.zeros(NDIM);
                dP[i] = dp;
                pm1 = pi - dP;  # print(xi,xm1,dX,dX.shape,xi.shape)
                #print('xtn:',xt[n])
                df_dp[i] = (losst(pi, xt[n], yt[n]) - losst(pm1, xt[n], yt[n])) / dp
                lossv(pi, xt[n], yt[n])
                lossv(pm1, xt[n], yt[n])
            # print(xi.shape,df_dx.shape)
            if(algo == 'GD'):xip1 = pi - LR * df_dp  # STEP
            if(algo == 'mom'):
                if (t == 1):
                    xip1 = pi - LR * df_dp
                else:
                    xip1 = pi - LR * df_dp - EDF * pj  # STEP
            if (t % 10 == 0):
                df = np.mean(np.absolute(losst(xip1, xt[n], yt[n]) - losst(pi, xt[n], yt[n])))
                lossv(xip1, xt[n], yt[n])
                lossv(pi, xt[n], yt[n])
                print(t, "	", pi, "	", "	", losst(pi, xt[n], yt[n]), df)

                if (df < tol):
                    print("STOPPING CRITERION MET (STOPPING TRAINING)")
                    break

            # UPDATE FOR NEXT ITERATION OF LOOP
            if(method == 'mom'):pj = pi
            pi = xip1

        return pi






