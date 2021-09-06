import json
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
from scipy.optimize import minimize
import func

class Data:
    # make the age-weight visualation
    def visual_function1(x, y):
        f, ax = plt.subplots()
        FS = 18  # FONT SIZE
        dt = 0.01
        plt.xlabel('age(years)', fontsize=FS)
        plt.ylabel('weight(lb)', fontsize=FS)
        plt.xticks([0, 20, 40, 60, 80, 100], fontsize=FS)
        plt.yticks([25, 50, 75, 100, 125, 150, 175, 200, 225], fontsize=FS)
        plt.plot(x, y, 'bo')
        plt.pause(dt)
        plt.show()

    # make the adult and children-weight visulation
    def visual_function2(y, is_adult):
        f, ax = plt.subplots()
        FS = 18  # FONT SIZE
        dt = 0.01
        plt.xlabel('weight(lb)', fontsize=FS)
        plt.ylabel('ADULT=1 CHILD=0', fontsize=FS)
        plt.xticks([25, 50, 75, 100, 125, 150, 175, 200, 225], fontsize=FS)
        plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=FS)
        plt.plot(y, is_adult, 'bo')
        plt.pause(dt)
        plt.show()

    # make the model visulization
    def visualization(modelx,modely,x,y,model,title):
        # DEFINE X DATA FOR PLOTTING
        N = 1000;
        xmin = 0;
        xmax = 100
        X = np.linspace(xmin, xmax, N)
        plt.figure()  # INITIALIZE FIGURE
        FS = 18  # FONT SIZE
        if(title == 'age'):
            plt.xlabel('age(years)', fontsize=FS)
            plt.ylabel('weight(lb)', fontsize=FS)
        if(title == 'is_adult'):
            plt.xlabel('weight(lb)', fontsize=FS)
            plt.ylabel('ADULT=1 CHILD=0', fontsize=FS)
        plt.plot(np.sort(modelx), modely, '-', label='Model', color='red')
        if(model == 'training'):plt.plot(x, y, 'bo', label='Trainng Set')
        if (model == 'test'): plt.plot(x, y, 'o', label='Test Set')
        plt.legend()
        plt.show()

    #read data from json file and compute it
    file = 'weight.json'
    with open(file) as f:
        json_data = json.load(f)
    xlabel = (json_data['xlabel'])
    ylabel = (json_data['ylabel'])
    is_adult = (json_data['is_adult'])
    x = (json_data['x'])
    y = (json_data['y'])


    visual_function1(x, y)
    visual_function2(y, is_adult)

    # Linear
    #partition data into test set and training set
    trainx, testx, trainy, testy = train_test_split(x, y, test_size=0.2)
    # let age < 18
    max_age = (18. - np.mean(x)) / np.std(x)
    #normolize data
    norx = func.normalize(trainx)
    nory = func.normalize(trainy)
    selectytrain = nory[norx[:] < max_age]
    selectxtrain = norx[norx[:] < max_age]
    # choose parameter
    p=np.random.uniform(5,1.,size=2)
    #optimize parameter
    res = minimize(func.loss, p, method='CG', tol=1e-5,args=(np.array(selectxtrain),np.array(selectytrain),"linear"))
    p = res.x
    #use the function to compute y
    ypred = func.model(np.array(np.sort(selectxtrain)),p,"linear")
    # unormolize ypred
    ypred = np.std(trainy)*ypred+np.mean(trainy)
    #unormolize selectxtrain
    selectxtrain = np.std(trainx)*selectxtrain+np.mean(trainx)
    #visualize model
    visualization(np.sort(selectxtrain),ypred,trainx,trainy,'training','age')
    visualization(np.sort(selectxtrain), ypred, testx, testy,'test','age')


    # Logistic
    #choose parameter
    p1 = np.random.uniform(0.5, 1., size=4)
    #optmize parameter
    res = minimize(func.loss, p1, method='CG', tol=1e-5, args=(np.array(norx), np.array(nory), "logistic"))
    p1 = res.x
    ypred = func.model(np.array(np.sort(norx)), p1, "logistic")
   #unormolize ypred
    ypred = np.std(trainy)*ypred+np.mean(trainy)
    #plt.plot(np.sort(norx), ypred, '-', norx, nory, 'bo')
    visualization(np.sort(trainx), ypred, trainx, trainy, 'training', 'age')
    visualization(np.sort(trainx), ypred, testx, testy, 'test', 'age')


    #logistic adult-children
    #partition data into test set and training set
    trainy, testy, trainis_adult, testis_adult = train_test_split(y, is_adult, test_size=0.2)
    #normolize data
    noris_adult = func.normalize(trainis_adult)
    nory = func.normalize(trainy)
    #choose parameter and optimize it 
    p2 = np.random.uniform(0.5, 1., size=4)
    res = minimize(func.loss, p2, method='Nelder-Mead', tol=1e-5, args=(np.array(nory), np.array(noris_adult), "logistic"))
    p2 = res.x
    is_adultprec = func.model(np.sort(nory), p2, "logistic")
    #unormolize is_adultpre
    is_adultprec = np.std(trainis_adult)*is_adultprec+np.mean(trainis_adult)
    visualization(np.sort(trainy), is_adultprec, trainy, trainis_adult, 'training', 'is_adult')
    visualization(np.sort(trainy), is_adultprec, testy, testis_adult, 'test', 'is_adult')

