#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 11 00:34:33 2018

@author: Amandeep
"""

from flask import Flask, request, jsonify
from PIL import Image
from base64 import decodestring
from io import BytesIO
import base64
import matplotlib.pyplot as plt
#from PIL import Image
import sklearn as sk
#import cv2
import pandas as pd
import numpy as np
app = Flask(__name__)


# root
@app.route("/")
def index():
    """
    this is a root dir of my server
    :return: str
    """
    return "This is root!!!!"


# GET
@app.route('/users/<user>')
def hello_user(user):
    """
    this serves as a demo purpose
    :param user:
    :return: str
    """
    return "Hello %s!" % user


# POST
@app.route('/api/post_some_data', methods=['POST'])
def get_text_prediction():
    """
    predicts requested text whether it is ham or spam
    :return: json
    """
    json = request.get_json()
    imgdata = base64.b64decode(json['text'])
    filename = 'some_image.jpg'  # I assume you have a way of picking unique filenames
    with open(filename, 'wb') as f:
        f.write(imgdata)
    df=pd.read_csv('project.csv')
    A=df.iloc[:,:].values
    #B=sk.preprocessing.normalize(A, norm='l2', axis=1, copy=True, return_norm=False)
    #X=np.copy(A)
    #X[:,:]=(A[:,:]-A[:,:].mean())/A[:,:].std()
    inp=Image.open('some_image.jpg').convert('LA')
    #plt.imshow(inp)
    #plt.show()
    #size=25,25
    #img1=inp.thumbnail(size, Image.ANTIALIAS)
    iimg=inp.resize((25,25))
    #plt.imshow(iimg)
    imge=iimg.load()
    #plt.imshow(imge)
    l=0
    arr=np.zeros((625,1)) 
    for h in range(0,25):
        for m in range(0,25):
            x,y=imge[h,m]
            arr[l]=x
            l=l+1
    arrt=np.transpose(arr)
    #C=sk.preprocessing.normalize(arrt, norm='l2', axis=1, copy=True, return_norm=False)
    #X1=np.copy(arrt)
    #X1[:,:]=(arrt[:,:]-arrt[:,:].mean())/arrt[:,:].std()
    #print(arrt)
    #%%
    list=['A','B','C','D','E','F','G','H','I','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y']
    Y=np.zeros((720))
    d1=0
    d2=31
    for i in range(24):
        Y[d1:d2]=i+1
        d1=d1+30
        d2=d2+30
    
    #%%
    #for i in range(24):
    #    #print(i)
    #    k=(i)*29
    #    #k=i*500+498
    #    #k=i*28
    #    print(k)
    #    I=X[k,:]
    #    Class=Y[k]
    #    print (Class)
    #    II=np.reshape(I,[25,25])
    #    plt.subplot(4,6,i+1)
    #    plt.imshow(II.T)
    #    plt.title('letter-'+str(Class))
    #    plt.show
        
    #%%
    from sklearn.model_selection import train_test_split
    ##from sklearn.metrics import classification_report
    from sklearn.svm import SVC
    #from sklearn.model_selection import GridSearchCV
    X_train,X_test,Y_train,Y_test=train_test_split(A,Y,test_size=0.2,random_state=0)
    svm=SVC(kernel='linear',C=1.0,random_state=0)
    #parameters={'C':(100,1e3,1e4,1e5),'gamma':(1e-08,1e-7,1e-6,1e-5)}
    #grid_search=GridSearchCV(svc,parameters,n_jobs=-1,cv=3)
        #start_time=timeit.default_timer()
    svm.fit(X_train,Y_train)
        #print("___%0.3fs seconds ___" %(timeit.default_timer()-start_time))
        
    ##accuracy=svm.score(X_test,Y_test)
    ##print('The accuracy on testing set is: {0:.1f}%' .format(accuracy*100))
    prediction=svm.predict(arrt)
    value=int(prediction)
    RESULT=list[value-1]
    ##report=classification_report(Y_test,prediction)
    ##print(report)
    print('Alphabet (by SVM)= ',RESULT)
    #%%
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import train_test_split
    ##from sklearn.metrics import classification_report
    
    #from sklearn.model_selection import GridSearchCV
    X_train,X_test,Y_train,Y_test=train_test_split(A,Y,test_size=0.2,random_state=21)
    #MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto',
    #       beta_1=0.9, beta_2=0.999, early_stopping=False,
    #       epsilon=1e-08, hidden_layer_sizes=(100), learning_rate='constant',
    #       learning_rate_init=0.001, max_iter=200, momentum=0.9,
    #       nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
    #       solver='lbfgs', tol=0.0001, validation_fraction=0.1, verbose=False,
    #       warm_start=False)
    clf=MLPClassifier(solver='lbfgs',alpha=1e-5,hidden_layer_sizes=(500),random_state=1)
    clf.fit(X_train, Y_train)
    prediction=clf.predict(arrt)
    value=int(prediction)
    RESULT+=list[value-1]
    print('Alphabet (by MLP)= ',RESULT)
    return jsonify(str(RESULT))
    #print('Success')
    ##report=classification_report(Y_test,prediction)
    ##print(report)
    ##accuracy=clf.score(X_test,Y_test)
    ##print(accuracy*100)


# running web app in local machine
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)