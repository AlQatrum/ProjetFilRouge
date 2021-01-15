# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 11:17:20 2021

@author: Hello
"""

#bibliothèques utiles
import math
import torch
import pandas as pd

import dlc_practical_prologue as prologue


#definition des fonctions utiles
######################################################################

def sigma(x):
    return x.tanh()

def dsigma(x):
    return 4 * (x.exp() + x.mul(-1).exp()).pow(-2)
    #1-x.tanh().pow(2) (pareil tanh=(e^x-e^-x)/(e^x+e^-x))


def loss(v, t):
    return (v - t).pow(2).sum()

def dloss(v, t):
    return 2 * (v - t)

######################################################################



#definition des passes avant et arrière
######################################################################
def forward_pass(w1, b1, w2, b2, w3 , b3, x):
    x0 = x
    s1 = w1.mv(x0) + b1
    x1 = sigma(s1)
    s2 = w2.mv(x1) + b2
    x2 = sigma(s2)
    s3 = w2.mv(x2) + b2
    x3 = sigma(s3)
    
    return x0, s1, x1, s2, x2, s3, x3

def backward_pass(w1, b1, w2, b2, w3, b3,
                  t,
                  x, s1, x1, s2, x2, s3, x3,
                  dl_dw1, dl_db1, dl_dw2, dl_db2, dl_dw3, dl_db3):
    x0 = x
    dl_dx3 = dloss(x2, t)
    dl_ds3 = dsigma(s3) * dl_dx3
    dl_dx2 = w3.t().mv(dl_ds3)
    dl_ds2 = dsigma(s2) * dl_dx2
    dl_dx1 = w2.t().mv(dl_ds2)
    dl_ds1 = dsigma(s1) * dl_dx1 
    
    dl_dw3.add_(dl_ds3.view(-1, 1).mm(x2.view(1, -1)))
    dl_db3.add_(dl_ds3)
    dl_dw2.add_(dl_ds2.view(-1, 1).mm(x1.view(1, -1)))
    dl_db2.add_(dl_ds2)
    dl_dw1.add_(dl_ds1.view(-1, 1).mm(x0.view(1, -1)))
    dl_db1.add_(dl_ds1)

######################################################################

#train_input, train_target, test_input, test_target = prologue.load_data(one_hot_labels = True,
#                                                                        normalize = True)

#importation des données
df=pd.read_csv('C:\\Users\\Hello\\Desktop\\IODAA\\fil_rouge\\to_aire.csv', sep=',', header=0 )
X=df.drop(columns=['age',"Unnamed: 0"])
Y=df["age"]


#fabrication des ensemble de test
from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(X, Y, test_size = 0.5)


nb_classes = train_target.size(1)
nb_train_samples = train_input.size(0)

zeta = 0.90

train_target = train_target * zeta
test_target = test_target * zeta

nb_hidden = 50
eta = 1e-1 / nb_train_samples
epsilon = 1e-6

w1 = torch.empty(nb_hidden, train_input.size(1)).normal_(0, epsilon)
b1 = torch.empty(nb_hidden).normal_(0, epsilon)
w2 = torch.empty(nb_classes, nb_hidden).normal_(0, epsilon)
b2 = torch.empty(nb_classes).normal_(0, epsilon)

dl_dw1 = torch.empty(w1.size())
dl_db1 = torch.empty(b1.size())
dl_dw2 = torch.empty(w2.size())
dl_db2 = torch.empty(b2.size())

for k in range(1000):

    # Back-prop

    acc_loss = 0
    nb_train_errors = 0

    dl_dw1.zero_()
    dl_db1.zero_()
    dl_dw2.zero_()
    dl_db2.zero_()

    for n in range(nb_train_samples):
        x0, s1, x1, s2, x2 = forward_pass(w1, b1, w2, b2, train_input[n])

        pred = x2.max(0)[1].item()
        if train_target[n, pred] < 0.5: nb_train_errors = nb_train_errors + 1
        acc_loss = acc_loss + loss(x2, train_target[n])

        backward_pass(w1, b1, w2, b2,
                      train_target[n],
                      x0, s1, x1, s2, x2,
                      dl_dw1, dl_db1, dl_dw2, dl_db2)

    # Gradient step

    w1 = w1 - eta * dl_dw1
    b1 = b1 - eta * dl_db1
    w2 = w2 - eta * dl_dw2
    b2 = b2 - eta * dl_db2

    # Test error

    nb_test_errors = 0

    for n in range(test_input.size(0)):
        _, _, _, _, x2 = forward_pass(w1, b1, w2, b2, test_input[n])

        pred = x2.max(0)[1].item()
        if test_target[n, pred] < 0.5: nb_test_errors = nb_test_errors + 1

    print('{:d} acc_train_loss {:.02f} acc_train_error {:.02f}% test_error {:.02f}%'
          .format(k,
                  acc_loss,
                  (100 * nb_train_errors) / train_input.size(0),
                  (100 * nb_test_errors) / test_input.size(0)))
