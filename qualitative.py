import pandas as pd
import re
import pickle
import numpy as np
import math
import os
import matplotlib.pyplot as plt
import argparse
from utils import *
from hyperparams import arch_hp

trackdir = '/data/aseewald/work/tracking'

def draw_rangealt(d,bbox=None,forwards=True,A=224,B=224,N=56,report_relative=False,border=0):
    if type(d) != dict:
        d = {'gp_X' : max(min(d[0],1),-1),'gp_Y' : max(min(d[1],1),-1),'sigmasqp' : d[2],'stridep' : min(d[3],0),'intensityp' : d[4]}
    if forwards:
        g_X_abs, g_Y_abs =  (d['gp_X'] + 1) * ((A+1)/2),(d['gp_Y'] + 1) * ((B+1)/2)
        if not report_relative: #return absolute coordinates, given bbox data.
            assert(bbox is not None)
            g_X = np.maximum(np.minimum((bbox[3] - bbox[2])/2 + (g_X_abs - (224/2)),224-border),border)
            g_Y = np.maximum(np.minimum((bbox[1] - bbox[0])/2 + (g_Y_abs - (224/2)),224-border),border)
        else:
            g_X,g_Y = (g_X_abs - (224/2)),(g_Y_abs - (224/2))
        return np.array([g_X,
                         g_Y,
                         math.exp(d['sigmasqp']),
                         math.exp(d['stridep']) * (max(A,B) - 1) / (N - 1),
                         math.exp(d['intensityp'])])
    else:
        return []

def iou_visualize(nickname,splitid,trial,tstep=None,window=10):
    ts = []
    if tstep != "postproc":
        for x in os.listdir(trackdir):
            m = re.match('saliency-quant_{}_{}_(\d+)_{}'.format(nickname,splitid,trial),x)
            if m:
                t = m.group(1)
                ts.append(int(t))
        t = max(ts)
    else:
        t = "final"
    fname,final = os.path.join(trackdir,"iou-quant_{}_{}_{}_{}".format(nickname,splitid,t,trial)),os.path.join(trackdir,"iou-quant_{}_{}_final_{}".format(nickname,splitid,trial))
    df = pickle.load(open(fname,'rb'))
    imgstep = 450 // math.sqrt(6)
    df,converted = possibly_convert_stats(df,imgstep)
    if converted:
        df.to_pickle(fname)
    if not os.path.exists(final):
        df.to_pickle(final)
    rand = df[df['selected'] == False]
    att = df[df['selected'] == True]
    df = df.sort('tstep')
    barw = 0.35
    row = 0
    bins = range(rand.iloc[0]['histo'][1].size - 1)
    while row < (len(att)-window): # don't make new windows if everything in the current window is from the same timestep.
        advance = 1
        crand = np.mean(np.array([x[0] for x in rand.iloc[row:(row+window)]['histo'].values]),axis=0,keepdims=True)
        catt =  np.mean(np.array([x[0] for x in att.iloc[row:(row+window)]['histo'].values]),axis=0,keepdims=True)
        tmin = rand.iloc[row]['tstep']
        while len(rand.iloc[row:(row+(advance * window))]['tstep'].unique()) == 1:
            crand = np.vstack([crand,np.mean(np.array([x[0] for x in rand.iloc[row:(row+window)].values]),axis=0)])
            catt = np.vstack([catt,np.mean(np.array([x[0] for x in att.iloc[row:(row+window)].values]),axis=0)])
            advance += 1
        tmax = rand.iloc[row+len(catt)-1]['tstep']
        crand,catt = np.mean(crand,axis=0), np.mean(catt,axis=0)
        fig,ax = plt.subplots()
        rects_att = ax.bar(bins,catt,barw,color='b')
        rects_rand = ax.bar(np.array(bins) + barw,crand,barw,color='r')
        oname = 'results/iouvis-{}_{}_{}_{}.png'.format(nickname,splitid,trial,tmin)
        print("saving to",oname)
        plt.savefig(oname)
        row += advance * window


def saliency_visualize(nickname,splitid,trial,tstep=None,window=10):
    '''
    The original data from arch.py has two column selected vs not selected and ran every `imgstep` steps.
    '''
    ts = []
    if tstep != "postproc":
        for x in os.listdir(trackdir):
            m = re.match('saliency-quant_{}_{}_(\d+)_{}'.format(nickname,splitid,trial),x)
            if m:
                t = m.group(1)
                ts.append(int(t))
        t = max(ts)
    else:
        t = "final"
    fname,final = os.path.join(trackdir,"saliency-quant_{}_{}_{}_{}".format(nickname,splitid,t,trial)),os.path.join(trackdir,"saliency-quant_{}_{}_final_{}".format(nickname,splitid,trial))
    df = pickle.load(open(fname,'rb'))
    imgstep = 450 // math.sqrt(6)
    df,converted = possibly_convert_stats(df,imgstep)
    if converted:
        df.to_pickle(fname)
    if not os.path.exists(final):
        df.to_pickle(final)
    rand = df[df['selected'] == False]
    att = df[df['selected'] == True]
    df = df.sort('tstep')
    barw = 0.35
    row = 0
    bins = range(rand.iloc[0]['histo'][1].size - 1)
    while row < (len(att)-window): # don't make new windows if everything in the current window is from the same timestep.
        advance = 1
        crand = np.mean(np.array([x[0] for x in rand.iloc[row:(row+window)]['histo'].values]),axis=0,keepdims=True)
        catt =  np.mean(np.array([x[0] for x in att.iloc[row:(row+window)]['histo'].values]),axis=0,keepdims=True)
        tmin = rand.iloc[row]['tstep']
        while len(rand.iloc[row:(row+(advance * window))]['tstep'].unique()) == 1:
            crand = np.vstack([crand,np.mean(np.array([x[0] for x in rand.iloc[row:(row+window)].values]),axis=0)])
            catt = np.vstack([catt,np.mean(np.array([x[0] for x in att.iloc[row:(row+window)].values]),axis=0)])
            advance += 1
        tmax = rand.iloc[row+len(catt)-1]['tstep']
        crand,catt = np.mean(crand,axis=0), np.mean(catt,axis=0)
        fig,ax = plt.subplots()
        plt.title('Saliency t={} to {}'.format(tmin,tmax))
        rects_att = ax.bar(bins,catt,barw,color='b')
        rects_rand = ax.bar(np.array(bins) + barw,crand,barw,color='r')
        plt.savefig('results/salvis-{}_{}_{}_{}.png'.format(nickname,splitid,trial,tmin))
        row += advance * window
        

def closest(nickname,splitid,trial,num_windows=2,tstep=None):
    '''

    '''
    ts = []
    if tstep is None:
        for x in os.listdir(trackdir):
            m = re.match('closest-quant_{}_{}_(\d+)_{}'.format(nickname,splitid,trial),x)
            if m:
                t = m.group(1)
                ts.append(int(t))
        t = max(ts)
    else:
        t = "final"
    fname,final = os.path.join(trackdir,"closest-quant_{}_{}_{}_{}".format(nickname,splitid,t,trial)),os.path.join(trackdir,"closest-quant_{}_{}_final_{}".format(nickname,splitid,trial))
    df = pickle.load(open(fname,'rb'))
    imgstep = 450 // math.sqrt(6) # the expression i happened to use during training.
    df,converted = possibly_convert_stats(df,imgstep)
    if converted:
        df.to_pickle(fname)
    if not os.path.exists(final):
        df.to_pickle(final)
    rand = df[df['selected'] == False]
    att = df[df['selected'] == True]
    window = len(att) // num_windows
    newbins = np.linspace(0,2.75,7)
    barw = 0.15
    row = 0
    while row <= (len(att)-window): # don't make new windows if everything in the current window is from the same timestep.
        advance = 1
        attcounts = np.zeros(newbins.size)
        randcounts = np.zeros(newbins.size)
        tmin = rand.iloc[row]['tstep']
        while len(rand.iloc[row:(row+(advance * window))]['tstep'].unique()) == 1:
            advance += 1
        tmax = rand.iloc[row+(advance * window)-1]['tstep']
        for j in range(row,row+(advance * window)):
            rcounts,rbins = rand.iloc[j]['histo']
            acounts,abins = att.iloc[j]['histo']
            for k in range(0,len(rbins)-1):
                for l in range(newbins.size-1):
                    if rbins[k] >= newbins[l] and rbins[k+1] <= newbins[l+1]:
                        randcounts[l] += rcounts[k]
                        break
            for k in range(0,len(abins)-1):
                for l in range(newbins.size-1):
                    if abins[k] >= newbins[l] and abins[k+1] <= newbins[l+1]:
                        attcounts[l] += acounts[k]
                        break
        fig,ax = plt.subplots()
        plt.title('Distance to closest object, t={} to {}'.format(tmin,tmax))
        attcounts /= attcounts.sum()
        randcounts /= randcounts.sum()
        rects_att = ax.bar(newbins,attcounts,barw,color='b')
        rects_rand = ax.bar(newbins + barw,randcounts,barw,color='r')
        plt.savefig('results/closestvis-{}_{}_{}_{}.png'.format(nickname,splitid,trial,row))
        row += advance * window

def final_attention(hyperparams,common_trial=128,common_tstep=29999,trange=200):
    '''
    A visualization of distribution of what attention features look like at a given time.
    '''
    X = readsql("select * from attentionvals WHERE trial = {} AND timestep = {}".format(common_trial,common_tstep))
    for nickname,df in X.groupby('nickname'):
        for splitid,df in df.groupby('splitid'):
            for atype,df in df.groupby('type'):
                fig.suptitle('name={} split={} param={}'.format(nickname,splitid,atype))
                sns.distplot(df['readable'])
                oname = 'results/final-att-{}_{}_{}_{}.png'.format(nickname,splitid,common_trial,atype)
                plt.savefig(oname)
                plt.close()

def attention(hyperparams,common_trial=128):
    '''
    A visualization of what attention features look like over time.
    '''
    X = readsql("select * from attentionvals WHERE trial = {}".format(common_trial),hyperparams)
    cmap = ['r','g','b','k','c','m']
    for nickname,df in X.groupby('nickname'):
        for splitid,df in df.groupby('splitid'):
            df = df.sort_values('timestep')
            mint,maxt = df['timestep'].min(), df['timestep'].max()
            occur = [(name,len(x)) for name,x in df.groupby('summaryimgname')]
            persistent = [name for name,count in occur if count == max(occur,key=lambda x:x[1])[1]]
            df = df[df['summaryimgname'].isin(persistent)]
            try: #not necessary.
                biases = pickle.load(open('/data/aseewald/work/tracking/attention-bias-history_{}_{}_final_{}'.format(nickname,splitid,common_trial),'rb'))
                biases = biases[biases['t'] == biases['t'].max()]
            except:
                biases = None
            for atype,df in df.groupby('type'):
                fig,axes = plt.subplots(nrows=len(df['filtid'].unique()))
                fig.suptitle('name={} split={} param={}'.format(nickname,splitid,atype))
                for filtid,df in df.groupby('filtid'):
                    i = 0
                    for imgname,df in df.groupby('summaryimgname'):
                        iname = os.path.split(imgname)[1]
                        axes[filtid].plot(df['timestep'],df['readable'],c=cmap[i],linestyle='-')
                        axes[filtid].plot(df['readable'] - df['readable_rel'],c=cmap[i],linestyle='--')
                        i += 1
                oname = 'results/att-{}_{}_{}_{}.png'.format(nickname,splitid,common_trial,atype)
                print("saving",oname)
                plt.savefig(oname)
                plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action')
    parser.add_argument('nicknames')
    parser.add_argument('splitid')
    parser.add_argument('trial')
    args = parser.parse_args()
    nicks = args.nicknames.split(',')
    hyperparams = arch_hp[nicks[0]] #any of them will do for these purposes.
    if args.action == 'attention':
        attention(hyperparams)
        #final_attention( )
    elif args.action == 'closest':
        for nick in nicks:
            closest(nick,args.splitid,args.trial)
    elif args.action == 'iou':
        for nick in nicks:
            try:
                iou_visualize(nick,args.splitid,args.trial)
            except:
                print("iou failed on nickname=",nick)
            #try:
                #saliency_visualize(nick,args.splitid,args.trial)
            #except:
                #print("failed on nick={},saliency".format(nick))
                #continue
