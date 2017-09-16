'''

This module is imported for visualizations.
Running this as a script will 

This program has an argparse-style command line interface. Running this program with the '--help' flag will describe the way to use it.
'''
import math
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import matplotlib.cm as cm
import seaborn as sns
import argparse
import memcache
from utils import *
from mytypes import *
from hyperparams import arch_hp

hyperparams = arch_hp['DRAW4-dual-shared'] #only need this for database handle.

def draw_rangealt(d,bbox=None,forwards=True,A=224,B=224,N=56,report_relative=False,border=0,img_s=224):
    if type(d) != dict:
        d = {'gp_X' : max(min(d[0],1),-1),'gp_Y' : max(min(d[1],1),-1),'sigmasqp' : d[2],'stridep' : min(d[3],0),'intensityp' : d[4]}
    if forwards:
        g_X_abs, g_Y_abs =  (d['gp_X'] + 1) * ((A+1)/2),(d['gp_Y'] + 1) * ((B+1)/2)
        if not report_relative: #return absolute coordinates, given bbox data.
            assert(bbox is not None)
            g_X = np.maximum(np.minimum((bbox[3] - bbox[2])/2 + (g_X_abs - (img_s/2)),img_s-border),border)
            g_Y = np.maximum(np.minimum((bbox[1] - bbox[0])/2 + (g_Y_abs - (img_s/2)),img_s-border),border)
        else:
            g_X,g_Y = (g_X_abs - (img_s/2)),(g_Y_abs - (img_s/2))
        return np.array([g_X,
                         g_Y,
                         math.exp(d['sigmasqp']),
                         math.exp(d['stridep']) * (max(A,B) - 1) / (N - 1),
                         math.exp(d['intensityp'])])
    else:
        return []

def legend_fmt(params,bias,filtid,imgname,nickname,splitid,t,bbox,trial,data_t='sampled',write=True,abbreviated=True) -> str:
    readable = draw_rangealt(params,bbox)
    gx,gy,stride,intensity,sigmasq = readable 
    dgx,dgy,dstride,dintensity,dsigmasq = readable - draw_rangealt(bias,bbox)
    # the ':+d' part has guaranteed plus or minus number formatting, to express difference of value from bias.
    if type(params) == dict:
        params = [params['gp_X'],params['gp_Y'],params['sigmasqp'],params['stridep'],params['intensityp']]
    if abbreviated:
        legend_txt = '''
    $\~g_X={:+.2f},g_X={:+.2f}$ 
    $\~g_Y={:+.2f},g_Y={:+.2f}$ 
    $\~\delta^2={:+.2f},\delta={:+.2f}$
    '''.format(params[0],gx,
               params[1],gy,
               params[2],stride)
    else:
        legend_txt = '''
    x mean:   $\~g_X={:+.2f}({:+.2f}),g_X={:+.2f}({:+.2f})$ 
    y mean:   $\~g_Y={:+.2f}({:+.2f}),g_Y={:+.2f}({:+.2f})$ 
    sigmasq:  $\~\sigma^2={:+.2f}({:+.2f}),\sigma^2={:+.2f}({:+.2f})$
    stride:   $\~\delta={:+.2f}({:+.2f}),\delta={:+.2f}({:+.2f})$
    intensity:$\~\gamma={:+.2f}({:+.2f}),\gamma={:+.2f}({:+.2f}) $
    '''.format(params[0],params[0] - bias[0],gx,dgx,
               params[1],params[1] - bias[1],gy,dgy,
               params[2],params[2] - bias[2],stride,dstride,
               params[3],params[3] - bias[3],intensity,dintensity,
               params[4],params[4] - bias[4],sigmasq,dsigmasq)
    if write:
        dosql(f"INSERT INTO attentionvals VALUES('{nickname}',{splitid},{t},'gx',{filtid},'{imgname}',{params[0]},{params[0] - bias[0]},{gx},{dgx},{trial},'{data_t}')",hyperparams)
        dosql(f"INSERT INTO attentionvals VALUES('{nickname}',{splitid},{t},'gy',{filtid},'{imgname}',{params[1]},{params[1] - bias[1]},{gy},{dgy},{trial},'{data_t}')",hyperparams)
        dosql(f"INSERT INTO attentionvals VALUES('{nickname}',{splitid},{t},'stride',{filtid},'{imgname}',{params[2]},{params[2] - bias[2]},{stride},{dstride},{trial},'{data_t}')",hyperparams)
        dosql(f"INSERT INTO attentionvals VALUES('{nickname}',{splitid},{t},'intensity',{filtid},'{imgname}',{params[3]},{params[3] - bias[3]},{intensity},{dintensity},{trial},'{data_t}')",hyperparams)
        dosql(f"INSERT INTO attentionvals VALUES('{nickname}',{splitid},{t},'sigmasq',{filtid},'{imgname}',{params[4]},{params[4] - bias[4]},{sigmasq},{dsigmasq},{trial},'{data_t}')",hyperparams)
    return legend_txt

def prepare_visualize_img(ds,batchid,posts,pre_full,bbox,biases,name,imgname,nickname,splitid,t,hyperparams,trial,include_center,chan=None,draw=False,displaying=False):
    pickle.dump((ds,batchid,posts,pre_full,bbox,biases,name,imgname,nickname,splitid,t,hyperparams,trial,include_center,chan,draw,displaying),open(name + '.pkl','wb'))

def visualize_img(ds,batchid,posts,pre_full,bbox,biases,name,imgname,nickname,splitid,t,hyperparams,trial,include_center,chan=None,draw=False,displaying=False,img_s=224,indicate_bias=False,abbreviated=True):
    '''
    Need to modify the leftmost column to be the full image with the candidate outlined in red and the patch outlined in blue.
    '''
    add_patch = True
    oname = name + ".jpg"
    if os.path.exists(oname):
        return
    gs = gridspec.GridSpec(len(ds.keys()),3)
    if displaying:
        plt.ion()
    fig = plt.gcf()
    fig.suptitle(os.path.split(imgname)[1])
    fig.set_size_inches(40,40)
    include_shift = int(not include_center) #if including center, shift over by one.
    txtsize = 56 if abbreviated else 42
    for filtid in ds.keys():
        dsf = {k : ds[filtid][k][batchid] for k in ds[filtid].keys()}
        readable = draw_rangealt(dsf,bbox)
        readable_bias = draw_rangealt(biases[filtid],bbox)
        print(f"readable={readable},readable_bias={readable_bias}")
        x_center,y_center = readable[0:2]
        x_bcenter,y_bcenter = readable_bias[0:2]
        intensity,stride,sigmasq = readable[2:]
        legend_txt = legend_fmt(dsf,biases[filtid],filtid,imgname,nickname,splitid,t,bbox,trial,write=True)
        raw = plt.subplot(gs[3 * filtid])
        raw.imshow(pre_full)
        raw.xaxis.set_ticklabels([])
        raw.grid(False)
        raw.yaxis.set_ticklabels([])
        patch = plt.subplot(gs[3 * filtid + 1])
        text = plt.subplot(gs[3 * filtid + 2])
        text.axis('off')
        patch.yaxis.set_ticklabels([])
        patch.grid(False)
        patch.xaxis.set_ticklabels([])
        if add_patch:
            candidate_rect = patches.Rectangle((bbox[2],bbox[0]),bbox[3] - bbox[2],bbox[1] - bbox[0],fill=False,edgecolor='r',linewidth=7)
            raw.add_patch(candidate_rect)
            y_s,x_s = img_s/4 * readable[3],img_s/4 * readable[3]
            y0,x0 = readable[0] - 0.5 * y_s,readable[1] - 0.5 * x_s
            attention_rect = patches.Rectangle((x0,y0),x_s,y_s,fill=False,edgecolor='b',linewidth=7)
            raw.add_patch(attention_rect)
            by_s,bx_s = img_s/4 * readable_bias[3],img_s/4 * readable_bias[3]
            by0,bx0 = readable_bias[0] - 0.5 * y_s,readable_bias[1] - 0.5 * x_s
            if indicate_bias:
                attention_bias_rect = patches.Rectangle((bx0,by0),bx_s,by_s,fill=False,edgecolor='cyan',linewidth=7)
                raw.add_patch(attention_bias_rect)
        print("bbox_coord={bbox}")
        patch.imshow(posts[filtid+include_shift])
        # used to be 42.
        text.text(0.5,0.5,legend_txt,ha='center',va='center',size=txtsize)
    gs.tight_layout(fig)
    fig.savefig(oname)
    print("saving as",oname)
    # these two lines together do a non-blocking show.
    if displaying:
        if 'DISPLAY' in os.environ.keys() and draw:
            plt.draw()
            plt.pause(0.1)
    fig.clear()
    plt.close()
    if displaying:
        plt.ioff()
    
def plot_qualitative(sess:tf.Session,filt:tf.Tensor,post:tf.Tensor,bboxs:tf.Tensor,alt_bboxs:tf.Tensor,feed_img:feed_t,t:int,splitid:int,\
                    nickname:str,imgs,parameters:parameters_t,imgnames:List[str],attend:tf.Tensor,hyperparams,trial,t_sess_dir,include_center=True,data_t='sampled',draw=False,mask=None,uuids=None,doing=False,prime=False) -> float:
    '''
    sess - this function evaluates tensors so it needs a session object.
    filt - 
    post - a tensor which shows the outputs of the DRAW layer (whether or not that ultimate output is used, which it is in the distinct model and just for sake of illustration in the shared model).
    bboxs - 
    alt_bboxs - 
    feed_img - the feed dict with which to do sess.run calls.
    t - current timestep
    splitid - 
    '''
    # This is not critical to training, it just is useful information so don't bail if it fails.
    try:
        if hyperparams.ctxop == 'DRAW':
            ks = ['gp_X','gp_Y','stridep','intensityp','sigmasqp']
            assert(all([k in attend[0].keys() for k in ks]))
            filters,posts = sess.run([tf.stack(filt),post],feed_dict=feed_img) 
            batchsize = posts.shape[0]
            if hyperparams.compute_t == 'shared':
                npatch = posts.shape[1]
            elif hyperparams.compute_t == 'full': #
                npatch = posts.shape[1]-1
            prop_zero = {k: [] for k in range(0,npatch)}
            # the conditional due to conditionally including vs excluding object candidate itself.
            biases = [sess.run(parameters[1][f'attention_{i}']) for i in range(0,npatch)]
            include_shift = int(not include_center) #if including center, shift over by one.
            nchan = 3
            attention = {}
            # saving individual featuremaps, because in the general case we don't have RGB, though we do save RGB channels in that case.
            for chan in range(nchan):
                for patchid in range(0,npatch):
                    attention[patchid] = {k : sess.run(attend[patchid][k],feed_dict=feed_img) for k in ks}
                    for batchid in range(batchsize):
                        F_X,F_Y = filters[patchid,0,batchid,:,:],filters[patchid,1,batchid,:,:]
                        F_out = posts[batchid,patchid+include_shift]
                        prop_zero[patchid].append(np.sum(F_out == np.zeros_like(F_out)) / F_out.size)
                        if patchid in [0,1] and batchid in [0,1]:
                            pass
            maybe_mkdir(t_sess_dir(t))
            d = f'{t_sess_dir(t)}/filter_vis'
            for batchid in range(batchsize):
                if mask is not None:
                    if not mask[batchid]:
                        continue
                # this changes.
                maybe_mkdir(d)
                names = f"{d}/{batchid}_{data_t}"
                if uuids is not None:
                    names = names + ',,' + uuids[batchid] + ',,' + str(prime)
                if doing:
                    visualize_img(attention,batchid,posts[batchid],imgs[batchid],alt_bboxs[batchid],biases,names,imgnames[batchid],nickname,splitid,t,hyperparams,trial,include_center,draw=draw)
                else:
                    prepare_visualize_img(attention,batchid,posts[batchid],imgs[batchid],alt_bboxs[batchid],biases,names,imgnames[batchid],nickname,splitid,t,hyperparams,trial,include_center,draw=draw)
            return prop_zero
        elif hyperparams.ctx_op in ['block_intensity','block_blur']:
            maybe_mkdir(t_sess_dir(t))
            d = f'{t_sess_dir(t)}/featmaps'
            maybe_mkdir(d)
            prevals = X
            rfilt, gfilt,bfilt, postvals = sess.run([filt[0],filt[1],filt[2],post],feed_dict=feed_img)
            for i in range(postvals.shape[0]):
                postimg = postvals[i]
                postmax = max(abs(np.min(postimg)), np.max(postimg))
                premax = max(abs(np.min(prevals)), np.max(prevals))
                imsave(f'{d}/conv0_pre_{i}_{data_t}.jpg',img_as_float(prevals[i] / premax))
                imsave(f'{d}/conv0_post_{i}_{data_t}.jpg',img_as_float(postimg / postmax))
                plt.matshow(rfilt[i].reshape((hyperparams.M,hyperparams.M)))
                plt.savefig(f'{d}/rfilt_{hyperparams.ctxop}_{i}_{data_t}')
                plt.close()
                if ctxconfirmed != 1:
                    if hyperparams.ctxop == "block_intensity":
                        if np.unique(rfilt[i]).size > 1:
                            ctxconfirmed = 1
                    elif hyperparams.ctxop == "block_blur":
                        if np.count_nonzero(rfilt[i]) > 1:
                            ctxconfirmed = 1
    except:
        print("WARNING failed to write out pre/post context.")
        return False

def purity_arch(trial,common_timestep,dataset,splitids):
    sns.set_style("whitegrid")
    # the grouped facetgrids.
    purity_cantype_foursquare(trial,common_timestep,dataset,splitids)
    purity_cantype_foursquare(trial,common_timestep,dataset,splitids)
    purity_canamount_foursquare(trial,common_timestep,dataset,splitids)
    # alltogether.
    purity_cantype(trial,common_timestep,dataset,splitids)
    purity_greedy( )
    purity_splitid_facet(trial,common_timestep,dataset,splitids)

def purity_cantype(trial,common_timestep,dataset,splitids):
    '''

    '''
    if type(common_timestep) == dict:
        print("haven't yet implemented this.")
    X = readsql(f"select * from arch_purity where trial = {trial} AND (timestep = {common_timestep} OR timestep = -1) AND dataset = '{dataset}'",hyperparams)
    Y = readsql(f"select netpurity,num_clusters,splitid,num_candidates AS num_samples,k_type AS nickname,quantile from greedy_purity where num_fields = 155 OR num_fields = -1 AND dataset = '{dataset}'",hyperparams)
    #X.loc[X['timestep'] == -1,'splitid'] = 7
    Y = Y[Y['nickname'].isin(['above_below','cnnapp','metric'])]
    #Y = Y[Y['num_samples'].isin([788,800,1200,1800,2600])]
    Y['nickname'].replace('cnnapp','vggnet',inplace=True)
    #Y['perfect'] = perfect
    #Y['even'] = even
    X = X.append(Y)
    X = X[X['splitid'].isin(splitids)]
    for ns,df in X.groupby('num_samples'):
        for q,dfp in df.groupby('quantile'):
            for splitid,dfpp in dfp.groupby('splitid'):
                num_unknown = readsql(f"select count(*) from splitcats where seen = 0 AND splitid = {splitid} AND dataset = '{dataset}'",hyperparams).iloc[0]['count']
                plt.gcf().set_size_inches(30,30)
                sns.set_style("whitegrid")
                g = sns.FacetGrid(dfpp,hue='nickname',col='even',row='perfect',hue_order=constants.hue_order,palette=constants.palette)
                g.map(sns.pointplot,'num_clusters','netpurity')
                for ax in g.axes.flat:
                    ax.set_ylim(0,0.8)
                oname = f'results/cantype-{ns}_{q}_{splitid}_factors.png'
                plt.savefig(oname)
                print("Saving as",oname)
                plt.close("all")
                col = ['purity','AUC']
                for perf,dd in dfpp.groupby('perfect'):
                    for even,ddd in dd.groupby('even'):
                        rowcolors,rowlabels,tab = [],[],[]
                        fig,ax = plt.subplots()
                        sns.reset_orig()
                        ax.xaxis.set_visible(False)
                        ax.yaxis.set_visible(False)
                        ax.spines["top"].set_visible(False);ax.spines["right"].set_visible(False);ax.spines["left"].set_visible(False);ax.spines["bottom"].set_visible(False);
                        ax.grid(False)
                        fig.suptitle('class balanced = {}, ground truth segmentations = {}'.format(bool(even),bool(perf)))
                        for name,dddd in ddd.groupby('nickname'):
                            pure = dddd[dddd['num_clusters'] == num_unknown]['netpurity'].mean()
                            gb = dddd.groupby('num_clusters').mean()['netpurity']
                            tab.append([pure,auc(gb.values,gb.index.values)])
                            idx = constants.hue_order.index(name)
                            rowcolors.append(constants.palette[idx])
                            rowlabels.append(constants.papernames[idx])
                        ax.table(cellText=tab,colLabels=col,rowColours=rowcolors,rowLabels=rowlabels,loc='center',colWidths=[0.25,0.25,0.25,0.25],rowLoc='right')
                        oname =  'results/cantypetable-{}_{}_{}_{}_{}_factors.png'.format(ns,q,splitid,even,perf)
                        plt.savefig(oname)
                        print("saving as",oname)
                plt.close("all")

def purity_cantype_foursquare(trial,common_timestep,dataset,splitids,perfect=1,even=0):
    '''

    '''
    if type(common_timestep) == dict:
        print("haven't yet implemented this.")
    X = readsql("select * from arch_purity where trial = {} AND (timestep = {} OR timestep = -1) AND dataset = '{}'".format(trial,common_timestep,dataset),hyperparams)
    Y = readsql("select netpurity,num_clusters,splitid,num_candidates AS num_samples,k_type AS nickname,quantile from greedy_purity where num_fields = 155 OR num_fields = -1 AND dataset = '{}'".format(dataset),hyperparams)
    #X.loc[X['timestep'] == -1,'splitid'] = 7
    Y = Y[Y['nickname'].isin(['above_below','cnnapp','metric'])]
    #Y = Y[Y['num_samples'].isin([788,800,1200,1800,2600])]
    Y['nickname'].replace('cnnapp','vggnet',inplace=True)
    Y['perfect'] = perfect
    Y['even'] = even
    X = X.append(Y)
    X = X[X['splitid'].isin(splitids)]
    for ns,df in X.groupby('num_samples'):
        for q,dfp in df.groupby('quantile'):
            for splitid,dfpp in dfp.groupby('splitid'):
                num_unknown = readsql("select count(*) from splitcats where seen = 0 AND splitid = {} AND dataset = '{}'".format(splitid,dataset),hyperparams).iloc[0]['count']
                plt.gcf().set_size_inches(30,30)
                sns.set_style("whitegrid")
                g = sns.FacetGrid(dfpp,hue='nickname',col='even',row='perfect',hue_order=constants.hue_order,palette=constants.palette)
                g.map(sns.pointplot,'num_clusters','netpurity')
                for ax in g.axes.flat:
                    ax.set_ylim(0,0.8)
                oname = 'results/cantype-{}_{}_{}_factors.png'.format(ns,q,splitid)
                plt.savefig(oname)
                print("Saving as",oname)
                plt.close("all")
                col = ['purity','AUC']
                for perf,dd in dfpp.groupby('perfect'):
                    for even,ddd in dd.groupby('even'):
                        rowcolors,rowlabels,tab = [],[],[]
                        fig,ax = plt.subplots()
                        sns.reset_orig()
                        ax.xaxis.set_visible(False)
                        ax.yaxis.set_visible(False)
                        ax.spines["top"].set_visible(False);ax.spines["right"].set_visible(False);ax.spines["left"].set_visible(False);ax.spines["bottom"].set_visible(False);
                        ax.grid(False)
                        fig.suptitle('class balanced = {}, ground truth segmentations = {}'.format(bool(even),bool(perf)))
                        for name,dddd in ddd.groupby('nickname'):
                            pure = dddd[dddd['num_clusters'] == num_unknown]['netpurity'].mean()
                            gb = dddd.groupby('num_clusters').mean()['netpurity']
                            tab.append([pure,auc(gb.values,gb.index.values)])
                            try:
                                idx = constants.hue_order.index(name)
                            except:
                                print(name,"not in hue_order")
                                continue
                            rowcolors.append(constants.palette[idx])
                            rowlabels.append(constants.papernames[idx])
                        ax.table(cellText=tab,colLabels=col,rowColours=rowcolors,rowLabels=rowlabels,loc='center',colWidths=[0.25,0.25,0.25,0.25],rowLoc='right')
                        oname =  'results/cantypetable-{}_{}_{}_{}_{}_factors.png'.format(ns,q,splitid,even,perf)
                        plt.savefig(oname)
                        print("saving as",oname)
                plt.close("all")

def purity_canamount_foursquare(trial,common_timestep,dataset,splitids):
    # Comapring Arch and its baselines.
    exclude_list = ",".join(["'{}'".format(x) for x in ['DRAW4-nocenter','DRAW5-nocener']])
    X = readsql("select * from arch_purity where timestep = {} OR timestep = -1 AND nickname NOT IN ({}) AND nickname not like '%nocenter%'".format(common_timestep,exclude_list),hyperparams)
    X = X[X['splitid'].isin(splitids)]
    for ev,df in X.groupby('even'):
        for perf,dfp in df.groupby('perfect'):
            for splitid,dfpp in dfp.groupby('splitid'):
                plt.gcf().set_size_inches(30,30)
                sns.set_style("whitegrid")
                g = sns.FacetGrid(dfp,col='quantile',row='num_samples',hue='nickname',hue_order=constants.hue_order,palette=constants.palette)
                g.map(sns.pointplot,"num_clusters",'netpurity')
                for ax in g.axes.flat:
                    ax.set_ylim( )
                    #box = ax.get_position()
                    #ax.set_position([box.x0,box.y0,box.width*0.85,box.height])
                #sns.plt.legend(loc='upper left',bbox_to_anchor=(1,0.5))
                #g.add_legend()
                oname = 'results/canamount-{}_{}_{}.png'.format(ev,perf,splitid)
                plt.savefig(oname)
                print("saving as",oname)
                plt.close("all")
                fig,ax = plt.subplots()
                ax.set_ylim(len(dfpp['nickname'].unique()) + 1)
                ax.xaxis.set_visible(False)
                ax.yaxis.set_visible(False)
                ax.grid(False)
                for i,nick in enumerate(dfpp['nickname'].unique()): #making separate legend to copy-and-paste to the side.
                    try:
                        c = constants.palette[constants.hue_order.index(nick)]
                    except:
                        print(nick," not in the palette")
                    ax.text(0,i+1,nick,ha='center',size='xx-large',bbox={'facecolor' : c,'alpha':0.8})
                oname = 'results/legendcanamount-{}_{}_{}_factors.png'.format(ev,perf,splitid)
                plt.savefig('results/legendcanamount-{}_{}_{}_factors.png'.format(ev,perf,splitid))
                plt.savefig(oname)
                plt.close("all")
    plt.close("all")

def purity_(common_timestep):
    include_list = ",".join(["'{}'".format(x) for x in ['DRAW4dual','DRAW3dual','above-below','vggnet']])

def purity_splitid_facet(common_timestep,dataset,splitids):
    include_list = ",".join(["'{}'".format(x) for x in ['DRAW4dual','DRAW3dual','above-below','vggnet']])
    X = readsql('''select * from arch_purity where timestep = {} OR timestep = -1 AND num_samples = 700 AND quantile = 1.0 AND even = 1 AND perfect = 1'''.format(common_timestep),hyperparams)
    X = X[X['splitid'].isin(splitids)]
    for ns,df in X.groupby('num_samples'):
        for q,dfp in df.groupby('quantile'):
            plt.gcf().set_size_inches(30,30)
            sns.set_style("whitegrid")
            g = sns.FacetGrid(dfp,col='splitid',hue='nickname',hue_order=constants.hue_order,palette=constants.palette)
            g.map(sns.pointplot,"num_clusters",'netpurity')
            #for ax in g.axes.flat:
                #box = ax.get_position()
                #ax.set_position([box.x0,box.y0,box.width*0.85,box.height])
            #sns.plt.legend(loc='upper left',bbox_to_anchor=(1,0.5))
            #g.add_legend()
            oname = 'results/cansplit-{}_{}.png'.format(ns,q)
            plt.savefig(oname)
            print("saving as",oname)

def purity_greedy( ):
    # comparing greedy methods.
    # looking at only the max numfields available.
    X = readsql("select * from purities_greedy G WHERE num_fields = -1 OR num_fields = (select max(num_fields) from purities_greedy where k_type = G.k_type)",hyperparams)
    for ns,df in X.groupby('num_candidates'):
        for q,dfp in df.groupby('quantile'):
            for splitid,dfpp in dfp.groupby('splitid'):
                g = sns.FacetGrid(dfpp,hue='k_type',col='nickname',hue_order=constants.hue_order,palette=constants.palette)
                g.map(sns.pointplot,'num_clusters','netpurity')
                #plt.title('num_samples={} ; quantile={}'.format(ns,q))
                plt.savefig('results/greedycantype-{}_{}-{}_factors.png'.format(ns,q,splitid))
                plt.close("all")
    # purity as a function of number of fields.
    Y = readsql("SELECT * FROM purities_greedy WHERE num_fields <> -1",hyperparams)
    for ns,df in Y.groupby('num_candidates'):
        for q,dfp in df.groupby('quantile'):
            for splitid,dfpp in dfp.groupby('splitid'):
                g = sns.FacetGrid(dfpp,hue='k_type',col='nickname',hue_order=constants.hue_order,palette=constants.palette)
                g.map(sns.pointplot,"num_fields","netpurity")
                plt.savefig('results/greedynumfields-{}_{}-{}_factors.png'.format(ns,q,splitid))
                plt.close("all")

def purity_greedyarch( ):
    X = readsql(''' select G.k_type AS nickname,G.num_candidates AS G.num_samples from purities_greedy G
                ''',hyperparams)
    Y = reasql("select * from arch_purity",hyperparams)
    sns.pointplot(X,x="numfields",y=" ") 
    plt.savefig('results/greed.png')

def complete_visualizations(sess_dir):
    ts = np.array(os.listdir(f'{sess_dir}/t'))
    np.random.shuffle(ts)
    j=0
    for t in ts:
        print(f'doing t={t}, finished {j} visualizations')
        vis_dir = f'{sess_dir}/t/{t}/filter_vis'
        ls = os.listdir(vis_dir)
        jpgs = [x for x in ls if '.jpg' in x]
        todo = list(set(ls) - set(jpgs))
        deletions = []
        for i,x in enumerate(todo):
            parts = x.split('_')
            # ignoring files which don't have names indicating they should be processed here.
            if len(parts) != 2: 
                deletions.append(i)
                continue
            # delete if matching jpg exists.
            if os.path.exists(f'{vis_dir}/{x}.jpg'):
                deletions.append(i)
        todo = np.delete(todo,deletions)
        np.random.shuffle(todo)
        mc = memcache.Client(['127.0.0.1:11211'])
        for i,x in enumerate(todo):
            if mc.get(x+t) is None:
                mc.set(x+t,0)
        for i,x in enumerate(todo):
            if mc.get(x+t) == 0:
                print("i={},done={}".format(i,i/len(todo)))
                try:
                    tup = pickle.load(open(f'{vis_dir}/{x}','rb'))
                    visualize_img(*tup)
                    j += 1
                    mc.set(x+t,1)
                except:
                    mc.set(x+t,2)
                    continue

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='action',help=" ")
    finalize_p = subparsers.add_parser('finalize_qual')
    arch_p = subparsers.add_parser('arch')
    greedy_p = subparsers.add_parser('greedy')
    arch_p.add_argument('--trial',type=int)
    arch_p.add_argument('--tstep',type=int)
    arch_p.add_argument('--dataset',default='COCO',type=str)
    finalize_p.add_argument('--trial',type=int)
    args = parser.parse_args();
    fmdir = '/{sess_dir}/featmaps'
    if args.action == 'arch':
        purity_arch(args.trial,args.tstep,args.dataset,[3,7])
    elif args.action == 'greedy':
        purity_greedy()
    elif args.action == 'finalize_qual':
        pass
