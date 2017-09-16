"""
Liu et al.
"Metric Learning from Relative Comparisons by Minimizing Squared Residual".
ICDM 2012.

Adapted from https://gist.github.com/kcarnold/5439917
Paper: http://www.cs.ucla.edu/~weiwang/paper/ICDM12.pdf
"""

from __future__ import print_function, absolute_import
import numpy as np
np.set_printoptions(threshold=np.nan)
import time
import sys
import copy
import random
import scipy.linalg
import scipy.stats as ss
from random import choice
from six.moves import xrange
from .base_metric import BaseMetricLearner

def regularization_loss(metric, prior_inv):
  sign, logdet = np.linalg.slogdet(metric)
  return np.sum(metric * prior_inv) - sign * logdet

class LSML(BaseMetricLearner):
  def __init__(self, tol=1e-3, max_iter=1000,regularization_lambda=1,chatty=False,minstep=1e-5):
    """Initialize the learner.

    Parameters
    ----------
    tol : float, optional
    max_iter : int, optional
    """
    self.tol = tol
    self.max_iter = max_iter
    self.reg_lambda = regularization_lambda
    self.minstep = minstep
    self.chatty = chatty

  def _prepare_inputs(self, X, constraints, weights, prior):
    self.X = X
    self.vab = np.diff(X[constraints[:,:2]], axis=1)[:,0]
    self.vcd = np.diff(X[constraints[:,2:]], axis=1)[:,0]
    if weights is None:
      self.w = np.ones(constraints.shape[0])
    else:
      self.w = weights
    self.w /= self.w.sum()  # weights must sum to 1
    if prior is None:
      self.M = np.cov(X.T)
    elif prior == "identity":
      self.M = np.eye(X.shape[1])
    else:
      self.M = prior

  def metric(self):
    return self.M
 
  def score(self,X,constraints,M,weights=None):
    self._prepare_inputs(X, constraints, weights, M)
    comp_loss = self._comparison_loss(self.M)
    if self.reg_lambda != 0:
      reg_loss = regularization_loss(self.M,np.eye(X.shape[1]))
    else:
      reg_loss = 0
    dabs = np.sum(self.vab.dot(self.M) * self.vab, axis=1)
    dcds = np.sum(self.vcd.dot(self.M) * self.vcd, axis=1)
    violations = dabs > dcds
    num_violations = np.sum(violations)
    return comp_loss,reg_loss,num_violations

  def fit(self, X, constraints, num_constraints, warm_start=None, weights=None, prior=None, verbose=False,num_steps=80,l_prev=1e-5,wall_timeout=350,skip_linesearch=False,poisson_k=0.1,skip_projection=False):
    """Learn the LSML model.

    Parameters
    ----------
    X : (n x d) data matrix
        each row corresponds to a single instance
    constraints : (m x 4) matrix of ints
        (a,b,c,d) indices into X, such that d(X[a],X[b]) < d(X[c],X[d])
    num_constraints: number to sample when taking a given gradient.
    weights : (m,) array of floats, optional
        scale factor for each constraint
    prior : (d x d) matrix, optional
        guess at a metric [default: covariance(X)]
    verbose : bool, optional
        if True, prints information while learning
    """
    self._prepare_inputs(X, constraints, weights, prior)
    if prior == "identity":
        prior_inv = self.M #because I = I^-1, and self.M has been set to np.eye(X.shape[1]).
    else:
        try:
            prior_inv = scipy.linalg.inv(self.M)
        except:
            w, v = scipy.linalg.eigh(self.M) #since I sometimes skip projection steps, do it here in the worst case.
            self.M = v.dot((np.maximum(w, 1e-8) * v).T)
    if warm_start != None:
        self.M = warm_start
    s_best = self._total_loss(self.M, prior_inv)
    exp_max = 0.7
    step_sizes = np.linspace(l_prev * exp_max, l_prev * (1/exp_max), num_steps)
    np.random.shuffle(step_sizes) #to avoid timeouts biasing the finding of 'optimal' stepsize.
    if self.chatty:
      print('initial loss', s_best)
    l_best = None
    tfit = time.time()
    grad_table = []
    for it in xrange(1, self.max_iter+1):
      grads,grad_norms,violations = [],[],[]
      if (time.time() - tfit) > wall_timeout:
        if self.chatty: print("Due to walltime constraints, stopping at it={}".format(it))
        break
      prv = ss.poisson(poisson_k,1)
      num_runs = prv.rvs()
      if self.chatty: print("poisson random variable is ",num_runs)
      for run in range(num_runs):
        grad,num_violations = np.nan_to_num(self._gradient(self.M, prior_inv,num_constraints))
        grads.append(grad)
        violations.append(num_violations)
        grad_norms.append(scipy.linalg.norm(grad))
      grad,grad_norm,num_violations = np.mean(grads,axis=0),np.mean(grad_norms),np.mean(violations)
      if grad_norm < self.tol:
        break
      if self.chatty:
        print('gradient norm', grad_norm)
      M_best,l_best = None,None
      if skip_linesearch:
        tpro = time.time()
        l_best = l_prev
        step_size = l_prev
        step_size /= grad_norm
        new_metric = self.M - max(step_size,self.minstep) * grad
        if not skip_projection:
          w, v = scipy.linalg.eigh(new_metric)
          new_metric = v.dot((np.maximum(w, 1e-8) * v).T)
        if self.chatty: print("projection took {} seconds".format(time.time() - tpro))
        self.M = new_metric
        break
      else:
        for step_size in step_sizes:
          if (time.time() - tfit) > wall_timeout:
            if self.chatty: print("Due to walltime constraints, stopping at it={}".format(it))
            break
          t0 = time.time()
          step_size /= grad_norm
          new_metric = self.M - max(step_size,self.minstep) * grad
          if not skip_projection:
            w, v = scipy.linalg.eigh(new_metric)
            new_metric = v.dot((np.maximum(w, 1e-8) * v).T)
          cur_s = self._total_loss(new_metric, prior_inv)
          if self.chatty: print("Step took {} seconds".format(time.time() - t0))
          if cur_s < s_best:
            l_best = step_size
            s_best = cur_s
            M_best = new_metric
            self.M = M_best
          if (time.time() - tfit) > wall_timeout:
            if self.chatty: print("Due to walltime constraints, stopping at it={}".format(it))
            break
      if M_best is None:
        break
      print('iter', it, 'cost', s_best, 'best step', l_best * grad_norm)
    else:
      print("Didn't converge after", it, "iterations. Final loss:", s_best)
    if l_best == None:
        l_best = (l_prev * exp_max) / 2.0 # no step was better, so decease the stepsize range.
    return max(l_best,self.minstep),num_violations,grads

  def _comparison_loss(self, metric):
    dab = np.sum(self.vab.dot(metric) * self.vab, axis=1)
    dcd = np.sum(self.vcd.dot(metric) * self.vcd, axis=1)
    violations = dab > dcd
    return self.w[violations].dot((np.sqrt(dab[violations]) -
                                   np.sqrt(dcd[violations]))**2)

  def _total_loss(self, metric, prior_inv):
    cl = self._comparison_loss(metric)
    if self.reg_lambda == 0:
      reg_loss = 0
    else:
      reg_loss = self.reg_lambda * regularization_loss(metric, prior_inv)
    return ( cl + reg_loss)

  def _gradient(self, metric, prior_inv,num_constraints,debug=False,epsilon=1e-8,skip_inv=True):
    # epsilon is a small non-zero value kludge to prevent division by zero errors.
    t0 = time.time()
    if skip_inv:
        dMetric = np.zeros_like(metric)
    else:
        dMetric = prior_inv - scipy.linalg.inv(metric)
    if debug:
        dMetric = copy.deepcopy(dMetric)
    assert(num_constraints <= self.vab.shape[0])
    current_rows = random.sample(list(range(self.vab.shape[0])),num_constraints)
    current_vab = self.vab[current_rows]
    current_vcd = self.vcd[current_rows]
    dabs = np.sum(current_vab.dot(metric) * current_vab, axis=1)
    dcds = np.sum(current_vcd.dot(metric) * current_vcd, axis=1)
    violations = dabs > dcds
    num_violations = np.sum(violations)
    t0 = time.time()
    for vab, dab, vcd, dcd in zip(current_vab[violations], dabs[violations],
                                  current_vcd[violations], dcds[violations]):
      dMetric += ((1-np.sqrt(dcd/np.maximum(dab,epsilon)))*np.outer(vab, vab) +
                  (1-np.sqrt(dab/np.maximum(dcd,epsilon)))*np.outer(vcd, vcd))
    if self.chatty: print("gradient took {} seconds,num_violations={}".format(time.time() - t0,num_violations))
    return dMetric,num_violations

  @classmethod
  def prepare_constraints(cls, labels, num_constraints):
    C = np.empty((num_constraints,4), dtype=int)
    a, c = np.random.randint(len(labels), size=(2,num_constraints))
    for i,(al,cl) in enumerate(zip(labels[a],labels[c])):
      C[i,1] = choice(np.nonzero(labels == al)[0])
      C[i,3] = choice(np.nonzero(labels != cl)[0])
    C[:,0] = a
    C[:,2] = c
    return C


