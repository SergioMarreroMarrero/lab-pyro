# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import torch
import pyro
pyro.set_rng_seed(101)

# %% [markdown]
# # An Introduction to Models in Pyro
#
# The basic unit of probabilistics programs is the stochastic function. This is an arbitrary Python callable that combines two ingredient:
#     
#     - deterministic Python code; and
#     - primitive stochastic functions that call a random number generator
#     
# Concretely, a stochastic function can be any Python object with a \__call__() method. 

# %% [markdown]
# ## Primitive Stochastic Functions
# Primitive stochastic functions, or distributions, are an importan class of sthocastic functions for which we can explicitly compute the probability of the outputs given the inputs.
#

# %%
# draw a sample from N(0, 1) and compute the log_prob

loc = 0 
scale = 1
normal = torch.distributions.Normal(loc, scale) # create a normal distribution object
x = normal.rsample()
print("sample", x)
print("log prob", normal.log_prob(x))


# %% [markdown]
# ## A simple model
#
# Be aware that this model don't relay on pyro

# %% [markdown]
# The output of this model are 2 random variables, (X, Y). 
# The function **weather** specifies the joint probability distribution over two named random variables: `cloudy` and `temp`. As such, it defines a probabilistic model that we can reason about using the techinques of probability theory.
#

# %%
def weather():
    cloudy = torch.distributions.Bernoulli(0.3).sample()
    cloudy = 'cloudy' if cloudy.item() == 1.0 else 'sunny'
    
    mean_temp = {'cloudy': 55.0, 'sunny': 75.0}[cloudy]
    scale_temp = {'cloudy': 10.0, 'sunny': 15.0}[cloudy]
    
    temp = torch.distributions.Normal(mean_temp, scale_temp).rsample()
    
    return cloudy, temp.item()


for _ in range(5):
    print(weather())

# %% [markdown]
# ## The pyro.sample Primitive

# %% [markdown]
# Be aware that we have moved from: __torch.distributions__ to __pyro.distributions__
#
# On other hand, it we have put a name in the backend:

# %%
x = pyro.sample("my_sample", pyro.distributions.Normal(loc, scale))
print(x)


# %%
def weather():
    cloudy = pyro.sample('cloudy', pyro.distributions.Bernoulli(0.3))
    cloudy = 'cloudy' if cloudy.item() == 1.0 else 'sunny'
    
    mean_temp = {'cloudy': 55.0, 'sunny': 75.0}[cloudy]
    scale_temp = {'cloudy': 10.0, 'sunny': 15.0}[cloudy]
    
    temp = pyro.sample('temp', pyro.distributions.Normal(mean_temp, scale_temp))
    
    return cloudy, temp.item()


for _ in range(5):
    print(weather())


# %% [markdown]
# ## Universality: Stochastic Recursion, Higher-order Stochastic Functions, and Random Control Flow

# %%
def ice_cream_sales():
    cloudy, temp = weather()
    expected_sales = 200.0 if cloudy == 'sunny' and temp > 80.0 else 50.0
    ice_cream = pyro.sample('ice_cream', pyro.distributions.Normal(loc= expected_sales, scale=10.0))
    return ice_cream

ice_cream_sales()


# %% [markdown]
# This kind of modilariy, familiar to any programmer, is obviously very powerful. But is it powerful enough to encomppass all the different kinds of models we'd like to express?
#
#
#

# %% [markdown]
# Define a geomtric distribution that counts the number of failures until the first success like so:

# %%
def geometric(p, t=None):
    
    if t is None:
        t = 0
    x = pyro.sample("x_{}".format(t), pyro.distributions.Bernoulli(p))
    if x.item() == 1:
        return 0
    else:
        return 1 + geometric(p, t+1)

print(geometric(0.5))


# %% [markdown]
# We are also free to define stochastic functions that accept as input or produce as output other stochastic functions:

# %%
def normal_product(loc, scale):
    z1 = pyro.sample("z1", pyro.distributions.Normal(loc, scale))
    z2 = pyro.sample("z2", pyro.distributions.Normal(loc, scale))
    
    y = z1 * z2
    
    return y

def make_normal_normal():
    mu_latent = pyro.sample("mu_laten", pyro.distributions.Normal(0, 1))
    fn = lambda scale: normal_product(mu_latent, scale)
    return fn

print(make_normal_normal()(1.))

# %% [markdown]
# Here `make_normal_normal` is a stochastic function that takes one argument and which, upon execution, generates three named random variables.
#
# The fact that Pyro supports arbitrary Python code like this -iteration, reursion, higher order functinos, etc. - in conjunction with random control flow means that Pyro stochatics functions are universal, i.e. they can be used to reprsent any computable probability distribution. As we will see in subsequent tutorials, this is incredible powerlful.
#
# It is worth emphasizing that this is one reason why Pyro is built on top of Pytorch: dynamy computational graphs are an importan ingredient in allowing for universal models that can benefit from GPU-accelaterd tensor math.

# %%
