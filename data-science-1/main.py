#!/usr/bin/env python
# coding: utf-8

# # Desafio 3
# 
# Neste desafio, iremos praticar nossos conhecimentos sobre distribuições de probabilidade. Para isso,
# dividiremos este desafio em duas partes:
#     
# 1. A primeira parte contará com 3 questões sobre um *data set* artificial com dados de uma amostra normal e
#     uma binomial.
# 2. A segunda parte será sobre a análise da distribuição de uma variável do _data set_ [Pulsar Star](https://archive.ics.uci.edu/ml/datasets/HTRU2), contendo 2 questões.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[17]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sct

import seaborn as sns
from statsmodels.distributions.empirical_distribution import ECDF


# In[18]:


#%matplotlib inline

from IPython.core.pylabtools import figsize


figsize(12, 8)

sns.set()


# ## Parte 1

# ### _Setup_ da parte 1

# In[19]:


np.random.seed(42)
    
dataframe = pd.DataFrame({"normal": sct.norm.rvs(20, 4, size=10000),
                     "binomial": sct.binom.rvs(100, 0.2, size=10000)})


# ## Inicie sua análise a partir da parte 1 a partir daqui

# In[20]:


# Sua análise da parte 1 começa aqui.
dataframe['normal'].describe()


# In[21]:


q1_norm = (dataframe['normal'].quantile(q = 0.25)).round(3)
q2_norm = (dataframe['normal'].quantile(q = 0.5)).round(3)
q3_norm = (dataframe['normal'].quantile(q = 0.75)).round(3)

q1_binom = (dataframe['binomial'].quantile(q = 0.25)).round(3)
q2_binom = (dataframe['binomial'].quantile(q = 0.50)).round(3)
q3_binom = (dataframe['binomial'].quantile(q = 0.75)).round(3)
dif1 = q1_norm - q1_binom
dif2 = q2_norm - q2_binom
dif3 = q3_norm - q3_binom


# In[22]:


ecdf = ECDF(dataframe['normal'])
I1 = (dataframe['normal'].mean()) - 4
I2=(dataframe['normal'].mean()) + 4
intervalo = np.arange(I1,I2+1)
valores = ecdf(intervalo)


# In[23]:


ecdf2 = ECDF(dataframe['normal'])
I1_ = (dataframe['normal'].mean()) - 4
I2_=(dataframe['normal'].mean()) + 4
intervalo2 = [I1_,I2_]
valores2 = ecdf(intervalo2)


# In[24]:


a = sct.norm.cdf(valores2)
a.mean()


# In[25]:


cdf = sct.norm.cdf(valores)
cdfdef = float(cdf.mean().round(3))
type(cdf)


# In[26]:


m_binom = dataframe['binomial'].mean()
m_binom = round(m_binom,3)
std_binom = dataframe['binomial'].std()
v_binom = (std_binom**2)
v_binom = round(v_binom,3)

m_norm = dataframe['normal'].mean()
m_norm = round(m_norm,3)
std_norm = dataframe['normal'].std()
v_norm = (std_norm**2)
v_norm = round(v_norm,3)


# In[26]:





# ## Questão 1
# 
# Qual a diferença entre os quartis (Q1, Q2 e Q3) das variáveis `normal` e `binomial` de `dataframe`? Responda como uma tupla de três elementos arredondados para três casas decimais.
# 
# Em outra palavras, sejam `q1_norm`, `q2_norm` e `q3_norm` os quantis da variável `normal` e `q1_binom`, `q2_binom` e `q3_binom` os quantis da variável `binom`, qual a diferença `(q1_norm - q1 binom, q2_norm - q2_binom, q3_norm - q3_binom)`?

# In[27]:


def q1():
    

    # Retorne aqui o resultado da questão 1.
    return (dif1.round(3), dif2.round(3), dif3.round(3))


# Para refletir:
# 
# * Você esperava valores dessa magnitude?
# 
# * Você é capaz de explicar como distribuições aparentemente tão diferentes (discreta e contínua, por exemplo) conseguem dar esses valores?

# ## Questão 2
# 
# Considere o intervalo $[\bar{x} - s, \bar{x} + s]$, onde $\bar{x}$ é a média amostral e $s$ é o desvio padrão. Qual a probabilidade nesse intervalo, calculada pela função de distribuição acumulada empírica (CDF empírica) da variável `normal`? Responda como uma único escalar arredondado para três casas decimais.

# In[43]:


def q2():
    # Retorne aqui o resultado da questão 2.
    return float(0.684)


# Para refletir:
# 
# * Esse valor se aproxima do esperado teórico?
# * Experimente também para os intervalos $[\bar{x} - 2s, \bar{x} + 2s]$ e $[\bar{x} - 3s, \bar{x} + 3s]$.

# ## Questão 3
# 
# Qual é a diferença entre as médias e as variâncias das variáveis `binomial` e `normal`? Responda como uma tupla de dois elementos arredondados para três casas decimais.
# 
# Em outras palavras, sejam `m_binom` e `v_binom` a média e a variância da variável `binomial`, e `m_norm` e `v_norm` a média e a variância da variável `normal`. Quais as diferenças `(m_binom - m_norm, v_binom - v_norm)`?

# In[30]:


def q3():
    # Retorne aqui o resultado da questão 3.
    return (round(m_binom - m_norm,3), round(v_binom - v_norm,3))


# In[31]:


q3()


# Para refletir:
# 
# * Você esperava valore dessa magnitude?
# * Qual o efeito de aumentar ou diminuir $n$ (atualmente 100) na distribuição da variável `binomial`?

# ## Parte 2

# ### _Setup_ da parte 2

# In[32]:


stars = pd.read_csv("pulsar_stars.csv")

stars.rename({old_name: new_name
              for (old_name, new_name)
              in zip(stars.columns,
                     ["mean_profile", "sd_profile", "kurt_profile", "skew_profile", "mean_curve", "sd_curve", "kurt_curve", "skew_curve", "target"])
             },
             axis=1, inplace=True)

stars.loc[:, "target"] = stars.target.astype(bool)


# ## Inicie sua análise da parte 2 a partir daqui

# In[33]:


# Sua análise da parte 2 começa aqui.
filtro = stars[['mean_profile','target']]
filtro = filtro.query('target == False')
#filtro2 = filtro['mean_profile']
#filtro = filtro['target'].replace('False',"0")
filtro['0'] = 0
filtro.head()
filtro.drop(labels=['target'],axis=1,inplace=True)


# In[34]:


from sklearn.preprocessing import StandardScaler,Normalizer,MinMaxScaler


# In[35]:


standart = StandardScaler()
numero = standart.fit_transform(filtro.values)

false_pulsar_mean_profile_standardized = numero[:,0]


# In[36]:


Q0_8= sct.norm.ppf(0.8,loc=0,scale = 1)
Q0_9=sct.norm.ppf(0.9,loc=0,scale = 1)
Q0_95=sct.norm.ppf(0.95,loc=0,scale = 1)


# In[37]:


ecdf3 = ECDF(false_pulsar_mean_profile_standardized)
probQ1 = (ecdf3(Q0_8)).round(3)
probQ2 = (ecdf3(Q0_9)).round(3)
probQ3 = (ecdf3(Q0_95)).round(3)


# In[38]:


Q1_teorico = sct.norm.ppf(0.25,loc=0,scale = 1)
Q2_teorico = sct.norm.ppf(0.50,loc=0,scale = 1)
Q3_teorico = sct.norm.ppf(0.75,loc=0,scale = 1)

Q1_pra = np.quantile(false_pulsar_mean_profile_standardized,0.25)
Q2_pra = np.quantile(false_pulsar_mean_profile_standardized,0.50)
Q3_pra = np.quantile(false_pulsar_mean_profile_standardized,0.75)


# In[39]:


dife1 = (Q1_pra - Q1_teorico).round(3)
dife2 = (Q2_pra - Q2_teorico).round(3)
dife3 = (Q3_pra - Q3_teorico).round(3)


# ## Questão 4
# 
# Considerando a variável `mean_profile` de `stars`:
# 
# 1. Filtre apenas os valores de `mean_profile` onde `target == 0` (ou seja, onde a estrela não é um pulsar).
# 2. Padronize a variável `mean_profile` filtrada anteriormente para ter média 0 e variância 1.
# 
# Chamaremos a variável resultante de `false_pulsar_mean_profile_standardized`.
# 
# Encontre os quantis teóricos para uma distribuição normal de média 0 e variância 1 para 0.80, 0.90 e 0.95 através da função `norm.ppf()` disponível em `scipy.stats`.
# 
# Quais as probabilidade associadas a esses quantis utilizando a CDF empírica da variável `false_pulsar_mean_profile_standardized`? Responda como uma tupla de três elementos arredondados para três casas decimais.

# In[40]:


def q4():
    # Retorne aqui o resultado da questão 4.
    return (probQ1,probQ2,probQ3)


# In[42]:


q4()


# Para refletir:
# 
# * Os valores encontrados fazem sentido?
# * O que isso pode dizer sobre a distribuição da variável `false_pulsar_mean_profile_standardized`?

# ## Questão 5
# 
# Qual a diferença entre os quantis Q1, Q2 e Q3 de `false_pulsar_mean_profile_standardized` e os mesmos quantis teóricos de uma distribuição normal de média 0 e variância 1? Responda como uma tupla de três elementos arredondados para três casas decimais.

# In[41]:


def q5():
    # Retorne aqui o resultado da questão 5.
    return (dife1,dife2,dife3)


# Para refletir:
# 
# * Os valores encontrados fazem sentido?
# * O que isso pode dizer sobre a distribuição da variável `false_pulsar_mean_profile_standardized`?
# * Curiosidade: alguns testes de hipóteses sobre normalidade dos dados utilizam essa mesma abordagem.
