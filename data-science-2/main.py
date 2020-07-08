#!/usr/bin/env python
# coding: utf-8

# # Desafio 4
# 
# Neste desafio, vamos praticar um pouco sobre testes de hipóteses. Utilizaremos o _data set_ [2016 Olympics in Rio de Janeiro](https://www.kaggle.com/rio2016/olympic-games/), que contém dados sobre os atletas das Olimpíadas de 2016 no Rio de Janeiro.
# 
# Esse _data set_ conta com informações gerais sobre 11538 atletas como nome, nacionalidade, altura, peso e esporte praticado. Estaremos especialmente interessados nas variáveis numéricas altura (`height`) e peso (`weight`). As análises feitas aqui são parte de uma Análise Exploratória de Dados (EDA).
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sct
import seaborn as sns


# In[2]:


#%matplotlib inline

from IPython.core.pylabtools import figsize


figsize(12, 8)

sns.set()


# In[30]:


athletes = pd.read_csv("athletes.csv")
athletes2 = pd.read_csv("athletes.csv")


# In[4]:


def get_sample(df, col_name, n=100, seed=42):
    """Get a sample from a column of a dataframe.
    
    It drops any numpy.nan entries before sampling. The sampling
    is performed without replacement.
    
    Example of numpydoc for those who haven't seen yet.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Source dataframe.
    col_name : str
        Name of the column to be sampled.
    n : int
        Sample size. Default is 100.
    seed : int
        Random seed. Default is 42.
    
    Returns
    -------
    pandas.Series
        Sample of size n from dataframe's column.
    """
    np.random.seed(seed)
    
    random_idx = np.random.choice(df[col_name].dropna().index, size=n, replace=False)
    
    return df.loc[random_idx, col_name]


# ## Inicia sua análise a partir daqui

# In[5]:


# Sua análise começa aqui.
athletes.head()


# In[6]:


athletes.fillna(0,inplace=True)


# In[7]:


sample = get_sample(athletes,"height",n=3000)


# In[8]:


x = sct.norm.rvs(sample)
shapiro_test = sct.shapiro(x)
shapiro_test


# In[9]:


jarque_bera_test = sct.jarque_bera(x)
jarque_bera_test


# In[10]:


sample2 = get_sample(athletes,"weight",n=3000)


# In[11]:


x2 = sct.norm.rvs(sample2)
d_agostino_pearson_test = sct.normaltest(x2)
d_agostino_pearson_test


# In[12]:


athletes["log_weight"] = np.log10((athletes["weight"]) + 1)
athletes.head()


# In[13]:


sample3 =get_sample(athletes,"log_weight",n=3000)
x3 = sct.norm.rvs(sample3)
d_agostino_pearson_test2 = sct.normaltest(x3)
d_agostino_pearson_test2


# In[14]:


athletes.head()


# In[31]:


athletes_filtro = athletes2
athletes_filtro = athletes_filtro[['nationality','height']]
athletes_filtro.query('nationality == "CAN" or nationality	== "USA" or nationality == "BRA"',inplace=True)
athletes_filtro.head()


# In[32]:


bra = athletes_filtro.query('nationality == "BRA"')
usa = athletes_filtro.query('nationality == "USA"')
can = athletes_filtro.query('nationality == "CAN"')


# In[33]:


media_bra = round(bra['height'].mean(),3)
media_usa = round(usa['height'].mean(),3)
media_can = round(can['height'].mean(),3)

vari_bra = round((bra['height'].std()**2),3)
vari_usa= round((usa['height'].std()**2),3)
vari_can= round((can['height'].std()**2),3)


# In[18]:


#bra_sample = get_sample(bra,'height',n=100)
#usa_sample = get_sample(usa,'height',n=100)
#can_sample = get_sample(can,'height',n=100)

#bra1 = sct.norm.rvs(bra_sample)
#usa1 = sct.norm.rvs(usa_sample)
#can1 = sct.norm.rvs(can_sample)


#bra1 = sct.norm.rvs(loc=1.7,scale=vari_bra,size=300)
#usa1 = sct.norm.rvs(loc=1.7,scale=vari_usa,size=300)
#can1 = sct.norm.rvs(loc=1.7,scale=vari_can,size=300)


# In[34]:


ttest_bra_usa = sct.ttest_ind(bra['height'],usa['height'],equal_var = False,nan_policy='omit')
ttest_bra_usa 


# In[35]:


ttest_bra_can = sct.ttest_ind(bra['height'],can['height'],equal_var = False,nan_policy='omit')
ttest_bra_can 


# In[36]:


ttest_usa_can = sct.ttest_ind(usa['height'],can['height'],equal_var = False,nan_policy='omit')
ttest_usa_can 


# In[37]:


pvalue_quest = round(ttest_usa_can[1],8)
pvalue_quest


# ## Questão 1
# 
# Considerando uma amostra de tamanho 3000 da coluna `height` obtida com a função `get_sample()`, execute o teste de normalidade de Shapiro-Wilk com a função `scipy.stats.shapiro()`. Podemos afirmar que as alturas são normalmente distribuídas com base nesse teste (ao nível de significância de 5%)? Responda com um boolean (`True` ou `False`).

# In[23]:


def q1():
    # Retorne aqui o resultado da questão 1.
    return False


# __Para refletir__:
# 
# * Plote o histograma dessa variável (com, por exemplo, `bins=25`). A forma do gráfico e o resultado do teste são condizentes? Por que?
# * Plote o qq-plot para essa variável e a analise.
# * Existe algum nível de significância razoável que nos dê outro resultado no teste? (Não faça isso na prática. Isso é chamado _p-value hacking_, e não é legal).

# ## Questão 2
# 
# Repita o mesmo procedimento acima, mas agora utilizando o teste de normalidade de Jarque-Bera através da função `scipy.stats.jarque_bera()`. Agora podemos afirmar que as alturas são normalmente distribuídas (ao nível de significância de 5%)? Responda com um boolean (`True` ou `False`).

# In[24]:


def q2():
    # Retorne aqui o resultado da questão 2.
    return False


# __Para refletir__:
# 
# * Esse resultado faz sentido?

# ## Questão 3
# 
# Considerando agora uma amostra de tamanho 3000 da coluna `weight` obtida com a função `get_sample()`. Faça o teste de normalidade de D'Agostino-Pearson utilizando a função `scipy.stats.normaltest()`. Podemos afirmar que os pesos vêm de uma distribuição normal ao nível de significância de 5%? Responda com um boolean (`True` ou `False`).

# In[25]:


def q3():
    # Retorne aqui o resultado da questão 3.
    return False


# __Para refletir__:
# 
# * Plote o histograma dessa variável (com, por exemplo, `bins=25`). A forma do gráfico e o resultado do teste são condizentes? Por que?
# * Um _box plot_ também poderia ajudar a entender a resposta.

# ## Questão 4
# 
# Realize uma transformação logarítmica em na amostra de `weight` da questão 3 e repita o mesmo procedimento. Podemos afirmar a normalidade da variável transformada ao nível de significância de 5%? Responda com um boolean (`True` ou `False`).

# In[26]:


def q4():
    # Retorne aqui o resultado da questão 4.
    return False


# __Para refletir__:
# 
# * Plote o histograma dessa variável (com, por exemplo, `bins=25`). A forma do gráfico e o resultado do teste são condizentes? Por que?
# * Você esperava um resultado diferente agora?

# > __Para as questão 5 6 e 7 a seguir considere todos testes efetuados ao nível de significância de 5%__.

# ## Questão 5
# 
# Obtenha todos atletas brasileiros, norte-americanos e canadenses em `DataFrame`s chamados `bra`, `usa` e `can`,respectivamente. Realize um teste de hipóteses para comparação das médias das alturas (`height`) para amostras independentes e variâncias diferentes com a função `scipy.stats.ttest_ind()` entre `bra` e `usa`. Podemos afirmar que as médias são estatisticamente iguais? Responda com um boolean (`True` ou `False`).

# In[27]:


def q5():
    # Retorne aqui o resultado da questão 5.
    return False


# ## Questão 6
# 
# Repita o procedimento da questão 5, mas agora entre as alturas de `bra` e `can`. Podemos afimar agora que as médias são estatisticamente iguais? Reponda com um boolean (`True` ou `False`).

# In[28]:


def q6():
    # Retorne aqui o resultado da questão 6.
    return True


# ## Questão 7
# 
# Repita o procedimento da questão 6, mas agora entre as alturas de `usa` e `can`. Qual o valor do p-valor retornado? Responda como um único escalar arredondado para oito casas decimais.

# In[29]:


def q7():
    # Retorne aqui o resultado da questão 7.
    return pvalue_quest


# __Para refletir__:
# 
# * O resultado faz sentido?
# * Você consegue interpretar esse p-valor?
# * Você consegue chegar a esse valor de p-valor a partir da variável de estatística?
