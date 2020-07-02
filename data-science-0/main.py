#!/usr/bin/env python
# coding: utf-8

# # Desafio 1
# 
# Para esse desafio, vamos trabalhar com o data set [Black Friday](https://www.kaggle.com/mehdidag/black-friday), que reúne dados sobre transações de compras em uma loja de varejo.
# 
# Vamos utilizá-lo para praticar a exploração de data sets utilizando pandas. Você pode fazer toda análise neste mesmo notebook, mas as resposta devem estar nos locais indicados.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Set up_ da análise

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


black_friday = pd.read_csv("black_friday.csv")


# ## Inicie sua análise a partir daqui

# In[3]:


black_friday.shape


# In[4]:


black_friday.head()


# In[5]:


#black_friday['Product_Category_2'].fillna(0,inplace=True)
#black_friday['Product_Category_3'].fillna(0,inplace=True)


# In[6]:


#(black_friday.query('Gender == "F" and Age == "26-35"')).shape[0]


# In[13]:


len((pd.unique(black_friday['User_ID'])))


# In[17]:


black_friday.info()


# In[24]:


black_friday['Gender'].dtype
black_friday['User_ID'].dtype


# In[26]:


black_friday['Product_Category_2'].isna().sum()


# In[27]:


black_friday['Product_Category_3'].isna().sum()


# In[31]:


black_friday['Product_Category_3'].value_counts()


# In[27]:


df_norm = (black_friday['Purchase'] - black_friday['Purchase'].mean()) / (black_friday['Purchase'].max() - black_friday['Purchase'].min())


# In[36]:


df_norm.mean()


# In[52]:


(df_norm.all() > -1 and df_norm.all() <1).sum()


# In[54]:


df_norm.value_counts()


# In[9]:


#black_friday.query('Product_Category_3 != "NaN" and Product_Category_2 == "NaN" ')


# ## Questão 1
# 
# Quantas observações e quantas colunas há no dataset? Responda no formato de uma tuple `(n_observacoes, n_colunas)`.

# In[ ]:


def q1():
    
    # Retorne aqui o resultado da questão 1.
    return black_friday.shape


# ## Questão 2
# 
# Há quantas mulheres com idade entre 26 e 35 anos no dataset? Responda como um único escalar.

# In[ ]:


def q2():
    (black_friday.query('Gender == "F" and Age == "26-35"')).shape[0]
    # Retorne aqui o resultado da questão 2.
    return (black_friday.query('Gender == "F" and Age == "26-35"')).shape[0]


# ## Questão 3
# 
# Quantos usuários únicos há no dataset? Responda como um único escalar.

# In[ ]:


def q3():
    # Retorne aqui o resultado da questão 3.
    return len((pd.unique(black_friday['User_ID'])))


# ## Questão 4
# 
# Quantos tipos de dados diferentes existem no dataset? Responda como um único escalar.

# In[ ]:


def q4():
    # Retorne aqui o resultado da questão 4.
    return 3


# ## Questão 5
# 
# Qual porcentagem dos registros possui ao menos um valor null (`None`, `ǸaN` etc)? Responda como um único escalar entre 0 e 1.

# In[13]:


def q5():
    sum = (black_friday['Product_Category_3'].isna().sum())
    total = black_friday.shape[0]
    porc = sum/total
    # Retorne aqui o resultado da questão 5.
    return float(porc)


# ## Questão 6
# 
# Quantos valores null existem na variável (coluna) com o maior número de null? Responda como um único escalar.

# In[15]:


def q6():
    maiorNull = black_friday['Product_Category_3'].isna().sum()
    # Retorne aqui o resultado da questão 6.
    return int(maiorNull)


# ## Questão 7
# 
# Qual o valor mais frequente (sem contar nulls) em `Product_Category_3`? Responda como um único escalar.

# In[22]:


def q7():
    # Retorne aqui o resultado da questão 7.
    return 16


# ## Questão 8
# 
# Qual a nova média da variável (coluna) `Purchase` após sua normalização? Responda como um único escalar.

# In[28]:


def q8():
    # Retorne aqui o resultado da questão 8.
    return float(df_norm.mean())


# ## Questão 9
# 
# Quantas ocorrências entre -1 e 1 inclusive existem da variáel `Purchase` após sua padronização? Responda como um único escalar.

# In[ ]:


def q9():
    # Retorne aqui o resultado da questão 9.
    pass


# ## Questão 10
# 
# Podemos afirmar que se uma observação é null em `Product_Category_2` ela também o é em `Product_Category_3`? Responda com um bool (`True`, `False`).

# In[ ]:


def q10():
    # Retorne aqui o resultado da questão 10.
    return True

