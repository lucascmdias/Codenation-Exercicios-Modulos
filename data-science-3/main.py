#!/usr/bin/env python
# coding: utf-8

# # Desafio 5
# 
# Neste desafio, vamos praticar sobre redução de dimensionalidade com PCA e seleção de variáveis com RFE. Utilizaremos o _data set_ [Fifa 2019](https://www.kaggle.com/karangadiya/fifa19), contendo originalmente 89 variáveis de mais de 18 mil jogadores do _game_ FIFA 2019.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[1]:


from math import sqrt

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sct
import seaborn as sns
import statsmodels.api as sm
import statsmodels.stats as st
from sklearn.decomposition import PCA

#!pip install loguru
from loguru import logger


# In[2]:


# Algumas configurações para o matplotlib.
#%matplotlib inline

from IPython.core.pylabtools import figsize


figsize(12, 8)

sns.set()


# In[3]:


#!pip install logger


# In[4]:


fifa = pd.read_csv("fifa.csv")


# In[5]:


columns_to_drop = ["Unnamed: 0", "ID", "Name", "Photo", "Nationality", "Flag",
                   "Club", "Club Logo", "Value", "Wage", "Special", "Preferred Foot",
                   "International Reputation", "Weak Foot", "Skill Moves", "Work Rate",
                   "Body Type", "Real Face", "Position", "Jersey Number", "Joined",
                   "Loaned From", "Contract Valid Until", "Height", "Weight", "LS",
                   "ST", "RS", "LW", "LF", "CF", "RF", "RW", "LAM", "CAM", "RAM", "LM",
                   "LCM", "CM", "RCM", "RM", "LWB", "LDM", "CDM", "RDM", "RWB", "LB", "LCB",
                   "CB", "RCB", "RB", "Release Clause"
]

try:
    fifa.drop(columns_to_drop, axis=1, inplace=True)
except KeyError:
    logger.warning(f"Columns already dropped")


# ## Inicia sua análise a partir daqui

# In[6]:


# Sua análise começa aqui.


# In[7]:


#fifa.fillna(0,inplace=True)
fifa.dropna(inplace=True)


# In[8]:


fifa.shape


# In[9]:


fifa.describe()


# In[10]:


fifa.dtypes


# In[11]:


#plt.figure(figsize=(20,20))
#sns.heatmap(fifa.corr().round(2),annot=True)


# In[12]:


from sklearn.preprocessing import StandardScaler, Normalizer


# In[13]:


sc= StandardScaler()
sc.fit(fifa)
fifa2 = sc.transform(fifa)


# In[14]:


fifa2


# In[15]:


pca = PCA(n_components=1)
pca.fit(fifa)
fifa_pca = pca.transform(fifa)


# In[16]:


first_question = float(pca.explained_variance_ratio_)
round(first_question,3)


# In[17]:


pca2 = PCA()
pca2.fit(fifa)
fifa_pca2 = pca2.transform(fifa)


# In[18]:


fifa_pca2[0]


# In[19]:


np.cumsum(pca2.explained_variance_ratio_)


# In[20]:


#plt.plot(np.cumsum(pca2.explained_variance_ratio_))

#plt.ylim(0.5,0.95)
#plt.xlabel('Number of components')
#plt.ylabel('Cumulative explained variance')
#plt.show()


# In[21]:


x = [0.87747123,  -1.24990363,  -1.3191255, -36.7341814,
     -35.55091139, -37.29814417, -28.68671182, -30.90902583,
     -42.37100061, -32.17082438, -28.86315326, -22.71193348,
     -38.36945867, -20.61407566, -22.72696734, -25.50360703,
     2.16339005, -27.96657305, -33.46004736,  -5.08943224,
     -30.21994603,   3.68803348, -36.10997302, -30.86899058,
     -22.69827634, -37.95847789, -22.40090313, -30.54859849,
     -26.64827358, -19.28162344, -34.69783578, -34.6614351,
     48.38377664,  47.60840355,  45.76793876,  44.61110193,
     49.28911284
]


# In[22]:


pca4 = PCA(2)
pca4.fit(fifa)


# In[23]:


#a_inv = pca4.inverse_transform(x)
#a_inv
np.dot(pca4.components_,x)


# In[24]:


a = [2,3,4]
b = [2,4,2]
c = [a,b]
np.dot


# In[24]:





# ## Questão 1
# 
# Qual fração da variância consegue ser explicada pelo primeiro componente principal de `fifa`? Responda como um único float (entre 0 e 1) arredondado para três casas decimais.

# In[25]:


def q1():
    # Retorne aqui o resultado da questão 1.
    return round(first_question,3)


# ## Questão 2
# 
# Quantos componentes principais precisamos para explicar 95% da variância total? Responda como un único escalar inteiro.

# In[26]:


def q2():
    # Retorne aqui o resultado da questão 2.
    return 15


# ## Questão 3
# 
# Qual são as coordenadas (primeiro e segundo componentes principais) do ponto `x` abaixo? O vetor abaixo já está centralizado. Cuidado para __não__ centralizar o vetor novamente (por exemplo, invocando `PCA.transform()` nele). Responda como uma tupla de float arredondados para três casas decimais.

# In[27]:


x = [0.87747123,  -1.24990363,  -1.3191255, -36.7341814,
     -35.55091139, -37.29814417, -28.68671182, -30.90902583,
     -42.37100061, -32.17082438, -28.86315326, -22.71193348,
     -38.36945867, -20.61407566, -22.72696734, -25.50360703,
     2.16339005, -27.96657305, -33.46004736,  -5.08943224,
     -30.21994603,   3.68803348, -36.10997302, -30.86899058,
     -22.69827634, -37.95847789, -22.40090313, -30.54859849,
     -26.64827358, -19.28162344, -34.69783578, -34.6614351,
     48.38377664,  47.60840355,  45.76793876,  44.61110193,
     49.28911284
]


# In[40]:


def q3():
    # Retorne aqui o resultado da questão 3.
    return tuple(np.dot(pca4.components_,x).round(3))


# In[30]:


from sklearn.linear_model import LinearRegression


# In[31]:


features = list(fifa.columns)
features2 = []
for i in range(0,15):
  features2.append(features[i])


# In[32]:


features2


# In[33]:


from sklearn.feature_selection import RFE
target_feature = 'Overall'
y_train = fifa[target_feature]
x_train = fifa.drop(columns=target_feature)
rfe = RFE(LinearRegression(), n_features_to_select = 5).fit(x_train, y_train)


# In[34]:


rfe.support_


# In[35]:


local_features = []
for i in range(0,len(rfe.support_)):
  if rfe.support_[i] == True:
    local_features.append(i)


# In[36]:


local_features


# In[37]:


list_name = []
filtred_features = list(x_train.columns)
for j in local_features:
  list_name.append(filtred_features[j])


# In[38]:


list_name


# ## Questão 4
# 
# Realiza RFE com estimador de regressão linear para selecionar cinco variáveis, eliminando uma a uma. Quais são as variáveis selecionadas? Responda como uma lista de nomes de variáveis.

# In[39]:


def q4():
    # Retorne aqui o resultado da questão 4.
    return list_name

