#!/usr/bin/env python
# coding: utf-8

# In[1]:


import requests
import http.client
import json
import uuid
import base64

client_id = '672ebfe2-b319-4d2f-a404-fcb7d1815ded'
client_key = 'V6nF2uI4aU0vI5tD1cI3rS5kU1fS2qY3uP4sV6qQ4nU7yV0hC0'
logins = [
    {'username':'SandBoxUser1','password':'4304e9ec8e13bd5497dba016d572007d00e37d030ec82b7fef8abaa5121b86d56e3dc7d62bce56f017cf2ec5a0c8fe892ab9009e99a9cb6fb9fc18fae431282f9cf6f3e633d4c36aa10a002713a474d4fd6ce78edc2e54c9d963f1c0ad9d72ed8acc7d6567abd18b3147fe415ae85182c6495b7d4c3960eed6f65ed2fbe2ecec5b7d90c6fc103300e1d683c2dcb2f59046a04656b8545669c9ca45c8e4dd255ce61a33e5b115461ae5950681552ef49025bd5fcd4d8da62506cdaa7d194d26832b0a36f7c162f524b7a4d5f3e899f0e5dcfb2d361c9fb6e6821d6c52cdd31aa055dad75aed027cfb308acdd4fbbe61ce93ecfc8657adc6c6a80b5d0006f96122'},
    {'username':'SandBoxUser2','password':'7d9599c90489459da45fc92bf6ff63338f76cbf85689ddc8d77ae1317a95bbf59b8770feb7485dae36196f4a49586fcd65883feff6869dc109d16ba730d7eb4335f6609ea3ca202731bc371009911b71b80a4cf64ab566500d05218137e7e071b80442faff6f1e5024a941b803429e43e42628dea3e513080d53a373e1293787aa7b6c631a6a2a5ca098e88ff7be5e0692b7f735f292e2a5022515e8a02d8bf9efc843bb8bd31a4f0c72ae5cb64661af282c98e2967f4c3569f1d10ccff4bc7f2126769ee633bf1da0688cb7edc1be31f9a2673b887a281dade13672f19f88836e6324120329ac0f42b460effedba44cfcc5da8dbd03f454703338129c3dfcad'},
    {'username':'SandBoxUser3','password':'272efd9e28b4c9f039ce4f68ecb9741dee8871579a413dc5fbe18224a353b37aeec0c94c56d9a018653594f1f0b422b39827a167b7f32cea4d72861de562a76950e7130b4bc5dfb05244850b194aa85a5972a433af2b9479ba4fa21781dd381b814e9c37ae575e10e50412ff3c84291cdb6714e26b180654b580951a9906be221cbca734485bb06bdc42b7d2dc8f35ad56378495a9c683d7ce1cebe6e7c7d05daae750228368b54bd3da42a687259f27fd5caaa5e8dc168b34ada6f8cdb577d979f25a83531bd95500ff16bf13ba04ec9bf21f486bd8aae9930d59a5297354e7079d7927e10e88097bffed5c870c53144ce419aaafd5a8934a14bdc72a5f252b'},
    {'username':'SandBoxUser4','password':'9978fc21ebc673db54dab722b445389583d349b17c77cd293a3ffd15a6f0c9fbebf301d887924271db9069ce4af89168c27f6ee686f68300c6e2316a7610542082256a271c19b968dd23cefbe9c4e00aa9cf2c477248972edf9b19044bfd48354edac68f86b87f244eedfe42f622fa457e7a5debef3d005bb5a7ddc71a14d2a0fe10848bb1e503ea7144f42c823615d18e058a78e6173e2f5d98b356c5169037a298a12c71e9449ab69254133e3d9cac26291e7f0ab67b0418327ee0b44480d420df8eb47d1f2b7f8669d2a5514904124ab8dd175dcccdfd4f407bae4db628516ccb79d2c0a42b6be9070d2a33d6f365ed53b50cb2e736ce606c40cc24de405a'},
    {'username':'SandBoxUser5','password':'4b383ff63feb386919832f0ebf5827ea574aa3f0693c7b21b56e52d5a236ca7b021293d9dc3d9e5afa7b31220f46ec1247151989d1fe6620516233a9286f26eba390dc955c41abe0ca2f5199a31d3deb520fd3208a30269d4dc684a7475b91a1f852c93c30fb4632b0acf424594eebb5b4d1d2ad695c4f4e128c32be3dec0568f9f2503c6dba05318a2a78a677e81610c8dcc4c7761beca2b4d964978802ee964f08d70cf9ae706da4e87d9f17f0a2b477c8e31967b86d529069730195ce1535f46e9b4f73705208ee8dccd274b62ba909d2d31e021e110cf7ef7fd447cb1b20dda92dcc1bb28c7122d10d707a73dc8292c2859c1da6b43f1e586387dcc0552a'},
]

class Context():
    pass
context = Context()


# ### step1GetAccessToken

# In[2]:



def step1GetAccessToken():
    encode_key = client_id+':'+client_key
    authorization = 'Basic '+ str(base64.b64encode(encode_key.encode('utf-8')),'utf-8')
    
    payload = {
        'grant_type':'client_credentials',
        'scope':'/api'
    }
    headers = {
        'accept': "application/json",
        'authorization': authorization,
        'content-type': "application/x-www-form-urlencoded"
    } 
    
    r = requests.post("https://sandbox.apihub.citi.com/gcb/api/clientCredentials/oauth2/token/hk/gcb", data = payload, headers = headers)
    data = json.loads(r.text)
    #print(data)
    context.access_token = data['access_token']
    return data['access_token']

#step1GetAccessToken()


# In[3]:


#context.access_token


# ### step2GetBizToken

# In[4]:


def step2GetBizToken():
    authorization = "Bearer " + context.access_token
    u = str(uuid.uuid1())
    headers = {
        'authorization': authorization,
        'client_id':client_id,
        'uuid': u,
        'content-type': "application/json"
    }
    r = requests.get("https://sandbox.apihub.citi.com/gcb/api/security/e2eKey", headers = headers)
    text = json.loads(r.text)
    modulus = text['modulus'];
    exponent = text['exponent'];
    bizToken = r.headers['bizToken'];
    eventId = r.headers['eventId'];
    context.eventId = eventId
    context.bizToken = bizToken
    return {
        'modulus':modulus,
        'exponent':exponent,
        'bizToken':bizToken,
        'eventId':eventId
    }
    
#step2GetBizToken()


# In[5]:


#context.eventId


# In[6]:


#context.bizToken


# ### step3GetRealAccessToken

# In[7]:


def step3GetRealAccessToken(user_index):
    encode_key = client_id+':'+client_key
    authorization = 'Basic '+ str(base64.b64encode(encode_key.encode('utf-8')),'utf-8')
    username = logins[user_index]['username']
    password = logins[user_index]['password']
    u = str(uuid.uuid1())
    
    step1GetAccessToken()
    step2GetBizToken()
    payload = {
        'grant_type':'password',
        'scope':'/api',
        'username':username,
        'password':password
    }
    headers = {
        'authorization':authorization,
        'bizToken':context.bizToken,
        'uuid':u,
        'content-type': "application/x-www-form-urlencoded",
        'accept':'application/json'
    }
    r = requests.post("https://sandbox.apihub.citi.com/gcb/api/password/oauth2/token/hk/gcb", data = payload, headers = headers)
    return json.loads(str(r.text))['access_token']
    
#access_token = step3GetRealAccessToken(2)
#access_token


# ### testRSAPassword()

# ### getCustomerInfo 

# In[8]:


def getCustomerInfo(access_token):
    authorization = 'Bearer '+ access_token
    u = str(uuid.uuid1())
    headers = {
        'authorization':authorization,
        'client_id':client_id,
        'uuid':u,
        'accept':'application/json'
    }
    r = requests.get("https://sandbox.apihub.citi.com/gcb/api/v1/customers/profiles", headers = headers)
    info = json.loads(str(r.text))
    
    #print(info.keys())
   #print(info["financialInformation"])
    if  'financialInformation' in info.keys():
        
        fixedAmount= info["financialInformation"]["incomeDetails"][0]["fixedAmount"]
        variableAmount= info["financialInformation"]["incomeDetails"][0]["variableAmount"]
        income=fixedAmount+variableAmount
        income=round(income,2)
    else:
        
        income =1
    #print(income)
    return income


# ### getAccounts

# In[9]:


def getAccounts(access_token):
#     global access_token
    authorization = 'Bearer '+ access_token
    u = str(uuid.uuid1())
    headers = {
        'authorization':authorization,
        'client_id':client_id,
        'uuid':u,
        'content-type': "application/json",
        'accept':'application/json'
    }
    r = requests.get("https://sandbox.apihub.citi.com/gcb/api/v1/accounts", headers = headers)
    info = json.loads(str(r.text))
    
    #print(info)  
    if 'accountGroupSummary' in info.keys():
        if 'totalCurrentBalance' in info["accountGroupSummary"][0].keys():
            #print(info["accountGroupSummary"][0]["totalCurrentBalance"])
            localCurrencyBalanceAmount = info["accountGroupSummary"][0]["totalCurrentBalance"]["localCurrencyBalanceAmount"]
            foreignCurrencyBalanceAmount  = info["accountGroupSummary"][0]["totalCurrentBalance"]["foreignCurrencyBalanceAmount"]
            totalCurrentBalance = localCurrencyBalanceAmount + foreignCurrencyBalanceAmount
            ##精确到两位小数
            totalCurrentBalance = round(totalCurrentBalance,2) 
            #print(totalCurrentBalance) 
        else:
            totalCurrentBalance = 1
    
        if 'totalAvailableBalance' in info["accountGroupSummary"][0].keys():
            #print(info["accountGroupSummary"][0]["totalAvailableBalance"])
            localCurrencyBalanceAmount = info["accountGroupSummary"][0]["totalAvailableBalance"]["localCurrencyBalanceAmount"]
            foreignCurrencyBalanceAmount  = info["accountGroupSummary"][0]["totalAvailableBalance"]["foreignCurrencyBalanceAmount"]
            totalAvailableBalance = localCurrencyBalanceAmount + foreignCurrencyBalanceAmount
            totalAvailableBalance = round(totalAvailableBalance,2)
        else:
            totalAvailableBalance = -1
    #print(totalAvailableBalance)
    else:
        totalCurrentBalance = 1
        totalAvailableBalance = -1
    return totalCurrentBalance,totalAvailableBalance


# In[ ]:





# ### getsumofTransactions

# In[10]:


def getsumofTransactions(access_token):
#     global access_token
    accountID='674d4a4f6a443741656e5a584a6f57665a444e685772393273615777397a4c665073305a5a2b51356f76513d'
    authorization = 'Bearer '+ access_token
    u = str(uuid.uuid1())
    
    headers = {
        'authorization':authorization,
        'client_id':client_id,
        'uuid':u,
        'accept':'application/json'
    }
    r = requests.get("https://sandbox.apihub.citi.com/gcb/api/v1/accounts/{accountID}/transactions", headers = headers)
    info = json.loads(str(r.text))
    
#getsumofTransactions()


# In[ ]:





# In[11]:


import numpy as np
from numpy.linalg import cholesky
#import matplotlib.pyplot as plt
def myrand2tran(average,var,sum):
    mu =average
    sigma = var
    sampleNo = sum
    np.random.seed(0)
    s = np.random.normal(mu, sigma, sampleNo)
    a = []
    for i in range(len(s)):
        val = s[i]
        while val < 0:
            h = np.random.normal(mu, sigma, 1)
            val = h[0]
        val = round(val,0)
        a.append(val)
    return a


# In[12]:


import numpy as np
from numpy.linalg import cholesky
#import matplotlib.pyplot as plt
def myrand(average,var,sum):
    #sampleNo = 10
    #mu = 100067.89
    #sigma = 800000
    mu =average
    sigma = var
    sampleNo = sum
    np.random.seed(0)
    s = np.random.normal(mu, sigma, sampleNo)
    a = []
    for i in range(len(s)):
        val = s[i]
        while val < 0:
            h = np.random.normal(mu, sigma, 1)
            val = h[0]
        val = round(val,2)
        a.append(val)
    return a


# ### buildVirUsers 

# In[13]:


import numpy as np
from numpy.linalg import cholesky
def buildVirUsers():
    sumTrUsers = len(Income)
    Income.extend(myrand(100067.89,800000,N-sumTrUsers))
    sumTransactions.extend(myrand2tran(2,6,N))
    CB.extend(myrand(1402150.57,900000,N-sumTrUsers))
    for i in range(N-1):
        r=np.random.uniform(0.2,1)
        ab = CB[i+1]*r
        AB.append(ab)
    seed=[-1,0,1]
    for i in range(N):
        r1=np.random.randint(1,10)
        r2=np.random.randint(1,10)
        r3=np.random.randint(1,10)
        r4=r2+np.random.choice(seed)
        if r4>10:
            r4=r4-1
        elif r4<1:
            r4=r4+1
        ExpectMoney.append(r1)
        ExpectReturn.append(r2)
        PlanTime.append(r3)
        RiskTol.append(r4)


# ### InData

# In[14]:


###录入txt文件 a+' '+b...每个数字相隔一个空格
import numpy as np
def inData(fname):
    f = open(fname, 'r')
    line = f.readline()
    data_list = []
    while line:
        num = list(map(float,line.split(' ')))
        data_list.append(num)
        line = f.readline()
    f.close()
    data_array = np.array(data_list)
    return data_array


# ### CalculateDist

# In[15]:


def dist(datas,indexs,centre):
    sum=0
    for i in range(7):
        if i not in indexs:
            sum=sum+(datas[i]-centre[i])*(datas[i]-centre[i])
    return sum


# ### For Default Expect

# In[16]:


def mypredict(mymodel):
    centroids=mymodel.cluster_centers_
    ###缺省值录入时以0填进去 txtname相应的改变
    txtname="D:\研一\花旗API\indata_default.txt"
    DataSet=inData(txtname)
    ###获取对应的缺省位置 txtname相应的改变
    index_name="D:\研一\花旗API\index_default.txt"
    IndexSet=inData(index_name)
    label=[]
    for i in range(len(DataSet)):
        mindist=100000
        minindex=-1
        for j in range(K):
            if dist(DataSet[i],IndexSet[i],centroids[j])<mindist:
                mindist = dist(DataSet[i],IndexSet[i],centroids[j])
                minindex=j
        label.append(minindex)
    return label


# ### K-Means and Visualization

# In[17]:


from sklearn.cluster import KMeans
from sklearn.decomposition import PCA 
import numpy as np
import matplotlib.pyplot as plt

def myKMeans(DataSet):
    X=np.array(DataSet)
    maxtimes =3000
    model=KMeans(n_clusters=K, max_iter=maxtimes, random_state=0).fit(DataSet)
    y_pred = model.labels_
    ###展示聚类结果
    #print(y_pred)
    pca=PCA(n_components=2)
    newData=pca.fit_transform(DataSet)
    color = ['HotPink', 'Aqua', 'Chartreuse', 'yellow', 'LightSalmon']
    colors=np.array(color)[y_pred]
    #plt.scatter(newData[:, 0], newData[:, 1], c=colors)
    ###展示聚类效果图
    #plt.show()
    return model


# ### TransferMoney

# In[18]:


def confirmInternalTransfer(controlFlowId,access_token):
    authorization = 'Bearer ' + access_token
    u = str(uuid.uuid1())
    payload = {
        "controlFlowId": controlFlowId
    }
    headers = {
        'authorization': authorization,
        'uuid': u,
        'accept': "application/json",
        'client_id': client_id,
        'content-type': "application/json"
    }
    r = requests.post("https://sandbox.apihub.citi.com/gcb/api/v1/moneyMovement/internalDomesticTransfers", data=json.dumps(payload), headers = headers)
    text = json.loads(r.text)
    if controlFlowId !="":
        return True
    else:
        return False


# In[19]:


def createInternalTransfer(sourceAccountId, access_token, invest_amount):
    authorization = 'Bearer ' + access_token
    u = str(uuid.uuid1())
    payload = {
        "sourceAccountId": sourceAccountId,
        "transactionAmount": invest_amount,
        "transferCurrencyIndicator": "SOURCE_ACCOUNT_CURRENCY",
        "payeeId": "7977557255484c7345546c4e53424766634b6c53756841672b556857626e395253334b70416449676b42673d",
        "chargeBearer": "BENEFICIARY",
        "fxDealReferenceNumber": "12345678",
        "remarks": "DOM Internal Transfer"
    }
    headers = {
        'authorization': authorization,
        'client_id': client_id,
        'uuid': u,
        'accept':'application/json',
        'content-type':'application/json'
    }
    r = requests.post("https://sandbox.apihub.citi.com/gcb/api/v1/moneyMovement/internalDomesticTransfers/preprocess", data=json.dumps(payload), headers = headers)
    text = json.loads(r.text)
    #print(text)
    return confirmInternalTransfer(text['controlFlowId'],access_token)



# In[20]:


###转账成功 createInternal..会调用confirm.然后返回True ，要改的话可以改这下面的转账金额 100 
def transfer():
    i=0
    access_token = step3GetRealAccessToken(i)
    return(createInternalTransfer('355a515030616a53576b6a65797359506a634175764a734a3238314e4668627349486a676f7449463949453d',access_token,100))


# In[21]:


def showNeighbors(mymodel,sample):
    s=[]
    s.append(sample)
    label = mymodel.predict(s)
    res=[]
    for i in range(N):
        t=[]
        t.append(DataSet[i])
        temp = mymodel.predict(t)
        if temp == label:
            res.append(DataSet[i])
    return res


# In[ ]:





# ### Main

# In[22]:


global Income
global AB
global CB
global sumTransactions
global N
global K

K=5
Income=[]
AB=[]
CB=[]
sumTransactions=[]

StoreWill=[]
Financialflex =[]
TransAct=[]
##期望投入资金
##期望收益率
##计划投资时间长度
##风险承受能力
ExpectMoney=[]
ExpectReturn=[]
PlanTime=[]
RiskTol=[]
DataSet=[]
N=50  ##模式1 50个数据
method=0 ## 两种模式 0外在获得输入 1 API获得

if __name__=="__main__":
    ###API获得数据
    for i in range(0,5):
        access_token = step3GetRealAccessToken(i)
        ##收入
        #print(access_token)
        income= getCustomerInfo(access_token)
        ##两种余额
        cb,ab = getAccounts(access_token)
        if income==1 or cb==1 or ab == -1:
            continue
        else:
            Income.append(income)
            AB.append(ab)
            CB.append(cb)
            
    ###创建虚拟用户
    buildVirUsers()
    maxTran = max (sumTransactions)
    minTran = min (sumTransactions)
    diff = maxTran - minTran
    for i in range(N):
    
        financialflex = round(AB[i]/CB[i],2)
        Financialflex.append(financialflex)
        ##总余额
        balance = round(CB[i] + AB[i],2)
        ##存储意愿
        storewill =round(balance / income ,2) 
        StoreWill.append(storewill)
        ##交易活跃度
        transact = (sumTransactions[i]-minTran)/ diff
        TransAct.append(round(transact,2))
        
    if method == 0 : 
        for i in range(N):
            temp=[]
            temp.append(TransAct[i])
            temp.append(StoreWill[i])
            temp.append(Financialflex[i])
            temp.append(ExpectMoney[i])
            temp.append(ExpectReturn[i])
            temp.append(PlanTime[i])
            temp.append(RiskTol[i])
            DataSet.append(temp)
    else:
        #txtname相应的改变
        txtname="D:\研一\花旗API\indata_nodefault.txt"
        DataSet=inData(txtname)
    
    #print(DataSet)
    ##得到此问题的聚类模型
    mymodel=myKMeans(DataSet)
    
    ##K-Means 模型，预测无缺省值时调用 mymodel.predict(..)即可
    #label_1 = mymodel.predict(...)
    ##K-Means 模型各类中心，预测有缺省值时，仅计算无缺省属性
    ##对于有缺省的预测时，缺省值以0写入文件，缺省位置也需要在另一个文件给出
    #label_2 = mypredict(mymodel)
    
    ##返回sample附近的点向量包括sample  sample格式[0.8, 28.02, 1.0, 1, 2, 3, 1]
    ##showNeighbors(mymodel,sample)
    


# In[ ]:





# In[ ]:




