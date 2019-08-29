import requests
import http.client
import json
import uuid
import base64


class Context():
    pass


context = Context()


def get_id_and_key():
    id = 'baecfb5e-74ae-449d-a715-56ff28148d28'
    key = 'P1gM8pQ2iP4tL8uX8hO5uO3jG7fC2dQ8bQ1oH1hP6xQ2mC2nE6'
    return id, key


client_id, client_key = get_id_and_key()


def get_logins():
    logins = [
        {'username': 'SandBoxUser1',
         'password': '4304e9ec8e13bd5497dba016d572007d00e37d030ec82b7fef8abaa5121b86d56e3dc7d62bce56f017cf2ec5a0c8fe892ab9009e99a9cb6fb9fc18fae431282f9cf6f3e633d4c36aa10a002713a474d4fd6ce78edc2e54c9d963f1c0ad9d72ed8acc7d6567abd18b3147fe415ae85182c6495b7d4c3960eed6f65ed2fbe2ecec5b7d90c6fc103300e1d683c2dcb2f59046a04656b8545669c9ca45c8e4dd255ce61a33e5b115461ae5950681552ef49025bd5fcd4d8da62506cdaa7d194d26832b0a36f7c162f524b7a4d5f3e899f0e5dcfb2d361c9fb6e6821d6c52cdd31aa055dad75aed027cfb308acdd4fbbe61ce93ecfc8657adc6c6a80b5d0006f96122'},
        {'username': 'SandBoxUser2',
         'password': '7d9599c90489459da45fc92bf6ff63338f76cbf85689ddc8d77ae1317a95bbf59b8770feb7485dae36196f4a49586fcd65883feff6869dc109d16ba730d7eb4335f6609ea3ca202731bc371009911b71b80a4cf64ab566500d05218137e7e071b80442faff6f1e5024a941b803429e43e42628dea3e513080d53a373e1293787aa7b6c631a6a2a5ca098e88ff7be5e0692b7f735f292e2a5022515e8a02d8bf9efc843bb8bd31a4f0c72ae5cb64661af282c98e2967f4c3569f1d10ccff4bc7f2126769ee633bf1da0688cb7edc1be31f9a2673b887a281dade13672f19f88836e6324120329ac0f42b460effedba44cfcc5da8dbd03f454703338129c3dfcad'},
        {'username': 'SandBoxUser3',
         'password': '272efd9e28b4c9f039ce4f68ecb9741dee8871579a413dc5fbe18224a353b37aeec0c94c56d9a018653594f1f0b422b39827a167b7f32cea4d72861de562a76950e7130b4bc5dfb05244850b194aa85a5972a433af2b9479ba4fa21781dd381b814e9c37ae575e10e50412ff3c84291cdb6714e26b180654b580951a9906be221cbca734485bb06bdc42b7d2dc8f35ad56378495a9c683d7ce1cebe6e7c7d05daae750228368b54bd3da42a687259f27fd5caaa5e8dc168b34ada6f8cdb577d979f25a83531bd95500ff16bf13ba04ec9bf21f486bd8aae9930d59a5297354e7079d7927e10e88097bffed5c870c53144ce419aaafd5a8934a14bdc72a5f252b'},
        {'username': 'SandBoxUser4',
         'password': '9978fc21ebc673db54dab722b445389583d349b17c77cd293a3ffd15a6f0c9fbebf301d887924271db9069ce4af89168c27f6ee686f68300c6e2316a7610542082256a271c19b968dd23cefbe9c4e00aa9cf2c477248972edf9b19044bfd48354edac68f86b87f244eedfe42f622fa457e7a5debef3d005bb5a7ddc71a14d2a0fe10848bb1e503ea7144f42c823615d18e058a78e6173e2f5d98b356c5169037a298a12c71e9449ab69254133e3d9cac26291e7f0ab67b0418327ee0b44480d420df8eb47d1f2b7f8669d2a5514904124ab8dd175dcccdfd4f407bae4db628516ccb79d2c0a42b6be9070d2a33d6f365ed53b50cb2e736ce606c40cc24de405a'},
        {'username': 'SandBoxUser5',
         'password': '4b383ff63feb386919832f0ebf5827ea574aa3f0693c7b21b56e52d5a236ca7b021293d9dc3d9e5afa7b31220f46ec1247151989d1fe6620516233a9286f26eba390dc955c41abe0ca2f5199a31d3deb520fd3208a30269d4dc684a7475b91a1f852c93c30fb4632b0acf424594eebb5b4d1d2ad695c4f4e128c32be3dec0568f9f2503c6dba05318a2a78a677e81610c8dcc4c7761beca2b4d964978802ee964f08d70cf9ae706da4e87d9f17f0a2b477c8e31967b86d529069730195ce1535f46e9b4f73705208ee8dccd274b62ba909d2d31e021e110cf7ef7fd447cb1b20dda92dcc1bb28c7122d10d707a73dc8292c2859c1da6b43f1e586387dcc0552a'}
    ]
    return logins


logins = get_logins()


def step1GetAccessToken():
    encode_key = client_id + ':' + client_key
    authorization = 'Basic ' + str(base64.b64encode(encode_key.encode('utf-8')), 'utf-8')

    payload = {
        'grant_type': 'client_credentials',
        'scope': '/api'
    }
    headers = {
        'accept': "application/json",
        'authorization': authorization,
        'content-type': "application/x-www-form-urlencoded"
    }
    r = requests.post("https://sandbox.apihub.citi.com/gcb/api/clientCredentials/oauth2/token/hk/gcb", data=payload,
                      headers=headers)
    data = json.loads(r.text)
    context.access_token = data['access_token']
    return data['access_token']


def step2GetBizToken():
    authorization = "Bearer " + context.access_token
    u = str(uuid.uuid1())
    headers = {
        'authorization': authorization,
        'client_id': client_id,
        'uuid': u,
        'content-type': "application/json"
    }
    r = requests.get("https://sandbox.apihub.citi.com/gcb/api/security/e2eKey", headers=headers)
    text = json.loads(r.text)
    modulus = text['modulus']
    exponent = text['exponent']
    bizToken = r.headers['bizToken']
    eventId = r.headers['eventId']
    context.eventId = eventId
    context.bizToken = bizToken
    return {
        'modulus': modulus,
        'exponent': exponent,
        'bizToken': bizToken,
        'eventId': eventId
    }


def step3GetRealAccessToken(user_index):
    encode_key = client_id + ':' + client_key
    authorization = 'Basic ' + str(base64.b64encode(encode_key.encode('utf-8')), 'utf-8')
    username = logins[user_index]['username']
    password = logins[user_index]['password']
    u = str(uuid.uuid1())

    payload = {
        'grant_type': 'password',
        'scope': '/api',
        'username': username,
        'password': password
    }
    headers = {
        'authorization': authorization,
        'bizToken': context.bizToken,
        'uuid': u,
        'content-type': "application/x-www-form-urlencoded",
        'accept': 'application/json'
    }
    r = requests.post("https://sandbox.apihub.citi.com/gcb/api/password/oauth2/token/hk/gcb", data=payload,
                      headers=headers)
    return json.loads(str(r.text))['access_token']


def getAccounts(access_token):# access_token = step3GetRealAccessToken(i)
    #     global access_token
    authorization = 'Bearer ' + access_token
    u = str(uuid.uuid1())
    headers = {
        'authorization': authorization,
        'client_id': client_id,
        'uuid': u,
        'content-type': "application/json",
        'accept': 'application/json'
    }
    r = requests.get("https://sandbox.apihub.citi.com/gcb/api/v1/accounts", headers=headers)
    return r.text


def retrievePayeeList(access_token):  # access_token = step3GetRealAccessToken(i)
    authorization = 'Bearer ' + access_token
    u = str(uuid.uuid1())
    headers = {
        'authorization': authorization,
        'client_id': client_id,
        'uuid': u,
        'accept': 'application/json'
    }
    params = {
        'paymentType': 'ALL'
    }
    r = requests.get("https://sandbox.apihub.citi.com/gcb/api/v1/moneyMovement/payees", params=params, headers=headers)
    return r.text


def retrieveDestSrcAcct(access_token):  # access_token = step3GetRealAccessToken(i)
    authorization = 'Bearer ' + access_token
    u = str(uuid.uuid1())
    headers = {
        'authorization': authorization,
        'client_id': client_id,
        'uuid': u,
        'accept': 'application/json'
    }
    params = {
        'paymentType': 'ALL'
    }
    r = requests.get("https://sandbox.apihub.citi.com/gcb/api/v1/moneyMovement/payees/sourceAccounts", params=params,
                     headers=headers)
    print(r.text)


def my_retrieveDestSrcAcct(access_token):
    authorization = 'Bearer ' + access_token
    u = str(uuid.uuid1())
    headers = {
        'authorization': authorization,
        'client_id': client_id,
        'uuid': u,
        'accept': 'application/json'
    }
    params = {
        'paymentType': 'ALL'
    }
    r = requests.get("https://sandbox.apihub.citi.com/gcb/api/v1/moneyMovement/payees/sourceAccounts", params=params,
                     headers=headers)
    text = json.loads(r.text)
    return text['sourceAccounts']


def createPersonalTransfer(sourceAccountId, access_token, invest_amount):
    authorization = 'Bearer ' + access_token
    u = str(uuid.uuid1())
    payload = {
        "sourceAccountId": sourceAccountId,
        "transactionAmount": invest_amount,
        "transferCurrencyIndicator": "SOURCE_ACCOUNT_CURRENCY",
        "payeeId": "7977557255484c7345546c4e53424766634b6c53756841672b556857626e395253334b70416449676b42673d",
        "chargeBearer": "BENEFICIARY",
        "fxDealReferenceNumber": "12345678",
        "remarks": "DOM Personal Transfer"
    }
    headers = {
        'authorization': authorization,
        'client_id': client_id,
        'uuid': u,
        'accept': 'application/json',
        'content-type': 'application/json'
    }
    r = requests.post("https://sandbox.apihub.citi.com/gcb/api/v1/moneyMovement/personalDomesticTransfers/preprocess",
                      data=json.dumps(payload), headers=headers)
    text = json.loads(r.text)
    return confirmInternalTransfer(text['controlFlowId'])


def confirmPersonalTransfer(controlFlowId, access_token):
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
    r = requests.post("https://sandbox.apihub.citi.com/gcb/api/v1/moneyMovement/personalDomesticTransfers",
                      data=json.dumps(payload), headers=headers)
    text = json.loads(r.text)
    return text


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
        'accept': 'application/json',
        'content-type': 'application/json'
    }
    r = requests.post("https://sandbox.apihub.citi.com/gcb/api/v1/moneyMovement/internalDomesticTransfers/preprocess",
                      data=json.dumps(payload), headers=headers)
    text = json.loads(r.text)
    return confirmInternalTransfer(text['controlFlowId'])


def confirmInternalTransfer(controlFlowId, access_token):
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
    r = requests.post("https://sandbox.apihub.citi.com/gcb/api/v1/moneyMovement/internalDomesticTransfers",
                      data=json.dumps(payload), headers=headers)
    text = json.loads(r.text)
    return text




