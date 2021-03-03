import json
import requests
import numpy as np
from random import sample

API_ENDPOINT = 'http://10.4.21.156'
MAX_DEG = 11

Secret_key = 'hckY8xfUMEtaX9mfpahSAngELplyFXTNXKyXPwYKlnhMAPsocG'

def urljoin(root, path=''):
    if path: root = '/'.join([root.rstrip('/'), path.rstrip('/')])
    return root

def send_request(id, vector, path):
    api = urljoin(API_ENDPOINT, path)
    vector = json.dumps(vector)
    response = requests.post(api, data={'id':id, 'vector':vector}).text
    if "reported" in response:
        print(response)
        exit()

    return response

def get_errors(id, vector):
    for i in vector: assert 0<=abs(i)<=10
    assert len(vector) == MAX_DEG

    return json.loads(send_request(id, vector, 'geterrors'))

def get_overfit_vector(id):
    return json.loads(send_request(id, [0], 'getoverfit'))

def genarate_population(initial_vector,popul_size):
    population = np.random.uniform(low=-10,high=10,size=(popul_size-1,len(initial_vector)))
    initial_vector = np.array(initial_vector)
    initial_vector = initial_vector.reshape(1,-1)
    population = np.append(initial_vector, population,axis=0)
    print (population)
    return population

def get_population_errors(population_vector):
    return np.array([get_errors(Secret_key,vector) for vector in population_vector])

def fitness_population(population_vector):
    population_errors = get_population_errors(population_vector)
    normal_error = population_errors[:,0] * (2/3) + population_errors[:,1] *(1/3)
    population_percent = np.array([(pow(2,-1*(x/1000))*100 for x in normal_error )])
    storebest_10(population_vector,population_errors,population_percent)
    return population_percent

def cross_over(p_vector):
    return list()

def mutation(p_vector):
    return list()

def storebest_10(p_v,p_e,p_per):
    index_p = np.argsort(p_per)
    p_v = p_v[index_p[::,-1]]
    p_e = p_e[index_p[::,-1]]
    p_per = p_per[index_p[::,-1]]
    file = open("my_outputs.txt","a")
    L = [p_v,p_e,p_per]
    file.writelines(L)
    file.close()


# Replace 'SECRET_KEY' with your team's secret key (Will be sent over email)
if __name__ == "__main__":
    initial_vector = get_overfit_vector(Secret_key)
    population = genarate_population(initial_vector,20)
    p_per = fitness_population(population)
    print(p_per)