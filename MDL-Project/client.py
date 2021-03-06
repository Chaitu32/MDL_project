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

def submit(id, vector):
    """
    used to make official submission of your weight vector
    returns string "successfully submitted" if properly submitted.
    """
    for i in vector: assert 0<=abs(i)<=10
    assert len(vector) == MAX_DEG
    return send_request(id, vector, 'submit')

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
    return np.array([(get_errors(Secret_key,list(vector))) for vector in population_vector])

def fitness_population(population_vector):
    population_errors = get_population_errors(population_vector)
    normal_error = population_errors[:,0] * (2/3) + population_errors[:,1] *(1/3)
    population_inverse = np.reciprocal(normal_error+1)
    new_population = storebest_10(population_vector,population_errors,population_inverse)
    return population_inverse,new_population

def cross_over(p_vector):
    return list()

def mutation(p_vector):
    return list()

def get_new_generation(population,p_inverse):
    half_num = int(population.shape[0]/2)
    new_genaration = np.zeros((half_num,population.shape[1]))
    survived_population = population[0:half_num]
    relative_probabilty = p_inverse[0:half_num]
    relative_probabilty = relative_probabilty / np.sum(relative_probabilty)
    for i in range(int(half_num/2)):
        parents = np.random.choice(survived_population,2,replace=False,p =relative_probabilty)
        children = cross_over(parents)
        new_genaration[i*2] = children[0]
        new_genaration[i*2+1] = children[1]
    new_genaration = mutation(new_genaration)
    population[half_num:] = new_genaration
    
    return population


def storebest_10(p_v,p_e,p_per):
    index_p = np.argsort(p_per)
    p_v = p_v[index_p[::-1]]
    p_e = p_e[index_p[::-1]]
    p_per = p_per[index_p[::-1]]
    file = open("my_outputs.txt","a")
    L = [str(p_v),str(p_e),str(p_per)]
    file.write("<--------------Start--------------->")
    file.writelines(L)
    file.write("<---------------End---------------->")
    file.close()
    return p_v


# Replace 'SECRET_KEY' with your team's secret key (Will be sent over email)
if __name__ == "__main__":
    initial_vector = get_overfit_vector(Secret_key)
    population = genarate_population(initial_vector,10)
    print(get_population_errors(population))
    p_per,p_newv = fitness_population(population)
    print(p_per)
    v = submit(Secret_key,initial_vector)