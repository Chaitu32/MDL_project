import json
import requests
import numpy as np
from random import sample

API_ENDPOINT = 'http://10.4.21.156'
MAX_DEG = 11
B10_V = np.zeros((10,MAX_DEG))
B10_E = np.zeros((10))

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
    return population

def get_population_errors(population_vector):
    return np.array([(get_errors(Secret_key,list(vector))) for vector in population_vector])

def fitness_population(population_vector):
    population_errors = get_population_errors(population_vector)
    normal_error = population_errors[:,0] * (2/3) + population_errors[:,1] *(1/3)
    population_inverse = np.reciprocal(normal_error+1)
    new_population = storebest_10(population_vector,population_errors,population_inverse)
    return population_inverse,new_population

def storebest_10(p_v,p_e,p_per):
    global B10_V,B10_E
    index_p = np.argsort(p_per)
    p_v = p_v[index_p[::-1]]
    p_e = p_e[index_p[::-1]]
    p_per = p_per[index_p[::-1]]

    if np.array_equal(B10_V,np.zeros((10,MAX_DEG))):
        B10_V = p_v[0:10]
        B10_E = p_e[0:10]
    else :
        temp_v = np.concatenate((p_v[0:10],B10_V))
        temp_errors = np.concatenate((p_e[0:10],B10_E))
        temp_indexs = np.argsort(temp_errors[:,0])
        temp_v = temp_v[temp_indexs[::-1]]
        temp_errors = temp_errors[temp_indexs[::-1]] 
        B10_V = temp_v[-10:]
        B10_E = temp_errors[-10:]
        B10_V = B10_V[::-1]
        B10_E = B10_E[::-1]

    file = open("my_generations4.txt","a")
    L = [str(p_v),str(p_e),str(p_per)]
    file.write("\n<--------------Start--------------->\n")
    file.write(L[0])
    file.write("\n")
    file.write(L[1])
    file.write("\n")
    file.write(L[2])
    file.write("\n<---------------End---------------->\n")
    file.write("\n<----------Best vectors------->\n")
    file.write(str(B10_V))
    file.write("\n")
    file.write(str(B10_E))
    file.write("\n<---------End--------->\n")
    file.close()
    return p_v

def cross_over(parents):
    children = np.array(parents)
    no_of_places = np.random.randint(low=1,high=parents.shape[1],size=1)
    selected_places = np.random.choice(parents.shape[1],no_of_places,replace=False)
    for i in selected_places:
        temp = children[0][i]
        children[0][i]=children[1][i]
        children[1][i]=temp
    return children

def mutation(p_vector):
    return list()

def get_new_generation(population,p_inverse):
    half_num = int(population.shape[0]/2)
    new_genaration = np.zeros((half_num,population.shape[1]))
    survived_population = population[0:half_num]
    relative_probabilty = p_inverse[0:half_num]
    relative_probabilty = relative_probabilty / np.sum(relative_probabilty)
    temp = list()
    for i in range(int(half_num/2)):
        parents_index = np.random.choice(survived_population.shape[0],2,replace=False,p =relative_probabilty)
        check=0
        for j in temp:
            if (j==parents_index).all():
                i -=1
                check=1
                break
        if check==1:
            continue
        temp.append(parents_index)
        children = cross_over(survived_population[parents_index])
        new_genaration[i*2] = children[0]
        new_genaration[i*2+1] = children[1]
    #new_genaration = mutation(new_genaration)
    population[half_num:] = new_genaration
    
    return population





# Replace 'SECRET_KEY' with your team's secret key (Will be sent over email)
if __name__ == "__main__":
    initial_vector = get_overfit_vector(Secret_key)
    population = genarate_population(initial_vector,20)
    #print(get_population_errors(population))
    new_p = population
    for _ in range(0,2):
        p_inv,p_newv = fitness_population(new_p)
        print(p_inv)
        new_p = get_new_generation(p_newv,p_inv)
    p_inv,p_newv = fitness_population(new_p)
    print(B10_V)
    print(B10_E)
    print(B10_V.shape)
    #submit(Secret_key,list(B10_V[0]))