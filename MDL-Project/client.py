import json
import requests
import numpy as np
from random import sample,uniform

API_ENDPOINT = 'http://10.4.21.156'
MAX_DEG = 11
MAX_POPULATION =16
B10_V = np.zeros((10,MAX_DEG))
B10_E = np.zeros((10))
Generations = [{
    "vectors":np.array([]),
    "vectors_error":np.array([]),
    "vectors_percent":np.array([])
}]
cur_generation = 0


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
    return new_population

def storebest_10(p_v,p_e,p_per):
    global B10_V,B10_E,Generations,cur_generation
    #if cur_generation!=0:
    percents = np.concatenate((Generations[cur_generation]["vectors_percent"],p_per))
    vectors =  np.concatenate((Generations[cur_generation]["vectors"],p_v))
    errors = np.concatenate((Generations[cur_generation]["vectors_error"],p_e))
    # else:
    #     percents = p_per
    #     vectors = p_v
    #     errors = p_e
    index_p = np.argsort(percents)
    vectors = vectors[index_p[::-1]]
    errors = errors[index_p[::-1]]
    percents = percents[index_p[::-1]]
    index2 = np.array(np.unique(vectors,axis=0,return_index=True)[1])
    index2 = np.sort(index2)
    vectors = vectors[index2]
    errors = errors[index2]
    percents = percents[index2]
    Generations.append({"vectors":np.array([]),
            "vectors_error":np.array([]),
            "vectors_percent":np.array([])
            })
    cur_generation +=1
    Generations[cur_generation]["vectors"] = vectors[0:MAX_POPULATION]
    Generations[cur_generation]["vectors_error"] = errors[0:MAX_POPULATION]
    Generations[cur_generation]["vectors_percent"] = percents[0:MAX_POPULATION]

    B10_V = vectors[0:10]
    B10_E = errors[0:10]
    name = "./output6/generations"+str(cur_generation)
    with open(name+".json",'w+') as file:
        temp = {"vectors":Generations[cur_generation]['vectors'].tolist(),
            "vectors_error":Generations[cur_generation]['vectors_error'].tolist(),
            "vectors_percent":Generations[cur_generation]['vectors_percent'].tolist()
            }
        json.dump(temp, file)
        file.close()
    with open(name+".txt",'w') as file:
        L = [str(vectors),str(errors),str(percents)]
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
    return Generations[cur_generation]["vectors"]

def cross_over(parents):
    children = np.array(parents)
    no_of_places = np.random.randint(low=3,high=MAX_DEG,size=1)
    selected_places = np.random.choice(parents.shape[1],no_of_places,replace=False)
    for i in selected_places:
        temp = children[0][i]
        children[0][i]=children[1][i]
        children[1][i]=temp
    return children

def mutation(p_vector):
    num = np.random.randint(low=0,high=100,size=p_vector.shape[0])
    for i in range(0,num.shape[0]):
        if num[i]<30:
            #print(cur_generation,i)
            place = np.random.randint(low=0,high=MAX_DEG,size=1)
            #print(place)
            num_muiltple = uniform(-2,2)
            temp = p_vector[i][place] *num_muiltple
            if abs(temp-p_vector[i][place]) < 0.1 :
                temp += uniform(-2,2)
            if temp<-10 or temp>10 :
                temp = uniform(-10,10)
            p_vector[i][place] =temp
    return p_vector

def get_new_generation():
    global Generations
    population = np.array(Generations[cur_generation]["vectors"])
    p_inverse = np.array(Generations[cur_generation]["vectors_percent"])
    half_num = int(MAX_POPULATION/2)
    new_genaration = np.zeros((half_num,MAX_DEG))
    survived_population = np.array(population[0:half_num])
    relative_probabilty = p_inverse[0:half_num]
    relative_probabilty = relative_probabilty / np.sum(relative_probabilty)
    temp = list()
    x = int(half_num/2)
    while(x!=0):
        x-=1
        parents_index = np.random.choice(survived_population.shape[0],2,replace=False,p =relative_probabilty)
        parents_index = np.sort(parents_index)
        temp.append(parents_index)
        children = cross_over(survived_population[parents_index])
        new_genaration[x*2] = children[0]
        new_genaration[x*2+1] = children[1]
    new_genaration = mutation(new_genaration)
    
    return new_genaration





# Replace 'SECRET_KEY' with your team's secret key (Will be sent over email)
if __name__ == "__main__":
    #initial_vector = get_overfit_vector(Secret_key)
    #population = genarate_population(initial_vector,MAX_POPULATION)
    #print(get_population_errors(population))
    with open('output5/generations30.json','r') as file:
        data = json.load(file)
    Generations[cur_generation]['vectors'] = np.array(data['vectors'])
    Generations[cur_generation]['vectors_error'] = np.array(data['vectors_error'])
    Generations[cur_generation]['vectors_percent'] = np.array(data['vectors_percent'])

    for _ in range(0,30):
        new_p = get_new_generation()
        #print(old_p)
        old_p = fitness_population(new_p)
    #p_newv = fitness_population(new_p)
    print(B10_V)
    print(B10_E)
    print(B10_V.shape)
    submit(Secret_key,list(B10_V[0]))