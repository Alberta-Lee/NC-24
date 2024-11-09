import asyncio
import numpy as np
import random
import demjson
import csv
from decimal import Decimal
import time
import scipy
from scipy.linalg import fractional_matrix_power

from gendata import init_vars, gen_synthetic_fourier, gen_data
from network import matrix_to_graph
from utils import predict_label_by_sign, cal_loss_and_gradient


class Admm:
    def __init__(self, id, host, port, t, w_t, **args):
        self.id = id
        self.host = host
        self.port = port
        self.t = t
        self.w_t = w_t
        self.num = args['num']
        self.error_count = 0
        self.error_rate = []
        self.f1 = []

        xs, ys = x.__next__()
        d = len(xs)
        self.node_nums = d

        self.names = self.__dict__
        names = self.__dict__
        for i in range(1, args['num']+1):
            names['node' + str(self.id)+'_neighbor'+str(i)+'_host'] = args[f'node{self.id}_neighbor{i}_host']
            names['node' + str(self.id) + '_neighbor' + str(i) + '_port'] = args[f'node{self.id}_neighbor{i}_port']
            names['node' + str(self.id) + '_neighbor' + str(i) + '_hat'] = args[f'node{self.id}_neighbor{i}_hat']
            names['node' + str(self.id) + '_neighbor' + str(i) + '_t'] = self.t

    async def server_init(self):
        self.server = await asyncio.start_server(self.server_callback, self.host, self.port)    # starting socket service
        addr = self.server.sockets[0].getsockname()
        print(f'Node {self.id} serving on {addr}')

        async with self.server:
            await self.server.serve_forever()

    async def run(self):
        server_coro = asyncio.create_task(self.server_init())   # create server concurrent
        await asyncio.sleep(0.01)
        client_coro = asyncio.create_task(self.client_method()) # create client concurrent
        await server_coro
        await client_coro

    def gen_multi_data(self, option):
        if option == 'synthetic_fourier':
            mu = 0  # mean
            sigma = 0.1  # standard deviation
            a, theta = init_vars(task_count, mu, sigma)

            global task_id
            
            task_id = np.random.randint(0, task_count)
            xs, ys = gen_synthetic_fourier(a, theta, True, task_id)
            print("round:{}, task_id:{}, label:{}".format(self.t, task_id, ys))
            return xs, ys

    async def server_callback(self, reader, writer):
        self.names = self.__dict__
        while True:
            data = await reader.read(40960)   # bytes
            await asyncio.sleep(random.uniform(0, 0.0001))
            message = data.decode()     # str
            arr_z = list(demjson.decode(message))   # list
            t = int(arr_z.pop())
            fmap = int(arr_z.pop())
            flag = int(num_map[self.id-1, fmap-1])

            self.names['node' + str(self.id) + '_neighbor' + str(flag) + '_t'] = t
            self.names['node' + str(self.id) + '_neighbor' + str(flag) + '_hat'] = arr_z

        print("Close the connection")

    async def client_method(self):
        names = self.__dict__
        self.names = self.__dict__
        for i in range(1, self.num+1):
            names[f'node{self.id}_neighbor{i}_reader'], names[f'node{self.id}_neighbor{i}_writer'] = \
                await asyncio.open_connection(names[f'node{self.id}_neighbor{i}_host'],
                                              names[f'node{self.id}_neighbor{i}_port'], limit=1024)
            print("init", i, names[f'node{self.id}_neighbor{i}_reader'], names[f'node{self.id}_neighbor{i}_writer'])
            await asyncio.sleep(0.1)
        await asyncio.sleep(0.4)

        w = np.zeros((hat.shape[0], task_count))
        z = np.zeros((hat.shape[0], task_count))
        v = np.zeros((hat.shape[0], task_count))
        I = np.identity(task_count)
        omega = (1 / task_count) * I

        TP = 0
        FN = 0
        FP = 0
        TN = 0
        bias = 0.000001

        for nt in range(1, T+1):
            xs, ys = self.gen_multi_data("synthetic_fourier")
            
            w_t = np.concatenate((w, z, v), axis=0)
            w_t = w_t.reshape(3, -1, task_count)

            for i in range(1, self.num+1):
                part = self.names['node' + str(self.id) + '_neighbor' + str(i) + '_hat']
                part = np.array(part)
                part = part.astype(np.float32)
                w_t = w_t + part
            w_t = w_t / (self.num+0.000001)
            self.w_t = w_t
            
            predict, val = predict_label_by_sign(self.w_t[0, :, task_id], xs)
            if predict != ys:
                self.error_count += 1

            if ys == 1:
                if ys != predict:
                    FN += 1
                else:
                    TP += 1
            elif ys == -1:
                if ys != predict:
                    FP += 1
                else:
                    TN += 1
            precision = TP / (TP+FP+bias)
            recall = TP / (TP+FN+bias)

            if nt % 100 == 0:
                print(f'DNode {self.id}  \n t:{nt}  \n error_count:{self.error_count}')

            loss, gradient = cal_loss_and_gradient(xs, ys, self.w_t[0, :, task_id])

            self.w_t[0, :, task_id] = (eta/(rho+eta)) * self.w_t[0, :, task_id] + (rho/(rho+eta)) * (u_t+self.w_t[2, :, task_id]) - (1/(rho+eta)) * (gradient+self.w_t[1, :, task_id])
            
            part1 = (lambda1+lambda3) * np.sum(self.w_t[1], axis=1)
            part2 = (lambda1+lambda3) * rho * np.sum(self.w_t[0], axis=1)
            sum = part1 + part2
            u_t_plus = (1/((lambda1+lambda3)*(lambda2+rho*task_count)+lambda2*rho)) * sum
            
            inv = np.linalg.pinv(omega)
            tmp = self.w_t[2] @ inv + self.w_t[2] @ np.transpose(inv)
            self.w_t[2, :, task_id] = (lambda2/(lambda2*(lambda1+lambda3+rho)+rho*task_count*(lambda1+lambda3))) * (self.w_t[1, :, task_id] + rho * self.w_t[0, :, task_id]) + (lambda4/2)*tmp[:, task_id]

            self.w_t[1, :, task_id] = self.w_t[1, :, task_id] + rho * (self.w_t[0, :, task_id] - u_t_plus - self.w_t[2, :, task_id])

            mat = np.transpose(self.w_t[2]) @ self.w_t[2]
            mul = fractional_matrix_power(mat, 1/2)
            trace = np.trace(mul)
            omega = mul / (trace+0.000001)

            w_t_plus = self.w_t
            message = w_t_plus
            message = message.tolist()
            for i in range(1, self.num + 1):
                names[f'message{i}'] = message[:]
                names[f'message{i}'].append(self.id)
                names[f'message{i}'].append(self.t)
                names[f'message{i}'] = demjson.encode(names[f'message{i}'])
                names[f'node{self.id}_neighbor{i}_writer'].write(names[f'message{i}'].encode())
                await names[f'node{self.id}_neighbor{i}_writer'].drain()
            await asyncio.sleep(random.uniform(0, 0.001))           

            self.t += 1
            self.w_t = w_t_plus
            
            F = (2*precision*recall) / (precision+recall+bias)
            self.f1.append(F)

            rate = self.error_count / self.t
            self.error_rate.append(rate)
            await asyncio.sleep(random.uniform(0, 0.001))

        print(f'Admm_Node {self.id}  \n error_count:{self.error_count} \n error_rate: {self.error_rate[-1]} ')
        print('Close the connection')

async def main(*nodes):
    await asyncio.gather(*[node.run() for node in nodes])

if __name__ == '__main__':

    global T
    global options
    global x
    global num_map
    global task_count

    task_count = 5
    rho = 5
    eta = 70
    lambda1 = 0.01
    lambda2 = 0.01
    lambda3 = 0.01
    lambda4 = 0.01

    T = 5000
    t = 1
    options = 'ring'

    x = gen_data('synthetic_fourier')
    hat, y = x.__next__()
    w = np.zeros((hat.shape[0], task_count))
    z = np.zeros((hat.shape[0], task_count))
    v = np.zeros((hat.shape[0], task_count))
    u_t = np.zeros(hat.shape[0])
    
    w_t = np.concatenate((w, z, v), axis=0)
    w_t = w_t.reshape(3, -1, task_count)

    num_map, G = matrix_to_graph(options)
    n = len(G.nodes())
    names = locals()

    for i in range(1, n + 1):
        D = {}
        D['num'] = G.nodes[i]['num_of_neighbors']
        for j in range(1, D['num'] + 1):
            D[f'node{i}_neighbor{j}_host'] = G.nodes[i][f'node{i}_neighbor{j}_host']
            D[f'node{i}_neighbor{j}_port'] = G.nodes[i][f'node{i}_neighbor{j}_port']
            D[f'node{i}_neighbor{j}_hat'] = w_t
        names[f'node{i}'] = Admm(i, G.nodes[i]['host'], G.nodes[i]['port'], t, w_t, **D)
    if options == 'ring':
        asyncio.run(main(node1, node2, node3, node4, node5))