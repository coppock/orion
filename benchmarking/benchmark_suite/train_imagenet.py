import argparse
import itertools
import json
import os
import random
import threading
import time
from ctypes import cdll

import numpy as np
import torch
from torchvision import models, datasets, transforms


def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


class DummyDataLoader():
    def __init__(self, batchsize):
        self.batchsize = batchsize
        self.data = torch.rand([self.batchsize, 3, 224, 224], pin_memory=True)
        self.target = torch.ones(
            [self.batchsize], pin_memory=True, dtype=torch.long)

    def __iter__(self):
        return self

    def __next__(self):
        return self.data, self.target


class RealDataLoader():
    def __init__(self, batchsize):
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
        )
        train_dataset = \
            datasets.ImageFolder(
                "/mnt/data/home/fot/imagenet/imagenet-raw-euwest4", transform=train_transform)
        self.train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batchsize, num_workers=8)

    def __iter__(self):
        print("Inside iter")
        return iter(self.train_loader)


def work(is_training):
    if is_training:
        optimizer.zero_grad()
        output = model(gpu_data)
        loss = criterion(output, gpu_target)
        loss.backward()
        optimizer.step()


def imagenet_loop(
    model_name,
    batchsize,
    train,
    num_iters,
    rps,
    uniform,
    dummy_data,
    local_rank,
    barrier,
    tid,
    input_file=''
):
    seed_everything(42)
    print(model_name, batchsize, local_rank, barrier, tid)
    backend_lib = cdll.LoadLibrary(os.getenv('LD_PRELOAD'))
    if rps > 0 and not input_file:
        sleep_times = [
            1/rps]*num_iters if uniform else np.random.exponential(scale=1/rps, size=num_iters)
    elif input_file:
        with open(input_file) as f:
            sleep_times = json.load(f)
    else:
        sleep_times = [0]*num_iters

    print(f"SIZE is {len(sleep_times)}")
    barrier.wait()

    print("-------------- thread id:  ", threading.get_native_id())

    if train and tid == 1:
        time.sleep(5)

    # data = torch.rand([batchsize, 3, 224, 224]).contiguous()
    # target = torch.ones([batchsize]).to(torch.long)
    model = models.__dict__[model_name](num_classes=1000)
    model = model.to(0)

    if train:
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        criterion = torch.nn.CrossEntropyLoss().to(local_rank)
    else:
        model.eval()

    train_loader = DummyDataLoader(
        batchsize) if dummy_data else RealDataLoader(batchsize)

    print("Enter loop!")

    # open loop
    next_startup = time.time()

    timings = []
    for batch_idx, batch in enumerate(itertools.islice(train_loader, num_iters)):
        # torch.cuda.profiler.cudart().cudaProfilerStart()
        print(f"Client {tid}, submit!, batch_idx is {batch_idx}")
        gpu_data, gpu_target = batch[0].to(local_rank), batch[1].to(local_rank)
        if train:
            # client_barrier.wait()
            # if tid==0 and batch_idx==20:
            #     torch.cuda.profiler.cudart().cudaProfilerStart()
            optimizer.zero_grad()
            output = model(gpu_data)
            loss = criterion(output, gpu_target)
            loss.backward()
            optimizer.step()
            backend_lib.block(batch_idx)
            iter_time = time.time() - next_startup
            timings.append(iter_time)
            # print(f"Client {tid} finished! Wait! It took {timings[batch_idx]}")
            if batch_idx == 10:  # for warmup
                barrier.wait()
                start = time.time()
            # if batch_idx==20:
            #     torch.cuda.profiler.cudart().cudaProfilerStart()
        else:
            with torch.no_grad():
                cur_time = time.time()
                if (cur_time >= next_startup):
                    if batch_idx == 100:
                        torch.cuda.profiler.cudart().cudaProfilerStart()
                    output = model(gpu_data)
                    if backend_lib: backend_lib.block(batch_idx)
                    # if batch_idx==250:
                    #     torch.cuda.profiler.cudart().cudaProfilerStop()
                    req_time = time.time()-next_startup
                    timings.append(req_time)
                    print(f"Client {tid} finished! Wait! It took {req_time}")
                    if batch_idx >= 10:
                        next_startup += sleep_times[batch_idx]
                    else:
                        next_startup = time.time()
                    if batch_idx == 10:
                        barrier.wait()
                        # hp starts after
                        if (batch_idx == 10):
                            next_startup = time.time()
                            start = time.time()
                    dur = next_startup-time.time()
                    if (dur > 0):
                        time.sleep(dur)
        if backend_lib and backend_lib.stop():
            print("---- STOP!")
            break

        print(f"Client {tid} at barrier!")
        barrier.wait()
        total_time = time.time() - start

        timings = timings[10:]
        timings = sorted(timings)

        if not train and len(timings) > 0:
            p50 = np.percentile(timings, 50)
            p95 = np.percentile(timings, 95)
            p99 = np.percentile(timings, 99)
            print(f"Client {tid} finished! p50: {
                  p50} sec, p95: {p95} sec, p99: {p99} sec")
            data = {
                'p50_latency': p50*1000,
                'p95_latency': p95*1000,
                'p99_latency': p99*1000,
                'throughput': (batch_idx-10)/total_time
            }
        else:
            data = {
                'throughput': (batch_idx-10)/total_time
            }
        with open(f'client_{tid}.json', 'w') as f:
            json.dump(data, f)

        print("Finished! Ready to join!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='resnet50')
    parser.add_argument('--batchsize', type=int, default=4)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--num-iters', type=int, default=100)
    parser.add_argument('--rps', type=float, default=30)
    parser.add_argument('--uniform', action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument('--dummy_data',
                        action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--input_file', default='')
    args = parser.parse_args()

    imagenet_loop(
        args.model,
        args.batchsize,
        args.train,
        args.num_iters,
        args.rps,
        args.uniform,
        args.dummy_data,
        args.local_rank,
        torch.multiprocessing.Barrier(1),
        0,
        args.input_file,
    )
