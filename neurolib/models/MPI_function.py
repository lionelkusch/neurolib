import time
import os
from mpi4py import MPI
import numpy as np
import sys

## MPI function for receive and send data
def init_mpi(path, logger):
    """
    initialise MPI connection
    :param path: path of the file for the port
    :param logger: logger of the modules
    :return:
    """
    while not os.path.exists(path + '.unlock'):  # FAT END POINT
        logger.info(path + '.unlock')
        logger.info(" not found yet, retry in 1 second")
        time.sleep(1)
    os.remove(path + '.unlock')
    fport = open(path, "r")
    port = fport.readline()
    fport.close()
    logger.info("wait connection " + port)
    comm = MPI.COMM_WORLD.Connect(port)
    logger.info('connect to ' + port)
    return comm


def send_mpi(comm, times, data, logger):
    """
    send mpi data
    :param comm: MPI communicator
    :param times: times of values
    :param data: rates inputs
    :param logger: logger of the modules
    :return:nothing
    """
    logger.info("start send")
    status_ = MPI.Status()
    # wait until the transformer accept the connections
    accept = False
    while not accept:
        req = comm.irecv(source=0, tag=0)
        accept = req.wait(status_)
        logger.info("receive accept")
    source = status_.Get_source()  # the id of the excepted source
    logger.info("get source : "+str(source))
    data = np.ascontiguousarray(data, dtype='d')  # format the rate for sending
    shape = np.array(data.shape[0], dtype='i')  # size of data
    times = np.array(times, dtype='d')  # time of starting and ending step
    print("time :",times)
    comm.Send([times, MPI.DOUBLE], dest=source, tag=0)
    comm.Send([shape, MPI.INT], dest=source, tag=0)
    comm.Send([data, MPI.DOUBLE], dest=source, tag=0)
    logger.info("end send")


def receive_mpi(comm, logger):
    """
        receive proxy values the
    :param comm: MPI communicator
    :param logger: logger of the modules
    :return: rate of all proxy
    """
    logger.info("start receive")
    status_ = MPI.Status()
    # send to the transformer : I want the next part
    req = comm.isend(True, dest=0, tag=0)
    req.wait()
    time_step = np.empty(2, dtype='d')
    comm.Recv([time_step, 2, MPI.DOUBLE], source=0, tag=MPI.ANY_TAG, status=status_)
    # get the size of the rate
    size = np.empty(1, dtype='i')
    comm.Recv([size, MPI.INT], source=0, tag=0)
    # get the rate
    rates = np.empty(size, dtype='d')
    comm.Recv([rates, size, MPI.DOUBLE], source=0, tag=MPI.ANY_TAG, status=status_)
    logger.info("end receive " + str(time_step))
    # print the summary of the data
    if status_.Get_tag() == 0:
        return time_step, rates
    else:
        return None


def end_mpi(comm, path, sending, logger):
    """
    ending the communication
    :param comm: MPI communicator
    :param path: for the close the port
    :param sending: if the transformer is for sending or receiving data
    :param logger: logger of the module
    :return: nothing
    """
    # read the port before the deleted file
    fport = open(path, "r")
    port = fport.readline()
    fport.close()
    # different ending of the transformer
    if sending:
        logger.info("close connection send " + port)
        sys.stdout.flush()
        status_ = MPI.Status()
        # wait until the transformer accept the connections
        logger.info("send check")
        accept = False
        while not accept:
            req = comm.irecv(source=0, tag=0)
            accept = req.wait(status_)
        logger.info("send end simulation")
        source = status_.Get_source()  # the id of the excepted source
        times = np.array([0., 0.], dtype='d')  # time of starting and ending step
        comm.Send([times, MPI.DOUBLE], dest=source, tag=1)
    else:
        logger.info("close connection receive " + port)
        # send to the transformer : I want the next part
        req = comm.isend(True, dest=0, tag=1)
        req.wait()
    logger.info("Barrier")
    comm.Barrier()
    # closing the connection at this end
    logger.info("disconnect communication")
    comm.Disconnect()
    logger.info("close " + port)
    MPI.Close_port(port)
    logger.info("close connection " + port)
    return

## Function for reshape the result of Neurolib
def reshape_result(result):
    """
    reshape the output of Neurolib for the
    :param result: output of Neurolib
    :return:
    """
    times = []
    values = []
    for (time, value) in result[0]:
        if time > 0.0:
            times.append(time)
            values.append(value)
    return ([np.array(times), np.expand_dims(np.concatenate(values), 1)],)
