import h5py


def read_in_data(path):
    data = h5py.File(path, 'r')
    result = []
    for i in list(data.keys()):
        images = data.get(i + '/images')[()]
        hist = data.get(i + '/hist')[()]
        labels = data.get(i + '/labels')[()]
        values = data.get(i + '/values')[()]
        folder = data.get(i + '/folder')[()]
        result.append([images, hist, labels, values, folder])
    return result
