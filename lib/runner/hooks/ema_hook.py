def get_ori_key(key):
    ori_key = key.split('.')
    ori_key[0] = ori_key[0][:-4]
    ori_key = '.'.join(ori_key)
    return ori_key
