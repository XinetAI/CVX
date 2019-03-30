import lmdb

class LmdbProgress:
    def __init__(self, root):
        self.root = root
        self.env = lmdb.open(self.root)
        
    def context(self):
        return self.env.begin(write = True) # 创建事务，并写入
        
    def toByte(self, inputs):
        if isinstance(inputs, int):
            return str(inputs).encode()
        elif isinstance(inputs, bytes):
            return inputs
        else:
            return inputs.encode()
        
    def insert(self, sid, name):
        sid = self.toByte(sid)
        name = self.toByte(name)
        txn = self.context()
        txn.put(sid, name)
        txn.commit()
        
    def delete(self, sid):
        txn = self.context()
        sid = self.toByte(sid)
        txn.delete(sid)
        txn.commit()
        
    def update(self, sid, name):
        txn = self.context()
        sid = self.toByte(sid)
        name = self.toByte(name)
        txn.put(sid, name)
        txn.commit()

    def search(self, sid):
        txn = self.env.begin()
        sid = self.toByte(sid)
        name = txn.get(sid)
        return name

    def display(self):
        txn = self.env.begin()
        cur = txn.cursor()
        for key, value in cur:
            print((key, value))
            
    def close(self):
        self.env.close()
        
    def reinit(self):
        '''
        重新打开 lmdb
        '''
        self.env = lmdb.open(self.root)