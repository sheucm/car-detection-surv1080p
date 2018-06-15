
class Counter (object):

    def __init__ (self):
        self.pre_id_list = list()
        self.total = 0

    def update (self, id_list):
        old_ids = list()
        new_ids = list()
        for id in id_list:
            if id in self.pre_id_list:
                old_ids.append(id)
            else:
                new_ids.append(id)

        self.total += len(new_ids)
        self.pre_id_list = id_list
        return self.total