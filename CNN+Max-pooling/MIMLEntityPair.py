# -*- coding: utf-8 -*-

import ConfigParser

ini_filename = 'ModelControl.ini'
conf = ConfigParser.ConfigParser()
conf.read(ini_filename)
_NA_AS_0s = conf.getboolean('mode', '_NA_AS_0s')

class MIMLEntityPair:
    ATT_SEP = "\t@\t"
    ER_SEP = "----------"
    MCOUNT_ALL = 0
    EPCOUNT_ALL = 0
    REL2ID_DICT = {'NA': 0} if not _NA_AS_0s else {}     # 如果将NA看作label, 那么label_index=0
    FEAT2ID_DICT = {}  # feature-idx

    def __init__(self, lines, ds_split, read_features = 0):
        ep_atts = lines[0].split(MIMLEntityPair.ATT_SEP)
        self.ep_sn = ep_atts[0].strip()
        self.ds_split = ds_split
        self.relations = ep_atts[1].strip().split(',')
        for rel in self.relations:
            if ds_split == 'test' and not MIMLEntityPair.REL2ID_DICT.has_key(rel) and rel != 'NA':
                print '    - Find new relation in testset: '+str(rel)
        self.rels_id = MIMLEntityPair.get_id_from_dict(self.relations, MIMLEntityPair.REL2ID_DICT, is_insert=True)
        self.e1_name = ep_atts[2].strip().split("List(")[1].split(")")[0].split(',')[0].strip()
        self.e2_name = ep_atts[3].strip().split("List(")[1].split(")")[0].split(',')[0].strip()
        self.e1_guid = ep_atts[4].strip()
        self.e2_guid = ep_atts[5].strip()
        self.e1_mid = ep_atts[6].strip()
        self.e2_mid = ep_atts[7].strip()
        self.mentions_info = list()

        for line in lines[1:]:
            m_atts = line.strip().split(MIMLEntityPair.ATT_SEP)
            features = (None if read_features == 0 else m_atts[5:])
            features_id_list = (None if features is None else MIMLEntityPair.get_id_from_dict(features, MIMLEntityPair.FEAT2ID_DICT, is_insert=True))
            self.mentions_info.append({
                'm_sn':m_atts[0].strip(),
                'sentence':m_atts[1].strip(),
                'id1':m_atts[2].strip(),
                'id2':m_atts[3].strip(),
                'filename': m_atts[4].strip(),
                'feat_id_list': features_id_list,
                'feats': features,
                'feat_for_loc': m_atts[11].strip(),
            })
            MIMLEntityPair.MCOUNT_ALL += 1
        MIMLEntityPair.EPCOUNT_ALL+=1
        if MIMLEntityPair.EPCOUNT_ALL%500 == 0:
            print '    - EP Count = ',MIMLEntityPair.EPCOUNT_ALL


    def has_same_ep_name(self, ep):
        assert isinstance(ep, MIMLEntityPair)
        if self.e1_guid == ep.e1_guid and self.e2_guid == ep.e2_guid:
            return True


    @staticmethod
    def get_id_from_dict(element_list, ele2id, is_insert):
        id_list = list()
        if is_insert:    # 如果是要新增, 就对每个rel检查在dict中是否已有, 若没有,新增idx; 若有, 取原来的idx.
            for ele in element_list:
                if ele == 'NA' and _NA_AS_0s:    # 如果选择'全0表示NA', 那么: EP的关系为NA则返回空list.
                    continue
                if not ele2id.has_key(ele):
                    ele2id[ele] = len(ele2id)
                id_list.append(ele2id[ele])
        else:    # 如果是要检索
            for ele in element_list:
                if ele in ele2id:
                    id_list.append(ele2id[ele])
                else:
                    id_list.append(None)
        return id_list


if __name__ == '__main__':
    file_path = "Z:/train-Multiple-out"
#    eps_with_mentions = read_from_file(file_path)
#    input_file =
#    for ep in eps_with_mentions:
#        sentences = [mention.sent for mention in ep]
#        ep_data = {"labels":ep.rels_idx_list,
#                   "e1":ep.e1mid,
#                   "e2":ep.e3mid,
#                   "sentences"                   }









