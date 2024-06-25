from posixpath import split
from MemPlannerAddonUtils import *
from utils import GetGeneralSplitSize, GetTrailingSplitSize

class BaseAddon:
    @classmethod
    # process reqs from op and tensor info to attach addtional_info, preproc() and postproc()
    def attach(cls, reqs, ops, params, tensors):
        raise(BaseException(f"Please use derived class and implement attach()."))

class InplaceAddon(BaseAddon):
    @classmethod
    def isTargetOp(cls, op) -> bool:
        # check input op is target op or not.
        raise(BaseException(f"Please use derived class and implement isTargetOp()."))
    @classmethod
    def GetTargetTensorID(cls, op, shared_params):
        # return (representative tensor's id, [tensor_id of tensors to be merged])
        raise(BaseException(f"Please use derived class and implement isTargetOp()."))
    @classmethod
    def TranslateTIDtoReqID(cls, target_tids, reqs):
        # Translaste the return values from GetTargetTensorID() to Requirment ID.
        raise(BaseException(f"Please use derived class and implement TranslateTIDtoReqID()."))
    @classmethod
    def AttachAdditionalInfo(cls, reqs, target_req_ids, tensors, target_tids):
        # Attach additional info to Requirement for preproc() or postproc() to preform mem planning for inplace implementation.
        raise(BaseException(f"Please use derived class and implement AttachAdditionalInfo()."))
    @classmethod
    def preproc(cls, planner, req_id, addon_id):        
        raise(BaseException(f"Please use derived class and implement preproc()."))
    @classmethod
    def postproc(cls, planner, req_id, addon_id):
        raise(BaseException(f"Please use derived class and implement post()."))
    @classmethod
    def AttachCallback(cls, reqs, target_req_id):
        # Attach callback function(preproc() and postproc()) to Requirement to preform mem planning for inplace implementation.
        raise(BaseException(f"Please use derived class and implement AttachCallback()."))
    @classmethod
    def attach(cls, reqs, ops, params, tensors):
        target_tid = {}
        for op in ops:
            if cls.isTargetOp(op):
                # 選該OP的input或output當代表當key，並把另一方包在list裡當成items
                # plan 好代表後，會嘗試將list裡的tensor plan在同個記憶體區塊
                # split為例：input當key，output tensors當items
                key, items = cls.GetTargetTensorID(op, params)
                if key in target_tid:
                    raise(BaseException("Duplicated key."))
                else:
                    target_tid[key] = items
        target_req_id = cls.TranslateTIDtoReqID(target_tid, reqs)
        cls.AttachAdditionalInfo(reqs, target_req_id, tensors, target_tid)
        cls.AttachCallback(reqs, target_req_id)


class InplaceSplitAddon(InplaceAddon):
    @classmethod
    def isTargetOp(cls, op) -> bool:
        # check input op is target op or not.
        return op.type == 'SPLIT'
    @classmethod
    def GetTargetTensorID(cls, op, shared_params):
        # return (representative tensor's id, [tensor_id of tensors to be merged])
        param = shared_params[op.shared_param_id]
        return (param.input, [param.output])
    @classmethod
    def TranslateTIDtoReqID(cls, target_tids, reqs):
        # Translaste the return values from GetTargetTensorID() to Requirment ID.
        translated_target_req_ids = {}
        for key, val in target_tids.items():
            translated_target_req_ids[FindReqIDByHandle(reqs, (key, -1))] = FindAllSplitReqIDByTid(reqs, val[0])
        return translated_target_req_ids
    @classmethod
    def AttachAdditionalInfo(cls, reqs, target_req_ids, tensors, target_tids):
        # Attach additional info to Requirement for preproc() or postproc() to preform mem planning for inplace implementation.
        for (key_rid, val_rid) , (key_tid, val_tid) in zip(target_req_ids.items(), target_tids.items()):
            additional_info = {}
            additional_info['split_req_ids'] = val_rid
            additional_info['split_height'] = tensors[val_tid[0]].split_height
            additional_info['split_size'] = GetGeneralSplitSize(tensors[val_tid[0]], tensors[val_tid[0]].split_height)
            additional_info['split_size_trailing'] = GetTrailingSplitSize(tensors[val_tid[0]], tensors[val_tid[0]].split_height)
            additional_info['split_cnt'] = len(val_rid)
            reqs[key_rid].additional_infos.append(additional_info)
            reqs[key_rid].priority = 9
    @classmethod
    def preproc(cls, planner, req_id, addon_id):        
        # No Operation
        return None
    @classmethod
    def postproc(cls, planner, req_id, addon_id):
        # Try to plan input tensors onto output tensor
        req_cur = planner.requirements[req_id]
        additional_info = req_cur.additional_infos[addon_id]
        addr_start = planner.calculated_offset[req_id]
        addr_cur = addr_start
        # backup original split req size
        split_req_size_ori = [planner.requirements[req_id_split].size for req_id_split in additional_info['split_req_ids']]
        # try all split
        for split_id, req_id_split in enumerate(additional_info['split_req_ids']):
            # get requirement instance of the split from planner
            req_split = planner.requirements[req_id_split]
            # calculate split addr range
            addr_split_head = addr_cur
            if split_id < additional_info['split_cnt']-1:
                addr_split_tail = addr_split_head + additional_info['split_size']
            else:
                addr_split_tail = addr_split_head + additional_info['split_size_trailing']
            # override req size
            req_split.size = addr_split_tail - addr_split_head
            # check overlapping in space and time
            inplace_is_save = True
            for candidate_id in planner.req_id_offset_sorted:
                if candidate_id == req_id:
                    continue
                req_candidate = planner.requirements[candidate_id]
                addr_candidate_head = planner.calculated_offset[candidate_id]
                if (isReqOverlapInTime(req_split, req_candidate) and
                     isReqOverlapInSpace(req_split, addr_split_head, req_candidate, addr_candidate_head)):
                    inplace_is_save = False
                    req_split.size = split_req_size_ori[split_id]
                    raise(BaseException("inplace split has addr conflict."))
                    break
            if inplace_is_save:
                save_result(planner, req_id_split, addr_split_head)
            addr_cur = addr_split_tail

    @classmethod
    def AttachCallback(cls, reqs, target_req_id):
        # Attach callback function(preproc() and postproc()) to Requirement to preform mem planning for inplace implementation.
        for key in target_req_id.keys():
            reqs[key].preprocs.append(cls.preproc)
            reqs[key].postprocs.append(cls.postproc)

class InplaceConcatAddon(InplaceAddon):
    @classmethod
    def isTargetOp(cls, op) -> bool:
        # check input op is target op or not.
        return op.type == 'CONCATENATION'
    @classmethod
    def GetTargetTensorID(cls, op, shared_params):
        # return (representative tensor's id, [tensor_id of tensors to be merged])
        param = shared_params[op.shared_param_id]
        return (param.output, [param.input])
    @classmethod
    def TranslateTIDtoReqID(cls, target_tids, reqs):
        # Translaste the return values from GetTargetTensorID() to Requirment ID.
        translated_target_req_ids = {}
        for key, val in target_tids.items():
            translated_target_req_ids[FindReqIDByHandle(reqs, (key, -1))] = FindAllSplitReqIDByTid(reqs, val[0])
        return translated_target_req_ids
    @classmethod
    def AttachAdditionalInfo(cls, reqs, target_req_ids, tensors, target_tids):
        # Attach additional info to Requirement for preproc() or postproc() to preform mem planning for inplace implementation.
        for (key_rid, val_rid) , (key_tid, val_tid) in zip(target_req_ids.items(), target_tids.items()):
            additional_info = {}
            additional_info['split_req_ids'] = val_rid
            additional_info['split_height'] = tensors[val_tid[0]].split_height
            additional_info['split_size'] = GetGeneralSplitSize(tensors[val_tid[0]], tensors[val_tid[0]].split_height)
            additional_info['split_size_trailing'] = GetTrailingSplitSize(tensors[val_tid[0]], tensors[val_tid[0]].split_height)
            additional_info['split_cnt'] = len(val_rid)
            reqs[key_rid].additional_infos.append(additional_info)
            reqs[key_rid].priority = 10
    @classmethod
    def preproc(cls, planner, req_id, addon_id):        
        # No Operation
        return None
    @classmethod
    def postproc(cls, planner, req_id, addon_id):
        # Try to plan input tensors onto output tensor
        req_cur = planner.requirements[req_id]
        additional_info = req_cur.additional_infos[addon_id]
        addr_start = planner.calculated_offset[req_id]
        addr_cur = addr_start
        # backup original split req size
        split_req_size_ori = [planner.requirements[req_id_split].size for req_id_split in additional_info['split_req_ids']]
        # try all split
        for split_id, req_id_split in enumerate(additional_info['split_req_ids']):
            # get requirement instance of the split from planner
            req_split = planner.requirements[req_id_split]
            # calculate split addr range
            addr_split_head = addr_cur
            if split_id < additional_info['split_cnt']-1:
                addr_split_tail = addr_split_head + additional_info['split_size']
            else:
                addr_split_tail = addr_split_head + additional_info['split_size_trailing']
            # override req size
            req_split.size = addr_split_tail - addr_split_head
            # check overlapping in space and time
            inplace_is_save = True
            for candidate_id in planner.req_id_offset_sorted:
                if candidate_id == req_id:
                    continue
                req_candidate = planner.requirements[candidate_id]
                addr_candidate_head = planner.calculated_offset[candidate_id]
                if (isReqOverlapInTime(req_split, req_candidate) and
                     isReqOverlapInSpace(req_split, addr_split_head, req_candidate, addr_candidate_head)):
                    inplace_is_save = False
                    req_split.size = split_req_size_ori[split_id]
                    raise(BaseException("inplace concat has addr conflict."))
                    break
            if inplace_is_save:
                save_result(planner, req_id_split, addr_split_head)
            addr_cur = addr_split_tail

    @classmethod
    def AttachCallback(cls, reqs, target_req_id):
        # Attach callback function(preproc() and postproc()) to Requirement to preform mem planning for inplace implementation.
        for key in target_req_id.keys():
            reqs[key].preprocs.append(cls.preproc)
            reqs[key].postprocs.append(cls.postproc)

AddonsTable = {
    "inplace_concat":InplaceConcatAddon,
    "inplace_split":InplaceSplitAddon
}
