def FindAllSplitReqIDByTid(reqs: list, tid):
    # Find all requirements of splits of target tensor by target's tensor id
    # Input: <all requirement list>, <target tensor id>, <target split id>
    # Output: <A list of tuples (req_id, split_id)>
    target_req_ids = [] # [(req_id, split_id), ...]
    for req_id, req in enumerate(reqs):
        # Online tensors
        if isinstance(req.handle, tuple):
            if req.handle[0] == tid:
                target_req_ids.append((req_id, req.handle[1]))
    if len(target_req_ids)-1 != max(target_req_ids, key=lambda x: x[1])[1]:
        if len(target_req_ids) != 1:
            raise(BaseException("len(target_req_ids) != max(target_req_ids)"))
    target_req_ids.sort(key=lambda x: x[1])
    return [x[0] for x in target_req_ids]

def FindReqIDByHandle(reqs, handle):
    for req_id, req in enumerate(reqs):
        if req.handle == handle:
            return req_id
    raise(BaseException("Target requirement not found."))

def isReqOverlapInTime(req_A, req_B):
    if req_A.first_time_used > req_B.last_time_used:
        return False
    elif req_B.first_time_used > req_A.last_time_used:
        return False
    return True

def isReqOverlapInSpace(req_A, addr_A, req_B, addr_B):
    if addr_A >= addr_B+req_B.size:
        return False
    elif addr_B >= addr_A+req_A.size:
        return False
    return True

def save_result(planner, req_id, offset):
    # first, validate result
        # assume no error, skip
    # second, save it in calculated_offset
    planner.calculated_offset[req_id] = offset
    # third, insert into planner.req_id_offset_sorted
    insert_point = 0
    for current_req_id in planner.req_id_offset_sorted:
        if planner.calculated_offset[current_req_id] > planner.calculated_offset[req_id]:
            break
        insert_point+=1
    planner.req_id_offset_sorted.insert(insert_point, req_id)

def align_addr(addr_base, alignment = 16):
    return (((addr_base-1)//alignment) + 1)*alignment
