import copy

class Node:
    def __init__(self, op_info, id):
        self.parents = []
        self.children = []
        self.info = copy.deepcopy(op_info)
        self.opid = id
    def append_children(self, children):
        self.children+=children

    def SplitOp(self, opid, opcode_index, inputA, inputB, outputs, num_splits):
        info = {
            "inputs": [
                inputA,
                inputB
            ],
            "outputs": outputs,
            "builtin_options_type": "SplitOptions",
            "builtin_options": {
                "num_splits": num_splits
            }
        }
        if opcode_index:
            info["opcode_index"] = opcode_index
        return Node(info, opid)


    def AddOp(self, opid, opcode_index, inputA, inputB, output, builtin_options):
        info = {
            "inputs": [
                inputA,
                inputB
            ],
            "outputs": [
                output
            ],
            "builtin_options_type": "AddOptions",
            "builtin_options": builtin_options
        }
        if opcode_index:
            info["opcode_index"] = opcode_index

        return Node(info, opid)

    def ConvOp(self, opid, opcode_index, inputs, outputs, conv_options):
        info =  {
            "inputs": inputs,
            "outputs": outputs,
            "builtin_options_type": "Conv2DOptions",
            "builtin_options": conv_options
        }
        if opcode_index:
            info["opcode_index"] = opcode_index

        return Node(info, opid)

class Graph:
    def __init__(self, ops, tensors, buffers, opcodes, inputs, outputs, exec_order):
        if len(inputs) > 1:
            raise "unsupported graph: 1 input tensor only."
        self.tensors = tensors
        self.buffers = buffers
        self.opcodes = opcodes
        self.inputs = copy.copy(inputs)
        self.outputs = copy.copy(outputs)
        self.exec_order = exec_order
        self.in_tensor_id = inputs[0]
        self.root_op_id = -1
        self.ops = None
        self.DFS_ordered = False
        self.BFS_ordered = False
        self.build_graph_from_ops(copy.deepcopy(ops))

    def build_DFS(self, current_id, op_lookup_input):
        if len(self.ops[current_id].children) > 0:
            return
        new_children = []
        for out in self.ops[current_id].info['outputs']:
            if out in op_lookup_input:
                new_children = list(set(new_children) | set(op_lookup_input[out]))
        if len(new_children) == 0:
            return
        self.ops[current_id].append_children(new_children)

        for child in self.ops[current_id].children:
            self.ops[child].parents.append(current_id)
            self.build_DFS(child, op_lookup_input)

    def build_graph_from_ops(self, ops):
        # init all nodes
        self.ops = [Node(op, i) for i, op in enumerate(ops)]

        # build input-op lookup table
        op_lookup_input = {}
        for opid, op in enumerate(ops):
            for in_id in op['inputs']:
                if in_id not in op_lookup_input.keys():
                    op_lookup_input[in_id] = []
                op_lookup_input[in_id].append(opid)
        self.op_lookup_input = op_lookup_input

        # find root
        self.root_op_id = op_lookup_input[self.in_tensor_id][0]

        # build the remainder
        self.build_DFS(self.root_op_id, op_lookup_input)

        # reorder execution ordering to the input exec_order
        self.ensure_order()


    def ensure_DFS_order(self):
        def DFS_ordering(current_id, DFS_orderred_operators:list):
            for parent in self.ops[current_id].parents:
                if parent not in DFS_orderred_operators:
                    return None
            DFS_orderred_operators.append(current_id)
            # To reach maximal performance of DFS ordering, sort children by output tensor ID of children op.
            for child in sorted(self.ops[current_id].children, key=lambda op_id: self.ops[op_id].info['outputs'][0]):
                if child not in DFS_orderred_operators:
                    DFS_ordering(child, DFS_orderred_operators)
        if self.DFS_ordered == False:
            self.DFS_ordered = True
            self.BFS_ordered = False
            new_operators = []
            for i, op in enumerate(self.ops):
                if self.in_tensor_id in op.info['inputs']:
                    start_id = i
            DFS_ordering(start_id, new_operators)
            self.operators = [self.ops[op].info for op in new_operators]

    def ensure_BFS_order(self):
        import collections
        grid, inp = {k.opid: [] for k in self.ops}, collections.Counter({k.opid: 0 for k in self.ops})
        prerequisites = []
        for current in self.ops:
            for child in self.ops[current.opid].children:
                prerequisites.append((child, current.opid))
            for parent in self.ops[current.opid].parents:
                prerequisites.append((current.opid, parent))
        [(grid[b].append(a), inp.update({a: 1})) for a, b in prerequisites]
        ans = [k for k, v in inp.items() if not v]
        [inp.update({kid: -1}) or inp[kid] == 0 and ans.append(kid) for node in ans for kid in grid[node]]
        self.operators = [self.ops[op].info for op in ans]

    def ensure_BFS_order_orig(self):
        import queue
        if self.BFS_ordered == False:
            self.BFS_ordered = True
            self.DFS_ordered = False
            BFS_queue = queue.Queue()
            BFS_orderred_operators = []
            for i, op in enumerate(self.ops):
                if self.in_tensor_id in op.info['inputs']:
                    start_id = i

            BFS_queue.put(start_id)
            while(not BFS_queue.empty()):
                current_id = BFS_queue.get()
                if current_id in BFS_orderred_operators:
                    continue
                for parent in self.ops[current_id].parents:
                    if parent not in BFS_orderred_operators:
                        continue
                BFS_orderred_operators.append(current_id)
                for child in self.ops[current_id].children:
                    if child not in BFS_orderred_operators:
                        BFS_queue.put(child)

            self.operators = [self.ops[op].info for op in BFS_orderred_operators]

    def recycle_tensors_buffers(self):
        new_tensor_ids = [-1 for _ in range(len(self.tensors))]
        new_buffer_ids = [-1 for _ in range(len(self.buffers))]
        cnt = 1

        # assign tensor id 0 to model input tensor
        new_tensor_ids[self.in_tensor_id] = 0

        # assign new tensor id to conv kernels
        for op in self.operators:
            if op.get("builtin_options_type","") == "Conv2DOptions":
                if new_tensor_ids[op['inputs'][1]] == -1:
                    new_tensor_ids[op['inputs'][1]] = cnt
                    cnt += 1

        # assign new tensor id to conv bias
        for op in self.operators:
            if op.get("builtin_options_type","") == "Conv2DOptions":
                if len(op['inputs']) >= 3:
                    if new_tensor_ids[op['inputs'][2]] == -1:
                        new_tensor_ids[op['inputs'][2]] = cnt
                        cnt += 1

        # assign new tensor id to conv padding param
        for op in self.operators:
            if op.get("builtin_options_type","") == "Conv2DOptions":
                if len(op['inputs']) >= 4:
                    if new_tensor_ids[op['inputs'][3]] == -1:
                        new_tensor_ids[op['inputs'][3]] = cnt
                        cnt += 1

        # assign new tensor id to the remainder
        for op in self.operators:
            for tensor_id in op['inputs']:
                if new_tensor_ids[tensor_id] == -1:
                    new_tensor_ids[tensor_id] = cnt
                    cnt += 1
            for tensor_id in op['outputs']:
                if new_tensor_ids[tensor_id] == -1:
                    new_tensor_ids[tensor_id] = cnt
                    cnt += 1

        # commit new tensor
        new_inputs = [new_tensor_ids[id] for id in self.inputs]
        new_outputs = [new_tensor_ids[id] for id in self.outputs]
        self.inputs = new_inputs
        self.outputs = new_outputs

        for op in self.operators:
            for i, ori_tensor_id in enumerate(op['inputs']):
                op['inputs'][i] = new_tensor_ids[ori_tensor_id]
            for i, ori_tensor_id in enumerate(op['outputs']):
                op['outputs'][i] = new_tensor_ids[ori_tensor_id]

        new_tensors = {}
        for ori_tensor_id, new_tensor_id in enumerate(new_tensor_ids):
            if new_tensor_id != -1:
                new_tensors[new_tensor_id] = self.tensors[ori_tensor_id]
        self.tensors = [new_tensors[id] for id in range(len(new_tensors))]


        # assign new buffer id
        cnt = 1
        new_buffer_ids[0] = 0
        for tensor in self.tensors:
            if new_buffer_ids[tensor['buffer']] == -1:
                new_buffer_ids[tensor['buffer']] = cnt
                cnt += 1

        # commit new buffer id
        for i in range(len(self.tensors)):
            self.tensors[i]['buffer'] = new_buffer_ids[self.tensors[i]['buffer']]

        new_buffers = {}
        for ori_buffer_id, new_buffer_id in enumerate(new_buffer_ids):
            if new_buffer_id != -1:
                new_buffers[new_buffer_id] = self.buffers[ori_buffer_id]
        self.buffers = [new_buffers[id] for id in range(len(new_buffers))]

    def remove_deprecated_op(self, deprecated):
        new_opid = []
        id = 0
        for i in range(len(self.ops)):
            if i in deprecated:
                new_opid.append(-1)
            else:
                new_opid.append(id)
                id += 1
        self.root_op_id = new_opid[self.root_op_id]
        for op in self.ops:
            op.opid = new_opid[op.opid]
            for i in range(len(op.parents)):
                op.parents[i] = new_opid[op.parents[i]]
            for i in range(len(op.children)):
                op.children[i] = new_opid[op.children[i]]
        self.ops = [op for op in self.ops if op.opid != -1]
        self.operators = [op for idx, op in enumerate(self.operators) if new_opid[idx] != -1]
    
    def ensure_order(self):
        if self.exec_order == 'DF':
            self.ensure_DFS_order()
        elif self.exec_order == 'BF':
            self.ensure_BFS_order()
        else:
            raise(BaseException("Unknown execution order setting."))

    def export(self):
        self.ensure_order()
        return self.buffers, self.tensors, self.inputs, self.outputs, self.operators, self.opcodes

