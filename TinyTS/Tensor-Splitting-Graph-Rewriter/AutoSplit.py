from MyGraph import Node,Graph
import copy

# class SplitNode:
#     def __init__(self, node:Node, op_id, split_id):
#         self.node = copy.deepcopy(node)
#         self.ori_opid = op_id
#         self.split_id = split_id
#         new_input_tensors = []
#         new_output_tensors = []

class SplitterNode:
    def __init__(self, node:Node):
        self.node = node
        self.split_id = []
        self.visited = False

class Splitter:
    def __init__(self,ori_graph:Graph, split_height:int):
        self.padding_param_tensors = {}
        self.ori_graph = ori_graph
        self.split_height = split_height
        self.opcodes =  copy.deepcopy(ori_graph.opcodes)
        self.tensors = copy.deepcopy(ori_graph.tensors)
        self.buffers = copy.deepcopy(ori_graph.buffers)
        self.operators = [copy.deepcopy(n.info) for n in self.ori_graph.ops]
        self.nodes = [SplitterNode(n) for n in self.ori_graph.ops]
        self.splittable_opcode_idxes = {}
        for i, opcode in enumerate(self.opcodes):
            if opcode.get("deprecated_builtin_code", 0) in [0, 34, 3, 4]:
                self.splittable_opcode_idxes[opcode.get("deprecated_builtin_code", 0)] = i

    def re_init(self, ori_graph):
        self.ori_graph = ori_graph
        self.opcodes =  copy.deepcopy(ori_graph.opcodes)
        self.tensors = copy.deepcopy(ori_graph.tensors)
        self.buffers = copy.deepcopy(ori_graph.buffers)
        self.operators = [copy.deepcopy(n.info) for n in self.ori_graph.ops]
        self.nodes = [SplitterNode(n) for n in self.ori_graph.ops]
        self.splittable_opcode_idxes = {}
        for i, opcode in enumerate(self.opcodes):
            if opcode.get("deprecated_builtin_code", 0) in [0, 34, 3, 4]:
                self.splittable_opcode_idxes[opcode.get("deprecated_builtin_code", 0)] = i

    def perform_split(self)->Graph:
        # vars init
        self.split_info = []
        self.split_tensors = []
        self.split_tensor_table = [list() for _ in range(len(self.tensors))]
        self.new_operators = []

        # Currently assume it's splittable from the root of the graph
        start_id = self.traverse_til_splittable(self.ori_graph.root_op_id)
        while( start_id is not None):
            # get splittable block
            end_id, splittables = self.traverse_til_not_splittable(start_id, [])

            input_tile_size = self.split_height
            output_tile_size = self.split_height

            # TODO split block input
            self.split_block_input(start_id, input_tile_size)

            # start split
            for op in splittables:
                self.split_one_node(op, input_tile_size, output_tile_size)

            # TODO concat block output
            self.concat_block_output(end_id)

            # start_id = self.traverse_til_splittable(end_id)
            # print("To avoid error, now compiler splits only one block.")
            start_id = self.traverse_til_end(end_id)

        new_graph = Graph ( self.new_operators, self.tensors, self.buffers,
                            self.opcodes, [self.ori_graph.in_tensor_id], self.ori_graph.outputs, self.ori_graph.exec_order)

        new_graph.recycle_tensors_buffers()

        return new_graph.export()

    def traverse_til_end(self,current_opid):
        for parent in self.nodes[current_opid].node.parents:
            if self.nodes[parent].visited == False:
                return None

        self.new_operators.append(self.nodes[current_opid].node.info)

        self.nodes[current_opid].visited = True
        for child in self.nodes[current_opid].node.children:
            # check if it is splittable op
            result = self.traverse_til_end(child)
            if result is not None:
                return result
        return None

    def traverse_til_splittable(self,current_opid):
        if self.nodes[current_opid].node.info.get("opcode_index",0) in self.splittable_opcode_idxes.values():
            return current_opid
        for parent in self.nodes[current_opid].node.parents:
            if self.nodes[parent].visited == False:
                return None

        self.new_operators.append(self.nodes[current_opid].node.info)

        self.nodes[current_opid].visited = True
        for child in self.nodes[current_opid].node.children:
            # check if it is splittable op
            result = self.traverse_til_splittable(child)
            if result is not None:
                return result
        return None

    def traverse_til_not_splittable(self,current_opid,splittables):
        for parent in self.nodes[current_opid].node.parents:
            if self.nodes[parent].visited == False:
                return None
        self.nodes[current_opid].visited = True
        splittables.append(current_opid)
        for child in self.nodes[current_opid].node.children:
            # check if it is splittable op
            if self.nodes[child].node.info.get("opcode_index",0) in self.splittable_opcode_idxes.values():
                result = self.traverse_til_not_splittable(child,splittables)
                if result is not None:
                    return result
            else:
                splittables.append(child)
                return (child, splittables)
        return None

    def split_tensor(self, tensor_id_in):
        tensor_info = self.tensors[tensor_id_in]
        buffer_id = len(self.buffers)
        tensor_id = len(self.tensors)
        new_tensor_info_base = copy.deepcopy(tensor_info)
        if 'shape_signature' in new_tensor_info_base:
            del new_tensor_info_base['shape_signature']
        new_tensor_info_base['shape'][1] = 1
        for i in range(tensor_info['shape'][1]):
            new_tensor_info = copy.deepcopy(new_tensor_info_base)
            new_tensor_info['buffer'] = buffer_id
            new_tensor_info['name'] += '_split_%d' % (i)
            self.buffers.append({})
            self.tensors.append(new_tensor_info)
            self.split_tensor_table[tensor_id_in].append(tensor_id)
            buffer_id += 1
            tensor_id += 1

    def split_tensor_by_n(self, tensor_id_in, tile_size):
        tensor_info = self.tensors[tensor_id_in]
        buffer_id = len(self.buffers)
        tensor_id = len(self.tensors)
        new_tensor_info_base = copy.deepcopy(tensor_info)
        if 'shape_signature' in new_tensor_info_base:
            del new_tensor_info_base['shape_signature']

        import math
        for i in range(0, math.ceil(tensor_info['shape'][1]/tile_size), 1):
            guard = min(tile_size, tensor_info['shape'][1] - i*tile_size)
            new_tensor_info = copy.deepcopy(new_tensor_info_base)
            new_tensor_info['shape'][1] = guard
            new_tensor_info['buffer'] = buffer_id
            new_tensor_info['name'] += '_split_%d' % (i)
            self.buffers.append({})
            self.tensors.append(new_tensor_info)
            self.split_tensor_table[tensor_id_in].append(tensor_id)
            buffer_id += 1
            tensor_id += 1

    def split_one_node(self, opid, input_split, output_split):
        opcode_idx = self.nodes[opid].node.info.get("opcode_index",0)
        if opcode_idx == self.splittable_opcode_idxes.get(0, -1):
            self.split_add(opid, output_split)
        elif opcode_idx == self.splittable_opcode_idxes.get(3 , -1):
            self.split_conv(opid, input_split, output_split)
        elif opcode_idx == self.splittable_opcode_idxes.get(4 , -1):
            self.split_dwconv(opid, input_split, output_split)
        elif opcode_idx == self.splittable_opcode_idxes.get(34, -1):
            self.split_pad(opid, output_split)

    def split_pad(self, opid, output_split):
        info = self.nodes[opid].node.info
        self.split_tensor_by_n(info['outputs'][0], output_split)

        inputs = info['inputs']
        outputs = info['outputs']

        if len(inputs) != 2:
            raise "wrong input number"
        elif len(outputs) != 1:
            raise "wrong output number"
        else:
            split_op_id = len(self.nodes)
            i_len = len(self.split_tensor_table[inputs[0]])
            for a, b, c in zip(self.split_tensor_table[inputs[0]],
                               [inputs[1] for i in range(i_len)],
                               self.split_tensor_table[outputs[0]]):
                new_op_info = copy.deepcopy(info)
                new_op_info['inputs'] = [a, b]
                new_op_info['outputs'] = [c]
                op = Node(new_op_info, split_op_id)

                self.nodes.append(SplitterNode(op))
                self.new_operators.append(new_op_info)
                self.nodes[opid].split_id.append(split_op_id)
                split_op_id+=1

    def split_conv(self, opid, input_split, output_split):
        info = self.nodes[opid].node.info
        self.split_tensor_by_n(info['outputs'][0], output_split)

        inputs = info['inputs']
        outputs = info['outputs']
        new_op_info_base = copy.deepcopy(info)

        if len(inputs) not in [2,3,4]:
            raise "wrong input number"
        elif len(outputs) != 1:
            raise "wrong output number"
        else:
            split_op_id = len(self.nodes)

            in_shape = self.tensors[inputs[0]]['shape']
            out_shape = self.tensors[outputs[0]]['shape']
            ker_shape = self.tensors[inputs[1]]['shape']
            stride_h = info['builtin_options']['stride_h']
            stride_w = info['builtin_options']['stride_w']

            if(len(inputs) == 4):
                tokens = self.tensors[info['inputs'][3]]['name'].split('_')
                if(tokens[0] != 'padding'):
                    raise "wrong custom padding setting tensor data"
                paddings_H = int(tokens[1])
                paddings_W = int(tokens[2])
            else:
                # calculate padding
                if info['builtin_options'].get('padding', 'SAME') == 'SAME':
                    total_padding_H = (out_shape[1] - 1) * stride_h + ker_shape[1] - in_shape[1]
                    total_padding_W = (out_shape[2] - 1) * stride_w + ker_shape[2] - in_shape[2]
                    paddings_H = total_padding_H // 2 if total_padding_H > 0 else 0
                    paddings_W = total_padding_W // 2 if total_padding_W > 0 else 0
                else:
                    paddings_H = 0
                    paddings_W = 0

            # generate splitted conv for each tile
            for out_y in range(0, out_shape[1], output_split):
                new_op_info = copy.deepcopy(new_op_info_base)

                guard_inner_y = min(output_split, out_shape[1] - out_y)

                new_inputs = []
                split_padding_H = -((out_y) * stride_h - paddings_H)
                split_padding_H = 0 if split_padding_H < 0 else split_padding_H

                # inference required in_y from this tile
                required = []
                for out_inner_y in range(guard_inner_y):
                    in_y_origin = (out_y + out_inner_y) * stride_h - paddings_H
                    for h in range(ker_shape[1]):
                        in_y = in_y_origin + h
                        if in_y >= 0 and in_y < in_shape[1] and (in_y//input_split) not in required:
                            required.append((in_y//input_split))

                # inputs
                for in_y in required:
                    new_inputs.append(self.split_tensor_table[inputs[0]][in_y])

                padding_param_tensor = self.get_padding_param_tensor(split_padding_H, paddings_W)
                if (len(inputs) == 4):
                    new_op_info['inputs'][3] = padding_param_tensor
                    new_op_info['inputs'] += new_inputs
                else:
                    new_op_info['inputs'] += [padding_param_tensor] + new_inputs
                new_op_info['inputs'][0] = new_op_info['inputs'][4]

                # outputs
                new_op_info['outputs'] = [self.split_tensor_table[outputs[0]][int(out_y)//input_split]]
                # for out_inner_y in range(guard_inner_y):
                #     new_op_info['outputs'].append(self.split_tensor_table[outputs[0]][out_y + out_inner_y])

                self.new_operators.append(new_op_info)
                op = Node(new_op_info, split_op_id)
                self.nodes.append(SplitterNode(op))
                self.nodes[opid].split_id.append(split_op_id)
                split_op_id += 1

    def split_dwconv(self, opid, input_split, output_split):
        info = self.nodes[opid].node.info
        self.split_tensor_by_n(info['outputs'][0], output_split)

        inputs = info['inputs']
        outputs = info['outputs']
        new_op_info_base = copy.deepcopy(info)

        if len(inputs) not in [2,3,4]:
            raise "wrong input number"
        elif len(outputs) != 1:
            raise "wrong output number"
        else:
            split_op_id = len(self.nodes)

            in_shape = self.tensors[inputs[0]]['shape']
            out_shape = self.tensors[outputs[0]]['shape']
            ker_shape = self.tensors[inputs[1]]['shape']
            stride_h = info['builtin_options']['stride_h']
            stride_w = info['builtin_options']['stride_w']

            if(len(inputs) == 4):
                tokens = self.tensors[info['inputs'][3]]['name'].split('_')
                if(tokens[0] != 'padding'):
                    raise "wrong custom padding setting tensor data"
                paddings_H = int(tokens[1])
                paddings_W = int(tokens[2])
            else:
                # calculate padding
                if info['builtin_options'].get('padding', 'SAME') == 'SAME':
                    total_padding_H = (out_shape[1] - 1) * stride_h + ker_shape[1] - in_shape[1]
                    total_padding_W = (out_shape[2] - 1) * stride_w + ker_shape[2] - in_shape[2]
                    paddings_H = total_padding_H // 2 if total_padding_H > 0 else 0
                    paddings_W = total_padding_W // 2 if total_padding_W > 0 else 0
                else:
                    paddings_H = 0
                    paddings_W = 0

            # generate splitted conv for each tile
            for out_y in range(0, out_shape[1], output_split):
                new_op_info = copy.deepcopy(new_op_info_base)

                guard_inner_y = min(output_split, out_shape[1] - out_y)

                new_inputs = []
                split_padding_H = -((out_y) * stride_h - paddings_H)
                split_padding_H = 0 if split_padding_H < 0 else split_padding_H

                # inference required in_y from this tile
                required = []
                for out_inner_y in range(guard_inner_y):
                    in_y_origin = (out_y + out_inner_y) * stride_h - paddings_H
                    for h in range(ker_shape[1]):
                        in_y = in_y_origin + h
                        if in_y >= 0 and in_y < in_shape[1] and (in_y//input_split) not in required:
                            required.append((in_y//input_split))


                # inputs
                for in_y in required:
                    new_inputs.append(self.split_tensor_table[inputs[0]][in_y])

                new_op_info = copy.deepcopy(new_op_info_base)
                padding_param_tensor = self.get_padding_param_tensor(split_padding_H, paddings_W)

                if (len(inputs) == 4):
                    new_op_info['inputs'][3] = padding_param_tensor
                    new_op_info['inputs'] += new_inputs
                else:
                    new_op_info['inputs'] += [padding_param_tensor] + new_inputs
                new_op_info['inputs'][0] = new_op_info['inputs'][4]

                # outputs
                # for out_inner_y in range(guard_inner_y):
                    # new_op_info['outputs'].append(self.split_tensor_table[outputs[0]][out_y + out_inner_y])

                new_op_info['outputs'] = [self.split_tensor_table[outputs[0]][out_y//input_split]]

                self.new_operators.append(new_op_info)
                op = Node(new_op_info, split_op_id)
                self.nodes.append(SplitterNode(op))
                self.nodes[opid].split_id.append(split_op_id)
                split_op_id += 1

    def split_add(self, opid, output_split):
        info = self.nodes[opid].node.info
        self.split_tensor_by_n(info['outputs'][0], output_split)

        inputs = info['inputs']
        outputs = info['outputs']

        if len(inputs) != 2:
            raise "wrong input number"
        elif len(outputs) != 1:
            raise "wrong output number"
        elif len(self.split_tensor_table[inputs[0]]) != len(self.split_tensor_table[inputs[1]]):
            raise BaseException("split number of two operand is not equal")
        else:
            split_op_id = len(self.nodes)
            for a,b,c in zip(self.split_tensor_table[inputs[0]],
                            self.split_tensor_table[inputs[1]],
                             self.split_tensor_table[outputs[0]]):
                new_op_info = copy.deepcopy(info)
                new_op_info['inputs'] = [a,b]
                new_op_info['outputs'] = [c]
                op = Node(new_op_info,split_op_id)
                self.nodes.append(SplitterNode(op))
                self.new_operators.append(new_op_info)
                self.nodes[opid].split_id.append(split_op_id)
                split_op_id+=1

    def split_block_input(self, start_opid, input_split):
        info = self.nodes[start_opid].node.info
        self.split_tensor_by_n(info['inputs'][0], input_split)
        axis_tensor = {
            "shape": [

            ],
            "type": "INT32",
            "buffer": len(self.buffers),
            "name": self.tensors[info['inputs'][0]]['name']+"_split_axis_tensor",
            "quantization": {
            },

          }
        axis_buffer = {
            "data": self.int_list_to_byte_list([1])
        }
        self.tensors.append(axis_tensor)
        self.buffers.append(axis_buffer)

        outputs = copy.deepcopy(self.split_tensor_table[info['inputs'][0]])
        # outputs = [x for i,x in enumerate(outputs) if i % input_split == 0]

        new_op_info = {
            "opcode_index": self.get_opcode_index(49),
            "inputs": [len(self.tensors)-1, info['inputs'][0]],
            "outputs": outputs,
            "builtin_options_type": "SplitOptions",
            "builtin_options": {
                "num_splits" : len(outputs)
            }
        }
        self.new_operators.append(new_op_info)

    def concat_block_output(self, end_opid):
        info = self.nodes[end_opid].node.info
        new_op_info = {
            "opcode_index": self.get_opcode_index(2),
            "inputs": copy.deepcopy(self.split_tensor_table[info['inputs'][0]]),
            "outputs": [
                info['inputs'][0]
            ],
            "builtin_options_type": "ConcatenationOptions",
            "builtin_options": {
                'axis': 1
            }
        }
        self.new_operators.append(new_op_info)

    def get_opcode_index(self, deprecated_builtin_code):
        for i, code in enumerate(self.opcodes):
            if code.get('deprecated_builtin_code', 0) == deprecated_builtin_code:
                return i
        raise 'opcode not found'

    def get_padding_param_tensor(self, padTop, padSide):
        if (padTop not in self.padding_param_tensors or
            padSide not in self.padding_param_tensors[padTop]):
            new_buffer_info = {
                "data": self.int_list_to_byte_list([padTop, padSide])
            }
            new_tensor_info = {
                    "shape": [
                        2
                    ],
                    "type": "INT32",
                    "buffer": len(self.buffers),
                    "name": "padding_%d_%d" % (padTop, padSide),

                    "quantization": {
                    }
                }

            if padTop not in self.padding_param_tensors:
                self.padding_param_tensors[padTop] = {padSide:len(self.tensors)}
            else:
                self.padding_param_tensors[padTop][padSide] = len(self.tensors)

            self.buffers.append(new_buffer_info)
            self.tensors.append(new_tensor_info)

        return self.padding_param_tensors[padTop][padSide]

    def PaddingFusion(self):
        def get_pad_param(pad_data):
            byte_data = bytes(pad_data)
            int_data = [ int.from_bytes(byte_data[4*i:4*i+4], byteorder='little') for i in range(len(byte_data)//4)]
            if False in [int_data[i] == 0 for i in [0,1,6,7]]:
                raise "pad in N or C"
            elif int_data[2]!=int_data[3] or int_data[4]!=int_data[5]:
                raise "asymmetric pad in H or W"
            return int_data[2], int_data[4]


        def apply_fusion(pad_opid, conv_opid):
            pad_op_info = self.ori_graph.ops[pad_opid].info
            conv_op_info = self.ori_graph.ops[conv_opid].info
            pad_H, pad_W = get_pad_param(self.buffers[self.tensors[pad_op_info['inputs'][1]]['buffer']]['data'])

            inputs = conv_op_info['inputs']
            outputs = conv_op_info['outputs']
            in_shape = self.tensors[pad_op_info['inputs'][0]]['shape']
            out_shape = self.tensors[outputs[0]]['shape']
            ker_shape = self.tensors[inputs[1]]['shape']
            stride_h = conv_op_info['builtin_options']['stride_h']
            total_padding_H = (out_shape[1] - 1) * stride_h + ker_shape[1] - in_shape[1]
            total_padding_W = (out_shape[2] - 1) * stride_h + ker_shape[2] - in_shape[2]
            calculated_padding_H = total_padding_H // 2 if total_padding_H > 0 else 0
            calculated_padding_W = total_padding_W// 2 if total_padding_W > 0 else 0

            conv_op_info['inputs'][0] = pad_op_info['inputs'][0]
            if calculated_padding_H==pad_H and calculated_padding_W==pad_W:
                conv_op_info['builtin_options']['padding'] = 'SAME'
            elif len(conv_op_info['inputs']) == 3:
                conv_op_info['inputs'].append(self.get_padding_param_tensor(pad_H,pad_W))
            elif len(conv_op_info['inputs']) < 3:
                 BaseException("worng inputs format: length < 3")
            elif len(conv_op_info['inputs']) > 3:
                 BaseException("overriding padding param already exist")



        def PaddingFusion_dfs(cur_opid, visited, deprecated):
            # check visited
            if visited[cur_opid]:
                return deprecated
            visited[cur_opid] = True

            # is cur_op a pad op?
            is_pad = self.opcodes[self.ori_graph.ops[cur_opid].info.get('opcode_index', 0)].get('deprecated_builtin_code', 0) == 34

            # visit all children
            is_deprecated = 0
            need_fusion = []
            for child_id in self.ori_graph.ops[cur_opid].children:
                if(is_pad and self.opcodes[self.ori_graph.ops[child_id].info.get('opcode_index', 0)].get('deprecated_builtin_code', 0) in [3, 4]):
                    need_fusion.append(child_id)
                    is_deprecated = 1 if is_deprecated == 0 else -1
                else:
                    is_deprecated = -1
                PaddingFusion_dfs(child_id, visited, deprecated)


            if is_deprecated == 1:
                deprecated.append(cur_opid)
                for opid in need_fusion:
                    apply_fusion(cur_opid, opid)

            return deprecated


        visited = [False for _ in range(len(self.ori_graph.ops))]
        deprecated = []
        while 1 :
            deprecated_new = []
            PaddingFusion_dfs(self.ori_graph.root_op_id, visited, deprecated_new)
            # TODO: remove deprecated operators
            if len(deprecated_new) == 0:
                break
            for opid in deprecated_new:
                if opid == self.ori_graph.root_op_id:
                    self.ori_graph.root_op_id = self.ori_graph.ops[opid].children[0]

                if len(self.ori_graph.ops[opid].parents) > 0:
                    parent = self.ori_graph.ops[opid].parents[0]
                    for i in range(len(self.ori_graph.ops[parent].children)):
                        if self.ori_graph.ops[parent].children[i] == opid:
                            del self.ori_graph.ops[parent].children[i]
                            self.ori_graph.ops[parent].children += self.ori_graph.ops[opid].children
                            break
                    for child in self.ori_graph.ops[opid].children:
                        for i in range(len(self.ori_graph.ops[child].children)):
                            if self.ori_graph.ops[child].parents[i] == opid:
                                self.ori_graph.ops[child].parents[i] = parent
                                break
                else:
                    for child in self.ori_graph.ops[opid].children:
                        for i in range(len(self.ori_graph.ops[child].children)):
                            if self.ori_graph.ops[child].parents[i] == opid:
                                del self.ori_graph.ops[child].parents[i]
                                break
            deprecated += deprecated_new
        self.ori_graph.buffers = self.buffers
        self.ori_graph.tensors = self.tensors
        self.ori_graph.remove_deprecated_op(deprecated)
        new_buffers, new_tensors, new_inputs, new_outputs, new_operators, new_opcodes = self.ori_graph.export()
        new_graph = Graph(new_operators, new_tensors, new_buffers, new_opcodes, new_inputs, new_outputs, self.ori_graph.exec_order)
        self.re_init(new_graph)

    def int_list_to_byte_list(self, ints):
        out = []
        for num in ints:
            if(type(num) != int):
                raise "int_list_to_byte_list: type error"
            out += [ b for b in (num).to_bytes(length=4, byteorder='little')]
        return out