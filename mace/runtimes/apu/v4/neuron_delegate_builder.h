    } else if (type == "Concat") {
      input_num = op_def.node_input_size() - 1;
    } else if (type == "DepthwiseConv2d") {
      input_num = 3;
    } else if (type == "Pad") {
      input_num = 1;
    } else if (type == "PRelu") {
      input_num = 2;
    } else if (type == "Reduce") {
      input_num = 1;
    } else if (type == "Reshape") {
      input_num = 1;
    } else if (type == "ResizeBilinear") {
      input_num = 1;
    } else if (type == "Deconv2D") {
      input_num = 2;
    } else if (type == "SpaceToDepth") {
      input_num = 1;
    } else if (type == "DepthToSpace") {
      input_num = 1;
    }

    else if (type == "Concat") {
      Argument axis = get_arg(op_def, "axis");
      AddScalarInt32Operand(axis.i());
      *nn_op_type = NEURON_CONCATENATION;
    } else if (type == "DepthwiseConv2d") {
      Argument padding_values = get_arg(op_def, "padding_values");
      AddScalarInt32Operand(padding_values.ints(3));  // padding on the left
      AddScalarInt32Operand(padding_values.ints(1));  // padding on the right
      AddScalarInt32Operand(padding_values.ints(0));  // padding on the top
      AddScalarInt32Operand(padding_values.ints(2));  // padding on the bottom
      Argument strides = get_arg(op_def, "strides");
      AddScalarInt32Operand(strides.ints(0));  // stride width
      AddScalarInt32Operand(strides.ints(1));  // stride height
      for (auto const_tensor : net_def_->tensors()) {
        if (op_def.input(1) == const_tensor.name()) {
          AddScalarInt32Operand(const_tensor.dims(0));  // depth_multiplier
          break;
        }
      }
      Argument activation = get_arg(op_def, "activation");
      Argument max_limit = get_arg(op_def, "max_limit");
      int act_mode;
      if (activation.s() == "RELU") {
        act_mode = NEURON_FUSED_RELU;
      } else if (activation.s() == "RELUX" && max_limit.f() == 6.0) {
        act_mode = NEURON_FUSED_RELU6;
      } else {
        act_mode = NEURON_FUSED_NONE;
      }
      act_mode = HandleFuseCode(op_def, act_mode);
      AddScalarInt32Operand(act_mode);
      AddScalarBoolOperand(false);  // Use NHWC format
      Argument dilations_values = get_arg(op_def, "dilations");
      AddScalarInt32Operand(dilations_values.ints(0));  // dilation width
      AddScalarInt32Operand(dilations_values.ints(1));  // dilation height
      *nn_op_type = NEURON_DEPTHWISE_CONV_2D;
    } else if (type == "Pad") {
      Argument paddings = get_arg(op_def, "paddings");
      const uint32_t kTensorRank = 2;
      uint32_t tensor_dims[kTensorRank];
      tensor_dims[0] = paddings.ints_size() / 2;
      tensor_dims[1] = 2;
      int size = tensor_dims[0] * tensor_dims[1];
      int32_t* padding_value = new int32_t[size];
      int32_buffers_->push_back(padding_value);
      for (auto i = 0 ; i < size ; i++) {
        padding_value[i] = paddings.ints(i);
      }
      AddArrayInt32Operand(padding_value, kTensorRank, tensor_dims);
      *nn_op_type = NEURON_PAD;
    } else if (type == "PRelu") {
      *nn_op_type = NEURON_PRELU;
    } else if (type == "Reduce") {
      Argument reduce_type = get_arg(op_def, "reduce_type");
      Argument axis = get_arg(op_def, "axis");
      Argument keepdims = get_arg(op_def, "keepdims");
      if (reduce_type.i() == 0) {
        *nn_op_type = NEURON_MEAN;
      } else {
        LOG(ERROR) << "Unsupport mace reduce mode";
        return false;
      }
      uint32_t axis_rank = static_cast<uint32_t>(axis.ints_size());
      int32_t *axis_value = nullptr;
      if (axis_rank > 0) {
        axis_value = new int32_t[axis_rank];
        for (auto i = 0 ; i < axis.ints_size() ; i++) {
          axis_value[i] = axis.ints(i);
        }
      }
      int32_buffers_->push_back(axis_value);
      AddVectorInt32Operand(axis_value, axis_rank);
      AddScalarInt32Operand(keepdims.i());
    } else if (type == "Reshape") {
      uint32_t output_shape_rank =
          static_cast<uint32_t>(op_def.output_shape(0).dims_size());
      int32_t *output_shape_value = nullptr;
      if (output_shape_rank > 0) {
        output_shape_value = new int32_t[output_shape_rank];
        for (auto i = 0 ; i < op_def.output_shape(0).dims_size() ; i++) {
          output_shape_value[i] = op_def.output_shape(0).dims(i);
        }
      }
      int32_buffers_->push_back(output_shape_value);
      AddVectorInt32Operand(output_shape_value, output_shape_rank);
      *nn_op_type = NEURON_RESHAPE;
    } else if (type == "ResizeBilinear") {
      Argument align_corners = get_arg(op_def, "align_corners");
      AddScalarInt32Operand(op_def.output_shape(0).dims(1));  // output width
      AddScalarInt32Operand(op_def.output_shape(0).dims(2));  // output height
      // set to true to specify NCHW data layout for input0 and output0.
      AddScalarBoolOperand(false);
      // align_corners
      AddScalarBoolOperand(static_cast<bool>(align_corners.i()));
      *nn_op_type = NEURON_RESIZE_BILINEAR;
    } else if (type == "Deconv2D") {
      // Since bias is the fourth input tensor, we add it here.
      int node_id = op_def.node_input(3).node_id();
      int neuron_tensor_index = operand_mapping_.mace_index_to_neuron(node_id);
      if (neuron_tensor_index == -1) {
        // Allocate a new tensor index
        neuron_tensor_index =
            operand_mapping_.add_new_neuron_tensor_index(node_id);
      }
      augmented_inputs_.push_back(neuron_tensor_index);
      Argument padding_values = get_arg(op_def, "padding_values");
      AddScalarInt32Operand(padding_values.ints(3));  // padding on the left
      AddScalarInt32Operand(padding_values.ints(1));  // padding on the right
      AddScalarInt32Operand(padding_values.ints(0));  // padding on the top
      AddScalarInt32Operand(padding_values.ints(2));  // padding on the bottom
      Argument strides = get_arg(op_def, "strides");
      AddScalarInt32Operand(strides.ints(0));  // stride width
      AddScalarInt32Operand(strides.ints(1));  // stride height
      Argument activation = get_arg(op_def, "activation");
      Argument max_limit = get_arg(op_def, "max_limit");
      int act_mode;
      if (activation.s() == "RELU") {
        act_mode = NEURON_FUSED_RELU;
      } else if (activation.s() == "RELUX" && max_limit.f() == 6.0) {
        act_mode = NEURON_FUSED_RELU6;
      } else {
        act_mode = NEURON_FUSED_NONE;
      }
      act_mode = HandleFuseCode(op_def, act_mode);
      AddScalarInt32Operand(act_mode);
      AddScalarBoolOperand(false);  // Use NHWC format
      *nn_op_type = NEURON_TRANSPOSE_CONV_2D;
    } else if (type == "SpaceToDepth") {
      Argument block_size = get_arg(op_def, "block_size");
      AddScalarInt32Operand(block_size.i());  // block_size
      *nn_op_type = NEURON_SPACE_TO_DEPTH;
    } else if (type == "DepthToSpace") {
      Argument block_size = get_arg(op_def, "block_size");
      AddScalarInt32Operand(block_size.i());  // block_size
      *nn_op_type = NEURON_DEPTH_TO_SPACE;
    }
