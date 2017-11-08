from symbol_mask_fpn import *

def get_resnet_humanpose_maskrcnn(num_classes=config.NUM_CLASSES):
    rcnn_feat_stride = config.RCNN_FEAT_STRIDE
    data = mx.symbol.Variable(name="data")
    rois = dict()
    label = dict()
    bbox_target = dict()
    bbox_weight = dict()
    mask_target = dict()
    mask_weight = dict()
    keypoint_label = dict()
    for s in rcnn_feat_stride:
        rois['rois_stride%s' % s] = mx.symbol.Variable(name='rois_stride%s' % s)
        label['label_stride%s' % s] = mx.symbol.Variable(name='label_stride%s' % s)
        bbox_target['bbox_target_stride%s' % s] = mx.symbol.Variable(name='bbox_target_stride%s' % s)
        bbox_weight['bbox_weight_stride%s' % s] = mx.symbol.Variable(name='bbox_weight_stride%s' % s)
        mask_target['mask_target_stride%s' % s] = mx.symbol.Variable(name='mask_target_stride%s' % s)
        mask_weight['mask_weight_stride%s' % s] = mx.symbol.Variable(name='mask_weight_stride%s' % s)
        keypoint_label['keypoint_label_stride%s' % s] = mx.symbol.Variable(name='keypoint_label_stride%s' % s)
        
    # reshape input
    for s in rcnn_feat_stride:
        rois['rois_stride%s' % s] = mx.symbol.Reshape(data=rois['rois_stride%s' % s],
                                                      shape=(-1, 5),
                                                      name='rois_stride%s_reshape' % s)

        label['label_stride%s' % s] = mx.symbol.Reshape(data=label['label_stride%s' % s], shape=(-1,), name='label_stride%s_reshape'%s)
        bbox_target['bbox_target_stride%s' % s] = mx.symbol.Reshape(data=bbox_target['bbox_target_stride%s' % s],
                                                                    shape=(-1, 4 * num_classes),
                                                                    name='bbox_target_stride%s_reshape'%s)
        bbox_weight['bbox_weight_stride%s' % s] = mx.symbol.Reshape(data=bbox_weight['bbox_weight_stride%s' % s],
                                                                    shape=(-1, 4 * num_classes),
                                                                    name='bbox_weight_stride%s_reshape'%s)
        mask_target['mask_target_stride%s' % s] = mx.symbol.Reshape(data=mask_target['mask_target_stride%s' % s],
                                                                    shape=(-1, num_classes, 28, 28),
                                                                    name='mask_target_stride%s_reshape'%s)
        mask_weight['mask_weight_stride%s' % s] = mx.symbol.Reshape(data=mask_weight['mask_weight_stride%s' % s],
                                                                    shape=(-1, num_classes, 1, 1),
                                                                    name='mask_weight_stride%s_reshape'%s)
        keypoint_label['keypoint_label_stride%s' % s] = mx.symbol.Reshape(data=mask_target['mask_target_stride%s' % s],
                                                                    shape=(-1, num_classes, 56, 56),
                                                                    name='mask_target_stride%s_reshape'%s)

    label_list = []
    bbox_target_list = []
    bbox_weight_list = []
    mask_target_list = []
    mask_weight_list = []
    keypoint_label_list = []
    for s in rcnn_feat_stride:
        label_list.append(label['label_stride%s' % s])
        bbox_target_list.append(bbox_target['bbox_target_stride%s' % s])
        bbox_weight_list.append(bbox_weight['bbox_weight_stride%s' % s])
        mask_target_list.append(mask_target['mask_target_stride%s' % s])
        mask_weight_list.append(mask_weight['mask_weight_stride%s' % s])
        keypoint_label_list.append(keypoint_label['keypoint_label_stride%s' % s])
        
        
    label = mx.symbol.concat(*label_list, dim=0)
    bbox_target = mx.symbol.concat(*bbox_target_list, dim=0)
    bbox_weight = mx.symbol.concat(*bbox_weight_list, dim=0)
    mask_target = mx.symbol.concat(*mask_target_list, dim=0)
    mask_weight = mx.symbol.concat(*mask_weight_list, dim=0)
    keypoint_label = mx.symbol.concat(*keypoint_label_list, dim=0)
     
    # shared convolutional layers, bottom up
    conv_feat = get_resnet_conv(data)

    # shared convolutional layers, top down
    conv_fpn_feat, _ = get_resnet_conv_down(conv_feat)

    # shared parameters for predictions
    rcnn_fc6_weight     = mx.symbol.Variable('rcnn_fc6_weight')
    rcnn_fc6_bias       = mx.symbol.Variable('rcnn_fc6_bias')
    rcnn_fc7_weight     = mx.symbol.Variable('rcnn_fc7_weight')
    rcnn_fc7_bias       = mx.symbol.Variable('rcnn_fc7_bias')
    rcnn_fc_cls_weight  = mx.symbol.Variable('rcnn_fc_cls_weight')
    rcnn_fc_cls_bias    = mx.symbol.Variable('rcnn_fc_cls_bias')
    rcnn_fc_bbox_weight = mx.symbol.Variable('rcnn_fc_bbox_weight')
    rcnn_fc_bbox_bias   = mx.symbol.Variable('rcnn_fc_bbox_bias')

    mask_conv_1_weight = mx.symbol.Variable('mask_conv_1_weight')
    mask_conv_1_bias   = mx.symbol.Variable('mask_conv_1_bias')
    mask_conv_2_weight = mx.symbol.Variable('mask_conv_2_weight')
    mask_conv_2_bias   = mx.symbol.Variable('mask_conv_2_bias')
    mask_conv_3_weight = mx.symbol.Variable('mask_conv_3_weight')
    mask_conv_3_bias   = mx.symbol.Variable('mask_conv_3_bias')
    mask_conv_4_weight = mx.symbol.Variable('mask_conv_4_weight')
    mask_conv_4_bias   = mx.symbol.Variable('mask_conv_4_bias')
    mask_deconv_1_weight = mx.symbol.Variable('mask_deconv_1_weight')
    mask_deconv_2_weight = mx.symbol.Variable('mask_deconv_2_weight')
    mask_deconv_2_bias = mx.symbol.Variable('mask_deconv_2_bias')

    keypoint_conv_1_weight = mx.symbol.Variable('keypoint_conv_1_weight')
    keypoint_conv_1_bias   = mx.symbol.Variable('keypoint_conv_1_bias')
    keypoint_conv_2_weight = mx.symbol.Variable('keypoint_conv_2_weight')
    keypoint_conv_2_bias   = mx.symbol.Variable('keypoint_conv_2_bias')
    keypoint_conv_3_weight = mx.symbol.Variable('keypoint_conv_3_weight')
    keypoint_conv_3_bias   = mx.symbol.Variable('keypoint_conv_3_bias')
    keypoint_conv_4_weight = mx.symbol.Variable('keypoint_conv_4_weight')
    keypoint_conv_4_bias   = mx.symbol.Variable('keypoint_conv_4_bias')
    keypoint_conv_5_weight = mx.symbol.Variable('keypoint_conv_5_weight')
    keypoint_conv_5_bias   = mx.symbol.Variable('keypoint_conv_5_bias')
    keypoint_conv_6_weight = mx.symbol.Variable('keypoint_conv_6_weight')
    keypoint_conv_6_bias   = mx.symbol.Variable('keypoint_conv_6_bias')
    keypoint_conv_7_weight = mx.symbol.Variable('keypoint_conv_7_weight')
    keypoint_conv_7_bias   = mx.symbol.Variable('keypoint_conv_7_bias')
    keypoint_conv_8_weight = mx.symbol.Variable('keypoint_conv_8_weight')
    keypoint_conv_8_bias   = mx.symbol.Variable('keypoint_conv_8_bias')
    
    rcnn_fc_keypoint_weight = mx.symbol.Variable('rcnn_fc_keypoint_weight')
    rcnn_fc_keypoint_bias = mx.symbol.Variable('rcnn_fc_keypoint_bias')
    
    rcnn_cls_score_list = []
    rcnn_bbox_pred_list = []
    mask_deconv_act_list = []
    keypoint_cls_score_list = []
    for stride in rcnn_feat_stride:
        if config.ROIALIGN:
            roi_pool = mx.symbol.ROIAlign(
                name='roi_pool', data=conv_fpn_feat['stride%s'%stride], rois=rois['rois_stride%s' % stride],
                pooled_size=(14, 14),
                spatial_scale=1.0 / stride)
        else:
            roi_pool = mx.symbol.ROIPooling(
                name='roi_pool', data=conv_fpn_feat['stride%s'%stride], rois=rois['rois_stride%s' % stride],
                pooled_size=(14, 14),
                spatial_scale=1.0 / stride)

        # classification with fc layers
        flatten = mx.symbol.Flatten(data=roi_pool, name="flatten")
        fc6     = mx.symbol.FullyConnected(data=flatten, num_hidden=1024, weight=rcnn_fc6_weight, bias=rcnn_fc6_bias)
        relu6   = mx.symbol.Activation(data=fc6, act_type="relu", name="relu6")
        drop6   = mx.symbol.Dropout(data=relu6, p=0.5, name="drop6")
        fc7     = mx.symbol.FullyConnected(data=drop6, num_hidden=1024, weight=rcnn_fc7_weight, bias=rcnn_fc7_bias)
        relu7   = mx.symbol.Activation(data=fc7, act_type="relu", name="relu7")

        # classification
        cls_score = mx.symbol.FullyConnected(data=relu7, weight=rcnn_fc_cls_weight, bias=rcnn_fc_cls_bias,
                                             num_hidden=num_classes)
        # bounding box regression
        bbox_pred = mx.symbol.FullyConnected(data=relu7, weight=rcnn_fc_bbox_weight, bias=rcnn_fc_bbox_bias,
                                             num_hidden=num_classes * 4)
        rcnn_cls_score_list.append(cls_score)
        rcnn_bbox_pred_list.append(bbox_pred)

        # Keypoint 
        keypoint_conv_1 = mx.symbol.Convolution(
            data=roi_pool, kernel=(3, 3), pad=(1, 1), num_filter=256, workspace=512, weight=keypoint_conv_1_weight, bias=keypoint_conv_1_bias,
            name="mask_conv_1")
        keypoint_conv_2 = mx.symbol.Convolution(
            data=keypoint_conv_1, kernel=(3, 3), pad=(1, 1), num_filter=256, workspace=512, weight=keypoint_conv_2_weight, bias=keypoint_conv_2_bias,
            name="mask_conv_1")
        keypoint_conv_3 = mx.symbol.Convolution(
            data=keypoint_conv_2, kernel=(3, 3), pad=(1, 1), num_filter=256, workspace=512, weight=keypoint_conv_3_weight, bias=keypoint_conv_3_bias,
            name="mask_conv_1")
        keypoint_conv_4  = mx.symbol.Convolution(
            data=keypoint_conv_3, kernel=(3, 3), pad=(1, 1), num_filter=256, workspace=512, weight=keypoint_conv_4_weight, bias=keypoint_conv_4_bias,
            name="mask_conv_1")
        keypoint_conv_5 = mx.symbol.Convolution(
            data=keypoint_conv_4, kernel=(3, 3), pad=(1, 1), num_filter=256, workspace=512, weight=keypoint_conv_5_weight, bias=keypoint_conv_5_bias,
            name="mask_conv_1")
        keypoint_conv_6 = mx.symbol.Convolution(
            data=keypoint_conv_5, kernel=(3, 3), pad=(1, 1), num_filter=256, workspace=512, weight=keypoint_conv_6_weight, bias=keypoint_conv_6_bias,
            name="mask_conv_1")
        keypoint_conv_7 = mx.symbol.Convolution(
            data=keypoint_conv_6, kernel=(3, 3), pad=(1, 1), num_filter=256, workspace=512, weight=keypoint_conv_7_weight, bias=keypoint_conv_7_bias,
            name="mask_conv_1")
        keypoint_conv_8 = mx.symbol.Convolution(
            data=keypoint_conv_7, kernel=(3, 3), pad=(1, 1), num_filter=256, workspace=512, weight=keypoint_conv_8_weight, bias=keypoint_conv_8_bias,
            name="mask_conv_1")
        
        keypoint_deconv_1 = mx.symbol.Deconvolution(data=keypoint_conv_8, kernel=(4, 4), stride=(2, 2), num_filter=256, pad=(1, 1), workspace=512, weight=mask_deconv_1_weight, name="mask_deconv1")
        keypoint_upsampling = mx.symbol.UpSampling(keypoint_deconv_1, scale=2, sample_type='nearest', workspace=512, name='keypoint_upsampling', num_args=1)
        
        keypoint_score = mx.symbol.FullyConnected(data=keypoint_upsampling, weight=rcnn_fc_keypoint_weight, bias=rcnn_fc_keypoint_bias, num_hidden=56*56)
        
        keypoint_cls_score_list.append(keypoint_score)
        
        # MASK
        mask_conv_1 = mx.symbol.Convolution(
            data=roi_pool, kernel=(3, 3), pad=(1, 1), num_filter=256, workspace=512, weight=mask_conv_1_weight, bias=mask_conv_1_bias,
            name="mask_conv_1")
        mask_relu_1 = mx.symbol.Activation(data=mask_conv_1, act_type="relu", name="mask_relu_1")
        mask_conv_2 = mx.symbol.Convolution(
            data=mask_relu_1, kernel=(3, 3), pad=(1, 1), num_filter=256, workspace=512, weight=mask_conv_2_weight, bias=mask_conv_2_bias,
            name="mask_conv_2")
        mask_relu_2 = mx.symbol.Activation(data=mask_conv_2, act_type="relu", name="mask_relu_2")
        mask_conv_3 = mx.symbol.Convolution(
            data=mask_relu_2, kernel=(3, 3), pad=(1, 1), num_filter=256, workspace=512, weight=mask_conv_3_weight, bias=mask_conv_3_bias,
            name="mask_conv_3")
        mask_relu_3 = mx.symbol.Activation(data=mask_conv_3, act_type="relu", name="mask_relu_3")
        mask_conv_4 = mx.symbol.Convolution(
            data=mask_relu_3, kernel=(3, 3), pad=(1, 1), num_filter=256, workspace=512, weight=mask_conv_4_weight, bias=mask_conv_4_bias,
            name="mask_conv_4")
        mask_relu_4 = mx.symbol.Activation(data=mask_conv_4, act_type="relu", name="mask_relu_4")
        mask_deconv_1 = mx.symbol.Deconvolution(data=mask_relu_4, kernel=(4, 4), stride=(2, 2), num_filter=256, pad=(1, 1),
                                                workspace=512, weight=mask_deconv_1_weight, name="mask_deconv1")
        mask_deconv_2 = mx.symbol.Convolution(data=mask_deconv_1, kernel=(1, 1), num_filter=num_classes,
                                              workspace=512, weight=mask_deconv_2_weight, bias=mask_deconv_2_bias, name="mask_deconv2")
        mask_deconv_act_list.append(mask_deconv_2)

    # concat output of each level
    cls_score_concat = mx.symbol.concat(*rcnn_cls_score_list, dim=0)  # [num_rois_4level, num_class]
    bbox_pred_concat = mx.symbol.concat(*rcnn_bbox_pred_list, dim=0)  # [num_rois_4level, num_class*4]
    keypoint_cls_score_concat = mx.symbol.concat(*keypoint_cls_score_list, dim=0)
    
    
    # loss
    cls_prob = mx.symbol.SoftmaxOutput(data=cls_score_concat,
                                           label=label,
                                           multi_output=True,
                                           normalization='valid', use_ignore=True, ignore_label=-1,
                                           name='rcnn_cls_prob')
    bbox_loss_ = bbox_weight * mx.symbol.smooth_l1(name='rcnn_bbox_loss_', scalar=1.0,
                                                   data=(bbox_pred_concat - bbox_target))

    bbox_loss = mx.sym.MakeLoss(name='bbox_loss', data=bbox_loss_, grad_scale=1.0 / config.TRAIN.BATCH_ROIS)
    
    keypoint_loss = mx.symbol.SoftmaxOutput(data=keypoint_cls_score_concat,
                                           label=keypoint_label,
                                           multi_output=True,
                                           normalization='valid', use_ignore=True, ignore_label=-1,
                                           name='rcnn_cls_prob')
    
    rcnn_group = [cls_prob, bbox_loss, keypoint_loss]
    for ind, name, last_shape in zip(range(len(rcnn_group)), ['cls_prob', 'bbox_loss', 
                                                              'keypoint_loss'], [num_classes, num_classes * 4]):
        rcnn_group[ind] = mx.symbol.Reshape(data=rcnn_group[ind], shape=(config.TRAIN.BATCH_IMAGES, -1, last_shape),
                                            name=name + '_reshape')

    mask_act_concat = mx.symbol.concat(*mask_deconv_act_list, dim=0)
    mask_prob = mx.symbol.Activation(data=mask_act_concat, act_type='sigmoid', name="mask_prob")
    mask_output = mx.symbol.Custom(mask_prob=mask_prob, mask_target=mask_target, mask_weight=mask_weight,
                                   label=label, name="mask_output", op_type='MaskOutput')
    mask_group = [mask_output]
    # group output
    group = mx.symbol.Group(rcnn_group+mask_group)
    return group
