import os
import json
import cv2
import numpy as np

import pycocotools.mask as mask_save_tool

def encode_masks(masks:np.array):
    enc_masks = np.asfortranarray((masks.astype(np.uint8)))
    enc_masks = mask_save_tool.encode(enc_masks)
    enc_masks = [{'size':d['size'], 'counts':bytes.decode(d['counts'])} for d in enc_masks]
    return enc_masks

def decode_masks(enc_masks:list):
    return mask_save_tool.decode(enc_masks)

# RLE_String=rleEncodedString,mask_size=[96,96],obj_bbox=[x,y,w,h],frame_size=target frame_size
def to_mask(rle_string,mask_size,obj_bbox,frame_size):
    # Build a dict to be able to use the pycocotools 
    rle_dict = {}
    rle_dict['counts'] = rle_string
    rle_dict['size'] = mask_size
    
    # 96x96 mask
    tmp_mask = mask_save_tool.decode([rle_dict])

    # When we generate the RLE in C++, we didnt do the fortran format transformation, so we convert the result back
    tmp_mask = tmp_mask.T.reshape((mask_size[1],mask_size[0]))
    tmp_mask = tmp_mask*255

    # bbox size_mask
    tmp_mask = cv2.resize(tmp_mask,(obj_bbox[2],obj_bbox[3]))

    ### Put bbox size_mask at target frame(not added)
    mask = np.zeros((frame_size[1], frame_size[0]), dtype=tmp_mask.dtype)
    mask[obj_bbox[1]:obj_bbox[1]+obj_bbox[3],obj_bbox[0]:obj_bbox[0]+obj_bbox[2]] = tmp_mask

    return mask

def _get_masks(rle_path_, start_frame_, finish_frame_, object_name):
    target_object = { 'name': object_name, 'id': None }

    # Read rail rle json data
    with open(rle_path_, 'r') as f:
        rle_data = json.load(f)

    # Check object name and get object id
    assert(target_object['name'] in rle_data['object_name_list'])
    index = rle_data['object_name_list'].index(target_object['name'])
    assert(index < len(rle_data['object_id_list']))
    target_object['id'] = rle_data['object_id_list'][index]
    
    start_frame = rle_data['data']['start_frame']
    finish_frame = rle_data['data']['finish_frame']
    frame_size = rle_data['target_frame_size']
    assert(start_frame <= start_frame_ and finish_frame_ <= finish_frame)

    masks = []
    for frame_number in range(start_frame_, finish_frame_ + 1):
        data_cur_frame = rle_data['data']['frame_%s' % frame_number]
        rail_data_cur_frame = [data for data in data_cur_frame if data['object_id'] == target_object['id']]
        mask = np.zeros((frame_size[1], frame_size[0]), dtype=np.uint8)
        for data in rail_data_cur_frame:
            mask_size = data['mask_size']
            obj_bbox = data['bbox']
            rle_string = data['rle_string']
            mask_ = to_mask(rle_string, mask_size, obj_bbox, frame_size)
            mask = np.maximum(mask, mask_)
        masks.append(mask)

    return masks

def get_rail_masks(rle_path, start_frame, finish_frame):
    return _get_masks(rle_path, start_frame, finish_frame, 'Rail')

def get_rail_pole_masks(rle_path, start_frame, finish_frame):
    return _get_masks(rle_path, start_frame, finish_frame, 'RailPole')

def get_masks_for_homograhy(rail_mask_path, rail_pole_mask_path, semantic_mask_path, start_frame, end_frame):
    '''
    Read the rail masks and the segmentation masks into memory and assemble them into a single mask per frame for optical flow estimation
    '''
    
    rail_data = json.load(open(rail_mask_path, "r"))
    rail_data = rail_data['data']

    pole_data = json.load(open(rail_pole_mask_path, "r"))
    pole_data = pole_data['data']
    
    seg_data = json.load(open(semantic_mask_path, "r"))
    seg_data = seg_data['data']
    
    masks = np.zeros((end_frame-start_frame+1, 1080, 1920), dtype=np.bool)
    
    print("Loading Segmenation Result. Takes a approximately a minute")
    
    for key in rail_data.keys():
      if key in ['start_frame', 'finish_frame']:
        continue
      frame = int(key.split("_")[-1]) - start_frame
      mask = np.zeros((1080, 1920), np.bool)
      kernel = np.ones((18,18),np.uint8)
      
      # load rail mask
      rail_mask_string = rail_data[key]
      encoded_mask = str.encode(rail_mask_string)
      m = np.asarray(mask_save_tool.decode({'size':[1080, 1920], 'counts': encoded_mask}), np.uint8)
      m = cv2.dilate(m,kernel)
      mask = np.logical_or(mask, m.astype(np.bool))

      # load rail pole mask
      rail_pole_mask_data = pole_data[key]
      for i in range(len(rail_pole_mask_data)):
        loc = rail_pole_mask_data[i]["bbox"]
        
        encoded_mask = str.encode(rail_pole_mask_data[i]["rle_string"])
        m = np.asarray(mask_save_tool.decode({'size':[loc[3], loc[2]], 'counts': encoded_mask}), np.uint8)
        m = cv2.dilate(m,kernel)
        mask[loc[1]:loc[1]+loc[3],loc[0]:loc[0]+loc[2]] = np.logical_or(mask[loc[1]:loc[1]+loc[3],loc[0]:loc[0]+loc[2]], m.astype(np.bool))
      
      # load seg mask keep background, foreground, rail and pole
      seg_mask_data = seg_data[key]
      for i in range(len(seg_mask_data)):
        object_id = seg_mask_data[i]["object_id"]   
        if object_id in [0, 3]:
          encoded_mask = str.encode(seg_mask_data[i]["rle_string"])
          decoded_mask = np.asarray(mask_save_tool.decode({'size':[1080, 1920], 'counts': encoded_mask}), np.uint8).astype(np.bool)
          mask[35:1080-35,35:1920-35] = np.logical_or(mask[35:1080-35,35:1920-35], decoded_mask[35:1080-35,35:1920-35])

      masks[frame,:,:] = mask

    return masks    

