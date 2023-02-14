import os, csv, copy, config
import pandas as pd
import numpy as np
from util.database import HorseRacingDB
from util.video_util import load_vid, find_fps
from ast import literal_eval
import gc


def get_directories(result_dir, tasks):
    return [os.path.join(result_dir, d) for task in tasks for d in config.directories[task].values()]

def create_dir(directory_list):
    for directory in directory_list:
        if not os.path.exists(directory):
            os.mkdir(directory)

def get_data_from_database(vid_names):
    '''
    connect to the mongoDB to obtain the jockeys and race info such as course, distance and track
    
    Parameters
    ----------
    vid_names : string
    
    Returns
    -------
    jockeys : dict
        key : Video name
        item: A list of jockey numbers for the jockeys that took part in the race and successfully finish
    race_info : dict
        key : Video name
        item: a dictionary {'track': track, 'distance': distance, 'course': course}
    '''
    jockeys, race_info = {}, {}
    # separate the video names into framesetid and racelabel. They will be processed differently
    framesetidlist = [int(vid_name) for vid_name in vid_names if len(vid_name) == 4]
    racelabellist = [vid_name for vid_name in vid_names if len(vid_name) != 4]
    db = HorseRacingDB()
    collection = db.conn.gtinfo['newfullraceinfo']

    # for framesetid
    race_objs = collection.find({'framesetid': { "$in": framesetidlist } })
    for obj in race_objs:
        track, distance, course = obj['trackabbr'], obj['length'], obj['railorcoursetype']
        race_info[str(obj["framesetid"]).zfill(4)] = {"track": track, "distance": distance, "course": course}
        if 'jockeys' in obj.keys():
            jockey_info = obj['jockeys']
            jockeys[str(obj['framesetid']).zfill(4)] = [ int(ji['jockeynumber']) for ji in jockey_info if ji['place'] is not None]

    # for racelabel
    race_objs = collection.find({'racelabel': { "$in": racelabellist } })
    for obj in race_objs:
        track, distance, course = obj['trackabbr'], obj['length'], obj['railorcoursetype']
        race_info[obj["racelabel"]] = {"track": track, "distance": distance, "course": course}   
        if 'jockeys' in obj.keys():
            jockey_info = obj['jockeys']
            jockeys[obj['racelabel']] = [ int(ji['jockeynumber']) for ji in jockey_info if ji['place'] is not None]
    
    return jockeys, race_info

def import_subsequences_results(path):
    # import trajectories from result
    f = open(path + ".csv", "r")
    csv_reader = csv.reader(f, delimiter=",")
    detected_boxes = []
    stf = 0
    edf = 0
    sf_flag = 0
    count = 0
    id_set = set()
    for row in csv_reader:
        if count < 3:
            count += 1
            continue
        if sf_flag == 0:
            stf = int(row[0].split("/img")[-1].split(".")[0])
            sf_flag = 1 
        edf = int(row[0].split("/img")[-1].split(".")[0])
        boxes = [ int(r) for r in row[3:] if r != ""]
        bxs = []
        for c in range(int(len(boxes) / 5)):
            idd = int(boxes[c*5])
            bxs.append([int((boxes[c*5+4])/2+boxes[c*5+2]),int((boxes[c*5+3])/2+boxes[c*5+1]), idd])
            if idd not in id_set:
                id_set.add(idd)
        detected_boxes.append(bxs)

    return detected_boxes

class Result:

    def __init__(self, result_dir, vid_dir, model_dir, tasks, label):
            
        '''
        This is the fucntion for generating the result object. This object is responsible for managing the input arguments such as model_dir, result_dir, track, etc. It is also responsible for the input/output of data. 
        
        Paramters:
        ----------
        result_dir : string
            the directory of the output
        vid_dir : string
            the directory for storing the videos (internal model) / the path to the video file (client mode)
        model_dir : string
            the directory for storing the trained deep learning models
        tasks : list of int
            task list, e.g. [0,1,2,6,7]
        label : the specific name that the client want for a particular output file name.
        
        Notes
        -----
        
        '''
        
        self.result_dir = result_dir
        self.vid_dir = vid_dir
        self.model_dir = model_dir
        
        self.tasks = tasks
        self.ontask = None # this variable is for keeping track of which task the system is running on. This is updated every new task
        self.processing_vid_name = None # this variable is for keeping track of which video the system is running on. This is updated every new video
        self.label = label
 
        # generate the directories for storing the results
        self.make_result_dir(result_dir, tasks)
        
        # get the list of video names, checking whether the vid_dir is a directory or path, and check whether the extension is valid
        self.vid_names = [ vid_name[:-4] for vid_name in os.listdir(vid_dir) if vid_name[-4:] in config.accepted_formats] if os.path.isdir(self.vid_dir) else [self.label] if self.label != "" else [vid_name[:-4] for vid_name in [self.vid_dir.split(os.sep)[-1]] if vid_name[-4:] in config.accepted_formats]
        
        # read the proccessed start/end/camera change
        #self.read_scene_classification_results()
        self.start_end_frames,self.cam_changes = self.get_scene_classification_results_from_cache(result_dir)

        #print("loaded start/end: ", self.start_end_frames)
        #print("loaded cc: ", self.cam_changes)
        # check which videos to process (exclude the previously processed videos)
        self.selected_vid_names = list(set([vid_name for task in tasks for vid_name in self.select_vid_names(result_dir, self.vid_names, task)]))    
        self.selected_vid_names.sort()

        # record for each video, which task is needed (exclude the previously processed task)
        self.selected_vid_names_by_task = {task: self.select_vid_names(result_dir, self.vid_names, task) for task in tasks}
        print("***Task arrangements: ", self.selected_vid_names_by_task)
        # find race info and number of jockeys from database
        self.jockeys, self.race_info = data = get_data_from_database(self.selected_vid_names) if os.path.isdir(self.vid_dir) else ({}, {})
        
        # frame data, this is used for checking whether to load frames. (If frames are loaded already, no need to load again)
        self.frames = None 
        self.frames_of_video = None
        self.frames_of_task = None
        
    # def read_scene_classification_results(self):
    #     file_path = self.result_dir + '/' + config.scene_classification_save_path
    #     if not os.path.exists(file_path):
    #         saved_preds = pd.DataFrame(columns=['video_name', 'start', 'end', 'cc_lst'])
    #         saved_preds.to_csv(file_path, index=False)
    #         self.start_end_frames, self.cam_changes= {}, {}
    #     else:
    #         self.start_end_frames, self.cam_changes= {}, {}
    #         saved_preds = pd.read_csv(file_path, dtype={'video_name': 'str'})
    #         #saved_preds = saved_preds.set_index('video_name')
    #         for _, row in saved_preds.iterrows():
    #             self.start_end_frames[row['video_name']] = [row['start'],row['end']]
    #             self.cam_changes[row['video_name']] = literal_eval(row['cc_lst'])

    def get_scene_classification_results_from_cache(self,result_dir):
        '''
        create the csv file for storing start-end frame and camera changes if not exist. If exist, load data into memory

        Paramters:
        ----------
        result_dir : string
            the directory of the output
        
        Notes
        -----

        '''

        ### Start-end frames
        if not os.path.exists(os.path.join(result_dir, config.video_processer_save_path)):
            with open(os.path.join(result_dir, config.video_processer_save_path), "w") as video_preprocess_file:
                video_preprocess_file.write("raceLabel,track,course,raceDist,md5,fps,coursePred,raceNumPred,raceDistPred,raceTimePred,startFrmNum,finishFrmNum\n")
        with open(os.path.join(result_dir, config.video_processer_save_path), "r") as video_preprocess_file:
            start_end_frames = list(csv.reader(video_preprocess_file, delimiter=","))[1:]
            start_end_frames = {sef[0]: [int(sef[-2]), int(sef[-1])] for sef in start_end_frames}

        ### Camera changes
        if not os.path.exists(os.path.join(result_dir, config.camerachange_save_path)):
            with open(os.path.join(result_dir, config.camerachange_save_path), "w") as cam_change_file:
                cam_change_file.write("raceLabel,cc0_start,cc0_end,cc1_start,cc1_end,cc2_start,cc2_end,cc3_start,cc3_end,...\n")
        with open(os.path.join(result_dir, config.camerachange_save_path), "r") as cam_change_file:
            cam_change_data = list(csv.reader(cam_change_file, delimiter=","))
            #print(cam_change_data)
            cam_changes = {data[0]: [[int(data[i]),int(data[i+1])] for i in range(1,len(data),2)] for did, data in enumerate(cam_change_data) if did > 0}
        return start_end_frames,cam_changes

    def load_frames(self, vid_name, start, end):
    
        '''
        load video frames, call load_vid from util.video_util
        
        Parameters
        ----------
        vid_name : string
            video name (with out extension)
        start : int
            the starting frame of the frame set for loading
        end : int
            the ending frame of the frame set for loading
        '''
        
        for form in config.accepted_formats: # check the video extension
            vid_path = os.path.join(self.vid_dir, vid_name + form) if os.path.isdir(self.vid_dir) else self.vid_dir
            if os.path.exists(vid_path):
                # do not reload frames if the whole video can be reused
                #if self.frames is None or self.frames_of_video != self.processing_vid_name or self.frames_of_task == config.Task.VIDEOPROCESSOR:
                if self.frames is None or self.frames_of_video != self.processing_vid_name or self.frames_of_task == config.Task.SCENECLASSIFY:
                    del self.frames # free memory below loading new frames
                    gc.collect()
                    self.frames = load_vid(vid_path, start, end) 
                self.frames_of_video = self.processing_vid_name
                self.frames_of_task = self.ontask
                return
        raise RuntimeError("Video Not Found Error")
    
    def get_fps(self, vid_name):
        
        '''
        get the fps of the video
        
        Parameters
        ----------
        vid_name : string
            video name (with out extension)
        
        Returns
        -------
        fps : int
        '''
    
        for form in config.accepted_formats:
            vid_path = os.path.join(self.vid_dir, vid_name + form) if os.path.isdir(self.vid_dir) else self.vid_dir
            if os.path.exists(vid_path):
                return find_fps(vid_path) # the find_fps function in util.video_util
        raise RuntimeError("Video Not Found Error")

    def make_result_dir(self, result_dir, tasks):
        ''' create directories for the result '''
        directory_list = [result_dir] + get_directories(result_dir, tasks)
        create_dir(directory_list)

    def select_vid_names(self, result_dir, vid_names, task):
        '''
        For a particular task, find the set of videos that need processing, exclude the videos that does not need to process
        '''
        if task == 0:
            videos = list(set.difference(set(vid_names), set(self.start_end_frames.keys()))) + \
                     list(set.difference(set(vid_names), set(self.cam_changes.keys())))
            return list(set(videos))
       
        else:
            directories = get_directories(result_dir, [task])
            differences = set()
            for directory in directories:
                if directory.split(os.sep)[-1] != config.directories[config.Task.TRACKING]['ROOT']:
                    # for tracking taskm if finaly result exists, it is done
                    if directory.split(os.sep)[-2] == 'Track_Cap' and directory.split(os.sep)[-1] != 'final_result':
                        continue
                    file_names = [name.split(".")[0].split("&")[0] for name in os.listdir(directory)]
                    difference = set.difference(set(vid_names), set(file_names))
                    differences = differences.union(difference)
            return list(differences)
    
    def convert_to_excel(self):
        '''convert the result to excel'''
        if not os.path.exists(os.path.join(self.result_dir, "xlsx_results")):
            os.mkdir(os.path.join(self.result_dir, "xlsx_results"))

        for vid_name in self.vid_names:

            # export race info such as start frame, end frame, camera change, distance, etc
            if vid_name not in self.start_end_frames.keys() or vid_name not in self.cam_changes.keys():
                continue
            start, end = self.start_end_frames[vid_name]
            scn_max = config.tracking[self.race_info[vid_name]['track']].scn_max
            data = [self.race_info[vid_name]['track'], "", str(self.race_info[vid_name]['distance']), "", start, end]
            column = ['track', 'raceNum', 'raceDist', 'raceTime', 'startFrame', 'endFrame']
            cam_changes = self.cam_changes[vid_name]
            subrace_data = [start] + np.array(cam_changes).flatten().tolist() + [end]
            data += [str(sd) for sd in subrace_data]
            for i in range(int(len(subrace_data)/2)):
                column += ['subrace' + str(len(cam_changes)-i+1) + '_start', 'subrace' + str(len(cam_changes)-i+1) + '_end']
            data.append("")
            column.append("end")
            df = pd.DataFrame([data], columns=column)

            writer = pd.ExcelWriter(os.path.join(self.result_dir, "xlsx_results", vid_name + ".xlsx"),engine='xlsxwriter')
            df.to_excel(writer, sheet_name='Race & Video Config', index=False)
            
            #export tracking result
            if not os.path.exists(os.path.join(self.result_dir, config.directories[config.Task.TRACKING]['FINALRESULT'], vid_name + ".csv")):
                writer.save()
                continue
            detected_boxes = import_subsequences_results(os.path.join(self.result_dir, config.directories[config.Task.TRACKING]['FINALRESULT'], vid_name))
            column = ['FrameNo', 'detectedJockeys'] + [k for j in [['cx'+str(i), 'cy'+str(i)] for i in range(scn_max)] for k in j]
            data = []
            for frm in range(len(detected_boxes)):
                caps = [["",""]]  * scn_max
                for cap in detected_boxes[frm]:
                    if cap[2] > scn_max:
                        continue
                    caps[cap[2]-1] = cap[:2]
                data.append([frm+start, len(detected_boxes[frm])] + [i for c in caps for i in c])
            df_track = pd.DataFrame(data, columns=column)
            
            # write to file
            df_track.to_excel(writer, sheet_name='Jockey Caps', index=False)
            writer.save()
            print("Finish writing", vid_name)

