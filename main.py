import time, argparse, os, config
import gc
import multiprocessing as mp
from util.result import Result
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

from memory_profiler import profile

#@profile(stream=open('task_functions.log','a'))
def task_functions(horseracingResult, task):

  '''
  Task Function will import the required dependency for each task, and run the function.
  Ideally, writing the task output to disk should be handled within the task itself. Therefore, there should be no data returned from executing the tasks.

  Parameters
  ----------
  horseracingResult : object
    horseracingResult is the class that manage the input arguments such as model_dir, result_dir, track, etc. It is also responsible for the input/output of data.
  task : int
    the task number
  '''

  if task == config.Task.FRAMEEXTRACTOR:
    from util import FrameSaver
    FrameSaver.save_frame(horseracingResult)
  elif task == config.Task.SCENECLASSIFY:
    from SceneClassifier.predict import find_start_end_cc
    find_start_end_cc(horseracingResult)
  elif task == config.Task.RAILMASK:
    from Segmentation.generate_rail_mask import generate_rail_mask
    generate_rail_mask(horseracingResult)
  elif task == config.Task.RAILPOLEMASK:
    from Segmentation.generate_rail_pole import generate_rail_pole
    generate_rail_pole(horseracingResult)
  elif task == config.Task.SEMANTICMASK:
    from Segmentation.generate_semantic_mask import generate_semantic_mask
    generate_semantic_mask(horseracingResult)
  elif task == config.Task.OPTICALFLOW:
    from OpticalFlow.calculate_homography import get_optical_flow
    get_optical_flow(horseracingResult)
  elif task == config.Task.DETECTION:
    from Detection.detect_cap_jockey_saddlecloth import detect_cap_jockey_saddlecloth
    detect_cap_jockey_saddlecloth(horseracingResult)
  elif task == config.Task.TRACKING:
    from Tracking.find_trajectories import get_trajectory
    get_trajectory(horseracingResult)
  
  return

#@profile(stream=open('launch_tasks.log','a'))
def launch_tasks(horseracingResult, tasks):
  
  '''
  Launch the task one by one from the task list
  
  Parameters
  ----------
  horseracingResult : object
    horseracingResult is the class that manage the input arguments such as model_dir, result_dir, track, etc. It is also responsible for the input/output of data.  
  tasks : list of int
    the task list. It gives the tasks to be run in order  
  
  Returns
  -------
  horseracingResult : object
    As indicated above
  '''
  for task in tasks:
    horseracingResult.ontask = task # set the ontask to the current task number, 
    if horseracingResult.processing_vid_name in horseracingResult.selected_vid_names_by_task[horseracingResult.ontask]:
      task_functions(horseracingResult, task)
      horseracingResult.selected_vid_names_by_task[horseracingResult.ontask].remove(horseracingResult.processing_vid_name)
  # clear frames after use for passing the result object back
  horseracingResult.frames = None
  return horseracingResult

#@profile(stream=open('run_horseracing.log','a'))
def run_horseracing(args):

  '''
  The main function for runing the code
  
  Parameters
  ----------
  
  args : object
    arguments input
    
  Notes
  -----
  
  This code has two modes: internal mode and the client mode. If the argument vid_dir is a folder of videos, then the system will know it is the internal mode. 
  Under the internal mode, the system will not need the arguments of 'track', 'distance', 'course', 'jockeys', 'racelabel' or 'xlsx'. The system will go to the MongoDB to look for the required information.
  If the argument vid_dir points to a video file, then the system runs in client mode and require the full set of arguments as input.
  This set up allows us to easier tune parameters or generate data release by runing the videos in batch, and allows us to draw data from mongoDB.
  '''
  
  # load arguments
  vid_dir, model_dir, result_dir, tasks, track, distance, course, jockeys, label, is_excel = args.video_path, args.model_dir, args.result_dir, args.tasks, args.track, args.distance, args.course, args.jockeys, args.racelabel, args.xlsx
  
  # dependencies record the set of tasks that need to be run if a particular task is selected, i.e. if args.tasks = [2, 7], we need to run [0, 2] for task 2, and [0, 1, 6, 7] for task 7. The combine set is run [0, 1, 2, 6, 7] as a result
  tasks_to_run = list(set([t for task in tasks for t in config.dependencies[int(task)]]))
  print("Runing Tasks: ", tasks_to_run)
  
  # initiate the result function
  horseracingResult = Result(result_dir, vid_dir, model_dir, tasks_to_run, label)
  
  # if the input video_path is a file, not a folder, 
  if not os.path.isdir(vid_dir) and len(horseracingResult.vid_names) > 0:
    horseracingResult.race_info[horseracingResult.vid_names[0]] = {"track": track, "distance": int(distance), 'course': course}
    if len(jockeys) != 0:
      horseracingResult.jockeys[horseracingResult.vid_names[0]] = [int(j) for j in jockeys]
  horseracingResult.selected_vid_names.sort()
  
  #pool = mp.get_context("spawn").Pool(processes=(8)) # comment out, for parameter tuning use only
  for vid_name in horseracingResult.selected_vid_names:
    print("----------------------- Processing Video: ", vid_name, ' -----------------------')
    horseracingResult.processing_vid_name = vid_name
    horseracingResult = launch_tasks(horseracingResult, tasks_to_run)
    #pool.apply_async(launch_tasks, args=(horseracingResult, tasks_to_run,), callback=None) # comment out, for parameter tuning use only
    gc.collect()
    #while len(pool._cache) >= 8: # comment out, for parameter tuning use only
    #  time.sleep(2) # comment out, for parameter tuning use only
  
  #pool.close() # comment out, for parameter tuning use only
  #pool.join() # comment out, for parameter tuning use only
  
  # output to excel in xlsx format
  if is_excel:
    horseracingResult.convert_to_excel()

if __name__ == "__main__":
 
  # necessary for making torch work fine. Torch is set to run in a separate process in the optical flow module. Setting start method to spawn is necessary for this purpose.
  import gc
  import sys
  mp.set_start_method('spawn') 

  if sys.platform in ['linux', 'darwin']:  # Linux or macOS
    import resource
    orig_open_file_lim = resource.getrlimit(resource.RLIMIT_NOFILE)
    if orig_open_file_lim != (4096, 4096):
      resource.setrlimit(resource.RLIMIT_NOFILE, (4096, 4096))  # enlarge read_file limit as to cope with multiprocessing error
      new_open_file_lim = resource.getrlimit(resource.RLIMIT_NOFILE)
      print(f'Changing open file limit (soft, hard): {orig_open_file_lim} --> {new_open_file_lim}')

  parser = argparse.ArgumentParser(description='HorseRacing Parser')
  parser.add_argument('--video_path', type=str, help='the dir to the videos')
  parser.add_argument('--model_dir', type=str, help='the dir to the models')
  parser.add_argument('--result_dir', type=str, help='the dir to the results', default="./Result")
  parser.add_argument('--tasks', nargs='+', type=int, help='what task to complete', default=[0])
  parser.add_argument('--track', type=str, help='track type, i.e. HVT/STT/SG')
  parser.add_argument('--distance', type=str, help='race distance, i.e. 1000/1200/1600')
  parser.add_argument('--course', type=str, help='race course, i.e. C+3')
  parser.add_argument('--jockeys', nargs='+', type=int, help='jockeys in the race', default=[])
  parser.add_argument('--racelabel', type=str, help='Any name specified by the user', default="")
  parser.add_argument('--xlsx', help='Output result as excel files', action='store_true')
  
  args = parser.parse_args()
  run_horseracing(args)