import warnings

import numpy as np
import time as time
import os
import matplotlib.pyplot as plt
import shutil
import ffmpeg
from PIL import Image

# Internal imports
from physr.physym import library as Lib
from physr.physym import program as Prog


# ----------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------- ANIMATE FUNCS ----------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

WHITE = (255, 255, 255)

def add_margin(pil_img, top, right, bottom, left, color):
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result

def proportional_resize (pil_img, target_height = 720):
    ratio = (target_height / float(pil_img.height))
    target_width = int(np.floor((float(pil_img.width) * float(ratio))))
    res = pil_img.resize((target_width, target_height), Image.ANTIALIAS)
    return res

def animate_prog_infix(my_program_str, my_lib, out_file="my_infix.mp4", framerate=1):
    # ------ Creating temp folder ------
    temp_folder = "temp_animation_infix"
    if temp_folder not in os.listdir():
        os.mkdir(temp_folder)
    # ------ Create VectPrograms ------
    program_idx = np.array([my_lib.lib_name_to_idx[tok_str] for tok_str in my_program_str])
    length = len(my_program_str)
    program_idx = program_idx[np.newaxis, :]
    my_programs = Prog.VectPrograms(batch_size=1, max_time_step=length, library=my_lib)
    prog_idx = 0
    # ------ Padding start/end so first/last frame stays longer ------
    pad_video_start = framerate*2  # pad for 1 sec
    pad_video_end   = framerate*2  # pad for 2 sec
    n_frames = length + 1 + pad_video_start + pad_video_end
    def temp_file_i_str(i):
        return str(i).zfill(len(str(n_frames)))
    # ------ temp file template ------
    temp_file = os.path.join(temp_folder, "prog_%i_" % prog_idx)
    # ------ Save function ------
    def save (i):
        # Temp file i name
        i_str = temp_file_i_str(i)
        temp_file_i = temp_file + "%s.png"%i_str
        # Generating image
        print("Saving image : %s/%i"%(i_str, length))
        print("  -> %s"%(temp_file_i))
        img = my_programs.get_infix_image( prog_idx             = prog_idx,
                                           replace_dummy_symbol = True,
                                           new_dummy_symbol     = "\square",
                                           do_simplify          = False,
                                           text_size            = 16,
                                           text_pos             = (0.0, 0.5),
                                           figsize              = (4, 1),
                                           dpi                  = 512,
                                           fpath                = temp_file_i,
                                          )
        return None
    # ------ Saving with 0 tokens ------
    save(i=0)
    first_frame_file = temp_file + "%s.png"%temp_file_i_str(0)
    # ------ 1st frame longer ------
    print("Copying first frame so it lasts longer...")
    for i in range (1, pad_video_start+1):
        i_str = temp_file_i_str(i)
        temp_file_i = temp_file + "%s.png" % i_str
        print("Saving image : %s/%i"%(i_str, n_frames))
        print("  -> %s"%(temp_file_i))
        shutil.copyfile(src=first_frame_file, dst=temp_file_i)
    # ------ Appending token one by one ------
    for i in range (1+pad_video_start, pad_video_start+length+1):
        # Appending next token
        j = i-1-pad_video_start # j = 0 to j = length
        my_programs.append(program_idx[:,j]) # 0th token appended -> save(1) as save(0) happens without any token
        # Saving
        save(i=i)
    # ------ Last frame longer ------
    # Copying last frame multiple times so it lasts longer
    print("Copying last frame so it lasts longer...")
    for i in range (pad_video_start+length+1, n_frames):
        i_str = temp_file_i_str(i)
        temp_file_i = temp_file + "%s.png" % temp_file_i_str(i)
        last_frame_file = temp_file + "%s.png"% temp_file_i_str(pad_video_start+length)
        print("Saving image : %s/%i"%(i_str, n_frames))
        print("  -> %s"%(temp_file_i))
        shutil.copyfile(src=last_frame_file, dst=temp_file_i)
    # ------ Animate ------
    print("Animating...")
    (
        ffmpeg
            .input(temp_file + "*.png", pattern_type='glob', framerate=framerate)
            .output(out_file)
            .run(overwrite_output=True)
    )
    # ------ Deleting temp folder ------
    shutil.rmtree(temp_folder)
    print("Animation func is done.")
    return None

def animate_prog_tree(my_program_str, my_lib, out_file="my_tree.mp4", target_height = None,  via_tex= False, framerate=1):
    # ------ Creating temp folder ------
    temp_folder = "temp_animation_tree"
    if temp_folder not in os.listdir():
        os.mkdir(temp_folder)
    # ------ Create VectPrograms ------
    program_idx = np.array([my_lib.lib_name_to_idx[tok_str] for tok_str in my_program_str])
    length = len(my_program_str)
    program_idx = program_idx[np.newaxis, :]
    my_programs = Prog.VectPrograms(batch_size=1, max_time_step=length, library=my_lib)
    prog_idx = 0
    # ------ Padding start/end so first/last frame stays longer ------
    pad_video_start = framerate*2  # pad for 2 sec
    pad_video_end   = framerate*2  # pad for 2 sec
    n_frames = length + 1 + pad_video_start + pad_video_end
    def temp_file_i_str(i):
        return str(i).zfill(len(str(n_frames)))
    # ------ temp file template ------
    temp_file = os.path.join(temp_folder, "prog_%i_" % prog_idx)
    # ------ Padding parameters ------
    pad_margin = 1.2
    padded_width, padded_height = 0,0
    do_pad = False
    # ------ Save function ------
    def save (i, my_progs):
        # Temp file i name
        i_str = temp_file_i_str(i)
        temp_file_i = temp_file + "%s.png"%i_str
        # Generating image
        print("Saving image : %s/%i"%(i_str, n_frames))
        print("  -> %s"%(temp_file_i))
        print(my_progs)
        if via_tex:
            img = my_progs.get_tree_image_via_tex (prog_idx = 0, fname = None, dpi = 600)
        else:
            img = my_progs.get_tree_image(prog_idx=0, fpath=None)
        # Preprocessing
        if do_pad:
            # Padding
            img = add_margin(pil_img=img,
                             right  = abs(padded_width-img.width),
                             bottom = abs(padded_height-img.height),
                             top=0, left=0, color=WHITE)
            if img.width != padded_width or img.height != padded_height:
                warnings.warn("Last frame height or width < intermediate frame => may result in low quality video, increase pad_margin ")
            # Resize
            if target_height is not None:
                img  = proportional_resize(img, target_height = target_height)
        # Saving
        print("  -> width = %i, height = %i"%(img.width, img.height))
        img.save(temp_file_i)
        return img
    # ------ Saving last frame first ------
    # Creating a complete VectPrograms
    my_programs_complete = Prog.VectPrograms(batch_size=1, max_time_step=length, library=my_lib)
    my_programs_complete.set_programs(program_idx)
    # Saving image
    last_frame = save(i=length, my_progs=my_programs_complete)
    # -> Getting padding parameters
    do_pad = True
    padded_width, padded_height = int(last_frame.width*pad_margin), int(last_frame.height*pad_margin) # 1.5 for margin
    # ------ Saving with 0 tokens ------
    save(i=0, my_progs=my_programs)
    first_frame_file = temp_file + "%s.png"%temp_file_i_str(0)
    # ------ 1st frame longer ------
    print("Copying first frame so it lasts longer...")
    for i in range (1, pad_video_start+1):
        i_str = temp_file_i_str(i)
        temp_file_i = temp_file + "%s.png" % i_str
        print("Saving image : %s/%i"%(i_str, n_frames))
        print("  -> %s"%(temp_file_i))
        shutil.copyfile(src=first_frame_file, dst=temp_file_i)
    # ------ Appending token one by one ------
    for i in range (1+pad_video_start, pad_video_start+length+1):
        # Appending next token
        j = i-1-pad_video_start # j = 0 to j = length
        my_programs.append(program_idx[:,j]) # 0th token appended -> save(1) as save(0) happens without any token
        # Saving
        save(i=i, my_progs=my_programs)
    # ------ Last frame longer ------
    # Copying last frame multiple times so it lasts longer
    print("Copying last frame so it lasts longer...")
    for i in range (pad_video_start+length+1, n_frames):
        i_str = temp_file_i_str(i)
        temp_file_i = temp_file + "%s.png" % i_str
        last_frame_file = temp_file + "%s.png"%temp_file_i_str(pad_video_start+length)
        print("Saving image : %s/%i"%(i_str, n_frames))
        print("  -> %s"%(temp_file_i))
        shutil.copyfile(src=last_frame_file, dst=temp_file_i)
    # ------ Animate ------
    print("Animating...")
    (
        ffmpeg
            .input(temp_file + "*.png", pattern_type='glob', framerate=framerate)
            .output(out_file)
            .run(overwrite_output=True)
    )
    # ------ Deleting temp folder ------
    shutil.rmtree(temp_folder)
    print("Animation func is done.")
    return None

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------- TEST CASE ------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

# LIBRARY CONFIG
args_make_tokens = {
                # operations
                "op_names"             : "all",  # or ["mul", "neg", "inv", "sin"]
                "use_protected_ops"    : True,
                # input variables
                "input_var_ids"        : {"x" : 0         , "t" : 1        },
                "input_var_units"      : {"x" : [1, 0, 0] , "t" : [0, 1, 0]},
                "input_var_complexity" : {"x" : 0.        , "t" : 0.       },
                # constants
                "constants"            : {"v_0" : 1.         },
                "constants_units"      : {"v_0" : [1, -1, 0]  },
                "constants_complexity" : {"v_0" : 0.         },
                    }
my_lib = Lib.Library(args_make_tokens = args_make_tokens,
                     superparent_units = [1, -1, 0], superparent_name = "v")


# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------- TEST CASE : ANIMATION ------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

animate_prog_infix (my_program_str=["add", "v_0", "div", "x", "t"], my_lib=my_lib, out_file="my_infix.mp4")

animate_prog_tree  (my_program_str=["add", "v_0", "div", "x", "t"], my_lib=my_lib, out_file="my_tree.mp4", target_height = 1440, via_tex = True)
# VLC -> ctrl+E / command+E -> enable magnification / zoom
# VLC -> E => frame by frame
