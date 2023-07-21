def make_jobfile_from_command_list(jobfile_path, commands):
    """
    Save a jobfile containing commands to run.
    Parameters
    ----------
    jobfile_path : str or path
    commands : list of str
    """
    # Creating a jobfile containing all commands to run
    jobfile_content = ''.join('%s\n' % com for com in commands)
    f = open(jobfile_path, "w")
    f.write(jobfile_content)
    f.close()
    return None