import subprocess
import os
from datetime import datetime

# tkinter allow us creating a GUI
import tkinter as tk
from tkinter import filedialog

# ntpath work for all paths on all platforms
import ntpath

def jupyter2md(file_path):

    # Run a bash command to convert .ipynb in .md and create a folder for images
    subprocess.Popen(["jupyter", "nbconvert", file_path, "--to", "markdown"]).communicate() 
    

def move_files(file_path):

    os.chdir(ntpath.dirname(file_path))

    # Get its folder
    jupyter_folder = ntpath.dirname(file_path)
    jupyter_folder = jupyter_folder.split("/")[-1]

    # Get the name of the file with its extension
    jupyter_file = ntpath.basename(file_path)
    
    # Eliminate the extension of the file
    jupyter_name = jupyter_file.split(".")[0]

    # Create the markdown file
    jupyter_md = jupyter_name + ".md"
    date = datetime.today().strftime("%Y-%m-%d-")

    # Get the name of the jupyter's images folder
    jupyter_img = jupyter_name + "_files"
    
    # Move the created files to their respective folders
    jupyter_md_folder = "../../_posts/" + jupyter_folder
    jupyter_img_folder = "../../assets/" + jupyter_folder
    
    subprocess.run(["mv", jupyter_md, jupyter_md_folder + "/" + date + jupyter_md])
    subprocess.run(["mv", jupyter_img, jupyter_img_folder])


# Frame
root = tk.Tk()
root.withdraw()

print("Select the Jupyter file path")
jupyter_path = filedialog.askopenfilename()

# Convert jupyter files to markdown
jupyter2md(jupyter_path)

# Move the created files to their respective folders
move_files(jupyter_path)

input("\nFile converted to markdown, press enter to exit")
