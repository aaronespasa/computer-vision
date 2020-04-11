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
    

def prepare_environment(file_path):

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
    md_file = date + jupyter_md

    subprocess.run(["mv", jupyter_md, jupyter_md_folder + "/" + md_file])
    subprocess.run(["mv", jupyter_img, jupyter_img_folder])

    # Add a front matter to the markdown file
    os.chdir(jupyter_md_folder)
    md_author = "aaron"
    jupyter_img_folder = "/".join(jupyter_img_folder.split("/")[2:])

    def entry2var():

        md_title = file_title.get()
        md_category = category.get()

        with open(md_file, 'r+') as f:
            file_content = f.read()
            with open(md_file, "w+") as f:
                f.write("---\n")
                f.write("layout: post\n")
                f.write("title: " + md_title + "\n")
                f.write("category: "+ md_category + "\n")
                f.write("author: " + md_author + "\n")
                f.write("---\n")
                f.write("{% assign imgUrl = " + "\"/" + jupyter_img_folder + "/" + jupyter_img + "/" + "\"" + " | prepend: site.baseurl%}\n")
                f.write(file_content)

        
    root2 = tk.Tk()

    canvas = tk.Canvas(root2, width=400, height=300)
    canvas.pack()

    root2.title("Inserta los datos del archivo Jupyter")
    
    title_label = tk.Label(root2, text="Título")
    canvas.create_window(200, 40, window=title_label)
    
    file_title = tk.Entry(root2)
    canvas.create_window(200, 80, window=file_title)

    category_label = tk.Label(root2, text="Categoría")
    canvas.create_window(200, 160, window=category_label)

    category = tk.Entry(root2)
    canvas.create_window(200, 200, window=category)

    button = tk.Button(root2, text="Ingresar datos", command=entry2var)
    canvas.create_window(200, 250, window=button)

    root2.mainloop()

# Frame
root1 = tk.Tk()
root1.withdraw()

print("Select the Jupyter file path")
JUPYTER_PATH = filedialog.askopenfilename()

# Convert jupyter files to markdown
jupyter2md(JUPYTER_PATH)

# Move the created files to their respective folders
prepare_environment(JUPYTER_PATH)

input("\nFile converted to markdown, press enter to exit")
