import shutil
import traceback
from os import listdir, makedirs, walk, getcwd, utime
from os.path import isfile, join, splitext, exists, getmtime, relpath
from tkinter.messagebox import askyesno, showinfo, showwarning, showerror
import pathlib
import tkinter.filedialog
import tkinter.ttk
import tkinter as tk
from PIL import ImageTk, Image
import json
import threading

def get_automatic_tags_from_txt_file(image_file):
    #If .txt available, read into automated caption
    txt_file = splitext(image_file)[0] + ".txt"
    try:
        with open(txt_file) as f:
            return ' '.join(f.read().split())
    except:
        pass
    return None

def import_tokenizer_reqs():
    try:
        global torch, open_clip, Image, model, preprocess, tokenizer, use_clip
        print("Importing Tokenizer...")
        import torch
        import open_clip
        from PIL import Image

        model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='laion400m_e32')
        tokenizer = open_clip.get_tokenizer('ViT-L-14')

        use_clip = True
        print("Done!")

    except:
        print("Done!")
        print(traceback.format_exc())
        print("Couldn't load torch or clip, falling back to tiktoken. Token count will be less accurate.")
        use_clip = False
        import tiktoken



#Return approximate number of tokens in string (seems to err on the high side)
#TODO: Add torch as dependency and query model directly for exact value?
def num_tokens_from_string(string: str, encoding_name: str= None) -> int:
    """Returns the number of tokens in a text string."""
    if use_clip:
        tokens = list(tokenizer([string])[0])
        while tokens[-1] == 0:
            tokens.pop()
        return len(tokens) - 2 #Always has a start/end token.
    else:
        encoding = tiktoken.get_encoding(encoding_name)
        num_tokens = len(encoding.encode(string))
        return num_tokens

def import_interrogators():
    try:
        global tagger, utils, interrogator, use_interrogate
        print("Importing automatic caption interrogators...")
        import tagger
        from tagger import utils
        from tagger import interrogator
        tagger.utils.refresh_interrogators()
        print("Done!")

    except:
        print(traceback.format_exc())
        print("Couldn't load clip interrogator. Won't interrogate images for automatic tags, only TXT.")
        use_interrogate = False

def do_interrogate(
        image: Image,

        interrogator: str,
        threshold: float,
        additional_tags: str,
        exclude_tags: str,
        sort_by_alphabetical_order: bool,
        add_confident_as_weight: bool,
        replace_underscore: bool,
        replace_underscore_excludes: str):
    
    if interrogator not in tagger.utils.interrogators:
        return ['', None, None, f"'{interrogator}' is not a valid interrogator"]

    interrogator: tagger.Interrogator = tagger.utils.interrogators[interrogator]

    postprocess_opts = (
        threshold,
        tagger.utils.split_str(additional_tags),
        tagger.utils.split_str(exclude_tags),
        sort_by_alphabetical_order,
        add_confident_as_weight,
        replace_underscore,
        tagger.utils.split_str(replace_underscore_excludes)
    )

    # single process
    if image is not None:
        ratings, tags = interrogator.interrogate(image)
        processed_tags = tagger.Interrogator.postprocess_tags(
            tags,
            *postprocess_opts
        )

        return [
            ', '.join(processed_tags),
            ratings,
            tags,
            ''
        ]
    return ['', None, None, '']
    
use_interrogate = True


def interrogate_automatic_tags(image_file):
    if use_interrogate:
        try:
            image = Image.open(image_file).convert('RGB')
            
            caption = do_interrogate(image, "wd14-vit-v2-git", 0.35, "", "", False, False, True, "0_0, (o)_(o), +_+, +_-, ._., <o>_<o>, <|>_<|>, =_=, >_<, 3_3, 6_9, >_o, @_@, ^_^, o_o, u_u, x_x, |_|, ||_||")[0]
            return caption
        except:
            print(traceback.format_exc())            
            return get_automatic_tags_from_txt_file(image_file)
    else:
       return get_automatic_tags_from_txt_file(image_file)
    

def truncate_string_to_max_tokens(string : str):
    while num_tokens_from_string(string.strip(), "gpt2") > 75:
        string = " ".join(string.split()[:-1])

    while string.endswith(","):
        string = string[:-1]
    return string

class manually_review_subset_popup(object):
    def __init__(self, parent, subset_path, image_files, review_all):
        self.parent = parent
        self.dataset_path = self.parent.parent.path
        self.subset_path = subset_path
        self.file_index = 0
        self.image_files = image_files
        if not review_all:
            for f in reversed(image_files):
                caption_file = "".join(splitext(f)[:-1]) + ".txt"
                caption = self.get_caption_from_file(caption_file)
                if num_tokens_from_string(caption, "gpt2") <= 75:
                    self.image_files.remove(f)


        self.create_ui()

    def create_ui(self):
        self.top = tk.Toplevel(self.parent.top)
        self.top.title("Manually review captions")
        self.create_primary_frame()
        self.set_ui(self.file_index)

 
    #Create primary frame
    def create_primary_frame(self):
        self.root_frame = tk.Frame(self.top)
        self.top.rowconfigure(0, weight=1)
        self.top.columnconfigure(0, weight=1)
        self.top.minsize(700, 400)

        self.top.wait_visibility()
        self.top.grab_set()
        self.top.transient(self.parent.top)

        self.root_frame.grid(padx=0, pady=0, sticky="nsew")
        self.root_frame.rowconfigure(0, weight = 1)
        self.root_frame.columnconfigure(0, weight = 2)
        self.root_frame.columnconfigure(1, weight = 0)
        self.root_frame.columnconfigure(1, minsize=400)

        self.create_image_frame()
        self.create_form_frame()
        self.statusbar_text = tk.StringVar()
        self.statusbar = tk.Label(self.top, 
                                  textvar=self.statusbar_text, 
                                  bd=1, 
                                  relief=tk.RAISED, 
                                  anchor=tk.W)
        self.statusbar.grid(row=1, column=0, sticky="ew")

    #Create the frame for image display
    def create_image_frame(self):
        self.image_frame = tk.Frame(self.root_frame, 
                              width=400, height=400, 
                              bd=2, 
                              relief=tk.SUNKEN)
        self.image_frame.grid(row=0, column=0, padx=0, pady=0, sticky="nsew")
        self.image_frame.rowconfigure(0, weight=1)
        self.image_frame.columnconfigure(0, weight=1)

        # Display image in image_frame
        self.image = Image.open("icon.png")
        self.framed_image = ImageTk.PhotoImage(self.image)
        self.sizer_frame = tk.Frame(self.image_frame,
                                    width=400, height=400,
                                    bd=0)
        self.sizer_frame.grid(row=0, column=0, sticky="nsew")
        self.sizer_frame.rowconfigure(0, weight=1)
        self.sizer_frame.columnconfigure(0, weight=1)

        self.image_label = tk.Label(self.sizer_frame, 
                                    image=self.framed_image, 
                                    bd=0)
        self.image_label.grid(row=0, column=0, sticky="nsew")
        self.sizer_frame.bind("<Configure>", self.image_resizer)

    #Create the frame for form display
    def create_form_frame(self):
        self.form_frame = tk.Frame(self.root_frame,
                               width=300, height=400, 
                               bd=1,
                               relief=tk.RAISED)
        self.form_frame.grid(row=0, column=1, 
                              padx=0, pady=0, 
                              sticky="nsew")
        
        self.form_frame.columnconfigure(1, weight = 1)

        caption_label = tk.Label(self.form_frame, text="Caption: ")
        caption_label.grid(row=4, column=0, padx=5, pady=(5,0), sticky="sw")

        self.token_count_label = tk.Label(self.form_frame, text="Tokens: 0 / 75")
        self.token_count_label.grid(row=4, column=1, padx=5, pady=(5,0), sticky="se")

        self.caption_textbox = tk.Text(self.form_frame, width=30, height=12, wrap=tk.WORD, spacing2=2, spacing3=2)

        self.caption_textbox.grid(row=5, column=0, 
                             columnspan=2, 
                             padx=5, pady=(0,5), 
                             sticky="ew")
        self.caption_textbox.bind("<Tab>", self.focus_next_widget)
        self.caption_textbox.bind('<Control-a>', self.select_all)
        self.caption_textbox.focus_set()
        self.caption_textbox.bind('<KeyRelease>', self.update_token_count)

        save_txt_btn = tk.Button(self.form_frame, 
                                  text="Auto truncate (Ctrl+T)", 
                                  command=self.auto_truncate)
        save_txt_btn.grid(row=12, column=0, 
                           columnspan=2, 
                           padx=5, pady=5, 
                           sticky="ew")
        self.caption_textbox.bind('<Control-t>', self.auto_truncate)


        save_txt_btn = tk.Button(self.form_frame, 
                                  text="Confirm file (Ctrl+S)", 
                                  command=self.save_txt)
        save_txt_btn.grid(row=13, column=0, 
                           columnspan=2, 
                           padx=5, pady=5, 
                           sticky="ew")
        self.top.bind("<Control-s>", self.save_txt)

        self.prev_file_btn = tk.Button(self.form_frame, 
                                  text="Previous (Ctrl+P/B)", 
                                  command=self.prev_file)
        self.prev_file_btn.grid(row=14, column=0, 
                           padx=5, pady=5, 
                           sticky="ew")
        self.top.bind("<Control-p>", self.prev_file)
        self.top.bind("<Control-b>", self.prev_file)

        self.next_file_btn = tk.Button(self.form_frame, 
                                  text="Next (Ctrl+N/F)", 
                                  command=self.next_file)
        self.next_file_btn.grid(row=14, column=1, 
                           padx=5, pady=5, 
                           sticky="ew")
        self.top.bind("<Control-n>", self.next_file)
        self.top.bind("<Control-f>", self.next_file)   
        self.top.bind("<Control-Home>", self.first_file )
        self.top.bind("<Control-End>", self.last_file)   


    #Create open dataset action
    def open_dataset(self, event = None):
        self.clear_ui()

        #Clear the UI and associated variables
        self.file_index = 0
        self.image_files = []


        #Get supported extensions
        exts = Image.registered_extensions()
        supported_exts = {ex for ex, f in exts.items() if f in Image.OPEN}

        #Get list of filenames matching those extensions
        files = [pathlib.Path(f).absolute()
                 for f in pathlib.Path(self.path).rglob("*")
                  if isfile(join(self.path, f))]
        
        self.image_files = [
            f for f in files if splitext(f)[1] in supported_exts]  

        self.image_files.sort()

        #Point UI to beginning of queue
        if(len(self.image_files) > 0):
            self.file_index = 0
            self.set_ui(self.file_index)

    #Ask user if they want to save if needed
    def save_unsaved_popup(self):
        if(len(self.image_files) == 0 or self.file_index >= len(self.image_files)):
            return False

        index = self.file_index
        caption_file = "".join(splitext(self.image_files[index])[:-1]) + ".txt"
        caption = self.get_caption_from_file(caption_file)

        if(self.get_caption_from_ui() != caption):
            answer = askyesno(parent=self.top,
                              title='Confirm image?',
                            message='You have changed the caption. Confirm now?')
            if answer:
                self.save_txt()
                return True
        return False

    def load_image(self, f):
        oldgeometry = self.top.geometry()
        oldminsize = self.top.minsize()
        oldmaxsize = self.top.maxsize()
        self.top.minsize(width=self.top.winfo_width(), 
                         height=self.top.winfo_height())
        self.top.maxsize(width=self.top.winfo_width(), 
                         height=self.top.winfo_height())
        tgt_width = self.sizer_frame.winfo_width()
        tgt_height = self.sizer_frame.winfo_height()
        try:
            self.image = Image.open(f)

            new_width = int(
                tgt_height * self.image.width / self.image.height)
            new_height = int(
                tgt_width * self.image.height / self.image.width)

            if new_width <= tgt_width:
                resized_image = self.image.resize(
                    (new_width, tgt_height), 
                    Image.LANCZOS)
                
            else:
                resized_image = self.image.resize(
                    (tgt_width, new_height), 
                    Image.LANCZOS)

            self.framed_image = ImageTk.PhotoImage(resized_image)
            self.image_label.configure(image=self.framed_image)
        except:
            self.image_label.configure(image='')

        self.top.geometry(oldgeometry)
        self.top.minsize(width=oldminsize[0], height=oldminsize[1])
        self.top.maxsize(width=oldmaxsize[0], height=oldmaxsize[1])
        self.image_resizer()

    #Resize image to fit resized window
    def image_resizer(self, e = None):
        tgt_width = self.image_frame.winfo_width() - 4
        tgt_height = self.image_frame.winfo_height() - 4

        if tgt_width < 1:
            tgt_width = 1
        if tgt_height < 1:
            tgt_height = 1

        new_width = int(tgt_height * self.image.width / self.image.height)
        new_height = int(tgt_width * self.image.height / self.image.width)

        if new_width < 1:
            new_width = 1
        if new_height < 1:
            new_height = 1

        if new_width <= tgt_width:
            resized_image = self.image.resize(
                (new_width, tgt_height), 
                Image.LANCZOS)
        else:
            resized_image = self.image.resize(
                (tgt_width, new_height), 
                Image.LANCZOS)
        self.framed_image = ImageTk.PhotoImage(resized_image)
        self.image_label.configure(image=self.framed_image)

    #Move the focus to the next item in the form
    def focus_next_widget(self, event):
        event.widget.tk_focusNext().focus()
        return("break")


    #Add UI elements for prev file button
    def prev_file(self, event = None):
        if self.file_index <= 0:
            self.file_index = 0
            return #Nothing to do if we're at first index.
        
        #Pop up unsaved data dialog if needed
        if self.save_unsaved_popup():
            self.file_index -= 1
            self.set_ui(self.file_index)
            return

        #Point UI to previous item in queue
        self.clear_ui()
        self.file_index -= 1
        self.set_ui(self.file_index)


        #Enable/disable buttons as appropriate
        if self.file_index > 0:
            self.prev_file_btn["state"] = "normal"
        else:
            self.prev_file_btn["state"] = "disabled"

        if self.file_index < len(self.image_files) - 1:
            self.next_file_btn["state"] = "normal"
        else:
            self.next_file_btn["state"] = "disabled"



    #Add UI elements for next file button
    def next_file(self, event = None):
        if self.file_index >= len(self.image_files) - 1:
            self.file_index = len(self.image_files) - 1
            return #Nothing to do if we're at first index.
                
        #Pop up unsaved data dialog if needed
        if self.save_unsaved_popup():
            return

        #Point UI to next item in queue
        self.clear_ui()
        self.file_index += 1
        self.set_ui(self.file_index)


    #Add UI elements for next file button
    def first_file(self, event = None):
        #Pop up unsaved data dialog if needed
        if self.save_unsaved_popup():
            return

        #Point UI to next item in queue
        self.clear_ui()
        self.file_index = 0
        self.set_ui(self.file_index)


    #Add UI elements for next file button
    def last_file(self, event = None):
        #Pop up unsaved data dialog if needed
        if self.save_unsaved_popup():
            return

        #Point UI to next item in queue
        self.clear_ui()
        self.file_index = len(self.image_files) - 1
        self.set_ui(self.file_index)


    #Clear the UI
    def clear_ui(self):
        if len(self.image_files) == 0:
            self.close()
            return
        self.form_frame.destroy()
        self.image = Image.open("icon.png")
        self.framed_image = ImageTk.PhotoImage(self.image)
        self.image_label.configure(image=self.framed_image)
        self.create_form_frame()
        self.form_frame.lift()
        self.statusbar_text.set("")

    def update_token_count(self, event = None):
        count = num_tokens_from_string(self.get_caption_from_ui(), "gpt2")
        self.token_count_label.configure(text=f"Tokens: {count} / 75")


    #Set the UI to the given item's values
    def set_ui(self, index: int):
        self.clear_ui()
        
        if(len(self.image_files) == 0 or self.file_index >= len(self.image_files)):
            return False


        caption_file = "".join(splitext(self.image_files[index])[:-1]) + ".txt"
        try:
            self.caption_textbox.insert(
                "1.0", 
                self.get_caption_from_file(caption_file))
        except:
            pass


        f = self.image_files[index]        
        self.load_image(f)
        
        self.statusbar_text.set(
            f"Image {1 + self.file_index}/{len(self.image_files)}: "
            f"{pathlib.Path(self.image_files[self.file_index]).name}")
        
        #Enable/disable buttons as appropriate
        if self.file_index > 0:
            self.prev_file_btn["state"] = "normal"
        else:
            self.prev_file_btn["state"] = "disabled"

        if self.file_index < len(self.image_files) - 1:
            self.next_file_btn["state"] = "normal"
        else:
            self.next_file_btn["state"] = "disabled"

        self.update_token_count()


    def select_all(self, event):
        # select text
        try:
            event.widget.select_range(0, 'end')
        except:
            event.widget.tag_add("sel", "1.0", "end")

        # move cursor to the end
        try:
            event.widget.icursor('end')
        except:
            event.widget.mark_set("insert", "end")

        #stop propagation
        return 'break'


    def get_caption_from_ui(self):
        return ' '.join(self.caption_textbox.get("1.0", "end").split())

    def get_caption_from_file(self, caption_file):
        caption = ""
        try:
            with open(caption_file) as f:
                caption = f.read()
        except:
            showwarning(parent=self.top,
                      title="Couldn't read caption",
                      message=f"Could not read TXT file {caption_file}")
            print(traceback.format_exc())

        return caption


    def write_caption_to_file(self, caption, caption_file):
        try:
            with open(caption_file, "w") as f:
                f.write(caption)
        except:
            showerror(parent=self,
                      title="Couldn't save caption",
                      message=f"Could not save TXT file {caption_file}")
            print(traceback.format_exc())

    def auto_truncate(self, event = None):
        caption = self.get_caption_from_ui()

        truncated = truncate_string_to_max_tokens(caption)
        self.caption_textbox.delete("1.0", "end")
        self.caption_textbox.insert("1.0", truncated)
        return "break"
        
    #Add UI elements for save JSON button
    def save_txt(self, event = None):
        self.write_caption_to_file(
            self.get_caption_from_ui(),
            "".join(splitext(self.image_files[self.file_index])[:-1]) + ".txt")
        del self.image_files[self.file_index]
        if self.file_index >= len(self.image_files):
            self.file_index -= 1
        if self.file_index < 0:
            self.close()
        self.set_ui(self.file_index)

    def close(self, event = None):
        self.top.grab_release()
        self.top.destroy()



class NumericEntry(tk.Entry):
    def __init__(self, master=None, **kwargs):
        self.var = tk.StringVar()
        tk.Entry.__init__(self, master, textvariable=self.var, **kwargs)
        self.old_value = ''
        self.var.trace('w', self.check)
        self.get, self.set = self.var.get, self.var.set

    def check(self, *args):
        if self.get().isdigit() or self.get() == "": 
            # the current value is only digits; allow this
            self.old_value = self.get()
        else:
            # there's non-digit characters in the input; reject this 
            self.set(self.old_value)


class generate_lora_subset_popup(object):
    def __init__(self, parent):
        self.parent = parent

        self.create_ui()

    def create_ui(self):
        self.top = tk.Toplevel(self.parent)
        self.top.title("Generate LoRA subset...")
        self.top.wait_visibility()
        self.top.grab_set()
        self.top.rowconfigure(0, weight=1)
        self.top.columnconfigure(0, weight=1)
        self.top.minsize(600, 400)
        self.top.transient(self.parent)

        self.form_frame = tk.Frame(self.top, 
                                   borderwidth=2,
                                   relief='raised',)
        self.form_frame.grid(row=0, column=0, 
                        padx=0, pady=0, 
                        sticky="nsew")
        
        self.form_frame.columnconfigure(1, weight=1)
        self.form_frame.rowconfigure(9, weight=1)

        #LoRA Output location
        output_path_label = tk.Label(self.form_frame, text="Output path: ")
        output_path_label.grid(row=0, column=0, padx=(5, 0), pady=5, sticky="e")

        self.output_path = tk.StringVar(None)
        self.output_path.set(pathlib.Path().absolute() / "lora_subsets")
        self.output_path.trace("w", lambda name, index, mode: 
                                self.populate_from_newest_subset())
        output_path_entry = tk.Entry(self.form_frame,
                                     textvariable=self.output_path, 
                                     justify="left")
        output_path_entry.grid(row=0, column=1, 
                               padx=(0, 5), pady=5, 
                               sticky="ew")
        output_path_entry.bind('<Control-a>', self.select_all)

        #Browse button
        browse_btn = tk.Button(self.form_frame, text='Browse...', 
                               command=self.browse)
        browse_btn.grid(row=0, column=2, padx=4, pady=4, sticky="sew")
        self.top.bind("<Control-b>", self.browse)

        #LoRA name
        lora_name_label = tk.Label(self.form_frame, text="LoRA name: ")
        lora_name_label.grid(row=1, column=0,
                             padx=(5, 0), pady=5, 
                             sticky="e")

        self.lora_name = tk.StringVar(None)
        self.lora_name.set("_".join(self.find_newest_subset("").split("_")[1:]))
        self.nametracer = self.lora_name.trace("w", lambda name, index, mode: 
                                self.populate_from_newest_subset())
        lora_name_entry = tk.Entry(self.form_frame,
                                     textvariable=self.lora_name, 
                                     justify="left")
        lora_name_entry.grid(row=1, column=1, 
                             columnspan=2,
                             padx=(0, 5), pady=5, 
                             sticky="ew")
        lora_name_entry.bind('<Control-a>', self.select_all)
        lora_name_entry.focus_set()
        lora_name_entry.select_range(0, 'end')


        #Labeled group for options
        settings_group = tk.LabelFrame(self.form_frame, 
                                    text="Settings")
        settings_group.grid(row=2, column=0, 
                            columnspan=3, 
                            padx=5, pady=5,
                            sticky="nsew")

        #Checkbox for inclusion of trigger word
        self.include_lora_name = tk.BooleanVar(None)
        self.include_lora_name.set(True)
        include_lora_name_chk = tk.Checkbutton(
            settings_group,
            var=self.include_lora_name,
            text="LoRA name included as trigger")
        include_lora_name_chk.grid(row=0, column=0, padx=5, pady=5, sticky="w")

        #Checkbox for inclusion of artist
        self.include_artist = tk.BooleanVar(None)
        self.include_artist.set(False)
        include_artist_chk = tk.Checkbutton(
            settings_group,
            var=self.include_artist,
            text="Artist included as trigger")
        include_artist_chk.grid(row=1, column=0, padx=5, pady=5, sticky="w")

        #Checkbox for inclusion of style
        self.include_style = tk.BooleanVar(None)
        self.include_style.set(False)
        include_style_chk = tk.Checkbutton(
            settings_group,
            var=self.include_style,
            text="Style included in caption")
        include_style_chk.grid(row=2, column=0, padx=5, pady=5, sticky="w")

        #Checkbox for inclusion of summary
        self.include_summary = tk.BooleanVar(None)
        self.include_summary.set(True)
        include_summary_chk = tk.Checkbutton(
            settings_group,
            var=self.include_summary,
            text="Summary included in caption")
        include_summary_chk.grid(row=3, column=0, padx=5, pady=5, sticky="w")

        #Checkbox for inclusion of this LoRA's associated description
        self.include_feature = tk.BooleanVar(None)
        self.include_feature.set(True)
        include_feature_chk = tk.Checkbutton(
            settings_group,
            var=self.include_feature,
            text="Feature for LoRA name included")
        include_feature_chk.grid(row=4, column=0, padx=5, pady=5, sticky="w")

        #Checkbox for inclusion of other feature descriptions
        self.include_other_features = tk.BooleanVar(None)
        self.include_other_features.set(True)
        include_other_features_chk = tk.Checkbutton(
            settings_group,
            var=self.include_other_features,
            text="Other features included")
        include_other_features_chk.grid(row=5, column=0, padx=5, pady=5, sticky="w")

        #Checkbox for inclusion of this LoRA's associated description
        self.include_automatic_tags = tk.BooleanVar(None)
        self.include_automatic_tags.set(True)
        include_automatic_tags_chk = tk.Checkbutton(
            settings_group,
            var=self.include_automatic_tags,
            text="Automatic tags included in caption")
        include_automatic_tags_chk.grid(row=6, column=0, padx=5, pady=5, sticky="w")

        #Checkbox to ask if captions should be manually reviewed if over token max
        review_group = tk.LabelFrame(settings_group, 
                                    text="Manual review options")
        review_group.grid(row=0, column=1, rowspan=3, columnspan=2,
                          sticky="ew")

        self.review_option = tk.IntVar()

        self.review_option.set(1)
        tk.Radiobutton(review_group, 
           text=f"None",
           variable=self.review_option, 
           value=0).grid(row=0, column=0, sticky="w")
        tk.Radiobutton(review_group, 
           text=f"Auto-truncate to 75 tokens",
           variable=self.review_option, 
           value=1).grid(row=1, column=0, sticky="w")
        tk.Radiobutton(review_group, 
           text=f"Review if over 75 tokens",
           variable=self.review_option, 
           value=2).grid(row=2, column=0, sticky="w")
        tk.Radiobutton(review_group, 
           text=f"Review all",
           variable=self.review_option, 
           value=3).grid(row=3, column=0, sticky="w")


        #Steps per image
        steps_per_image_label = tk.Label(settings_group, text="Steps per image: ")
        steps_per_image_label.grid(row=3, column=1,
                             padx=(10, 0), pady=5, 
                             sticky="w")

        self.steps_per_image_entry = NumericEntry(settings_group,
                                     justify="left")
        self.steps_per_image_entry.set(100)
        self.steps_per_image_entry.grid(row=3, column=2, 
                             padx=(0, 5), pady=5, 
                             sticky="ew")
        self.steps_per_image_entry.bind('<Control-a>', self.select_all)

        # Cancel button
        cancel_btn = tk.Button(self.form_frame, text='Cancel', 
                               command=self.cancel)
        cancel_btn.grid(row=10, column=0, padx=4, pady=4, sticky="sew")
        self.top.bind("<Escape>", self.cancel)
        self.top.bind("<Control-g>", self.generate)

        # Generate button
        generate_btn = tk.Button(self.form_frame, text='Generate (Ctrl+G)', 
                               command=self.generate)
        generate_btn.grid(row=10, column=1,
                          columnspan=2,
                          padx=4, pady=4, 
                          sticky="sew")
        
        self.populate_from_newest_subset()

    def select_all(self, event):
        # select text
        try:
            event.widget.select_range(0, 'end')
        except:
            event.widget.tag_add("sel", "1.0", "end")

        # move cursor to the end
        try:
            event.widget.icursor('end')
        except:
            event.widget.mark_set("insert", "end")

        #stop propagation
        return 'break'

    def close(self):
        self.top.grab_release()
        self.top.destroy()

    def browse(self, event = None):
        #Popup folder selection dialog
        default_dir = pathlib.Path().absolute() / "lora_subsets"
        try:
            path = tk.filedialog.askdirectory(
                parent=self.top, 
                initialdir=default_dir,
                title="Select a location for subset output")
        except:
            return

        if path:
            pl_path = pathlib.Path(path)
            pl_parent_path = pathlib.Path(self.parent.path)
            if(pl_path in pl_parent_path.parents 
               or pl_parent_path in pl_path.parents
               or pl_path == pl_parent_path):
                showerror(message=
                          "Output path may not be an ancestor of dataset path, "
                          "nor may it be within the dataset tree. This is to "
                          "reduce the chances of clobbering the dataset with "
                          "the subset.")
                self.output_path.set(default_dir)
            else:
                self.output_path.set(path)

    def cancel(self, event = None):
        self.close()

    #Save an info JSON for this subset to identify it as ours
    def save_subset_info(self, path):
        info_path = path / "LoRA_info.json"
        info = {
            "lora_tag_helper_version": 1,
            "name": self.lora_name.get(),
            "include_lora_name": self.include_lora_name.get(),
            "include_artist": self.include_artist.get(),
            "include_style": self.include_style.get(),
            "include_summary": self.include_summary.get(),
            "include_feature": self.include_feature.get(),
            "include_other_features": self.include_other_features.get(),
            "include_automatic_tags": self.include_automatic_tags.get(),
            "review_option": self.review_option.get(),
            "steps_per_image": self.steps_per_image_entry.get(),
        }
        with open(info_path, "w") as f:
            json.dump(info, f, indent=4)
        utime(path)

    #Load info JSON from a subset
    def load_subset_info(self, path):
        info_path = pathlib.Path(path) / "LoRA_info.json"
        try:
            with open(info_path) as f:
                info = json.load(f)
            return info
        except:
            return None
        

    def find_newest_subset(self, lora_name):
        #Find the newest subset matching the current lora name.
        try:
            dirs = [x for x in next(walk(self.output_path.get()))[1]]
            matching_dirs = [pathlib.Path(self.output_path.get()) / x 
                             for x in dirs if x.endswith(lora_name)]
            
            ordered_dirs = sorted(matching_dirs, key=getmtime)

            ordered_info = [self.load_subset_info(x) for x in ordered_dirs]

            for dir, info in zip(reversed(ordered_dirs), reversed(ordered_info)):
                if info:
                    try:
                        if("name" in info
                           and "include_lora_name" in info
                           and "include_artist" in info
                           and "include_style" in info
                           and "include_summary" in info
                           and "include_feature" in info
                           and "include_other_features" in info
                           and "include_automatic_tags" in info
                           and "review_option" in info
                           and "steps_per_image" in info):
                            return pathlib.Path(dir).name
                    except:
                        pass
        except:
            pass
        return "100_default"
        
    #Find the newest appropriate subset in the path and populate info from it
    def populate_from_newest_subset(self):
        #Find the newest subset matching the current lora name.
        try:
            subset = self.find_newest_subset(self.lora_name.get())
            subset_path = pathlib.Path(self.output_path.get()) / subset
            info = self.load_subset_info(subset_path)
            if info:
                if(self.lora_name.get() == ""
                   or self.lora_name.get() == "default"):
                    self.lora_name.set(info["name"])
                self.include_lora_name.set(info["include_lora_name"])
                self.include_artist.set(info["include_artist"])
                self.include_style.set(info["include_style"])
                self.include_summary.set(info["include_summary"])
                self.include_feature.set(info["include_feature"])
                self.include_other_features.set(info["include_other_features"])
                self.include_automatic_tags.set(info["include_automatic_tags"])
                self.review_option.set(info["review_option"])
                self.steps_per_image_entry.set(info["steps_per_image"])
        except:
            pass
        

    def generate(self, event = None):
        #Validate output path
        default_dir = pathlib.Path().absolute() / "lora_subsets"
        output_path = pathlib.Path(self.output_path.get())
        dataset_path = pathlib.Path(self.parent.path)        
        if(output_path in dataset_path.parents 
           or dataset_path in output_path.parents
           or output_path == dataset_path):
            showerror(message=
                  "Output path must not be an ancestor of dataset path, "
                  "nor may it be within the dataset tree. This is to "
                  "reduce the chances of clobbering the dataset with "
                  "the subset.")
            self.output_path.set(default_dir)
            return

        #Validate LoRA name
        self.lora_name.trace_vdelete("w", self.nametracer)
        self.lora_name.set('_'.join(self.lora_name.get().split()))
        self.nametracer = self.lora_name.trace("w", lambda name, index, mode: 
                                self.populate_from_newest_subset())

        #  Make num_name folder or error if it already exists and isn't labeled
        #  as LoRA subset
        subset_path = (pathlib.Path(self.output_path.get())
                         / '_'.join([self.steps_per_image_entry.get(),
                                    self.lora_name.get()]))
        
        if not exists(subset_path):
            makedirs(subset_path)
        else:
            info = self.load_subset_info(subset_path)
            if not info:
                showerror(message=
                          "The output directory exists, but does not have "
                          "valid subset information. Aborting to avoid "
                          "clobbering non-subset directory.")                
                return

        self.save_subset_info(subset_path)

        popup = tk.Toplevel(self.top)
        tk.Label(popup, text="Processing subset images...").grid(row=0,column=0)
        progress_var = tk.DoubleVar()
        progress_var.set(0)
        progress_bar = tk.ttk.Progressbar(popup, variable=progress_var, maximum=100)
        progress_bar.grid(row=1, column=0)#.pack(fill=tk.X, expand=1, side=tk.BOTTOM)
        popup.pack_slaves()
        i = 0
        output_images = []
        #For each image
        for path in self.parent.image_files:
            #Update progress bar
            progress_var.set(100 * i / len(self.parent.image_files))
            popup.update()
            i = i + 1

            #Load associated JSON and/or TXT as normal
            item = self.parent.get_item_from_file(path)

            #Get unique flat name for file
            tgt_image = relpath(pathlib.Path(path), self.parent.path)
            tgt_parent= relpath(pathlib.Path(tgt_image).parent)
            tgt_basename = relpath(pathlib.Path(tgt_image).name)
            tgt_image = tgt_basename
            tgt_name = "".join(tgt_basename)[:-1]
            tgt_ext = splitext(tgt_image)[-1]

            if(tgt_name != item["title"] and item["title"]):
                tgt_image = str(pathlib.Path(tgt_parent) / item["title"]) + tgt_ext

            tgt_image = tgt_image.replace("/", "_")
            i = 2
            while exists(tgt_image):
                tgt_image = splitext(tgt_image)[0] + f"_{i}" + tgt_ext

            tgt_prefix = splitext(tgt_image)[0]


            #TODO: Trim to token max if requested
            
            #Crop image and output to subset folder
            crop = item["crop"]
            if crop != [0, 0, 1, 1]:            
                with Image.open(path) as cropped_img:
                    cropped_img = cropped_img.crop(
                        (crop[0] * cropped_img.width,
                         crop[1] * cropped_img.height,
                         crop[2] * cropped_img.width, 
                         crop[3] * cropped_img.height))
                    cropped_img.save(subset_path / tgt_image)
            else:
                shutil.copy2(path, subset_path / tgt_image)

            output_images.append(subset_path / tgt_image)

            #Copy JSON to subset folder
            json_file = "".join(splitext(path)[:-1]) + ".json"
            shutil.copy2(json_file, str(subset_path / tgt_prefix) + ".json")

            #Save .txt to subset folder
            caption = ""
            if self.include_lora_name.get():
                caption += self.lora_name.get() + ", "
            
            if self.include_style.get() and item["style"]:
                caption += item["style"]

                if self.include_artist.get():
                    caption += " by "
                else:
                    caption += ", "

            if(self.include_artist.get()
               and item["artist"] 
               and item["artist"] != "unknown"):
                caption += item["artist"] + ", "
            
            if self.include_summary.get() and item["summary"]:
                caption += item["summary"] + ", "
            
            if(self.include_feature.get()
               and self.lora_name.get() in item["features"]
               and item["features"]):
                caption += item["features"][self.lora_name.get()] + ", "

            if self.include_other_features.get():
                for f in item["features"]:
                    if f != self.lora_name.get() and item["features"][f]:
                        caption += item["features"][f] + ", "
            
            if self.include_automatic_tags.get() and item["automatic_tags"]:
                caption += item["automatic_tags"]
            
            if caption.endswith(", "):
                caption = caption[:-2]


            components = caption.lower().split(",")

            unique_components_forward = []
            for c in components:
                found = False
                for u_c in unique_components_forward:
                    if c.strip() in u_c.strip():
                        found = True
                if not found:
                    unique_components_forward.append(c.strip())

            unique_components = []
            for c in reversed(unique_components_forward):
                found = False
                for u_c in unique_components:
                    if c in u_c:
                        found = True
                if not found:
                    unique_components.append(c)

            caption = ", ".join(reversed(unique_components))

            if self.review_option.get() == 1: #Auto-truncate
                caption = truncate_string_to_max_tokens(caption)
            with open(str(subset_path / tgt_prefix) + ".txt", "w") as f:
                f.write(" ".join(caption.split()))

        popup.destroy()

        #Pop up box for manual review
        try:
            print(f"About to wait for window: {self.review_option.get()}")
            if self.review_option.get() > 1:                    
                self.top.wait_window(manually_review_subset_popup(
                        self,
                        subset_path,
                        output_images,
                        self.review_option.get() > 2).top)

        except:
            print(traceback.format_exc())

        #Message showing stats of created subset
        try:
            showinfo(parent=self.top,
                     title="Subset written",
                     message=f"Wrote {len(self.parent.image_files)} txt captions "
                              "to subset folder.")

            self.close()
        except:
            pass
        

# the given message with a bouncing progress bar will appear for as long as func is running, returns same as if func was run normally
# a pb_length of None will result in the progress bar filling the window whose width is set by the length of msg
# Ex:  run_func_with_loading_popup(lambda: task('joe'), photo_img)  
def run_func_with_loading_popup(parent, func, msg, window_title = None, bounce_speed = 8, pb_length = None):
    func_return_l = []
    top = tk.Toplevel(parent)

    if isinstance(parent, lora_tag_helper):
        x = parent.winfo_x() + 200
        y = parent.winfo_y() + 200
    
        top.geometry(f"+{x}+{y}")

    
    
    class _main_frame(object):
        def __init__(self, top, window_title, bounce_speed, pb_length):
            print('top of Main_Frame')
            self.func = func
            # save root reference
            self.top = top
            # set title bar
            self.top.title(window_title)

            self.bounce_speed = bounce_speed
            self.pb_length = pb_length

            self.msg_lbl = tk.Label(top, text=msg)
            self.msg_lbl.pack(padx = 10, pady = 5)

            # the progress bar will be referenced in the "bar handling" and "work" threads
            self.load_bar = tk.ttk.Progressbar(top)
            self.load_bar.pack(padx = 10, pady = (0,10))

            self.bar_init()


        def bar_init(self):
            # first layer of isolation, note var being passed along to the self.start_bar function
            # target is the function being started on a new thread, so the "bar handler" thread
            self.start_bar_thread = threading.Thread(target=self.start_bar, args=())
            # start the bar handling thread
            self.start_bar_thread.start()

        def start_bar(self):
            # the load_bar needs to be configured for indeterminate amount of bouncing
            self.load_bar.config(mode='indeterminate', maximum=100, value=0, length = self.pb_length)
            # 8 here is for speed of bounce
            self.load_bar.start(self.bounce_speed)            
#             self.load_bar.start(8)            

            self.work_thread = threading.Thread(target=self.work_task, args=())
            self.work_thread.start()

            # close the work thread
            self.work_thread.join()


            self.top.destroy()
#             # stop the indeterminate bouncing
#             self.load_bar.stop()
#             # reconfigure the bar so it appears reset
#             self.load_bar.config(value=0, maximum=0)

        def work_task(self):
            func_return_l.append(func())

    # call Main_Frame class with reference to root as top
    _main_frame(top, window_title, bounce_speed, pb_length)
    parent.wait_window(top)
    return func_return_l[0]


#Application class
class lora_tag_helper(tk.Tk):

    #Constructor
    def __init__(self):
        super().__init__()
        self.already_initialized = False
        self.image_files = []
        self.file_index = 0
        self.features = []
        self.l_pct = 0
        self.t_pct = 0
        self.r_pct = 1
        self.b_pct = 1

        self.create_ui()
        self.wm_protocol("WM_DELETE_WINDOW", self.quit)
        self.bind("<Visibility>", self.import_reqs)

    def import_reqs(self, event):
        if not self.already_initialized and event.widget is not self:
            self.already_initialized = True
            run_func_with_loading_popup(
                    self,
                    lambda: import_tokenizer_reqs(),
                    "Importing Tokenizer Requirements...", 
                    "Importing Tokenizer Requirements...")

            run_func_with_loading_popup(
                    self,
                    lambda: import_interrogators(), 
                    "Importing Interrogator Requirements...", 
                    "Importing Interrogator Requirements...")

    #Create all UI elements
    def create_ui(self):
        # Set window info
        self.iconphoto(self, tk.PhotoImage(file="icon_256.png")) 
        self.title("LoRA Tag Helper")

        self.create_menu()
        self.create_primary_frame()

 
    #Create primary frame
    def create_primary_frame(self):
        self.root_frame = tk.Frame(self)
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)
        self.minsize(500, 670)
        self.root_frame.grid(padx=0, pady=0, sticky="nsew")
        self.root_frame.rowconfigure(0, weight = 1)
        self.root_frame.columnconfigure(0, weight = 2)
        self.root_frame.columnconfigure(1, weight = 0)
        self.root_frame.columnconfigure(1, minsize=400)

        self.create_image_frame()
        self.create_form_frame()
        self.create_initial_frame()
        self.statusbar_text = tk.StringVar()
        self.statusbar = tk.Label(self, 
                                  textvar=self.statusbar_text, 
                                  bd=1, 
                                  relief=tk.RAISED, 
                                  anchor=tk.W)
        self.statusbar.grid(row=1, column=0, sticky="ew")

    # Create main menu bar
    def create_menu(self):
        menu_bar = tk.Menu(self)
        file_menu = tk.Menu(menu_bar, tearoff=0)


        file_menu.add_command(label="Open dataset...", 
                              command=self.open_dataset, 
                              underline=0, 
                              accelerator="Ctrl+O")
        self.bind("<Control-o>", self.open_dataset)

        file_menu.add_command(label="Generate Lora subset...", 
                              command=self.generate_lora_subset, 
                              underline=10, 
                              accelerator="Ctrl+L")
        self.bind("<Control-l>", self.generate_lora_subset)

        file_menu.add_command(label="Interrogate all automatic tags...", 
                              command=self.update_all_automatic_tags, 
                              underline=10, 
                              accelerator="Ctrl+Shift+T")
        self.bind("<Control-T>", self.update_all_automatic_tags)


        file_menu.add_command(label="Exit", 
                              command=self.quit, 
                              underline=1, 
                              accelerator="Ctrl+Q")
        self.bind_all("<Control-q>", self.quit)

        #Add the complete menu bar to the file menu
        menu_bar.add_cascade(label="File", menu=file_menu, underline=0)
        self.config(menu=menu_bar)

    #Create the frame for image display
    def create_image_frame(self):
        self.image_frame = tk.Frame(self.root_frame, 
                              width=400, height=400, 
                              bd=2, 
                              relief=tk.SUNKEN)
        self.image_frame.grid(row=0, column=0, padx=0, pady=0, sticky="nsew")
        self.image_frame.rowconfigure(0, weight=1)
        self.image_frame.columnconfigure(0, weight=1)

        # Display image in image_frame
        self.image = Image.open("icon.png")
        self.framed_image = ImageTk.PhotoImage(self.image)
        self.sizer_frame = tk.Frame(self.image_frame,
                                    width=400, height=400,
                                    bd=0)
        self.sizer_frame.grid(row=0, column=0, sticky="nsew")
        self.sizer_frame.rowconfigure(0, weight=1)
        self.sizer_frame.columnconfigure(0, weight=1)

        self.sizer_frame.bind("<Configure>", self.image_resizer)

        self.x = self.y = 0
        self.canvas = tk.Canvas(self.sizer_frame, cursor="cross")

        self.canvas.grid(row=0,column=0,sticky="nswe")

        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_move_press)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)

        self.rect = None

        self.start_x = None
        self.start_y = None

        center_x = self.sizer_frame.winfo_width() / 2
        center_y = self.sizer_frame.winfo_height() / 2

        self.canvas.create_image(center_x, center_y, anchor="center",image=self.framed_image)

    def on_button_press(self,event):
        # save mouse drag start position
        self.start_x = self.canvas.canvasx(event.x)
        self.start_y = self.canvas.canvasy(event.y)
        self.on_move_press(event)
        
    def on_move_press(self, event):
        curx = self.canvas.canvasx(event.x)
        cury = self.canvas.canvasy(event.y)

        l_pct, t_pct = self.coord_to_pct(curx, cury)
        r_pct, b_pct = self.coord_to_pct(self.start_x, self.start_y)

        self.l_pct = min(l_pct, r_pct)
        self.t_pct = min(t_pct, b_pct)
        self.r_pct = max(l_pct, r_pct)
        self.b_pct = max(t_pct, b_pct)

        self.l_pct = max(0, min(self.l_pct, 1))
        self.t_pct = max(0, min(self.t_pct, 1))
        self.r_pct = max(0, min(self.r_pct, 1))
        self.b_pct = max(0, min(self.b_pct, 1))

        # expand rectangle as you drag the mouse
        self.generate_crop_rectangle()
    
    def coord_to_pct(self, x, y):
        w = self.image_frame.winfo_width() - 4
        h = self.image_frame.winfo_height() - 4
        x_offset = (w - self.image_width) / 2
        y_offset = (h - self.image_height) / 2
        return ((x - x_offset) / self.image_width, 
                (y - y_offset) / self.image_height)

    def pct_to_coord(self, x_pct, y_pct):
        w = self.image_frame.winfo_width() - 4
        h = self.image_frame.winfo_height() - 4
        x_offset = (w - self.image_width) / 2
        y_offset = (h - self.image_height) / 2
        return (int(x_pct * self.image_width + x_offset),
                int(y_pct * self.image_height + y_offset))

    def generate_crop_rectangle(self):
        f_w = self.image_frame.winfo_width() - 4
        f_h = self.image_frame.winfo_height() - 4
        x_offset = (f_w - self.image_width) / 2
        y_offset = (f_h - self.image_height) / 2

        try:
            if self.crop_left_area:
                self.canvas.delete(self.crop_left_area)
            if self.crop_top_area:
                self.canvas.delete(self.crop_top_area)
            if self.crop_right_area:
                self.canvas.delete(self.crop_right_area)
            if self.crop_bottom_area:
                self.canvas.delete(self.crop_bottom_area)
        except:
            pass

        l, t = self.pct_to_coord(self.l_pct, self.t_pct)
        r, b = self.pct_to_coord(self.r_pct, self.b_pct)

        
        w = int(l - x_offset)
        h = int(f_h - 2 * y_offset)
        fill = self.canvas.winfo_rgb("red") + (int((0.5 if w > 1 else 0) * 255),)
        image = Image.new('RGBA', (w, h), fill)
        self.crop_left_image = ImageTk.PhotoImage(image)
        self.crop_left_area = self.canvas.create_image(x_offset, y_offset, image=self.crop_left_image, anchor="nw")

        w = int(r - l)
        h = int(t - y_offset)
        fill = self.canvas.winfo_rgb("red") + (int((0.5 if h > 1 else 0) * 255),)
        image = Image.new('RGBA', (w, h), fill)
        self.crop_top_image = ImageTk.PhotoImage(image)
        self.crop_top_area = self.canvas.create_image(l, y_offset, image=self.crop_top_image, anchor="nw")

        w = int(f_w - r - x_offset)
        h = int(f_h - 2 * y_offset)
        fill = self.canvas.winfo_rgb("red") + (int((0.5 if w > 1 else 0) * 255),)
        image = Image.new('RGBA', (w, h), fill)
        self.crop_right_image = ImageTk.PhotoImage(image)
        self.crop_right_area = self.canvas.create_image(r, y_offset, image=self.crop_right_image, anchor="nw")

        w = int(r - l)
        h = int(f_h - b - y_offset)
        fill = self.canvas.winfo_rgb("red") + (int((0.5 if h > 1 else 0) * 255),)
        image = Image.new('RGBA', (w, h), fill)
        self.crop_bottom_image = ImageTk.PhotoImage(image)
        self.crop_bottom_area = self.canvas.create_image(l, b, image=self.crop_bottom_image, anchor="nw")


    def on_button_release(self, event):
        coord1 = self.pct_to_coord(self.l_pct, self.t_pct)
        coord2 = self.pct_to_coord(self.r_pct, self.b_pct)
        if(coord2[0] - coord1[0] < 5
           and coord2[1] - coord1[1] < 5):
            self.l_pct = 0
            self.t_pct = 0
            self.r_pct = 1
            self.b_pct = 1
            self.canvas.delete(self.crop_left_area)
            self.canvas.delete(self.crop_top_area)
            self.canvas.delete(self.crop_right_area)
            self.canvas.delete(self.crop_bottom_area)

    #Create the initial frame display
    def create_initial_frame(self):
        self.initial_frame = tk.Frame(self.root_frame,
                               width=300, height=400, 
                               bd=1, 
                               relief=tk.RAISED)
        self.initial_frame.columnconfigure(0, weight = 1)
        self.initial_frame.rowconfigure(0, weight = 1)
        self.initial_frame.rowconfigure(2, weight = 1)
        self.add_initial_buttons()
        self.show_initial_frame()

    #Create the frame for form display
    def create_form_frame(self):
        self.form_frame = tk.Frame(self.root_frame,
                               width=300, height=400, 
                               bd=1,
                               relief=tk.RAISED)
        self.form_frame.columnconfigure(1, weight = 1)

        self.add_artist_entry()
        self.add_style_entry()
        self.add_title_entry()
        self.add_rating_entry()
        self.add_summary_text()
        self.add_features_table()
        self.add_automatic_tags_text()        
        self.add_form_buttons()
        self.show_form_frame()

    #Hide the right hand form controls
    def hide_form_frame(self):
        self.form_frame.grid_forget()

    #Show the right hand form controls
    def show_form_frame(self):
        self.form_frame.grid(row=0, column=1, 
                              padx=0, pady=0, 
                              sticky="nsew")

    #Hide the initial "Open a dataset" prompt
    def hide_initial_frame(self):
        self.initial_frame.grid_forget()

    #Show the initial "Open a dataset" prompt
    def show_initial_frame(self):
        self.initial_frame.grid(row=0, column=1, 
                              padx=0, pady=0, 
                              sticky="nsew")
        self.initial_frame.lift()

    #Add the star rating query to the form
    def add_rating_entry(self):
        #Labeled group for rating
        rating_group = tk.LabelFrame(self.form_frame, 
                                    text="Quality for Training")
        rating_group.grid(row=3, column=0, 
                          columnspan=2, 
                          padx=5, pady=5,
                          sticky="nsew")

        self.rating = tk.IntVar()
        self.rating.set(0)
        tk.Radiobutton(rating_group, 
           text=f"Not rated",
           variable=self.rating, 
           value=0).grid(row=0, column=0, sticky="w")
        for i in range(1, 6):
            tk.Radiobutton(rating_group, 
               text=f"{i}",
               variable=self.rating, 
               value=i).grid(row=0, column=i, sticky="w")


    #Add the artist query to the form
    def add_artist_entry(self):
        artist_name_label = tk.Label(self.form_frame, text="Artist: ")
        artist_name_label.grid(row=0, column=0, padx=0, pady=5, sticky="e")

        self.artist_name = tk.StringVar(None)
        self.artist_name.set("unknown")
        artist_name_entry = tk.Entry(self.form_frame,
                                     textvariable=self.artist_name, 
                                     justify="left")
        artist_name_entry.grid(row=0, column=1, padx=(0, 5), pady=5, sticky="ew")
        artist_name_entry.bind('<Control-a>', self.select_all)
        artist_name_entry.focus_set()
        artist_name_entry.select_range(0, 'end')

    #Add the style query to the form
    def add_style_entry(self):
        style_label = tk.Label(self.form_frame, text="Style: ")
        style_label.grid(row=1, column=0, padx=0, pady=5, sticky="e")

        self.style = tk.StringVar(None)
        self.style.set("")
        style_entry = tk.Entry(self.form_frame,
                                     textvariable=self.style, 
                                     justify="left")
        style_entry.grid(row=1, column=1, padx=(0, 5), pady=5, sticky="ew")
        style_entry.bind('<Control-a>', self.select_all)

    #Add the title query to the form
    def add_title_entry(self):
        title_label = tk.Label(self.form_frame, text="Title: ")
        title_label.grid(row=2, column=0, padx=0, pady=5, sticky="e")

        self.title_var = tk.StringVar(None)
        self.title_var.set("untitled")
        title_entry = tk.Entry(self.form_frame, textvariable=self.title_var, justify="left")
        title_entry.grid(row=2, column=1, padx=(0, 5), pady=5, sticky="ew")
        title_entry.bind('<Control-a>', self.select_all)

    #Move the focus to the next item in the form
    def focus_next_widget(self, event):
        event.widget.tk_focusNext().focus()
        return("break")

    #Add the summary text box to the form
    def add_summary_text(self):
        summary_label = tk.Label(self.form_frame, text="Summary: ")
        summary_label.grid(row=4, column=0, padx=5, pady=(5,0), sticky="sw")

        self.summary_textbox = tk.Text(self.form_frame, width=30, height=4, wrap=tk.WORD, spacing2=2, spacing3=2)
        self.summary_textbox.grid(row=5, column=0, 
                             columnspan=2, 
                             padx=5, pady=(0,5), 
                             sticky="ew")
        self.summary_textbox.bind("<Tab>", self.focus_next_widget)
        self.summary_textbox.bind('<Control-a>', self.select_all)

    #Add the features table to the form
    def add_features_table(self):
        self.features_group = tk.LabelFrame(self.form_frame, 
                                    text="Features")
        self.features_group.grid(row=6, column=0, 
                            columnspan=3, 
                            padx=5, pady=5,
                            sticky="nsew")

        self.features_group.rowconfigure(1, weight=1)
        self.features_group.columnconfigure(0, weight=1)
        self.features_group.columnconfigure(1, weight=3)

        features_name_label = tk.Label(self.features_group, text="Name")
        features_name_label.grid(row=0, column=0, padx=5, pady=0, sticky="ew")
        features_desc_label = tk.Label(self.features_group, text="Description")
        features_desc_label.grid(row=0, column=1, 
                                 padx=5, pady=0, 
                                 sticky="ew")

        #Populate feature table
        for _ in range(2):
            self.add_row()

    #Add the automated tag text box to the form
    def add_automatic_tags_text(self):
        automatic_tags_label = tk.Label(self.form_frame, text="Automated tags: ")
        automatic_tags_label.grid(row=9, column=0, padx=5, pady=(5, 0), sticky="sw")

        self.automatic_tags_textbox = tk.Text(self.form_frame, width=30, height=4, wrap=tk.WORD, spacing2=2, spacing3=2)
        self.automatic_tags_textbox.grid(row=10, column=0, 
                                    columnspan=2, 
                                    padx=5, pady=(0, 5), 
                                    sticky="ew")

        self.automatic_tags_textbox.bind("<Tab>", self.focus_next_widget)
        self.automatic_tags_textbox.bind('<Control-a>', self.select_all)

        self.form_frame.rowconfigure(11, weight=1)

        import_tags_btn = tk.Button(self.form_frame, 
                                  text="Import automatic tags from TXT (Ctrl+T)", 
                                  command=self.update_ui_automatic_tags)
        import_tags_btn.grid(row=12, column=0, 
                             columnspan=2, 
                             padx=5, pady=5, 
                             sticky="ew")
        
        self.bind("<Control-t>", self.update_ui_automatic_tags)

        save_json_btn = tk.Button(self.form_frame, 
                                  text="Save JSON (Ctrl+S)", 
                                  command=self.save_json)
        save_json_btn.grid(row=13, column=0, 
                           columnspan=2, 
                           padx=5, pady=5, 
                           sticky="ew")
        self.bind("<Control-s>", self.save_json)


    #Add the "Open a dataset" prompt button to the initial display
    def add_initial_buttons(self):
        self.initial_btn = tk.Button(self.initial_frame, 
                                  text="Open Dataset... (Ctrl+O)", 
                                  command=self.open_dataset)
        self.initial_btn.grid(row=2, column=0, 
                           padx=5, pady=5, 
                           sticky="ew")
                                  
    #Add the save and navigation buttons to the right-hand form
    def add_form_buttons(self):
        self.prev_file_btn = tk.Button(self.form_frame, 
                                  text="Previous (Ctrl+P/B)", 
                                  command=self.prev_file)
        self.prev_file_btn.grid(row=14, column=0, 
                           padx=5, pady=5, 
                           sticky="ew")
        self.bind("<Control-p>", self.prev_file)
        self.bind("<Control-b>", self.prev_file)

        self.next_file_btn = tk.Button(self.form_frame, 
                                  text="Next (Ctrl+N/F)", 
                                  command=self.next_file)
        self.next_file_btn.grid(row=14, column=1, 
                           padx=5, pady=5, 
                           sticky="ew")
        self.bind("<Control-n>", self.next_file)
        self.bind("<Control-f>", self.next_file)        
        

    #Clear the UI
    def clear_ui(self):
        self.form_frame.destroy()
        self.features = []
        self.image = Image.open("icon.png")
        self.framed_image = ImageTk.PhotoImage(self.image)
        self.canvas.delete(self.image_handle)
        self.create_form_frame()
        self.form_frame.lift()
        self.statusbar_text.set("")

    #Set the UI to the given item's values
    def set_ui(self, index: int):
        self.clear_ui()
        item = self.get_item_from_file(self.image_files[index])
        f = self.image_files[index]        
        self.load_image(f)

        try: 
            self.artist_name.set(item["artist"])
        except: 
            pass

        try: 
            self.style.set(item["artist"])
        except: 
            pass

        try:
            self.title_var.set(item["title"])
        except:
            pass


        try:
            self.style.set(item["style"])
        except:
            pass

        try:
            self.rating.set(item["rating"])
        except:
            pass

        try:
            self.summary_textbox.insert("1.0", item["summary"])
        except:
            pass

        try:
            self.l_pct = item["crop"][0]
            self.t_pct = item["crop"][1]
            self.r_pct = item["crop"][2]
            self.b_pct = item["crop"][3]
        except:
            pass

        self.generate_crop_rectangle()

        try:
            i = 0
            for k, v in item["features"].items():
                if(i >= len(self.features)):
                    self.add_row()                
                self.features[i][0]["var"].set(k)
                self.features[i][1]["var"].set(v)
                i += 1
        except:
            pass

        try:
            self.automatic_tags_textbox.insert("1.0", item["automatic_tags"])
        except:
            pass
        
        self.statusbar_text.set(f"Image {1 + self.file_index}/{len(self.image_files)}: "
                                f"{pathlib.Path(self.image_files[self.file_index]).name}")
       

    #Create open dataset action
    def open_dataset(self, event = None):
        self.clear_ui()
        self.show_initial_frame()

        #Clear the UI and associated variables
        self.file_index = 0
        self.image_files = []

        #Popup folder selection dialog
        try:
            self.path = tk.filedialog.askdirectory(
                parent=self, 
                initialdir="./dataset",
                title="Select a dataset")
            if not self.path:
                return
        except:
            return
        

        #Get supported extensions
        exts = Image.registered_extensions()
        supported_exts = {ex for ex, f in exts.items() if f in Image.OPEN}

        #Get list of filenames matching those extensions
        files = [pathlib.Path(f).absolute()
                 for f in pathlib.Path(self.path).rglob("*")
                  if isfile(join(self.path, f))]
        
        self.image_files = [
            f for f in files if splitext(f)[1] in supported_exts]  

        self.image_files.sort()

        #Populate JSONs
        for path in self.image_files:
            json_file = splitext(path)[0] + ".json"
            self.write_item_to_file(self.get_item_from_file(path), json_file)

        #Enable/disable buttons as appropriate
        if self.file_index > 0:
            self.prev_file_btn["state"] = "normal"
        else:
            self.prev_file_btn["state"] = "disabled"

        if self.file_index < len(self.image_files) - 1:
            self.next_file_btn["state"] = "normal"
        else:
            self.next_file_btn["state"] = "disabled"

        #Point UI to beginning of queue
        if(len(self.image_files) > 0):
            self.file_index = 0
            self.set_ui(self.file_index)
            self.hide_initial_frame()
        else:
            showwarning(parent=self,
                        title="Empty Dataset",
                        message="No supported images found in dataset")
            self.show_initial_frame()


    #Create open dataset action
    def generate_lora_subset(self, event = None):
        if len(self.image_files) > 0:
            self.save_unsaved_popup()
            #Pop up dialog to gather information and perform generation
            generate_lora_subset_popup(self)

    def load_image(self, f):
        oldgeometry = self.geometry()
        oldminsize = self.minsize()
        oldmaxsize = self.maxsize()
        self.minsize(width=self.winfo_width(), height=self.winfo_height())
        self.maxsize(width=self.winfo_width(), height=self.winfo_height())
        tgt_width = self.sizer_frame.winfo_width()
        tgt_height = self.sizer_frame.winfo_height()
        try:
            self.image = Image.open(f)

            new_width = int(
                tgt_height * self.image.width / self.image.height)
            new_height = int(
                tgt_width * self.image.height / self.image.width)

            if new_width <= tgt_width:
                self.image_width = new_width
                self.image_height = tgt_height
            else:
                self.image_width = tgt_width
                self.image_height = new_height
            resized_image = self.image.resize(
                (new_width, tgt_height), 
                Image.LANCZOS)

            self.framed_image = ImageTk.PhotoImage(resized_image)
            center_x = self.sizer_frame.winfo_width() / 2
            center_y = self.sizer_frame.winfo_height() / 2

            try:
                self.canvas.delete(self.image_handle)
            except:
                pass
            self.image_handle = self.canvas.create_image(center_x, center_y, anchor="center",image=self.framed_image)
        except:
            pass
        self.geometry(oldgeometry)
        self.minsize(width=oldminsize[0], height=oldminsize[1])
        self.maxsize(width=oldmaxsize[0], height=oldmaxsize[1])
        self.image_resizer()

    #Resize image to fit resized window
    def image_resizer(self, e = None):
        try:
            l, t = self.pct_to_coord(self.l_pct, self.t_pct)
            r, b = self.pct_to_coord(self.r_pct, self.b_pct)
        except:
            pass

        tgt_width = self.image_frame.winfo_width() - 4
        tgt_height = self.image_frame.winfo_height() - 4

        if tgt_width < 1:
            tgt_width = 1
        if tgt_height < 1:
            tgt_height = 1

        new_width = int(tgt_height * self.image.width / self.image.height)
        new_height = int(tgt_width * self.image.height / self.image.width)

        if new_width < 1:
            new_width = 1
        if new_height < 1:
            new_height = 1

        if new_width <= tgt_width:
            self.image_width = new_width
            self.image_height = tgt_height
        else:
            self.image_width = tgt_width
            self.image_height = new_height
        resized_image = self.image.resize(
            (self.image_width, self.image_height), 
            Image.LANCZOS)
        self.framed_image = ImageTk.PhotoImage(resized_image)
        #self.image_label.configure(image=self.framed_image)
        center_x = self.sizer_frame.winfo_width() / 2
        center_y = self.sizer_frame.winfo_height() / 2

        try:
            self.canvas.delete(self.image_handle)
        except:
            pass
        self.image_handle = self.canvas.create_image(center_x, center_y, anchor="center",image=self.framed_image)

        try:
            self.generate_crop_rectangle()
        except:
            pass
    #Remove row from feature table
    def remove_row(self, i: int):
        self.features[i][0]["entry"].destroy()
        self.features[i][1]["entry"].destroy()
        del self.features[i]

    #Callback for when feature is modified
    def feature_modified(self, var: str):
        found_i = None
        for i in range(len(self.features)):
            for j in range(len(self.features[i])):
                if self.features[i][j]["var"] is var:
                    found_i = i

        while(len(self.features) > 2
            and not self.features[-1][0]["var"].get()
            and not self.features[-1][1]["var"].get()
            and not self.features[-2][0]["var"].get()
            and not self.features[-2][1]["var"].get()):
             if found_i == len(self.features) - 1:
                 self.remove_row(len(self.features) - 2)
             else:
                 self.remove_row(len(self.features) - 1)
        if(self.features[-1][0]["var"].get()
           or self.features[-1][1]["var"].get()):
            self.add_row()
            
    #Add entry to feature table
    def add_entry(self, i: int, j: int):
        s = tk.StringVar(None)
        s.trace("w", 
                lambda name, index, mode, var=s: self.feature_modified(var))
        if j == 0:
            e = tk.Entry(self.features_group, 
                         textvariable=s, 
                         width=1,
                         justify="right")
        else:
            e = tk.Entry(self.features_group, 
                         textvariable=s, 
                         width=3,
                         justify="left")
        e.grid(row=i + 1, column=j, 
               sticky="ew")
        e.bind('<Control-a>', self.select_all)        
        return {"var":s, "entry":e}

    #Add row to feature table
    def add_row(self):
        row = []
        for j in range(2):
            row.append(self.add_entry(len(self.features), j))
        self.features.append(row)


    def get_item_from_ui(self):
        item = {"lora_tag_helper_version": 1}

        try: 
            item["artist"] = self.artist_name.get()
        except: 
            pass

        try: 
            item["style"] = self.style.get()
        except: 
            pass

        try:
            item["title"] = self.title_var.get()
        except:
            pass

        try:
            item["rating"] = self.rating.get()
        except:
            pass

        try:
            item["summary"] = ' '.join(self.summary_textbox.get("1.0", "end").split())
        except:
            pass

        try:
            item["crop"] = [
                self.l_pct,
                self.t_pct,
                self.r_pct,
                self.b_pct
            ]
        except:
            pass

        try:
            features = {}
            for i in range(len(self.features)):
                if(self.features[i][0]["var"].get()):
                    features.update({self.features[i][0]["var"].get():
                                     self.features[i][1]["var"].get()})
            item["features"] = features
        except:
            print(traceback.format_exc())

        try:
            item["automatic_tags"] = ' '.join(self.automatic_tags_textbox.get("1.0", "end").split())
        except:
            pass

        return item

    def get_item_from_file(self, path):
        #Read filename into title
        item = {"lora_tag_helper_version": 1,
                "title":splitext(pathlib.Path(path).name)[0],
                "artist": "unknown",
                "style": "unknown",
                "rating": 0,
                "summary": "",
                "features": {},
                "crop": [0, 0, 1, 1],
                "automatic_tags": ""}

        #If .txt available, read into automated caption
        txt_file = splitext(path)[0] + ".txt"
        try:
            with open(txt_file) as f:
                item["automatic_tags"] = ' '.join(f.read().split())
        except:
            pass

        #If available, parse JSON into fields
        json_file = splitext(path)[0] + ".json"
        try:
            with open(json_file) as f:
                json_item = json.load(f)
                item.update(json_item)
        except:
            pass

        try:
            if item["lora_tag_helper_version"] > 1:
                print("Warning: file generated by newer version of lora_tag_helper")
        except:
            pass

        return item


    def write_item_to_file(self, item, json_file):
        try:
            with open(json_file, "w") as f:
                json.dump(item, f, indent=4)
        except:
            showerror(parent=self,
                      title="Couldn't save JSON",
                      message="Could not save JSON file")
            print(traceback.format_exc())
        

    #Add UI elements for save JSON button
    def save_json(self, event = None):
        self.write_item_to_file(
            self.get_item_from_ui(),
            splitext(self.image_files[self.file_index])[0] + ".json")

    #Update automatic tags in JSON for image file
    def update_automatic_tags(self, path, popup=False):
        json_file = splitext(path)[0] + ".json"
        item = self.get_item_from_file(json_file)

        if popup:
            item["automatic_tags"] = run_func_with_loading_popup(
                self,
                lambda: interrogate_automatic_tags(path), 
                "Interrogating Image...", 
                "Interrogating Image...")            
        else:
            item["automatic_tags"] = interrogate_automatic_tags(path)

        self.write_item_to_file(item, json_file)

    #Update automatic tags in all JSON files
    def update_all_automatic_tags(self, event = None):
        self.save_unsaved_popup()
        popup = tk.Toplevel(self)
        tk.Label(popup, text="Processing subset images...").grid(row=0,column=0)
        progress_var = tk.DoubleVar()
        progress_var.set(0)
        progress_bar = tk.ttk.Progressbar(popup, variable=progress_var, maximum=100)
        progress_bar.grid(row=1, column=0)#.pack(fill=tk.X, expand=1, side=tk.BOTTOM)
        popup.pack_slaves()
        i = 0
        for f in self.image_files:
            #Update progress bar
            progress_var.set(100 * i / len(self.image_files))
            popup.update()
            i = i + 1            
            self.update_automatic_tags(f, popup=False)
        popup.destroy()
        self.update_ui_automatic_tags()

    #Update automatic tags in all JSON files
    def update_ui_automatic_tags(self, event = None):
        if len(self.image_files) > 0:
            self.save_unsaved_popup()
            self.update_automatic_tags(self.image_files[self.file_index])
            self.set_ui(self.file_index)


    #Add UI elements for prev file button
    def prev_file(self, event = None):
        if self.file_index <= 0:
            self.file_index = 0
            return #Nothing to do if we're at first index.
        
        #Pop up unsaved data dialog if needed
        self.save_unsaved_popup()

        #Point UI to previous item in queue
        self.clear_ui()
        self.file_index -= 1
        self.set_ui(self.file_index)


        #Enable/disable buttons as appropriate
        if self.file_index > 0:
            self.prev_file_btn["state"] = "normal"
        else:
            self.prev_file_btn["state"] = "disabled"

        if self.file_index < len(self.image_files) - 1:
            self.next_file_btn["state"] = "normal"
        else:
            self.next_file_btn["state"] = "disabled"



    #Add UI elements for next file button
    def next_file(self, event = None):
        if self.file_index >= len(self.image_files) - 1:
            self.file_index = len(self.image_files) - 1
            return #Nothing to do if we're at first index.
                
        #Pop up unsaved data dialog if needed
        self.save_unsaved_popup()

        #Point UI to next item in queue
        self.clear_ui()
        self.file_index += 1
        self.set_ui(self.file_index)
        
        #Enable/disable buttons as appropriate
        if self.file_index > 0:
            self.prev_file_btn["state"] = "normal"
        else:
            self.prev_file_btn["state"] = "disabled"

        if self.file_index < len(self.image_files) - 1:
            self.next_file_btn["state"] = "normal"
        else:
            self.next_file_btn["state"] = "disabled"

    #Ask user if they want to save if needed
    def save_unsaved_popup(self):
        if(len(self.image_files) == 0):
           return
        
        json_file = "".join(splitext(self.image_files[self.file_index])[:-1]) + ".json"
        if(self.get_item_from_ui() != self.get_item_from_file(json_file)):
            answer = askyesno(parent=self,
                              title='Save unsaved data?',
                            message='You have unsaved changes. Save JSON now?')
            if answer:
                self.write_item_to_file(self.get_item_from_ui(), json_file)

    def select_all(self, event):
        # select text
        try:
            event.widget.select_range(0, 'end')
        except:
            event.widget.tag_add("sel", "1.0", "end")

        # move cursor to the end
        try:
            event.widget.icursor('end')
        except:
            event.widget.mark_set("insert", "end")

        #stop propagation
        return 'break'

    #Create quit action
    def quit(self, event = None):
        self.save_unsaved_popup()

        self.destroy()




#Application entry point
if __name__ == "__main__":
    #Instantiate the application
    app = lora_tag_helper()
    app.wait_visibility()
    #Let the user do their thing
    app.mainloop()
