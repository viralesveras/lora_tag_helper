from os import listdir, makedirs, walk, getcwd, utime, remove
from os.path import isfile, join, splitext, exists, getmtime, relpath
import time
import threading
import shutil
import pathlib
import re
import traceback
from PIL import ImageTk, Image
import json

from tkinterdnd2 import DND_FILES, TkinterDnD
from tkinter.messagebox import askyesno, showinfo, showwarning, showerror
import tkinter.filedialog
import tkinter.ttk
import tkinter.font
import tkinter as tk
import tkinter as tk
from tkinter import ttk

import pynput
from pprint import pprint

import spacy

import tagger

treeview_separator = "\u2192"

BALLOT_BOX = "\u2610"
BALLOT_BOX_WITH_X = "\u2612"

#TODO:
#Eventually: Batch rename/delete feature...Alt click on feature?
#Eventually: generate output dataset optionally without .jsons, organized in various ways
#Eventually: use PNG info as alternative (read-only) source of data, and allow writing it during LoRA subset generation
#Eventually: search for images with feature (i.e. active filter in main window?)

class TtkCheckList(ttk.Treeview):
    def __init__(self, master=None, width=200, clicked=None, separator='.',
                 unchecked=BALLOT_BOX, checked=BALLOT_BOX_WITH_X, **kwargs):
        """
        :param width: the width of the check list
        :param clicked: the optional function if a checkbox is clicked. Takes a
                        `iid` parameter.
        :param separator: the item separator (default is `'.'`)
        :param unchecked: the character for an unchecked box (default is
                          "\u2610")
        :param unchecked: the character for a checked box (default is "\u2612")

        Other parameters are passed to the `TreeView`.
        """
        if "selectmode" not in kwargs:
            kwargs["selectmode"] = "none"
        if "show" not in kwargs:
            kwargs["show"] = "tree"
        ttk.Treeview.__init__(self, master, **kwargs)
        self._separator = separator
        self._unchecked = unchecked
        self._checked = checked
        self._clicked = self.toggle if clicked is None else clicked
        self.parent_frame = master
        self.column('#0', width=width, stretch=tk.YES)
        self.bind("<Button-1>", self._item_click, True)

    def _item_click(self, event):
        assert event.widget == self
        x, y = event.x, event.y
        element = self.identify("element", x, y)
        if element == "text":
            iid = self.identify_row(y)
            self._clicked(iid)
            return "break"

    def add_item(self, item):
        """
        Add an item to the checklist. The item is the list of nodes separated
        by dots: `Item.SubItem.SubSubItem`. **This item is used as `iid`  at
        the underlying `Treeview` level.**
        """
        try:
            parent_iid, text = item.rsplit(self._separator, maxsplit=1)
        except ValueError:
            parent_iid, text = "", item

        def in_tree(item, root = ''):
            children = self.get_children(root)
            if item in children:
                return True
            for child in children:
                if in_tree(item, child):
                    return True
            return False

        if(not in_tree(item)):
            self.insert(parent_iid, index='end', iid=item,
                        text=self._unchecked+" "+text, open=True)

    def autofit(self):
        minwidth = 200
        font = tk.font.nametofont("TkTextFont")
        for item in self.get_children():
            minwidth = max(minwidth, min(400, 40 + font.measure(self.item(item, "text"))))
            for child in self.get_children(item):
                minwidth = max(minwidth, min(400, 60 + font.measure(self.item(child, "text"))))
                for grandchild in self.get_children(child):
                    minwidth = max(minwidth, min(400, 80 + font.measure(self.item(grandchild, "text"))))

        self.parent_frame.columnconfigure(0, minsize=minwidth)

    def toggle(self, iid):
        """
        Toggle the checkbox `iid`
        """
        text = self.item(iid, "text")
        if text[0] == self._checked:
            self.uncheck(iid)
        else:
            self.check(iid)


    def checked(self, iid):
        """
        Return True if checkbox `iid` is checked
        """
        text = self.item(iid, "text")
        return text[0] == self._checked

    def check(self, iid):
        """
        Check the checkbox `iid`
        """
        text = self.item(iid, "text")
        if text[0] == self._unchecked:
            self.item(iid, text=self._checked+text[1:])

        #If an item is checked, all its ancestors should be as well.        
        parent_iid = self.parent(iid)
        if parent_iid:
            self.check(parent_iid)

    def uncheck(self, iid):
        """
        Uncheck the checkbox `iid`
        """
        text = self.item(iid, "text")
        if text[0] == self._checked:
            self.item(iid, text=self._unchecked+text[1:])
        
        #If an item is unchecked, all its descendants should be as well.        
        children = self.get_children(iid)
        for c in children:
            self.uncheck(c)


def get_automatic_tags_from_txt_file(image_file):
    #If .txt available, read into automated caption
    txt_file = splitext(image_file)[0] + ".txt"
    try:
        with open(txt_file) as f:
            return ' '.join(f.read().split())
    except:
        print(traceback.format_exc())
    return None

use_clip = False
tokenizer_ready = False
def import_tokenizer_reqs():
    global tokenizer_ready
    try:
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
            import tiktoken
    except:
        print(traceback.format_exc())
    tokenizer_ready = True

nlp = None
def do_get_pos(string):
    global nlp
    if not nlp:
        print("Loading natural language processing model...")
        nlp = spacy.load("en_core_web_sm")        
    return nlp(string)

#Return approximate number of tokens in string (seems to err on the high side)
def num_tokens_from_string(string: str, encoding_name: str= None) -> int:
    """Returns the number of tokens in a text string."""
    if use_clip:
        def raw_get_tokens(strings):
            token_list = [list(x) for x in list(tokenizer(strings))]
            for tl in token_list:
                while tl[-1] == 0:
                    tl.pop()
            return token_list

        #The tokenizer saturates at 77 tokens. Therefore, split the string
        #until each returned value is less than 77 tokens to get a valid answer.
        chunks = [string]
        token_chunks = raw_get_tokens(chunks)
        found_77 = len(token_chunks[0]) == 77

        while found_77:
            new_chunks = []
            for s in chunks:
                split_s = s.split()
                joiner = " "
                if len(split_s) == 1:
                    split_s = s
                    joiner = ""
                left_s = joiner.join(split_s[:int(len(split_s) / 2)])
                right_s =joiner.join(split_s[int(len(split_s) / 2):])
                new_chunks.append(left_s)
                new_chunks.append(right_s)
            chunks = new_chunks

            new_token_chunks = raw_get_tokens(chunks)
            token_chunks = new_token_chunks
            found_77 = False
            for t in token_chunks:
                if len(t) == 77:
                    found_77 = True

        sum_tokens = 0
        for t in token_chunks:
            sum_tokens += len(t) - 2 #Each chunk has a start/end token.

        return sum_tokens
    else:
        encoding = tiktoken.get_encoding(encoding_name)
        num_tokens = len(encoding.encode(string))
        return num_tokens


def import_interrogators():
    try:
        try:
            global tagger, utils, interrogator, use_interrogate, interrogator_ready
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
    except:
        print(traceback.format_exc())
    interrogator_ready = True
    
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
interrogator_ready = False

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


class save_defaults_popup(object):
    def __init__(self, parent):
        self.parent = parent

        self.create_ui()

    def create_ui(self):
        self.top = tk.Toplevel(self.parent)
        self.top.title("Save defaults for path in dataset...")
        self.top.wait_visibility()
        self.top.grab_set()
        self.top.rowconfigure(0, weight=1)
        self.top.columnconfigure(0, weight=1)
        self.top.minsize(600, 400)
        self.top.transient(self.parent)

        self.form_frame = tk.Frame(self.top, 
                                   borderwidth=2,
                                   relief='raised',)
        
        self.form_frame.columnconfigure(1, weight=1)
        self.form_frame.rowconfigure(5, weight=1)

        #Defaults output location
        output_path_label = tk.Label(self.form_frame, text="Path: ")
        output_path_label.grid(row=0, column=0, padx=(5, 0), pady=5, sticky="e")

        self.output_path = tk.StringVar(None)
        self.output_path.set(relpath(pathlib.Path(self.parent.image_files[self.parent.file_index]).parent, self.parent.path))
        set_output_path_entry = tk.Entry(self.form_frame,
                                     textvariable=self.output_path, 
                                     justify="left")
        set_output_path_entry.grid(row=0, column=1, 
                               padx=(0, 5), pady=5, 
                               sticky="ew")
        set_output_path_entry.bind('<Control-a>', self.select_all)

        #Browse button
        browse_btn = tk.Button(self.form_frame, text='Browse...', 
                               command=self.browse)
        browse_btn.grid(row=0, column=2, padx=4, pady=4, sticky="sew")
        self.top.bind("<Control-b>", self.browse)


        #Labeled group for options
        settings_group = tk.LabelFrame(self.form_frame, 
                                    text="Settings")
        settings_group.grid(row=2, column=0, 
                            columnspan=3, 
                            padx=5, pady=5,
                            sticky="nsew")

        settings_group.columnconfigure(1, weight=1)
        defaults = self.parent.get_defaults()
        #Checkbox for inclusion of artist
        self.set_artist = tk.BooleanVar(None)
        self.set_artist.set(self.parent.artist_name.get() != defaults["artist"])
        set_artist_chk = tk.Checkbutton(
            settings_group,
            var=self.set_artist,
            text=f"Set artist:")
        set_artist_chk.grid(row=0, column=0, padx=5, pady=5, sticky="w")

        self.artist = tk.StringVar(None)
        self.artist.set(self.parent.artist_name.get())
        
        set_artist_entry = tk.Entry(settings_group,
                                    textvariable=self.artist, 
                                    justify="left")
        set_artist_entry.grid(row=0, column=1, 
                               padx=(0, 5), pady=5, 
                               sticky="ew")
        set_artist_entry.bind('<Control-a>', self.select_all)
        self.artist.trace("w", lambda name, index, mode: 
                                self.set_artist.set(True))


        #Checkbox for inclusion of style
        self.set_style = tk.BooleanVar(None)
        self.set_style.set(self.parent.style.get() != defaults["style"])
        set_style_chk = tk.Checkbutton(
            settings_group,
            var=self.set_style,
            text=f"Set style:")
        set_style_chk.grid(row=1, column=0, padx=5, pady=5, sticky="w")

        self.style = tk.StringVar(None)
        self.style.set(self.parent.style.get())
        
        set_style_entry = tk.Entry(settings_group,
                                    textvariable=self.style, 
                                    justify="left")
        self.style.trace("w", lambda name, index, mode: 
                                self.set_style.set(True))
        set_style_entry.grid(row=1, column=1, 
                               padx=(0, 5), pady=5, 
                               sticky="ew")
        set_style_entry.bind('<Control-a>', self.select_all)

        #Checkbox for inclusion of features
        self.set_features = tk.BooleanVar(None)
        self.set_features.set(False)
        set_features_chk = tk.Checkbutton(
            settings_group,
            var=self.set_features,
            text=f"Set features:")
        set_features_chk.grid(row=2, column=0, padx=5, pady=5, sticky="w")

        self.features = tk.StringVar(None)
        self.features.set(json.dumps({f[0]["var"].get(): "" for f in self.parent.features if f[0]["var"].get()}))
        
        set_features_entry = tk.Entry(settings_group,
                                    textvariable=self.features, 
                                    justify="left")
        set_features_entry.grid(row=2, column=1, 
                               padx=(0, 5), pady=5, 
                               sticky="ew")
        self.features.trace("w", lambda name, index, mode: 
                                self.set_features.set(True))
        set_features_entry.bind('<Control-a>', self.select_all)

        self.include_feature_descriptions = tk.BooleanVar(None)
        self.include_feature_descriptions.set(False)
        self.include_feature_descriptions.trace("w", lambda name, index, mode: 
                                self.toggle_feature_descs())
        include_feature_descriptions_chk = tk.Checkbutton(
            settings_group,
            var=self.include_feature_descriptions,
            text=f"Include feature descriptions")
        include_feature_descriptions_chk.grid(row=3, column=1, padx=5, pady=5, sticky="w")


        #Labeled group for rating
        self.set_rating = tk.BooleanVar(None)
        self.set_rating.set(False)
        set_rating_chk = tk.Checkbutton(
            settings_group,
            var=self.set_rating,
            text=f"Set quality:")
        set_rating_chk.grid(row=4, column=0, padx=5, pady=5, sticky="w")

        rating_group = tk.LabelFrame(settings_group, 
                                    text="Quality for Training")
        rating_group.grid(row=4, column=1, 
                          padx=5, pady=5,
                          sticky="nsew")

        self.rating = tk.IntVar()
        self.rating.set(False)
        tk.Radiobutton(rating_group, 
           text=f"Not rated",
           variable=self.rating, 
           value=0).grid(row=0, column=0, padx=5, pady=5, sticky="w")

        for i in range(1, 6):
            tk.Radiobutton(rating_group, 
               text=f"{i}",
               variable=self.rating, 
               value=i).grid(row=0, column=i, sticky="w")

        self.rating.trace("w", lambda name, index, mode: 
                                self.set_rating.set(True))

        # Cancel button
        cancel_btn = tk.Button(self.form_frame, text='Cancel', 
                               command=self.cancel)
        cancel_btn.grid(row=6, column=0, padx=4, pady=4, sticky="sew")
        self.top.bind("<Escape>", self.cancel)
        self.top.bind("<Control-s>", self.save)

        # Save button
        save_btn = tk.Button(self.form_frame, text='Save (Ctrl+S)', 
                               command=self.save)
        save_btn.grid(row=6, column=1,
                          columnspan=2,
                          padx=4, pady=4, 
                          sticky="sew")

        self.form_frame.grid(row=0, column=0, 
                        padx=0, pady=0, 
                        sticky="nsew")


    def toggle_feature_descs(self):
        if self.include_feature_descriptions.get():
            self.features.set(json.dumps({f[0]["var"].get(): f[1]["var"].get() for f in self.parent.features if f[0]["var"].get()}))
        else:
            self.features.set(json.dumps({f[0]["var"].get(): "" for f in self.parent.features if f[0]["var"].get()}))

    def select_all(self, event):
        # select text
        try:
            event.widget.select_range(0, 'end')
        except:
            print(traceback.format_exc())
            event.widget.tag_add("sel", "1.0", "end")

        # move cursor to the end
        try:
            event.widget.icursor('end')
        except:
            print(traceback.format_exc())
            event.widget.mark_set("insert", "end")

        #stop propagation
        return 'break'

    def get_defaults_from_ui(self):
        defaults = {}
        if self.set_artist.get():
            defaults["artist"] = self.artist.get()
        if self.set_style.get():
            defaults["style"] = self.style.get()
        if self.set_features.get():
            try:
                defaults["features"] = json.loads(self.features.get())
            except:
                print(traceback.format_exc())
                showerror(parent=self.top, title="Error", message="Features must be valid json dict.")
                return
        if self.set_rating.get():
            defaults["rating"] = self.rating.get()
        return defaults           

    def save(self, event = None):
        dataset_path = self.parent.path
        path= self.output_path.get()
        if path.startswith('/'):
            path = relpath(path, dataset_path)

        if path.startswith('..'):
            showerror(parent=self.top, title="Error", message="Output path must be in dataset")
            return

        abs_path = pathlib.Path(dataset_path) / path
        if not exists(abs_path):
            showerror(parent=self.top, title="Error", message="Output path must exist")
            return

        with open(abs_path / "defaults.json", "w") as f:
            json.dump(self.get_defaults_from_ui(), f, indent=4)
        self.close()

    def cancel(self, event = None):
        self.close()
        return "break"

    def close(self):
        self.top.grab_release()
        self.top.destroy()
        return "break"

    def browse(self, event = None):
        #Popup folder selection dialog
        default_dir = self.parent.image_files[self.parent.file_index].parent
        try:
            path = tk.filedialog.askdirectory(
                parent=self.top, 
                initialdir=default_dir,
                title="Select a location for subset output")
        except:
            print(traceback.format_exc())
            return

        if path:
            pl_path = pathlib.Path(relpath(path, self.parent.path))
            if str(pl_path).startswith(".."):
                showerror(message=
                          "Output path must be in dataset")
                self.output_path.set(relpath(default_dir, self.parent.path))
            else:
                self.output_path.set(str(pl_path))
        return "break"



class manually_review_subset_popup(object):
    def __init__(self, parent, subset_path, image_files, review_all):
        try:
            if not tokenizer_ready:
                showerror(parent=parent.top,
                            title="Not ready",
                            message="The tokenizer is not yet ready.")
                self.top = tk.Toplevel(self.parent.top)
                self.close()
                return
            self.parent = parent
            self.dataset_path = self.parent.parent.path
            self.subset_path = subset_path
            self.file_index = 0
            self.image_files = image_files.copy()
            self.icon_image = Image.open("icon.png")
            if not review_all:
                for f in reversed(image_files):
                    caption_file = "".join(splitext(f)[:-1]) + ".txt"
                    caption = self.get_caption_from_file(caption_file)
                    if num_tokens_from_string(caption, "gpt2") <= 75:
                        self.image_files.remove(f)

            if len(self.image_files) == 0:
                showinfo(parent=parent.top,
                         title="No such files",
                         message="No images had more than 75 tokens.")
                self.top = tk.Toplevel(self.parent.top)
                self.close()
                return

            self.icon_image = Image.open("icon.png")

            self.create_ui()
        except:
            print(traceback.format_exc())
           
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
        self.image = self.icon_image
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
        self.image = Image.open(f)
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

    #Move the focus to the prev item in the form
    def focus_prev_widget(self, event):
        event.widget.tk_focusPrev().focus()
        return("break")


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
        self.image = self.icon_image
        self.framed_image = ImageTk.PhotoImage(self.image)
        self.image_label.configure(image=self.framed_image)
        self.caption_textbox.delete("1.0", "end")
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
            print(traceback.format_exc())


        f = self.image_files[index]        
        self.load_image(f)
        
        self.statusbar_text.set(
            f"Image {1 + self.file_index}/{len(self.image_files)}: "
            f"{relpath(pathlib.Path(self.image_files[self.file_index]), self.parent.parent.path)}")
        
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
        self.top.update_idletasks()


    def select_all(self, event):
        # select text
        try:
            event.widget.select_range(0, 'end')
        except:
            print(traceback.format_exc())
            event.widget.tag_add("sel", "1.0", "end")

        # move cursor to the end
        try:
            event.widget.icursor('end')
        except:
            print(traceback.format_exc())
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


        #Checkbox to enable filtering
        self.enable_filtering = tk.BooleanVar(None)
        self.enable_filtering.set(False)
        enable_filtering_chk = tk.Checkbutton(
            settings_group,
            var=self.enable_filtering,
            text="Enable filtering")        
        enable_filtering_chk.grid(row=4, column=1, padx=5, pady=5, sticky="w")

        self.filter = tk.StringVar(None)
        self.filter.set("")
        self.filter_entry = tk.Entry(settings_group,
                                     textvariable=self.filter,
                                     justify="left")
        self.filter_entry.grid(row=4, column=2, 
                             padx=(0, 5), pady=5, 
                             sticky="ew")
        self.filter_entry.bind('<Control-a>', self.select_all)        

        self.enable_filtering.trace("w", 
                lambda name, index, mode: self.on_enable_filtering_modified())

        #Labeled group for rating
        self.filter_rating = tk.BooleanVar(None)
        self.filter_rating.set(False)
        filter_rating_chk = tk.Checkbutton(
            settings_group,
            var=self.filter_rating,
            text=f"Filter quality >= ")
        filter_rating_chk.grid(row=5, column=1, padx=5, pady=5, sticky="w")

        rating_group = tk.LabelFrame(settings_group, 
                                    text="Minimum Quality")
        rating_group.grid(row=5, column=2, 
                          padx=5, pady=1,
                          sticky="nsew")

        self.minimum_rating = tk.IntVar()
        self.minimum_rating.set(False)
        tk.Radiobutton(rating_group, 
           text=f"Not rated",
           variable=self.minimum_rating, 
           value=0).grid(row=0, column=0, padx=5, pady=1, sticky="w")

        for i in range(1, 6):
            tk.Radiobutton(rating_group, 
               text=f"{i}",
               variable=self.minimum_rating, 
               value=i).grid(row=0, column=i, pady=1, sticky="w")
        self.minimum_rating.trace("w", lambda name, index, mode: 
                        self.filter_rating.set(True))

        #Checkbox for fetching automatic tags if empty
        self.interrogate_automatic_tags = tk.BooleanVar(None)
        self.interrogate_automatic_tags.set(False)
        interrogate_automatic_tags_chk = tk.Checkbutton(
            settings_group,
            var=self.interrogate_automatic_tags,
            text="Interrogate image if automatic tags empty")
        interrogate_automatic_tags_chk.grid(row=6, column=1, columnspan=2, padx=5, pady=5, sticky="w")

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
            print(traceback.format_exc())
            event.widget.tag_add("sel", "1.0", "end")

        # move cursor to the end
        try:
            event.widget.icursor('end')
        except:
            print(traceback.format_exc())
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
            print(traceback.format_exc())
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

    def on_enable_filtering_modified(self):
        if self.enable_filtering.get():
            self.filter_entry.config(state="normal")
        else:
            self.filter_entry.config(state="disabled")

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
            "interrogate_automatic_tags": False,
            "review_option": self.review_option.get(),
            "steps_per_image": self.steps_per_image_entry.get(),
            "enable_filtering": self.enable_filtering.get(),
            "filter": self.filter.get(),
            "filter_rating": self.filter_rating.get(),
            "minimum_rating": self.minimum_rating.get()
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
            print(traceback.format_exc())
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
                        print(traceback.format_exc())
        except:
            print(traceback.format_exc())
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
                try:
                    self.enable_filtering.set(info["enable_filtering"])
                except KeyError:
                    pass
                try:
                    self.filter.set(info["filter"])
                except KeyError:
                    pass
                try:
                    self.filter_rating.set(info["filter_rating"])
                except KeyError:
                    pass
                try:
                    self.minimum_rating.set(info["minimum_rating"])
                except KeyError:
                    pass
                try:
                    self.interrogate_automatic_tags.set(info["interrogate_automatic_tags"])
                except KeyError:
                    pass

        except:
            print(traceback.format_exc())
        

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
            else:
                exts = Image.registered_extensions()
                supported_exts = {ex for ex, f in exts.items() if f in Image.OPEN}
                supported_exts.update({".txt", ".json"})
                stale_files = [pathlib.Path(f).absolute()
                         for f in pathlib.Path(subset_path).rglob("*")
                          if isfile(join(subset_path, f))]
                msg_box = tk.messagebox.askyesnocancel('Existing Subset', f"{len(stale_files)} files already exist in '{subset_path}'." 
                                                    "\nDelete them?",
                                                    parent=self.top,
                                                    icon='warning')
                if msg_box is not None:
                    if msg_box == True:
                        for f in stale_files:
                            remove(f)
                else:
                    showinfo(parent=self.top,
                             title="Generation canceled",
                             message=f"Generation canceled.")
                    return
                        
                
        

        self.save_subset_info(subset_path)

        popup = tk.Toplevel(self.top)
        tk.Label(popup, text="Processing subset images...").grid(row=0,column=0)
        progress_var = tk.DoubleVar()
        progress_var.set(0)
        progress_bar = tk.ttk.Progressbar(popup, variable=progress_var, maximum=100)
        progress_bar.grid(row=1, column=0)#.pack(fill=tk.X, expand=1, side=tk.BOTTOM)
        popup.pack_slaves()
        current_image_index = 0
        output_images = []
        #For each image
        for path in self.parent.image_files:
            #Update progress bar
            progress_var.set(100 * current_image_index / len(self.parent.image_files))
            popup.update()
            current_image_index += 1

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
               and item["artist"]):
                caption += item["artist"] + ", "
            
            if self.include_summary.get() and item["summary"]:
                caption += item["summary"] + ", "
            
            if(self.include_feature.get()
               and self.lora_name.get() in item["features"]):
                feature = item["features"][self.lora_name.get()]
                if feature == "":
                    feature = self.lora_name.get()
                caption += feature + ", "

            if self.include_other_features.get():
                for f in item["features"]:
                    if f != self.lora_name.get() and item["features"][f]:
                        feature = item["features"][f]
                        if feature == "":
                            feature = f
                        caption += feature + ", "

            try:
                if self.interrogate_automatic_tags.get() and not item["automatic_tags"]:
                    item["automatic_tags"] = interrogate_automatic_tags(path)
            except:
                print(traceback.format_exc())
            
            if self.include_automatic_tags.get() and item["automatic_tags"]:
                caption += item["automatic_tags"]
            
            if caption.endswith(", "):
                caption = caption[:-2]


            components = caption.split(",")

            unique_components_forward = []
            for c in components:
                found = False
                for u_c in unique_components_forward:
                    if c.strip().lower() in u_c.strip().lower():
                        found = True
                if not found:
                    unique_components_forward.append(c.strip())

            unique_components = []
            for c in reversed(unique_components_forward):
                found = False
                for u_c in unique_components:
                    if c.strip().lower() in u_c.strip().lower():
                        found = True
                if not found:
                    unique_components.append(c.strip())

            caption = ", ".join(reversed(unique_components))

            if self.enable_filtering.get():
                filtered_components_or = re.split(",| OR ", self.filter.get())
                match = False

                for c in filtered_components_or:
                    #Handle the AND operator
                    match_and = True
                    component_and = c.split(" AND ")
                    for c_and in component_and:
                        invert = False
                        while c_and.strip().startswith("NOT "):
                            c_and = c_and[4:]
                            invert = not invert

                        #Handle the NOT operator
                        cur_match_and = c_and.strip().lower() in caption.lower()
                        if invert:
                            cur_match_and = not cur_match_and
                        match_and &= cur_match_and

                    #Handle the OR operator or comma (treated equivalently)
                    match |= match_and

                #If this item doesn't match the filter, skip it.
                if not match:
                    continue

            if self.filter_rating.get() and item["rating"] < self.minimum_rating.get():
                continue

            if self.review_option.get() == 1: #Auto-truncate
                caption = truncate_string_to_max_tokens(caption)
            with open(str(subset_path / tgt_prefix) + ".txt", "w") as f:
                f.write(" ".join(caption.split()))

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
            target_json = str(subset_path / tgt_prefix) + ".json"
            if isfile(json_file):
                shutil.copy2(json_file, target_json)
            else:
                self.parent.write_item_to_file(
                    self.parent.get_item_from_file(path),
                    target_json
                    )


        popup.destroy()

        if len(output_images) == 0:
            showwarning(parent=self.top,
                        title="Empty Dataset",
                        message="No images matched filter.")
            return
            

        #Pop up box for manual review
        try:
            print(f"About to wait for window: {self.review_option.get()}")
            if self.review_option.get() > 1: 
                self.top.wait_window(manually_review_subset_popup(
                        self,
                        subset_path,
                        output_images.copy(),
                        self.review_option.get() > 2).top)

        except:
            print(traceback.format_exc())

        #Message showing stats of created subset
        try:
            showinfo(parent=self.top,
                     title="Subset written",
                     message=f"Wrote {len(output_images)} images+jsons+captions "
                              "to subset folder.")

            self.close()
        except:
            print(traceback.format_exc())
        

# the given message with a bouncing progress bar will appear for as long as func is running, returns same as if func was run normally
# a pb_length of None will result in the progress bar filling the window whose width is set by the length of msg
# Ex:  run_func_with_loading_popup(lambda: task('joe'), photo_img)  
def run_func_with_loading_popup(parent, func, msg, window_title = None, bounce_speed = 8, pb_length = None):
    func_return_l = []
    top = tk.Toplevel(parent)

    if isinstance(parent, lora_tag_helper):
        x = parent.winfo_x() + 50
        y = parent.winfo_y() + 50
    
        top.geometry(f"+{x}+{y}")

    
    
    class _main_frame(object):
        def __init__(self, top, window_title, bounce_speed, pb_length):
            self.done = False
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
            try:
                # the load_bar needs to be configured for indeterminate amount of bouncing
                self.load_bar.config(mode='indeterminate', maximum=100, value=0, length = self.pb_length)
                # 8 here is for speed of bounce
                self.load_bar.start(self.bounce_speed)            
    #             self.load_bar.start(8)            

                self.work_thread = threading.Thread(target=self.work_task, args=())
                self.work_thread.start()

                # close the work thread
                self.work_thread.join()

                self.done = True
                self.top.destroy()
    #             # stop the indeterminate bouncing
    #             self.load_bar.stop()
    #             # reconfigure the bar so it appears reset
    #             self.load_bar.config(value=0, maximum=0)
            except:
                print(traceback.format_exc())
        def work_task(self):
            func_return_l.append(func())

    # call Main_Frame class with reference to root as top
    frame = _main_frame(top, window_title, bounce_speed, pb_length)
    parent.update()
    if not frame.done:
        parent.wait_window(top)
    parent.update()
    if len(func_return_l) == 1:
        return func_return_l[0]
    else:
        return func_return_l


        


#Application class
class lora_tag_helper(TkinterDnD.Tk):

    #Constructor
    def __init__(self):
        super().__init__()

        self.image_width = 1
        self.image_height = 1
        self.image_handle = None
        self.crop_left_area = None
        self.crop_top_area = None
        self.crop_right_area = None
        self.crop_bottom_area = None
        self.already_initialized = False
        self.image_files = []
        self.file_index = 0
        self.features = []
        self.l_pct = 0
        self.t_pct = 0
        self.r_pct = 1
        self.b_pct = 1
        self.feature_count = 0
        self.features = []
        self.icon_image = Image.open("icon.png")
        self.ctrl_pressed = False
        self.prompt = ""

        self.feature_checklist = []
        self.geometry("1200x600")

        self.create_ui()
        self.wm_protocol("WM_DELETE_WINDOW", self.quit)
        self.listener = pynput.keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release)
        self.listener.start()
        self.update()
        self.after(1000, self.import_reqs)

    def import_reqs(self, event = None):
        try:
            if not self.already_initialized and (event is None or event.widget is not self):
                self.already_initialized = True
                run_func_with_loading_popup(
                        self,
                        lambda: import_interrogators(), 
                        "Importing Interrogator Requirements...", 
                        "Importing Interrogator Requirements...")
                run_func_with_loading_popup(
                        self,
                        lambda: import_tokenizer_reqs(),
                        "Importing Tokenizer Requirements...", 
                        "Importing Tokenizer Requirements...")
        except:
            print(traceback.format_exc())
    #Create all UI elements
    def create_ui(self):
        # Set window info
        self.iconphoto(self, tk.PhotoImage(file="icon_256.png")) 
        self.title("LoRA Tag Helper")

        self.create_menu()
        self.create_primary_frame()

        self.drop_target_register(DND_FILES)
        self.dnd_bind('<<Drop>>', self.handle_drop)

    def handle_drop(self, event):
        file = pathlib.Path(event.data)
        if len(self.image_files) > 0:
            self.go_to_image(None, file)
        elif file.is_dir():
            self.open_dataset(None, file)

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
        self.root_frame.columnconfigure(1, minsize=600)

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

        file_menu.add_command(label="Go to specific image in dataset...", 
                              command=self.go_to_image, 
                              underline=0, 
                              accelerator="Ctrl+G")
        self.bind("<Control-g>", self.go_to_image)

        file_menu.add_command(label="Reset this image to defaults...", 
                              command=self.reset, 
                              underline=0, 
                              accelerator="Ctrl+Shift+R")
        self.bind("<Control-R>", self.reset)

        file_menu.add_command(label="Save as Default...", 
                              command=self.save_defaults, 
                              underline=0, 
                              accelerator="Ctrl+Shift+S")
        self.bind("<Control-S>", self.save_defaults)


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
        self.image = self.icon_image
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
            print(traceback.format_exc())

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

        self.top_group = tk.LabelFrame(self.form_frame, 
                                    text="")
        self.top_group.columnconfigure(1, weight = 1)

        self.top_group.grid(row=0, column=0, 
                          columnspan=2, 
                          padx=5, pady=5,
                          sticky="nsew")        

        self.add_artist_entry()
        self.add_style_entry()
        self.add_title_entry()
        self.add_rating_entry()
        self.add_summary_text()
        self.add_features_table()
        self.add_automatic_tags_text()        
        self.add_form_buttons()
        self.add_feature_checklist()
        self.show_form_frame()

    #Hide the right hand form controls
    def hide_form_frame(self):
        self.form_frame.grid_remove()

    #Show the right hand form controls
    def show_form_frame(self):
        self.form_frame.grid(row=0, column=1, 
                              padx=0, pady=0, 
                              sticky="nsew")

    #Hide the initial "Open a dataset" prompt
    def hide_initial_frame(self):
        self.initial_frame.grid_remove()

    #Show the initial "Open a dataset" prompt
    def show_initial_frame(self):
        self.initial_frame.grid(row=0, column=1, 
                              padx=0, pady=0, 
                              sticky="nsew")
        self.initial_frame.lift()
        self.image_resizer()

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
        self.rating.set(self.get_defaults()["rating"])
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
        artist_name_label = tk.Label(self.top_group, text="Artist: ")
        artist_name_label.grid(row=0, column=0, padx=0, pady=5, sticky="e")

        self.artist_name = tk.StringVar(None)
        self.artist_name.set(self.get_defaults()["artist"])
        self.artist_name_entry = tk.Entry(self.top_group,
                                     textvariable=self.artist_name, 
                                     justify="left")
        self.artist_name_entry.grid(row=0, column=1, padx=(0, 5), pady=5, sticky="ew")
        self.artist_name_entry.bind('<Control-a>', self.select_all)
        self.artist_name_entry.focus_set()
        self.artist_name_entry.select_range(0, 'end')

    #Add the style query to the form
    def add_style_entry(self):
        style_label = tk.Label(self.top_group, text="Style: ")
        style_label.grid(row=1, column=0, padx=0, pady=5, sticky="e")

        self.style = tk.StringVar(None)
        self.style.set(self.get_defaults()["style"])
        self.style_entry = tk.Entry(self.top_group,
                                     textvariable=self.style, 
                                     justify="left")
        self.style_entry.grid(row=1, column=1, padx=(0, 5), pady=5, sticky="ew")
        self.style_entry.bind('<Control-a>', self.select_all)

    #Add the title query to the form
    def add_title_entry(self):
        title_label = tk.Label(self.top_group, text="Title: ")
        title_label.grid(row=2, column=0, padx=0, pady=5, sticky="e")

        self.title_var = tk.StringVar(None)
        self.title_var.set(self.get_defaults()["title"])
        self.title_entry = tk.Entry(self.top_group, textvariable=self.title_var, justify="left")
        self.title_entry.grid(row=2, column=1, padx=(0, 5), pady=5, sticky="ew")
        self.title_entry.bind('<Control-a>', self.select_all)

    #Move the focus to the prev item in the form
    def focus_prev_widget(self, event):
        event.widget.tk_focusPrev().focus()
        return("break")

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
        self.summary_textbox.bind('<KeyRelease>', self.add_features_from_summary)

    #Add the features table to the form
    def add_features_table(self):
        self.features_group = tk.LabelFrame(self.form_frame, 
                                    text="Features")
        self.features_group.grid(row=6, column=0, 
                            columnspan=2, 
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

        self.form_frame.rowconfigure(9, weight=1)

        import_tags_btn = tk.Button(self.form_frame, 
                                  text="Import automatic tags (Ctrl+T)", 
                                  command=self.update_ui_automatic_tags)
        import_tags_btn.grid(row=12, column=0, 
                             columnspan=2, 
                             padx=5, pady=5, 
                             sticky="ew")
        
        #On Linux at least, Ctrl-t has a really annoying default behavior that swaps two characters.
        self.bind("<Control-t>", self.update_ui_automatic_tags)
        self.artist_name_entry.bind("<Control-t>", self.update_ui_automatic_tags)
        self.style_entry.bind("<Control-t>", self.update_ui_automatic_tags)
        self.title_entry.bind("<Control-t>", self.update_ui_automatic_tags)
        self.summary_textbox.bind("<Control-t>", self.update_ui_automatic_tags)
        self.automatic_tags_textbox.bind("<Control-t>", self.update_ui_automatic_tags)

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


    def on_press(self, key):
        if key == pynput.keyboard.Key.ctrl:
            self.ctrl_pressed = True

    def on_release(self, key):
        if key == pynput.keyboard.Key.ctrl:
            self.ctrl_pressed = False

    def feature_clicked(self, iid):        
        self.disable_feature_tracing()
        try:            
            tv = self.feature_checklist_treeview                

            deleting = False
            if self.ctrl_pressed:
                print(f"Delete {iid}")
                deleting = True
                tv.uncheck(iid)
            else:
                tv.toggle(iid)


            feature_iids = tv.get_children()
            noun_iids = []
            desc_iids = []
            for feature in feature_iids:
                for noun in tv.get_children(feature):
                    noun_iids.append(noun)
            for noun in noun_iids:
                for desc in tv.get_children(noun):
                    desc_iids.append(desc)

            tree = [iid]
            while tv.parent(tree[0]):
                tree.insert(0, tv.parent(tree[0]))


            #Find the row that matches this feature (if any)
            for row in range(self.feature_count):
                if(self.features[row][0]["var"].get().strip() == tree[0].strip()):
                    break


            #Find the last component that matches this noun (if any)
            desc = self.features[row][1]["var"].get()
            components = []
            this_component = ""
            feature = tv.item(tree[0], "text")[2:].strip()
            noun = ""
            adjective = ""
            if len(tree) > 1:
                noun = tv.item(tree[1], "text")[2:].strip()
                if row < self.feature_count:
                    components = [c.strip() for c in desc.split(",")]
                    for c in reversed(components):
                        if c.endswith(noun):
                            this_component = c
                            break
            if len(tree) > 2:
                adjective = tv.item(tree[2], "text")[2:].strip()

            if tv.checked(iid):

                #Make a new feature row if necessary and set it.
                if row == self.feature_count:
                    for row in range(self.feature_count):
                        if(self.features[row][0]["var"] == ""
                           and self.features[row][1]["var"]):
                            break

                self.features[row][0]["var"].set(feature)

                #If this is a noun, and the noun isn't already in the
                #description, then add it.
                if len(tree) > 1 and not this_component.endswith(noun):
                    if desc != "":
                        desc += f", {noun}"
                    else:
                        desc = f"{noun}"
                    self.features[row][1]["var"].set(desc)
                    this_component = f"{noun}"
                    if components != ['']:
                        components.append(this_component)
                    else:
                        components = [this_component]

                #If this is an adjective, and it isn't already in the component,
                #then prepend it before the noun.
                if len(tree) > 2 and adjective not in this_component:
                    new_component = f"{adjective} {noun}".join(this_component.rsplit(noun, 1))
                    for i in reversed(range(len(components))):
                        if components[i] == this_component:
                            components[i] = new_component
                    self.features[row][1]["var"].set(", ".join(components))

            else:
                #If this is a feature, remove the entire feature row.
                if len(tree) == 1 and row != self.feature_count:
                    self.remove_row(row)

                #If this is a noun, remove the relevant component.
                if len(tree) == 2 and this_component != "":
                    components.remove(this_component)
                    self.features[row][1]["var"].set(", ".join(components))

                #If this is an adjective, remove it from the relevant component.
                if len(tree) == 3 and adjective in this_component:
                    new_component = this_component.replace(f"{adjective} ", "")
                    for i in reversed(range(len(components))):
                        if components[i] == this_component:
                            components[i] = new_component.strip()
                    self.features[row][1]["var"].set(", ".join(components))

            if deleting:
                tv.delete(iid)
                relative_path = relpath(pathlib.Path(self.image_files[self.file_index]).absolute(), self.path)
                parents = [str(p) for p in pathlib.Path(relative_path).parents]
                for p in self.known_feature_checklists:
                    if p in parents:
                        self.known_feature_checklists[p] = [x for x in self.known_feature_checklists[p] if not x[0].startswith(iid)]
                pprint(self.known_feature_checklists)
                          
        except:
            print(traceback.format_exc())
        self.enable_feature_tracing()
        self.feature_modified(self.features[0][0]["var"].get())


    def add_feature_checklist(self):
        self.feature_checklist_group = tk.LabelFrame(self.form_frame, 
                                    text="Features")
        self.feature_checklist_group.grid(row=0, column=2, 
                            rowspan=15, 
                            padx=5, pady=5,
                            sticky="nsew")

        self.feature_checklist_group.rowconfigure(0, weight=1)
        self.feature_checklist_group.columnconfigure(0, weight=1,minsize=200)

        bgcolor = self.feature_checklist_group["background"]
        ttk.Style().configure("Treeview", borderwidth=0, relief=tk.FLAT, background=bgcolor, fieldbackground=bgcolor)
        self.feature_checklist_treeview = TtkCheckList(self.feature_checklist_group,
                                                       height=self.feature_count, 
                                                       separator=treeview_separator,
                                                       clicked=self.feature_clicked)
        self.feature_checklist_treeview.grid(row=0, column=0, padx=5, pady=5, sticky="news")
        self.feature_checklist_treeview.rowconfigure(0, weight=1)
        self.feature_checklist_treeview.columnconfigure(0, weight=1)
        # Constructing vertical scrollbar
        # with treeview
        self.verscrlbar = ttk.Scrollbar(self.feature_checklist_group,
                           orient ="vertical",
                           command = self.feature_checklist_treeview.yview)
        self.verscrlbar.grid(row=0, column=1, sticky="nes")
        self.feature_checklist_treeview.configure(yscrollcommand = self.verscrlbar.set)
        self.update_checklist()
        

    #Add the save and navigation buttons to the right-hand form
    def update_checklist(self):
        for item in self.feature_checklist_treeview.get_children():
           self.feature_checklist_treeview.delete(item)        
        for item in self.feature_checklist:
            self.feature_checklist_treeview.add_item(item[0])
            if item[1]:
                self.feature_checklist_treeview.check(item[0])
        self.feature_checklist_treeview.autofit()

    def disable_feature_tracing(self):    
        for i in range(len(self.features)):
            for j in range(2):
                if self.features[i][j]["trace"]:
                    self.features[i][j]["var"].trace_vdelete("w", self.features[i][j]["trace"])
                    self.features[i][j]["trace"] = None

    def enable_feature_tracing(self):
        self.disable_feature_tracing()
        for i in range(len(self.features)):
            for j in range(2):
                self.features[i][j]["trace"] = self.features[i][j]["var"].trace("w",
                    lambda name, index, mode, var=self.features[i][j]["var"]: self.feature_modified(var))

    #Clear the UI
    def clear_ui(self):
        self.summary_textbox.delete("1.0", "end")
        self.automatic_tags_textbox.delete("1.0", "end")
        self.image = self.icon_image
        self.framed_image = ImageTk.PhotoImage(self.image)
        self.canvas.delete(self.image_handle)

        self.disable_feature_tracing()

        if self.feature_count > 0:
            while self.feature_count > 1:
                self.remove_row(self.feature_count - 1)
            self.features[0][0]["var"].set("")
            self.features[0][1]["var"].set("")

        self.enable_feature_tracing()

        self.statusbar_text.set("")

    #Set the UI to the given item's values
    def set_ui(self, index: int, item = None):
        if len(self.image_files) == 0 or index > len(self.image_files):
            return
        
        self.clear_ui()

        f = self.image_files[index]
        self.load_image(f)

        if item is None:
            item = self.get_item_from_file(self.image_files[index])


        try: 
            self.artist_name.set(item["artist"])
        except: 
            print(traceback.format_exc())

        try: 
            self.style.set(item["artist"])
        except: 
            print(traceback.format_exc())

        try:
            self.title_var.set(item["title"])
        except:
            print(traceback.format_exc())


        try:
            self.style.set(item["style"])
        except:
            print(traceback.format_exc())

        try:
            self.rating.set(item["rating"])
        except:
            print(traceback.format_exc())

        try:
            self.summary_textbox.insert("1.0", item["summary"])
        except:
            print(traceback.format_exc())

        try:
            self.l_pct = item["crop"][0]
            self.t_pct = item["crop"][1]
            self.r_pct = item["crop"][2]
            self.b_pct = item["crop"][3]
        except:
            print(traceback.format_exc())

        self.generate_crop_rectangle()



        self.disable_feature_tracing()

        try:
            i = 0
            for k, v in item["features"].items():
                if(i >= self.feature_count):
                    self.add_row()    
                self.disable_feature_tracing()            
                self.features[i][0]["var"].set(k)
                self.features[i][1]["var"].set(v)
                i += 1
        except:
            print(traceback.format_exc())

        if len(self.features) > 0:
            self.feature_modified(self.features[0][0]["var"])

        try:
            self.automatic_tags_textbox.insert("1.0", item["automatic_tags"])
        except:
            print(traceback.format_exc())

        #Enable/disable buttons as appropriate
        if self.file_index > 0:
            self.prev_file_btn["state"] = "normal"
        else:
            self.prev_file_btn["state"] = "disabled"

        if self.file_index < len(self.image_files) - 1:
            self.next_file_btn["state"] = "normal"
        else:
            self.next_file_btn["state"] = "disabled"

        
        self.statusbar_text.set(f"Image {1 + self.file_index}/{len(self.image_files)}: "
                                f"{relpath(pathlib.Path(self.image_files[self.file_index]), self.path)}")

        self.update_idletasks()
        
       

    #Gather known feature set
    def update_known_features(self, file, item):
        relative_path = pathlib.Path(relpath(pathlib.Path(file), self.path))
        parents = [str(p) for p in relative_path.parents]
        for p in parents:
            if p not in self.known_features:
                self.known_features[p] = {}

        p = parents[0]

        combined_features = {}
        if p in self.known_features:
            combined_features = self.known_features[p]

        if "features" in item:
            for feature in item["features"]:
                components = item["features"][feature].split(",")
                if feature not in combined_features:
                    combined_features[feature] = ""
                combined_components = combined_features[feature].split(",")
                    
                for c in components:
                    c = c.strip()
                    if c not in combined_components:
                        combined_features[feature] += ", " + c                            

        self.known_features.update({p: combined_features})
                    

    def build_known_feature_checklists(self):
        self.known_feature_checklists = {}
        for path in self.known_features:
            known_checklist = []
            p = str(path)
            for name in self.known_features[p]:
                if name != '':
                    desc = self.known_features[p][name]
                    new_item = (name, False)

                    found = False
                    for parent in pathlib.Path(path).parents:
                        parent = str(parent)
                        if parent not in self.known_feature_checklists:
                            self.known_feature_checklists[parent] = {}
                        if new_item in self.known_feature_checklists[parent]:
                            found = True

                    if not found and new_item not in known_checklist:
                        known_checklist.append(new_item)
                    components = desc.split(",")
                    for c in components:
                        c = c.strip()
                        if c != '' and c != name:
                            for c_split in self.split_component(c):
                                new_item = (name + treeview_separator + c_split, False)
                                found = False
                                for parent in pathlib.Path(path).parents:
                                    if new_item in self.known_feature_checklists[str(parent)]:
                                        found = True
                                if not found and new_item not in known_checklist:
                                    known_checklist.append(new_item)
            known_checklist.sort()
            self.known_feature_checklists[path] = known_checklist

        self.known_features = {}



    #Create open dataset action
    def open_dataset(self, event = None, directory = None):
        self.known_features = {}
        self.clear_ui()
        self.show_initial_frame()

        #Clear the UI and associated variables
        self.file_index = 0
        self.image_files = []

        #Popup folder selection dialog
        if directory is None:
            try:
                self.path = tk.filedialog.askdirectory(
                    parent=self, 
                    initialdir="./dataset",
                    title="Select a dataset")
                if not self.path:
                    return
            except:
                print(traceback.format_exc())
                return
        else:
            self.path = directory                    

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
            item = self.get_item_from_file(path)
            self.update_known_features(path, item)
            #self.write_item_to_file(item, json_file)
        self.build_known_feature_checklists()

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
            self.update()
            self.wait_window(generate_lora_subset_popup(self).top)
            self.update()

    def load_image(self, f):
        try:
            self.image = Image.open(f)
            self.image_resizer()

            prompt = ""
            try:
                self.image.load()  # Needed only for .png EXIF data
                prompt = " ".join(self.image.info['parameters'].split("Negative prompt: ")[0].split())
                prompt = re.sub(r"<.*>", "", prompt).strip().strip(",").strip()
            except:
                pass

            self.prompt = prompt
            print(f"Prompt: '{prompt}'")
        except:
            print(traceback.format_exc())


    #Resize image to fit resized window
    def image_resizer(self, e = None):
        try:
            l, t = self.pct_to_coord(self.l_pct, self.t_pct)
            r, b = self.pct_to_coord(self.r_pct, self.b_pct)
        except:
            print(traceback.format_exc())

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
            if self.image_handle:
                self.canvas.delete(self.image_handle)
        except:
            print(traceback.format_exc())
        self.image_handle = self.canvas.create_image(center_x, center_y, anchor="center",image=self.framed_image)

        try:
            self.generate_crop_rectangle()
        except:
            print(traceback.format_exc())

    #Remove row from feature table
    def remove_row(self, i: int):
        self.feature_count -= 1
        if i != self.feature_count:
            for j in range(i, self.feature_count):
                self.features[j][0]["var"].set(self.features[j + 1][0]["var"].get())
                self.features[j][1]["var"].set(self.features[j + 1][1]["var"].get())
        self.features[self.feature_count][0]["entry"].grid_remove()
        self.features[self.feature_count][1]["entry"].grid_remove()
        self.features[self.feature_count][0]["var"].set("")
        self.features[self.feature_count][1]["var"].set("")

    def split_component(self, c):
        #Get the parts of speech
        pos = do_get_pos(c)

        wasalnum = False
        def rejoin(tokens):
            joined = ""
            first = True
            for t in tokens:
                if t.text != "":
                    if not first and str.isalnum(t.text[0]) and wasalnum:
                        joined += " "
                    first = False
                    joined += t.text
                    wasalnum = str.isalnum(t.text[-1])
            return joined

        #Rejoin any tokens that were split by a hyphen.
        fixed_pos = []
        it = iter(range(len(pos)))
        for i in it:
            try:
                if i < len(pos) - 2:
                    if pos[i + 1].text[0] == '-':
                        class token_imitator():
                            def __init__(self, pos_, text):
                                self.pos_ = pos_
                                self.text = text
                        joined_item = token_imitator(pos[i + 2].pos_, pos[i].text + "-" + pos[i + 2].text)
                        fixed_pos.append(joined_item)
                        next(it, None)
                        next(it, None)
                    else:
                        fixed_pos.append(pos[i])
                else:
                    fixed_pos.append(pos[i])
            except:
                print(traceback.format_exc())

        #If the final component is recognized as any kind of noun,
        #then use that as the parent.
        last_word_index = -1
        for last_word_token in reversed(fixed_pos):
            if str.isalnum(last_word_token.text[0]):
                break
            last_word_index -= 1
        if len(fixed_pos) > 1 and last_word_token.pos_ in ["NOUN", "PROPN"]:
            parent = last_word_token.text

            #Split by adjective
            splits = []
            this_split = []
            for c in fixed_pos[:last_word_index]:
                this_split.append(c)
                if c.pos_ in ["ADJ", "NUM", "NOUN", "PROPN"]:
                    splits.append(this_split)
                    this_split = []
            
            if this_split != []:
                splits.append(this_split)
                this_split = []

            retval = [parent]
            for s in splits:
                retval.append(parent + treeview_separator + rejoin([x for x in s]))
            
            return retval

        #If nothing else was detected, return the entire component.
        return [c]

    def build_checklist_from_features(self):
        path = relpath(pathlib.Path(self.image_files[self.file_index]).absolute().parent, self.path)
        parents = [str(p).strip() for p in pathlib.Path(path).parents]
        parents.insert(0, str(path).strip())
        self.feature_checklist = []
        for p in parents:
            for x in self.known_feature_checklists[str(p)]:
                if x not in self.feature_checklist:
                    self.feature_checklist.append(x)

        for row in self.features:
            name = row[0]["var"].get().strip()
            if name != '':
                desc = row[1]["var"].get().strip()            
                self.feature_checklist.append((name,True))
                components = desc.split(",")
                for c in components:
                    c = c.strip()
                    if c != '' and c != name:
                        for c_split in self.split_component(c):
                            self.feature_checklist.append(
                                (name + treeview_separator + c_split, True))
        self.feature_checklist.sort()

        self.update_checklist()

    #Callback for when feature is modified
    def feature_modified(self, var: str):
        self.disable_feature_tracing()
        found_i = None
        for i in range(self.feature_count):
            for j in range(len(self.features[i])):
                if self.features[i][j]["var"] is var:
                    found_i = i
                    found_j = j

        for i in range(self.feature_count - 1):
            if(not self.features[i][0]["var"].get()
              and not self.features[i][0]["var"].get()):
                self.remove_row(i)
                if i < found_i:
                    self.features[found_i - 1][found_j - 1]["entry"].focus()
                else:
                    self.features[found_i][found_j]["entry"].focus()


        if(self.features[self.feature_count - 1][0]["var"].get()
           or self.features[self.feature_count - 1][1]["var"].get()):
            self.add_row()


        self.build_checklist_from_features()
        self.enable_feature_tracing()


    #Add entry to feature table
    def add_entry(self, i: int, j: int):
        s = tk.StringVar(None)
        t = s.trace("w", 
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
        e.bind("<Control-t>", self.update_ui_automatic_tags)
        return {"var":s, "entry":e, "trace": t}

    #Add row to feature table
    def add_row(self):
        self.feature_count += 1
        if self.feature_count > len(self.features):
            row = []
            for j in range(2):
                row.append(self.add_entry(self.feature_count, j))
            self.features.append(row)
        else:
            self.features[self.feature_count - 1][0]["entry"].grid()
            self.features[self.feature_count - 1][1]["entry"].grid()


    def get_item_from_ui(self):
        item = self.get_defaults()
        try: 
            item["artist"] = self.artist_name.get()
        except: 
            print(traceback.format_exc())

        try: 
            item["style"] = self.style.get()
        except: 
            print(traceback.format_exc())

        try:
            item["title"] = self.title_var.get()
        except:
            print(traceback.format_exc())

        try:
            item["rating"] = self.rating.get()
        except:
            print(traceback.format_exc())

        try:
            item["summary"] = ' '.join(self.summary_textbox.get("1.0", "end").split())
        except:
            print(traceback.format_exc())

        try:
            item["crop"] = [
                self.l_pct,
                self.t_pct,
                self.r_pct,
                self.b_pct
            ]
        except:
            print(traceback.format_exc())

        try:
            features = {}
            for i in range(self.feature_count):
                key = self.features[i][0]["var"].get()
                if(key):
                    extant = ""
                    val = self.features[i][1]["var"].get()
                    if key in features:
                        extant = features[key] + ", "
                    features.update({key: extant + val})
            item["features"] = features
        except:
            print(traceback.format_exc())

        try:
            item["automatic_tags"] = ' '.join(self.automatic_tags_textbox.get("1.0", "end").split())
        except:
            print(traceback.format_exc())

        return item
    
    def get_defaults(self, path = None):
        if path is None:
            if len(self.image_files) == 0:
                path = "./dataset/default.png"
            else:
                path = self.image_files[self.file_index]
                
        defaults = {"lora_tag_helper_version": 1,
                    "title":splitext(pathlib.Path(path).name)[0],
                    "artist": "unknown",
                    "style": "photo",
                    "rating": 0,
                    "summary": self.prompt,
                    "features": {},
                    "crop": [0, 0, 1, 1],
                    "automatic_tags": ""}

        if len(self.image_files) == 0:
            return defaults
        
        path = pathlib.Path(path)        
        paths = [p for p in reversed(pathlib.Path(path).parents) if p not in pathlib.Path(self.path).parents]

        for p in paths:
            json_file = p / "defaults.json"
            if isfile(json_file):
                with open(json_file) as f:
                    defaults.update(json.load(f))
        return defaults
    
    def get_item_from_file(self, path):
        #Read filename into title
        item = self.get_defaults(path)

        #If .txt available, read into automated caption
        txt_file = splitext(path)[0] + ".txt"
        try:
            with open(txt_file) as f:
                item["automatic_tags"] = ' '.join(f.read().split())
        except (FileNotFoundError) as error:
            pass

        #If available, parse JSON into fields
        json_file = splitext(path)[0] + ".json"
        try:
            with open(json_file) as f:
                json_item = json.load(f)
                item.update(json_item)
        except FileNotFoundError:
            pass

        try:
            if item["lora_tag_helper_version"] > 1:
                print("Warning: file generated by newer version of lora_tag_helper")
        except:
            print(traceback.format_exc())

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
        
    def update_known_feature_checklists(self):
        path = relpath(pathlib.Path(self.image_files[self.file_index]).absolute().parent, self.path)
        temp_checklist = self.feature_checklist
        for i in range(len(temp_checklist)):
            temp_checklist[i] = (temp_checklist[i][0], False)
        self.known_feature_checklists[path] = temp_checklist
        

    #Add UI elements for save JSON button
    def save_json(self, event = None):
        file = self.image_files[self.file_index]
        self.write_item_to_file(
            self.get_item_from_ui(),
            splitext(file)[0] + ".json")
        self.update_known_feature_checklists()

    #Update automatic tags in JSON for image file
    def update_automatic_tags(self, path, popup=False):
        if not interrogator_ready:
            showwarning(parent=self,
                        title="Not ready",
                        message="The interrogator is not yet ready.")
            return        
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
        if not interrogator_ready:
            showwarning(parent=self,
                        title="Not ready",
                        message="The interrogator is not yet ready.")
            return        

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
        if not interrogator_ready:
            showwarning(parent=self,
                        title="Not ready",
                        message="The interrogator is not yet ready.")
            return "break"    
        if len(self.image_files) > 0:
            self.save_unsaved_popup()
            self.update_automatic_tags(self.image_files[self.file_index])
            self.set_ui(self.file_index)
        return "break"


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
        self.artist_name_entry.focus()
        class event_imitator():
            def __init__(self, widget):
                self.widget = widget
        self.select_all(event_imitator(self.artist_name_entry))




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
        self.artist_name_entry.focus()
        class event_imitator():
            def __init__(self, widget):
                self.widget = widget
        self.select_all(event_imitator(self.artist_name_entry))

    def save_defaults(self, event = None):
        if len(self.image_files) == 0:
            showerror(parent=self, title="Error", message="Dataset must be open")
            return
        #Pop up dialog to save default settings for path
        self.update()
        self.wait_window(save_defaults_popup(self).top)
        self.update()

    def add_features_from_summary(self, event = None):
        text = self.summary_textbox.get("1.0", "end")
        components = text.split(',')
        words = []
        for c in components:
            words.extend(c.split())

        features = {f[0] for f in self.feature_checklist}
        active_features = {f[0] for f in self.feature_checklist if f[1]}

       
        features_to_add = {f for f in features if f in words and f not in active_features}

        self.disable_feature_tracing()
        i = self.feature_count - 1
        try:
            for f in features_to_add:
                self.add_row()
                self.features[i][0]["var"].set(f)
                i += 1
        except:
            print(traceback.format_exc())
        self.enable_feature_tracing()
        self.feature_modified(self.features[0][0]["var"])

        



    def reset(self, event = None):
        self.set_ui(self.file_index, self.get_defaults())
        

    def go_to_image(self, event = None, file = None):
        if not file:
            file = tk.filedialog.askopenfilename(parent=self.root_frame, initialdir=self.path, title="Select an image in the dataset", filetypes =[('Supported images', [f"*{x}" for x in Image.registered_extensions()])])
            if file:
                file = pathlib.Path(file).absolute()
        if file.is_dir():
            i = 0
            for f in self.image_files:
                if str(f).startswith(str(file)):
                    self.file_index = i
                    self.set_ui(i)
                    break
                i += 1
        elif file:
            try:
                print(f"File in go_to_image: {file}")
                i = self.image_files.index(pathlib.Path(file))
                print(f"Index in go_to_image: {i}")
                self.file_index = i
                self.set_ui(i)
            except ValueError:
                print(f"Warning: Supplied path {file} is not an image in the dataset. Ignoring.")


        
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
                self.update_known_feature_checklists()

    def select_all(self, event):
        # select text
        try:
            event.widget.select_range(0, 'end')
        except:
            print(traceback.format_exc())
            event.widget.tag_add("sel", "1.0", "end")

        # move cursor to the end
        try:
            event.widget.icursor('end')
        except:
            print(traceback.format_exc())
            event.widget.mark_set("insert", "end")

        #stop propagation
        return 'break'

    #Create quit action
    def quit(self, event = None):
        self.save_unsaved_popup()

        self.destroy()


#Application entry point
if __name__ == "__main__":
    global app
    #Instantiate the application
    app = lora_tag_helper()
    #Let the user do their thing
    app.mainloop()
