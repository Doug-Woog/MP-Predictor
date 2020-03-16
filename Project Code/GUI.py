import tkinter as tk
from tkinter import filedialog,CENTER,S
from random import randint
from Model_Training import model
from rdkit import Chem
from rdkit.Chem import Draw
from mordred import descriptors
import mordred as m
import numpy as np
import pandas as pd
from PIL import ImageTk,Image
from pubchemprops.pubchemprops import get_second_layer_props
import pubchempy as pcp

testcalc = m.Calculator(descriptors)
alldesc = dict((descript.__str__(),descript) for descript in testcalc.descriptors)

class Overview(tk.Tk):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        container = tk.Frame(self)
        self.attributes("-fullscreen",True)
        self.title("Chemical Melting Point Prediction Program")
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)
        self.frames = {}
        for F in (mainPage,predictionPage,examplesPage):
            page_name = F.__name__
            frame = F(parent=container, controller=self)
            self.frames[page_name] = frame

            # put all of the pages in the same location;
            # the one on the top of the stacking order
            # will be the one that is visible.
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame("mainPage")
    
    def show_frame(self, page_name):
        '''Show a frame for the given page name'''
        for frame in self.frames.values():
            frame.grid_remove()
        frame = self.frames[page_name]
        frame.grid()
        
        #frame = self.frames[page_name]
        #frame.tkraise()

    def get_page(self, classname):
        '''Returns an instance of a page given it's class name as a string'''
        for page in self.frames.values():
            if str(page.__class__.__name__) == classname:
                return page
        return None


class mainPage(tk.Frame):
    def __init__(self, parent,controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        self.create_widgets()
        
        
    
    def create_widgets(self):
        self.screentext= tk.Label(self,text= "Welcome to the Chemical Melting Point Prediction Program\nProceed by pressing the button below",font=('TkDefaultFont','25'),anchor=CENTER)
        self.screentext.pack(side='top',expand='yes',fill='both')
        self.quit = tk.Button(self, text="QUIT", fg="red",
                              command=self.controller.destroy,font=('TkDefaultFont','12'))
        
        self.predictbutton = tk.Button(self,command=self.proceed,font=('TkDefaultFont','20'))
        self.predictbutton["text"] = "Predict Chemical Melting Points"
        self.predictbutton.pack(side='top',expand='yes')
        self.gotoexamples = tk.Button(self,command=self.examples,text="Example Chemicals",font=('TkDefaultFont','20'))
        self.gotoexamples.pack()
        self.quit.pack()


    def proceed(self):
        """Go to Predictions Page"""
        self.controller.show_frame("predictionPage")

    def examples(self):
        """Go to Examples Page"""
        self.controller.show_frame("examplesPage")

class predictionPage(tk.Frame):
    """Page for predicting melting points"""
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        
        label = tk.Label(self, text="Please enter the SMILES string for the molecule whose melting point you would like to be predicted",font=('TkDefaultFont','20'))
        label.pack(side="top", fill="x", pady=10)
        button = tk.Button(self, text="Go to the start page",
                           command=lambda: controller.show_frame("mainPage"),font=('TkDefaultFont','15'))
        button.pack(side="bottom")
        self.SMILES = tk.StringVar()
        self.SMILES.set("Enter SMILES")
        #self.test = tk.Label(self,text=self.SMILES.get())
        #self.test.pack()
        self.molimg = tk.Label(self)
        self.molimg.pack(expand='yes')
        self.pred = tk.StringVar()
        self.pred.set("Predicted Melting Point: Please enter SMILES string")
        self.modelname = tk.Label(self,text="Model Name: No model loaded",font=('TkDefaultFont','12'))
        self.modelname.pack()
        self.modelChoose = tk.Button(self,text="Select Model File",command=lambda: self.find_model(),font=('TkDefaultFont','13'))
        self.modelChoose.pack()
        self.mp = tk.Label(self,text=self.pred.get(),font=('TkDefaultFont','15'),fg='red')
        self.mp.pack(side="bottom")
        self.MPbutton = tk.Button(self,text="Find Actual Melting Point",command=lambda: self.getMP(),font=('TkDefaultFont','13'))
        self.MPbutton.pack()
        self.MPActual = tk.Label(self,text="Actual Melting Point: Click button to find Actual Melting Point",font=('TkDefaultFont','12'))
        self.MPActual.pack(side='bottom')

        def callback(self,sv):
            """Called on text input change"""
            inputMol = sv.get()
            #self.test['text']= inputMol
            self.MPActual['text']= "Actual Melting Point: Click button to find Actual Melting Point"
            if inputMol != '':
                mol = Chem.MolFromSmiles(inputMol)
            else:
                mol = None

            if mol == None:
                self.mp['text'] = "Please enter valid SMILES string"
            else:
                self.mp['text'] = self.predictMP(mol)
                img = Chem.Draw.MolToImage(mol)
                tkimg = ImageTk.PhotoImage(img)
                self.molimg['image'] = tkimg
                self.molimg.image = tkimg
              


        self.SMILES.trace("w", lambda name, index, mode, sv=self.SMILES: callback(self,self.SMILES))
        tk.Entry(self, width = 70, font = ('Arial', 15), textvariable = self.SMILES).pack()

    def find_model(self):
        """Finds and loads a model"""
        modelFilepath =  filedialog.askopenfile()
        self.modelname['text'] = "Model Name: "+modelFilepath.name[:-7].split('/')[-1]+"\n"
        self.model = model('load')
        self.model.load_model(modelFilepath.name)

        self.modelname['text'] += "Model Type: "+ str(self.model.information['Training']['Model Type'])+"\n"
        self.modelname['text'] += "Model Parameters: "+ str(self.model.information['Training']['Model Parameters'])+"\n"
        self.modelname['text'] += "PCA: "+ str(self.model.information['Training']['PCA'])+"\n"
        self.modelname['text'] += "Features: "+ str(self.model.information['Training']['Features'])+"\n"
        self.modelname['text'] += "Training Samples: "+ str(self.model.information['Training']['Samples'])+"\n"
        self.modelname['text'] += "Training RMSE: "+ str(self.model.information['Training']['RMSE'])+" ºC\n"
        self.modelname['text'] += "Test Samples: "+ str(self.model.information['Testing']['Samples'])+"\n"
        self.modelname['text'] += "Test RMSE: "+ str(self.model.information['Testing']['RMSE'])+" ºC"

        
        self.calc = m.Calculator()
        for desc in self.model.getDescriptors():
            self.calc.register(alldesc[desc])

        mol = Chem.MolFromSmiles(self.SMILES.get())
        if mol == None:
            self.mp['text'] = "Please enter valid SMILES string"
        else:
            self.mp['text'] = self.predictMP(mol)

    def predictMP(self,smilesstring):
        """Uses currently loaded model to predict MP of inputted chemical"""
        try:
            calcdesc = pd.DataFrame(self.calc(smilesstring)).to_numpy().reshape(1,-1)
        except:
            return "Calculator Error: Make sure valid model has been selected"

        try:
            return "Predicted Melting Point: "+str(self.model.predictSingle(calcdesc)[0]) + " ºC"
        except:
            return "Molecule is missing descriptors"

    def getMP(self):
        """Checks Pubchem for actual MP of inputted chemical"""
        smiles= self.SMILES.get()
        try:
            results = pcp.get_compounds(smiles, 'smiles')
            resultid = results[0].cid
        except:
            resultid = None
        found = False
        if resultid:
            prop = get_second_layer_props(resultid,['Melting Point'])
            if 'Melting Point' in prop.keys():
                for measure in prop['Melting Point']:
                    if 'StringWithMarkup' in measure['Value'].keys():
                        if "°C" in measure['Value']['StringWithMarkup'][0]['String']:
                            MP = measure['Value']['StringWithMarkup'][0]['String']
                            found = True
                            break
                    elif 'Unit' in measure['Value'].keys():
                        if 'Number' in measure['Value'].keys():
                            if "°C" in measure['Value']['Unit']:
                                MP = str(measure['Value']['Number'][0]) +" °C"
                                found = True
                                break
        if found == True:
            self.MPActual['text']= "Actual Melting Point: "+MP
        else:
            self.MPActual['text']= "Actual Melting Point: Unknown"

class examplesPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        title = tk.Label(self,text="Example Chemicals\nClick to copy",font=('TkDefaultFont','20'))
        title.pack()
        button = tk.Button(self, text="Go to the start page",
                           command=lambda: controller.show_frame("mainPage"),font=('TkDefaultFont','17'))
        button.pack(side='bottom')
        def setmol(molname):
            page_one = self.controller.get_page("predictionPage")
            page_one.SMILES.set(molname)
            controller.show_frame("predictionPage")
        examples = {
        'Propane':'CCC',
        'Cocaine':'CN1C2CCC1C(C(C2)OC(=O)C3=CC=CC=C3)C(=O)OC',
        'Salicylic acid':'C1=CC=C(C(=C1)C(=O)O)O',
        'Paracetamol':'CC(=O)NC1=CC=C(C=C1)O',
        'Glucose':'C([C@@H]1[C@H]([C@@H]([C@H](C(O1)O)O)O)O)O',
        'THC':'CCCCCC1=CC(=C2[C@@H]3C=C(CC[C@H]3C(OC2=C1)(C)C)C)O',
        'LSD':'CCN(CC)C(=O)[C@H]1CN([C@@H]2CC3=CNC4=CC=CC(=C34)C2=C1)C',
        'Aspirin':'CC(=O)OC1=CC=CC=C1C(=O)O',
        'Guanine':'C1=NC2=C(N1)C(=O)NC(=N2)N',
        'Caffeine':'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',
        'Dopamine':'C1=CC(=C(C=C1CCN)O)O',
        'Adrenaline':'CNC[C@@H](C1=CC(=C(C=C1)O)O)O'
        }
        
        for mol in examples.keys():
            tk.Button(self,text=mol,command=lambda mol=mol:setmol(examples[mol]),font=('TkDefaultFont','15')).pack()
        

if __name__ == "__main__":
    app = Overview()
    app.mainloop()