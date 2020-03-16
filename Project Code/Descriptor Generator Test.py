from Descriptor_Making_Machine import generateDescriptors,PCA_fit
import mordred as m
import pandas as pd

descrips = [m.ABCIndex,m.Chi,m.EState,m.AtomCount,m.ZagrebIndex,m.BondCount,m.KappaShapeIndex]
directory = "C:\\Marcus Stuff\\EPQ\\"
#patents+ochem+enamine+bradley+bergstrom fixed

if __name__ == "__main__":
    generateDescriptors(dataset=directory+"Datasets\\patents+ochem+enamine+bradley+bergstrom fixed.csv",filename=directory+"Descriptor Datasets\\patents+ochem+enamine+bradley+bergstrom all",descs=None,big=30000)
"""
dataset = directory+"Descriptor Datasets\\patents+ochem+enamine+bradley+bergstrom descriptors = ['ABCIndex', 'Chi', 'EStat', 'AtomCount', 'ZagrebIndex', 'BondCount', 'KappaShapeIndex'] .csv"
#PCA_fit(dataset,0.99,save_path=directory+"PCA\\Patents Not All Descriptors 99% variance")
df = pd.read_csv(directory+"Datasets\\patents+ochem+enamine+bradley+bergstrom fixed.csv")
for chem in df['Melting Point {measured, converted}']:
    if chem > 300:
        print(chem)
"""