### Sort images to the correct folders for classification
### The code sort the 2000 images based based on whether a shadow is identified or not

import os
import shutil
import pandas as pd


### Input parameters

imagePath = '~\\Image Data\\AllImages\\'
outPath = '~\\Image Data\\AllImageSplit\\'
file_name = '~\\Image Data\\Planilha_diagnostic.xlsx'
sheetList = ['Planilha 1', 'Planilha 2', 'Planilha 3', 'Planilha 4', 'Worksheet 5', 'Planilha 6', 'Worksheet 7', 'Worksheet 8', 'Worksheet 9', 'Planilha 10']

### Initialise the image control spreadsheet
conditionDF = pd.DataFrame(columns = ['File', 'Condition1', 'Condition2', 'Condition3', 'Condition4'])

### Load in image information
for i in sheetList:
    df = pd.read_excel(io=file_name,
                       sheet_name=i,
                       skiprows=7,
                       header=None,
                       usecols="A:E",
                       names=['File', 'Condition1', 'Condition2', 'Condition3', 'Condition4']
                       )

    conditionDF = conditionDF.append(df)

print(conditionDF.head(5))

### Determine Image Type
ShadowDF = conditionDF[(conditionDF.Condition1 == 1) | (conditionDF.Condition2 == 1) | (conditionDF.Condition3 == 1) | (conditionDF.Condition4 == 1) | (conditionDF.Condition1 == '1') | (conditionDF.Condition2 == '1') | (conditionDF.Condition3 == '1') | (conditionDF.Condition4 == '1')]
nonShadowDF = conditionDF[(conditionDF.Condition1 != 1) & (conditionDF.Condition2 != 1) & (conditionDF.Condition3 != 1) & (conditionDF.Condition4 != 1) & (conditionDF.Condition1 != '1') & (conditionDF.Condition2 != '1') & (conditionDF.Condition3 != '1') & (conditionDF.Condition4 != '1')]

ShadowList = [(i[:i.find("_", i.find("_")+1)], i[i.find("_", i.find("_")+1)+1:i.find(",")], i[:i.find("_", i.find("_")+1)]+i[i.find("."):i.find(",")]) for i in ShadowDF.File]
nonShadowList = [(i[:i.find("_", i.find("_")+1)], i[i.find("_", i.find("_")+1)+1:i.find(",")], i[:i.find("_", i.find("_")+1)]+i[i.find("."):i.find(",")]) for i in nonShadowDF.File]

for fld, img, name in ShadowList:
    shutil.copy2(imagePath+fld+'\\'+img, outPath+'Shadow\\'+name)

for fld, img, name in nonShadowList:
    shutil.copy2(imagePath+fld+'\\'+img, outPath+'nonShadow\\'+name)