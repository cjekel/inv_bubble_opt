from abaqus import *
from abaqusConstants import *
from caeModules import *
from driverUtils import executeOnCaeStartup
import os.path as path
import numpy as np
executeOnCaeStartup()

o3 = session.openOdb(name='model.odb')
total_frames = len(o3.steps['Step-1'].frames)
session.viewports['Viewport: 1'].setValues(displayedObject=o3)
odb = session.odbs['model.odb']
session.fieldReportOptions.setValues(reportFormat=COMMA_SEPARATED_VALUES)
session.writeFieldReport(fileName='BubbleTest/L00000.csv', append=OFF, 
    sortItem='Node Label', odb=odb, step=0, frame=total_frames - 1, outputPosition=NODAL, 
    variable=(('U', NODAL, ((COMPONENT, 'U1'), (COMPONENT, 'U2'), (COMPONENT, 'U3'), )), ))

#===============================================================================================
    # Generate the reports for the further increment in loads at each time step = 1 
#===============================================================================================

saved_strings = list()
load = np.linspace(0.001, 3.0, 200)
load = load * 0.0001
for i in range(1, len(load)):
        
    i_string = str(i)
    step_string_i = 'Step-' + i_string
    i_plus_1_string = str(i + 1)
    step_string_i_plus_1 = 'Step-' + i_plus_1_string
    saved_strings.append([step_string_i, step_string_i_plus_1])

for i in range(0, len(saved_strings)):
    step_string_i = saved_strings[i][1]
    folderPath_string = 'BubbleTest/'
    digits = len(str(abs(i+1))) 
    L_string = 'L'
    for j in range(1, 6-digits): L_string += '0'
    L_string += str(i+1)
    L_string += '.csv'
    filePath_string = path.join(folderPath_string, L_string)
    total_frames = len(o3.steps[step_string_i].frames)
    session.writeFieldReport(fileName=filePath_string, append=OFF, 
        sortItem='Node Label', odb=odb, step=i+1, frame=total_frames - 1, outputPosition=NODAL,
        variable=(('U', NODAL, ((COMPONENT, 'U1'), (COMPONENT, 'U2'), (COMPONENT, 'U3'), )), ))