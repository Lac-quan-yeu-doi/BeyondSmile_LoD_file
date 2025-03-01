import pandas as pd
import numpy as np
import os

# set up image size 1*1* 128dpi
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn.preprocessing import MinMaxScaler,StandardScaler





def polygon_area(x, y):
    """
    Calculates the area of a polygon given arrays of x and y coordinates.

    Args:
      x: A NumPy array containing the x-coordinates of the vertices.
      y: A NumPy array containing the y-coordinates of the vertices.

    Returns:
      The area of the polygon.
    """

    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
def calculate_angle(x,y):
    vector_1_x=x[0]-x[1]
    vector_1_y=y[0]-y[1]
    vector_2_x=x[2]-x[1]
    vector_2_y=y[2]-y[1]
    cosine=np.dot([vector_1_x,vector_1_y],[vector_2_x,vector_2_y])\
    /np.sqrt(np.dot([vector_1_x,vector_1_y],[vector_1_x,vector_1_y]))\
    /np.sqrt(np.dot([vector_2_x,vector_2_y],[vector_2_x,vector_2_y]))
    return np.arccos(cosine)/np.pi*180
def calculate_distance(x,y,is_last=0):
    if is_last==0:
        i=-1
        j=0
    elif is_last==1:
        j=0
        i=len(x)//2
    else:
        j=len(x)//4
        i=3*j
    vector_1_x=x[j]-x[i]
    vector_1_y=y[j]-y[i]
    return np.sqrt(np.dot([vector_1_x,vector_1_y],[vector_1_x,vector_1_y]))


# directory=['P13.json','P08.json','P20.json','P18.json','P19.json','P30.json']
directory=os.listdir('./data')
print(directory)


mpl.rcParams[ 'figure.figsize' ] = ( 1, 1 )
mpl.rcParams[ 'figure.dpi' ] = 40
label_df=pd.read_csv('./groundtruth/phq9.csv')
label_df['start_ts']=pd.to_datetime(label_df['start_ts'])
label_df['end_ts']=pd.to_datetime(label_df['end_ts'])


import datetime
def to_depress(ID,time):
    labels=label_df[label_df['pid']==ID]
    labels=labels.reset_index(drop=True)
#     print(labels.shape[0])
    for i in range(labels.shape[0]):
        if (time>labels.loc[i,'start_ts']) and(time<labels.loc[i,'end_ts']):
            return labels.loc[i,'depression_episode']
    return np.nan



for file in directory:
    img_name=[]
    time_stamps=[]
    lop=[]
    rop=[]
    smile=[]
    
    Left_eye=[]
    Right_eye=[]
    lip_uppper=[]
    lip_lower=[]
    Left_eyebrow_ang1=[]
    Right_eyebrow_ang1=[]
    Left_eyebrow_ang2=[]
    Right_eyebrow_ang2=[]
    Left_eyebrow_ang3=[]
    Right_eyebrow_ang3=[]
    Left_eyebrow_ang4=[]
    Right_eyebrow_ang4=[]
    l_eye_brown_len=[]
    r_eye_brown_len=[]
    l_eye_len_h=[]
    r_eye_len_h=[]
    l_eye_len_v=[]
    r_eye_len_v=[]
    lips_ang_up=[]
    lips_ang_low=[]
    
    if not file.endswith('json'):
        continue
    user_ID=file.split('.')[0]
    print(user_ID)
    dataframe=pd.read_json(f'./data/{file}')
    dataframe=dataframe.sort_values('timestamp',ascending=True)
    
    smilingProbability=[classify.get('smilingProbability') for classify \
                    in dataframe['classification'].to_list()]
    rightEyeOpenProbability=[classify.get('rightEyeOpenProbability') for classify \
                            in dataframe['classification'].to_list()]
    leftEyeOpenProbability=[classify.get('leftEyeOpenProbability') for classify \
        in dataframe['classification'].to_list()]
    try:
        os.mkdir(f'./image/{user_ID}')
    except:
        pass
    for i in range(dataframe.shape[0]):
        if i%10!=0:
            continue
        lop.append(leftEyeOpenProbability[i])
        rop.append(rightEyeOpenProbability[i])
        smile.append(smilingProbability[i])
        print(user_ID, '_________',i)
        data_img_points=dataframe.loc[i,'contours']
        time_stamp=int(dataframe.loc[i,'timestamp'].timestamp()*1000)
        time_stamps.append(dataframe.loc[i,'timestamp'])
        
#         time_stamp=datetime.datetime.strptime(time, "%Y-%m-%d, %H:%M:%S.%f").timestamp()
        contours = data_img_points

        x_contour = [point['x'] for point in contours]
        y_contour = [point['y'] for point in contours]

        # Face parts definition: (start_index, end_index, label, color)
        # If a single index or a small set, we'll treat them individually.
        face_parts = [
        ((0, 35),    'Face oval',           'red'),
        ((36, 40),   'Left eyebrow upper',  'blue'),
        ((41, 45),   'Left eyebrow','blue'),
        ((46, 50),   'Right eyebrow upper', 'purple'),
        ((51, 55),   'Right eyebrow','purple'),
        ((56, 71),   'Left eye',            'green'),
        ((72, 87),   'Right eye',           'green'),
        ((88, 96),   'lip_uppper',  'orange'),
        ((97, 105),  'Lower lip',     'orange'),
        ((106, 116), 'Upper lip',     'cyan'),
        ((117, 125), 'lip_lower',  'cyan')
    ]
        
        
        # Plot each part
        for (start, end), label, color in face_parts:
            # If start == end, it's a single point
            x_part = np.array(x_contour[start:end+1])
            y_part = np.array(y_contour[start:end+1])
# Calculate features
            if label=='Left eye':
                area = polygon_area(x_part, y_part)
                Left_eye.append(area)
                l_eye_length=calculate_distance(x_part, y_part,is_last=1)
                l_eye_len_h.append(l_eye_length)
                l_eye_length_v=calculate_distance(x_part, y_part,is_last=2)
                l_eye_len_v.append(l_eye_length_v)
            if label=='Right eye':
                area = polygon_area(x_part, y_part)
                Right_eye.append(area)
                r_eye_length=calculate_distance(x_part, y_part,is_last=1)
                r_eye_len_h.append(r_eye_length)
                r_eye_length_v=calculate_distance(x_part, y_part,is_last=2)
                r_eye_len_v.append(r_eye_length_v)
            if label=='lip_uppper':
                area = polygon_area(x_part, y_part)
                lip_uppper.append(area)
                lips_angle_up = calculate_angle(x_part[[4,5,6]], y_part[[4,5,6]])
                lips_ang_up.append(lips_angle_up)
            if label=='lip_lower':
                area = polygon_area(x_part, y_part)
                lip_lower.append(area)
                lips_angle_low = calculate_angle(x_part[[3,4,5]], y_part[[3,4,5]])
                lips_ang_low.append(lips_angle_low)
    #         Calulate angle
            if label=='Left eyebrow upper':
                angle_l1=calculate_angle(x_part, y_part)
                Left_eyebrow_ang1.append(angle_l1)
                
                angle_l2=calculate_angle(x_part[[0,1,3]], y_part[[0,1,3]])
                Left_eyebrow_ang2.append(angle_l2)
                
                angle_l3=calculate_angle(x_part[[0,1,4]], y_part[[0,1,4]])
                Left_eyebrow_ang3.append(angle_l3)
                
                angle_l4=calculate_angle(x_part[[1,2,3]], y_part[[1,2,3]])
                Left_eyebrow_ang4.append(angle_l4)
                
                l_eye_brown_length=calculate_distance(x_part, y_part)
                l_eye_brown_len.append(l_eye_brown_length)
            if label=='Right eyebrow upper':
                angle_r1=calculate_angle(x_part, y_part)
                Right_eyebrow_ang1.append(angle_r1)
                
                angle_r2=calculate_angle(x_part[[0,1,3]], y_part[[0,1,3]])
                Right_eyebrow_ang2.append(angle_r2)
                
                angle_r3=calculate_angle(x_part[[0,1,4]], y_part[[0,1,4]])
                Right_eyebrow_ang3.append(angle_r3)
                
                angle_r4=calculate_angle(x_part[[1,2,3]], y_part[[1,2,3]])
                Right_eyebrow_ang4.append(angle_r4)
                

                
                
                r_eye_brown_length=calculate_distance(x_part, y_part)
                r_eye_brown_len.append(l_eye_brown_length)
                
                
                
####################PLOTING_IMG################################

        plt.figure(figsize=(10, 12))

        # Plot each part
        for (start, end), label, color in face_parts:
            # If start == end, it's a single point
            x_part = x_contour[start:end+1]
            y_part = y_contour[start:end+1]

            # Plot the points for this part
            plt.scatter(x_part, y_part, c=color, label=label, edgecolor='black')

            # Optionally, connect them with a line if it's a range > 1 point
            if end > start:
                # Connect the points in order
                plt.plot(x_part, y_part, c=color)
            if label == 'Left eyebrow':
                # 36,41,40,45,... is the position of transition of top and bottom part position
                plt.plot([x_contour[i] for i in [36,41]], [y_contour[i] for i in [36,41]], c=color) 
                plt.plot([x_contour[i] for i in [40,45]], [y_contour[i] for i in [40,45]], c=color)
            if label == 'Right eyebrow':
                plt.plot([x_contour[i] for i in [46,51]], [y_contour[i] for i in [46,51]], c=color)
                plt.plot([x_contour[i] for i in [50,55]], [y_contour[i] for i in [50,55]], c=color)
            if label == 'Face oval':
                plt.plot([x_contour[i] for i in [0,35]], [y_contour[i] for i in [0,35]], c=color)
            if label == 'lip':
                plt.plot([x_contour[i] for i in [88,125]], [y_contour[i] for i in [88,125]], c=color)
                plt.plot([x_contour[i] for i in [96,117]], [y_contour[i] for i in [96,117]], c=color)
        #     if label == 'Upper lip':
        #         plt.plot([x_contour[i] for i in [88,106]], [y_contour[i] for i in [88,106]], c=color)
        #         plt.plot([x_contour[i] for i in [96,116]], [y_contour[i] for i in [96,116]], c=color)
        #     if label == 'Lower lip':
        #         plt.plot([x_contour[i] for i in [97,117]], [y_contour[i] for i in [97,117]], c=color)
        #         plt.plot([x_contour[i] for i in [105,125]], [y_contour[i] for i in [105,125]], c=color)
        # Invert y-axis to match typical image coordinates
        plt.gca().invert_yaxis()

        # plt.title('Facial Contours by Region')
        # plt.xlabel('X')
        # plt.ylabel('Y')
        # plt.grid(True)
        plt.axis('off')
        # plt.legend(loc='upper left', bbox_to_anchor=(1,1))
        plt.tight_layout()
        plt.savefig(f'./image/{user_ID}/{user_ID}_{time_stamp}.png',dpi=20)
        plt.clf()
        plt.close()

        img_name.append(f'./image/{user_ID}/{user_ID}_{time_stamp}.png')
    for i in [Left_eye,Right_eye,lip_uppper,lip_lower,lips_ang_up,lips_ang_low,Left_eyebrow_ang1,Left_eyebrow_ang2,
    Left_eyebrow_ang3,Left_eyebrow_ang4,l_eye_brown_len,l_eye_len_h,l_eye_len_v,Right_eyebrow_ang1,Right_eyebrow_ang2,
    Right_eyebrow_ang3,Right_eyebrow_ang4,r_eye_brown_len,r_eye_len_h,r_eye_len_v]:
        print(len(i))
    feature1=pd.DataFrame(np.array([
                                    Left_eye,
                                    Right_eye,
                                    
                                    lip_uppper,
                                   lip_lower,
                                    lips_ang_up,
                                    lips_ang_low,
                                    
                                    Left_eyebrow_ang1,
                                    Left_eyebrow_ang2,
                                    Left_eyebrow_ang3,
                                    Left_eyebrow_ang4,
                                    l_eye_brown_len,
                                    l_eye_len_h,
                                    l_eye_len_v,
                                    
                                    Right_eyebrow_ang1,
                                    Right_eyebrow_ang2,
                                   Right_eyebrow_ang3,
                                    Right_eyebrow_ang4,
                                   r_eye_brown_len,
                                    r_eye_len_h,
                                   r_eye_len_v]).T,
                      columns=[
#                                eye_fewatures
                               'Left_eye',
                               'Right_eye',
#                             lips features
                               'lip_uppper',
                              'lip_lower',
                              'lips_angle_up',
                                'lips_angle_low',
#                               Left eye brown features
                               'Left_eyebrow_ang1',
                               'Left_eyebrow_ang2',
                               'Left_eyebrow_ang3',
                               'Left_eyebrow_ang4',
#                                eye_features
                               'l_eye_brown_len',
                               'l_eye_len_h',
                               'l_eye_len_v',
#                                Right eye brown  Features
                               'Right_eyebrow_ang1',
                               'Right_eyebrow_ang2',
                              'Right_eyebrow_ang3',
                               'Right_eyebrow_ang4',
#                                eye_features
                              'r_eye_brown_len',
                               'r_eye_len_h',
                              'r_eye_len_v'])    
    # features_scale=feature1.copy()
    scaler=StandardScaler()#MinMaxScaler()
    features_scale=scaler.fit_transform(feature1)

    features_final=pd.DataFrame(features_scale,columns=feature1.columns.to_numpy())
    features=pd.DataFrame(np.array(img_name).T,
                      columns=['img_name'])
    features['ID']=user_ID
    features['timestamp']=time_stamps
    features['smile']=smile
    features['right_eye_open']=rop
    features['left_eye_open']=lop
    features['level']=features[['ID','timestamp']].apply(lambda x: \
                                                    to_depress(x['ID'],x['timestamp']),axis=1)
    features=pd.concat([features,features_final],axis=1)
    features.to_csv(f'./data_img_info_csv/{user_ID}.csv')
      
    


























# In[ ]:





# In[ ]:




