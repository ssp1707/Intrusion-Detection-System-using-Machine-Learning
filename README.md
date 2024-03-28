# Intrusion-Detection-System-using-Machine-Learning

This project detects if a ntwork is intrusive or not by utilizing the power of one-hot encoding and random forest classifier.

## About

The project Intrusion Detection System using Machine Learning aims to develop a robust system capable of identifying and mitigating malicious activities within computer networks through the application of advanced machine learning techniques. By leveraging historical network traffic data and labeled intrusion examples, the system will employ various machine learning algorithms such as decision trees, support vector machines, and neural networks to detect anomalies indicative of unauthorized access, data breaches, or other security breaches. Through extensive training and validation processes, the system will continuously refine its models to adapt to emerging threats and enhance detection accuracy, ultimately bolstering the overall cybersecurity posture of the network environment.

## Features Used
1. Unusual Traffic Patterns: Unusual traffic patterns can be identified through features such as duration, src_bytes, dst_bytes, and count. A sudden spike in count (number of connections to the same host) or abnormally large values in src_bytes or dst_bytes may indicate suspicious activity.

2. Anomalies in Protocol Usage: Protocol usage can be inferred from features such as protocol_type, service, and flag. For example, unexpected values in the protocol_type or flag features could indicate protocol misuse or unusual behavior.

3. Repeated Failed Login Attempts: Failed login attempts can be detected through features related to authentication, such as num_failed_logins. Multiple failed login attempts from the same source IP address may be indicative of a brute-force attack.

4. Unusual System Activities: System activity can be represented by features such as num_file_creations, num_shells, num_root, etc. Unusual values in these features, such as a high number of file creations or shells opened, may indicate malicious activity.

## Requirements

- Python 3.x
- Required Python packages: pandas and sklearn

## Installation

1. Clone the repository:

   ```shell
   git clone [https://github.com/ssp1707/Intrusion-Detection-System-using-Machine-Learning.git]

2. Install the required packages:

   ```shell
   pip install -r requirements.txt

3. Download the pre-trained ids model and classify if a network is intrusive or not.

## Usage

1. Run the python program
   
2. The program itself selects a random range of sample data to detect.
   
3. The model then detects if the network activity is intrusive or non-intrusive.

## System Architecture
![image](https://github.com/ssp1707/Intrusion-Detection-System-using-Machine-Learning/assets/75234965/d954af6f-b933-4697-b021-1d1d80cb0fa9)

## Program
```
# Import the prerequisite libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
# Load the dataset
column_names = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes',
                'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
                'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
                'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
                'num_access_files', 'num_outbound_cmds', 'is_host_login',
                'is_guest_login', 'count', 'srv_count', 'serror_rate',
                'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
                'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
                'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
                'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
                'dst_host_serror_rate', 'dst_host_srv_serror_rate',
                'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'class']
df = pd.read_csv("kddcup.data_10_percent.gz", header=None, names=column_names)te
# Preprocessing
df = pd.get_dummies(df, columns=['protocol_type', 'service', 'flag'])
X = df.drop('class', axis=1)
y = df['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Training the model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)def check_pout(mouth_right,mouth_left,orig_mouth_dist):
# Evaluation
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
# Sample network activity data
intrusive_count = 0
non_intrusive_count = 0
for i in range(len(X_test)):
    sample_data = X_test.iloc[i].values.reshape(1, -1)
    actual_label = y_test.iloc[i]
    prediction = rf.predict(sample_data)
    # Print intrusive instances
    if prediction != 'normal.' and intrusive_count < 5:  
        print("Sample", i+1, ": The network activity is intrusive.")
        intrusive_count += 1
    # Print not intrusive instances
    if prediction == 'normal.' and non_intrusive_count < 5: 
        print("Sample", i+1, ": The network activity is not intrusive.")
        non_intrusive_count += 1
    if intrusive_count >= 5 and non_intrusive_count >= 5:  
        break    
```
## Output

### Confusion Matrix
![image](https://github.com/ssp1707/Intrusion-Detection-System-using-Machine-Learning/assets/75234965/82c44905-3a2b-49aa-a830-a16509843eae)

### Classification Report
![image](https://github.com/ssp1707/Intrusion-Detection-System-using-Machine-Learning/assets/75234965/84f3713e-604d-42d7-b7fe-1e2038accbad)

### Predicition
![image](https://github.com/ssp1707/Intrusion-Detection-System-using-Machine-Learning/assets/75234965/b106bc0d-024c-4f25-9b64-62f55f0239f8)
![image](https://github.com/ssp1707/Intrusion-Detection-System-using-Machine-Learning/assets/75234965/06aa285f-ddfa-47b9-ae0e-4c7f5a3554b7)

## Result
This project Intrusion Detection System using Machine Learning has successfully developed a robust system to detect the intrusive network activities. This model was trained using KDD Cup 1999 dataset and is evaluated to check its accuracy and performance.

## Articles Published/References
1. Hung-Jen Len and Ying-Chih Lin, “Intrusion detection system: A comprehensive review”, 2013.

2. Nathan Shone, "A Deep Learning Approach for Network Intrusion Detection System.", 2018.

3. Mohammad Almseidin and Maen Alzubi, "Evaluation of machine learning algorithms forintrusion detection system", 2017.

4. Anish Halimaa A, "Machine Learning Based Intrusion Detection System", 2017.

5. R. Vinayakumar; Mamoun Alazab and K. P. Soman, "Deep Learning Approach for Intelligent Intrusion Detection System”, 2019.

6. Lirim Ashiku, Cihan Dagli, "Network Intrusion Detection System using Deep Learning", 2021.

7. D. Shreeya Jain, Pranav M. Pawar and Raja Muthalagu, "Hybrid Intrusion Detection System for Internet of Things (IoT)",2022.

8. Zeeshan Ahmad and Adnan Shahid Khan, “Network intrusion detection system: A systematic 	study of machine learning and deep learning approaches”, 2020.
