#  Healthcare Workforce Burnout Analysis & Power BI Dashboard

##  Project Overview

Employee well-being has become one of the biggest challenges in modern healthcare organizations. Long shifts, emotional pressure, staff shortages, and work-life imbalance often lead to stress, burnout, absenteeism, and high turnover.

This project was created to analyze workforce mental health patterns using **Python** and transform the findings into an interactive **Power BI Dashboard** for decision-makers.

Using a dataset of **5,000 healthcare employees**, this analysis explores how stress levels, burnout frequency, job satisfaction, department type, and turnover intention are connected.

The goal is simple:

 Turn raw employee data into actionable business insights that help organizations improve employee wellness, retention, and productivity.

---

#  Tools & Technologies Used

###  Python (Data Analytics)

* Pandas
* NumPy
* Matplotlib
* Seaborn

###  Dashboarding

* Power BI

###  Other Tools

* Jupyter Notebook
* Excel / CSV

---

#  Dataset Information

* **Domain:** Healthcare / HR Analytics
* **Records:** 5,000 Employees
* **Format:** CSV
* **Source:** Kaggle (Healthcare Workforce Mental Health Dataset)

### Key Features Included:

* Employee ID
* Employee Type
* Department
* Workplace Factor
* Stress Level
* Burnout Frequency
* Job Satisfaction
* Access to EAPs
* Mental Health Absences
* Turnover Intention

---

#  Business Problem Statement

Healthcare organizations often lose talent due to unmanaged burnout and workplace stress.

Without analytics, leaders struggle to answer:

* Which departments are under the most pressure?
* How does stress affect employee satisfaction?
* Who is likely to leave the company?
* Which workplace factors create the highest burnout risk?

This project solves those questions using data.

---

#  Data Cleaning & Preprocessing

Before analysis, the dataset was cleaned and prepared professionally.

### Steps Performed:

 Removed duplicate records
 Checked missing values
 Standardized column names
 Converted Yes/No columns into numerical values
 Encoded burnout frequency levels
 Removed unnecessary columns like Employee ID
 Created custom features:

* **Stress Category** → Low / Medium / High
* **Burnout Risk Score** → Combined score using stress + satisfaction + burnout frequency

---

#  Exploratory Data Analysis (EDA)

## Univariate Analysis

Used to understand one variable at a time.

### Visuals Created:

* Histogram of Stress Level
* Countplot of Stress Category
* Boxplot of Stress Distribution

### Findings:

* Most employees fall in medium to high stress range
* Several employees show extreme stress outliers
* High stress is not isolated — it affects a large workforce segment

---

## Bivariate Analysis

Used to study relationships between two variables.

### Visuals Created:

* Scatterplot: Stress vs Mental Health Absences
* Barplot: Stress by Employee Role
* Correlation Heatmap

### Findings:

* Higher stress leads to more absences
* Job satisfaction decreases when stress increases
* Certain roles consistently experience higher pressure

---

## Multivariate Analysis

Used to study multiple variables together.

### Visuals Created:

* Pairplot
* Pivot Heatmaps
* Grouped Burnout Risk Charts

### Findings:

* High stress + low satisfaction = highest burnout risk
* Some departments are repeatedly high-risk across metrics
* Turnover intention is strongly associated with burnout

---

#  Key Business Insights

##  1. Burnout is a Major Organizational Risk

* **66% of employees** fall in Critical or Warning burnout zones.

This suggests burnout is no longer an individual issue — it is a system-wide workforce problem.

---

##  2. Stress Levels Are Alarmingly High

* Average stress score = **7.33 / 10**

This indicates many employees are operating under continuous pressure.

---

##  3. Job Satisfaction Has Direct Impact

Employees with lower satisfaction scores consistently show:

* Higher stress
* More burnout
* Greater turnover intention

---

##  4. Highest Risk Departments

Departments requiring immediate intervention:

* Radiology
* ICU
* Specialty Care

These units show repeated high burnout patterns.

---

##  5. Turnover Risk is Serious

* Around **67% employees** expressed intention to leave.

Burnout appears to be a major driver behind retention issues.

---

#  Power BI Dashboard

After Python analysis, the cleaned data was transformed into an interactive dashboard in **Power BI**.

### Dashboard Features:

* Cards
* Department-wise Stress Tracking
* Burnout Risk Segmentation
* Turnover Monitoring
* Filters by Role / Department
* Trend Visualizations
* Executive Summary Insights

### Why Power BI?

Power BI helps non-technical managers understand complex workforce issues quickly and take action faster.

---

#  Recommendations for Management

## Immediate Actions

* Launch wellness programs in ICU, Radiology, Specialty Care
* Provide counselling support
* Reduce excessive workload

## Short-Term Actions

* Improve scheduling flexibility
* Introduce recognition programs
* Improve manager communication

## Long-Term Actions

* Build predictive burnout monitoring systems
* Use HR analytics monthly
* Integrate live dashboards with HR systems

---

#  Sample Visualizations Included

* Histogram
* Boxplot
* Scatterplot
* Heatmap
* Pairplot
* Burnout Risk Dashboard (Power BI)

---
## 🔗 Project Repository

[View Full Project Here]()

```
```

---

#  Project Structure

```bash
healthcare-burnout-analysis/
│── main.py
│── Project word document
│── README.md
│── mental_health_data.csv
```

---

#  What I Learned From This Project

Through this project, I improved my skills in:

* Data Cleaning
* Feature Engineering
* Data Visualization
* Business Insight Generation
* Dashboard Design
* Storytelling with Data

---

#  Future Improvements

* Machine Learning burnout prediction model
* Real-time HR dashboard
* Employee sentiment analysis
* Department benchmarking system

---

#  Author

**Mugamad Hamdhan**

Aspiring Data Analyst passionate about solving business problems using data.

---

#  If You Like This Project

Please give this repository a star and connect with me.
