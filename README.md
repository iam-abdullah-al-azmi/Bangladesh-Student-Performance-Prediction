# Bangladesh Student Performance Prediction

## Summary

In this machine learning project, different machine learning algorithm has been
implemented to observe the relationship between the hsc_result of a student and
other features.

## Dataset Overview

Here, the top 5 rows of the dataset provided for having an overview.

### First Portion

|    date    | gender | age | address | famsize | Pstatus  | M_Edu | F_Edu |
| :--------: | :----: | :-: | :-----: | :-----: | :------: | :---: | :---: |
| 29/04/2018 |   M    | 18  |  Rural  |   GT3   | Together |   3   |   2   |
| 29/04/2018 |   F    | 19  |  Rural  |   LE3   |  Apart   |   0   |   4   |
| 29/04/2018 |   F    | 19  |  Rural  |   GT3   | Together |   0   |   3   |
| 29/04/2018 |   F    | 19  |  Rural  |   LE3   |  Apart   |   2   |   3   |
| 29/04/2018 |   M    | 17  |  Rural  |   GT3   | Together |   1   |   1   |

### Second Portion

|  M_Job  |  F_Job   | relationship | smoker | tuition_fee | time_friends |
| :-----: | :------: | :----------: | :----: | :---------: | :----------: |
| At_home |  Farmer  |      No      |   No   |    71672    |      4       |
|  Other  |  Health  |     Yes      |   No   |    26085    |      5       |
| Teacher | Services |      No      |   No   |    40891    |      3       |
| At_home | Business |      No      |   No   |    50600    |      2       |
| At_home |  Farmer  |      No      |   No   |    62458    |      2       |

### Third Portion

| ssc_result | hsc_result |
| :--------: | :--------: |
|    4.22    |    3.72    |
|    3.47    |    2.62    |
|    3.32    |    2.56    |
|    4.57    |    4.17    |
|    4.5     |    3.94    |

## Feature Description

### First Portion

|   famsize   |    Pstatus    |      M_Edu       |      F_Edu       |
| :---------: | :-----------: | :--------------: | :--------------: |
| Family Size | Parent Status | Mother Education | Father Education |

### Second Portion

|   M_Job    |   F_Job    |
| :--------: | :--------: |
| Father Job | Father Job |

### famsize

|      GT3       |     LE3     |
| :------------: | :---------: |
| Greater Than 3 | Less Than 3 |

### M_Edu

|      0       |  1  |  2  |  3  |
| :----------: | :-: | :-: | :-: |
| No Education | JSC | SSC | HSC |

### Abbreviation Explanation

| Abbreviation |         Description          |
| :----------: | :--------------------------: |
|     JSC      | Junior Secondary Certificate |
|     SSC      | Secondary School Certificate |
|     HSC      | Higher Secondary Certificate |

## time_friends

|     1     |     2     |      3      |     4      |     5      |     6     |
| :-------: | :-------: | :---------: | :--------: | :--------: | :-------: |
| One Hours | Two Hours | Three Hours | Four Hours | Five Hours | Six Hours |

## Repository Tree

## Results
