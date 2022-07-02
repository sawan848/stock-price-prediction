   # [Stock-Market-Price-Prediction](https://docs.google.com/document/d/16XxmrRX6agrSqefl0qFdFBFmF1XjJZQLceU5Uqhtt4w) this is based [python](https://python.com)programming language i have used a  [Tensorflow](https://tensorflow.com/) python-library



## I have used Followwing library

<ol>
   <li>Numpy</li>
   <li>Pandas</li>
   <li>Datetime</li>
   <li>Matplotlib</li>
   <li>Pandas_datareader</li>
   <li>Seaborn</li>
   <li>Tensorflow</li>
   <li>Keras</li>
   <li>Streamlit</li>
   <li>Cufflinks</li>
   <li>Plotly</li>
   <li>Prophet</li>
   <li>Streamlit</li>
   <li>Yahoo Finance</li>  
</ol>


I have used  [Anaconda](https://www.anaconda.com) for Machine learning model-building and model-training,model-compiling,[Visual Studio Code](https://code.visualstudio.com/) for text-edifing [PyCharm](https://www.jetbrains.com) for devloped a web-page 


```bash
         choco install python
```
# [DateTime](https://docs.python.org/3/library/datetime.html)
Encapsulate of date/time values
<br/><br/>
### Class DateTime

---


DateTime objects represent instants in time and provide interfaces for
controlling its representation without affecting the absolute value of
the object.

DateTime objects may be created from a wide variety of string or
numeric data, or may be computed from other DateTime objects.
DateTimes support the ability to convert their representations to many
major timezones, as well as the ability to create a DateTime object
in the context of a given timezone.

DateTime objects provide partial numerical behavior:

* Two date-time objects can be subtracted to obtain a time, in days
  between the two.

* A date-time object and a positive or negative number may be added to
  obtain a new date-time object that is the given number of days later
  than the input date-time object.

* A positive or negative number and a date-time object may be added to
  obtain a new date-time object that is the given number of days later
  than the input date-time object.

* A positive or negative number may be subtracted from a date-time
  object to obtain a new date-time object that is the given number of
  days earlier than the input date-time object.

DateTime objects may be converted to integer, long, or float numbers
of days since January 1, 1901, using the standard int, long, and float
functions (Compatibility Note: int, long and float return the number
of days since 1901 in GMT rather than local machine timezone).
DateTime objects also provide access to their value in a float format
usable with the python time module, provided that the value of the
object falls in the range of the epoch-based time module.

A DateTime object should be considered immutable; all conversion and numeric
operations return a new DateTime object rather than modify the current object.

A DateTime object always maintains its value as an absolute UTC time,
and is represented in the context of some timezone based on the
arguments used to create the object.  A DateTime object's methods
return values based on the timezone context.

Note that in all cases the local machine timezone is used for representation if no timezone is specified.

### install DateTime
```bash 
         pip install DateTime
```
# [Seaborn](https://seaborn.pydata.org/)
Seaborn is a library for making statistical graphics in [Python](https://python.com). It builds on top of matplotlib and integrates closely with pandas data structures.

Seaborn helps you explore and understand your data. Its plotting functions operate on dataframes and arrays containing whole datasets and internally perform the necessary semantic mapping and statistical aggregation to produce informative plots. Its dataset-oriented, declarative API lets you focus on what the different elements of your plots mean, rather than on the details of how to draw them

## Install seaborn
```bash
         conda install -c anaconda seaborn
```

# [Plotly](https://plotly.com/python/)
The plotly Python library is an interactive, open-source plotting library that supports over 40 unique chart types covering a wide range of statistical, financial, geographic, scientific, and 3-dimensional use-cases.

Built on top of the Plotly JavaScript library (plotly.js), plotly enables Python users to create beautiful interactive web-based visualizations that can be displayed in Jupyter notebooks, saved to standalone HTML files, or served as part of pure Python-built web applications using Dash. The plotly Python library is sometimes referred to as "plotly.py" to differentiate it from the JavaScript library.

Thanks to deep integration with our Kaleido image export utility, plotly also provides great support for non-web contexts including desktop editors (e.g. QtConsole, Spyder, PyCharm) and static document publishing (e.g. exporting notebooks to PDF with high-quality vector images).
## install plotly
```bash
   conda install -c plotly plotly
   conda install -c plotly/label/test plotl
```
# pandas-datareader
Up to date remote data access for pandas, works for multiple versions of pandas.
## Install pandas_datareader
```bash
         conda install -c anaconda pandas-datareader
```
# [Streamlit](https://docs.streamlit.io/)
Streamlit is an open-source Python library that makes it easy to create and share beautiful, custom web apps for machine learning and data science. In just a few minutes you can build and deploy powerful data apps.

## Install Streamlit
```bash
         pip install streamlit
```
### run streamlit web app
```bash
         streamlit run myfile.py
```
# [Prophet](https://facebook.github.io/prophet/docs/quick_start.html) 
Prophet follows the sklearn model API. We create an instance of the Prophet class and then call its fit and predict methods.
<br/>

The input to Prophet is always a dataframe with two columns: ds and y. The ds (datestamp) column should be of a format expected by Pandas, ideally YYYY-MM-DD for a date or YYYY-MM-DD HH:MM:SS for a timestamp. The y column must be numeric, and represents the measurement we wish to forecast.

## Install Prophet
```bash
         python -m pip install prophet
         conda install -c conda-forge prophet

```
![An old rock in the desert](https://facebook.github.io/prophet/static/quick_start_files/quick_start_12_0.png)
## Install yahoo-finance
```bash
         pip install yahoo-finance
```

## Install [Cufflinks](https://github.com/santosjorge/cufflinks)
```bash
         pip install cufflinks
```

# [Tensorflow](https://www.tensorflow.org/learn) 
TensorFlow makes it easy for beginners and experts to create machine learning models for desktop, mobile, web, and cloud. See the sections below to get started.


## Intall [tensorflow](https://www.tensorflow.org/tutorials) 
```bash
         conda install -c conda-forge tensorflow
         conda install -c conda-forge/label/broken tensorflow
         conda install -c conda-forge/label/cf201901 tensorflow
         conda install -c conda-forge/label/cf202003 tensorflow
```

