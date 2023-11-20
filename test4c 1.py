import io
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from streamlit import session_state
from st_aggrid.grid_options_builder import GridOptionsBuilder
from st_aggrid import AgGrid
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from statsmodels.tsa.arima_model import ARIMA


# def home_button():
#     # st.write("""
#     # <style>
#     #     .home-button {
#     #         position: absolute;
#     #         top: 10px;
#     #         right: 10px;
#     #         z-index: 1; /* To make sure it's above other elements */
#     #     }
#     # </style>
#     # """)
#     st.write('<a href="http://localhost:8501/?page=page1" class="home-button">Home</a>', unsafe_allow_html=True)

class ForecastApp:
    def __init__(self):
        self.page = st.experimental_get_query_params().get("page", ["page1"])[0]
        self.logo_style = """
            position: absolute;
            top: 10px;
            right: 10px;
        """
        self.uploaded_file = None
        self.df = None
        self.validation_data = None
        self.test_data = None
        st.session_state.dataframe= None
        self.validation_start_date = None
        self.validation_end_date = None
        self.best_model = None
        self.models = {
            "Multiple Regression": {"Adjusted R2": round(-250.9134186960262, 2), "RMSE": 0.09414145363023743, "MAPE": 117.51883866689889},
            "Decision Tree Regression": {"Adjusted R2": 0.8163828997617839, "RMSE": 0.0024650003002872397, "MAPE": 3.172768626883775},
            "XGBoost Regression": {"Adjusted R2": 0.8536747810522771, "RMSE": 0.002212808585344038, "MAPE": 2.832314211680962},
            "Moving Average Time Series": {"Adjusted R2": 5.2857373198933466e-05, "RMSE": 0.006537465436224878, "MAPE": 7.424492427333727},
            "ARIMA Time Series": {"Adjusted R2": 0.1607017791725205, "RMSE": 0.00590263053440146, "MAPE": 6.801996404020479},
            "Random Forest Regression": {"Adjusted R2": 0.8598778932608623, "RMSE": 0.002162839459028616, "MAPE": 2.771629463227229},
        }

    def run(self):
        st.set_page_config(page_title='Pangea Forecast Accelerator', page_icon="🧊")
        background_color = "#c7ecee"
        page_bg_css = f"""
            <style>
            body {{
                background-color: {background_color};
            }}
            </style>
        """
        st.markdown(page_bg_css, unsafe_allow_html=True)

        if self.page == 'page1':
            self.page1()

        elif self.page == 'page2':
            self.page2()

    def page1(self):
        st.title("Forecast Accelerator")
        st.write("Welcome to Forecast Accelerator by Pangea!")

        self.uploaded_file = st.sidebar.file_uploader("Upload your data files (csv)", type=['csv'])

        if self.uploaded_file is not None:
            self.df = self.load_data(self.uploaded_file)

            st.subheader("Data Overview")
            self.display_head_table(self.df)

            st.subheader('Data Summary')
            self.display_descriptive_statistics_table(self.df)

            self.select_features()

            if st.button("Run the Models"):
                st.experimental_set_query_params(page="page2")

        # home_button()

    def load_data(self, uploaded_file):
        file_extension = uploaded_file.name.split(".")[-1]

        if file_extension.lower() == 'csv':
            df = pd.read_csv(uploaded_file, encoding='latin-1')

        elif file_extension.lower() == 'xlsx':
            df = pd.read_excel(uploaded_file)

        elif file_extension.lower() == 'utf-8':
            df = pd.read_csv(uploaded_file, encoding='utf-8')

        return df

    def display_head_table(self, df):
        pd.options.display.float_format = '{:.2f}'.format
        table_style = """
        <style>
        table {
            border-collapse: collapse;
            border: 3px solid black;
            width: 100%;
        }
        th, td {
            border: 2px solid black;
            padding: 8px;
            text-align: left;
        }
        </style>
        """
        st.markdown(table_style, unsafe_allow_html= True)
        head_table = df.head()
        head_table = head_table.style.apply(lambda x: ['background-color: lightblue' if i % 2 != 0 else '' for i in range(len(x))], axis=0)
        head_table = head_table.set_table_styles([{'selector': 'th', 'props': [('font-weight', 'bold'), ('background-color', 'turquoise')]}])
        st.table(head_table)

    def display_descriptive_statistics_table(self, dataframe):
        pd.options.display.float_format = '{:.2f}'.format
        descriptive_stats = dataframe.describe(include='all')
        data_types = pd.DataFrame(data={'data types': dataframe.dtypes})
        descriptive_stats = descriptive_stats.fillna('-')

        descriptive_stats = descriptive_stats.applymap(lambda value: np.round(value, 2) if isinstance(value, (float, np.floating)) else value)
        descriptive_stats = pd.concat([data_types.T, descriptive_stats])
        # descriptive_stats = pd.concat([data_types.T, descriptive_stats]).applymap(lambda value: round(value, 2) if isinstance(value, float) else value)
        
        descriptive_stats = descriptive_stats.loc[['data types', 'count', 'min', '25%', '50%', '75%', 'max', 'mean', 'std']]

        descriptive_stats = descriptive_stats.style.apply(lambda x: ['background-color: lightblue' if i % 2 != 0 else '' for i in range(len(x))], axis=0)
        descriptive_stats = descriptive_stats.set_table_styles([{'selector': 'th', 'props': [('font-weight', 'bold'), ('background-color', 'turquoise')]}])
        st.table(descriptive_stats)

    def select_features(self):
        st.sidebar.write("Step 1: Select features")
        date_column = st.sidebar.selectbox("Select the Date column", ['None'] + self.df.columns.to_list(), index=0)

        if date_column == 'None':
            st.sidebar.warning("Please select a valid date column")
        else:
            st.sidebar.write("Step 2: Select target column")
            columns_without_date = [col for col in self.df.columns if col != date_column]
            target_column = st.sidebar.selectbox("Select the Target column", ['None'] + columns_without_date, index=0)

            if target_column == 'None':
                st.sidebar.warning("Please select a valid target column")

            else:
                st.sidebar.write("Step 3: Select lead/lag columns")
                available_columns = [col for col in self.df.columns if col not in [date_column, target_column]]
                lead_lag_columns = st.sidebar.multiselect("Select lead/lag columns:", ['None'] + available_columns)
                if 'None' in lead_lag_columns:
                    st.sidebar.warning("Please select a valid Lead/Lag column")

                else:
                    self.display_feature_engineering()

    def display_feature_engineering(self):
        st.subheader('Feature Engineering')

        index_df = pd.DataFrame(index=self.df.columns, columns=['Features'])
        index_df['Features'] = self.df.columns

        st.session_state.dataframe = pd.concat([index_df, pd.DataFrame(index=self.df.columns, columns=['Type','Null values imputation','Encoding', 'Scaling'])], axis=1)


        #determine the data types
        for col in self.df.columns:
            if self.df[col].dtypes == 'object':
                st.session_state.dataframe.loc[col, 'Type'] = 'Categorical'
                #Limit options for Null values imputation
                st.session_state.dataframe.loc[col, 'Null values imputation'] = 'None'
                #Limit option for Encoding
                st.session_state.dataframe.loc[col, 'Encoding'] = 'None'
                st.session_state.dataframe.loc[col, 'Scaling'] = 'None'


            else:
                st.session_state.dataframe.loc[col, 'Type'] = 'Continuous'
                st.session_state.dataframe.loc[col, 'Scaling'] = 'None'
                st.session_state.dataframe.loc[col, 'Null values imputation'] = 'None'
                st.session_state.dataframe.loc[col, 'Encoding'] = 'None'





        # st.session_state.dataframe[['Null values imputation','Encoding', 'Scaling']] = 'None'

        # type_dropdownlist = ('Continuous', 'Categorical')
        encoding_dropdownlist = ('None', 'One-Hot Encoding', 'Label Encoding')
        scaling_dropdownlist = ('None', 'Min-Max', 'Standardisation')
        null_values_dropdownlist = ('None', 'Mean', 'Median', 'Mode')

        gb = GridOptionsBuilder.from_dataframe(st.session_state.dataframe)
        gb.configure_default_column(editable=True, min_column_width=10)

        # gb.configure_column('False', editable=False, cellEditor='agSelectCellEditor',
        #                     cellEditorParams={'values': self.df.columns.tolist()}, singleClickEdit=True)

        # Set 'Type' column as non-editable and don't provide options to the user
        gb.configure_column('Type', editable=False, cellEditor='agSelectCellEditor',
                            cellEditorParams={'values': ['Categorical' if self.df[col].dtypes == 'object' else 'Continuous' for col in self.df.columns] 
                                              }, singleClickEdit=True)

        gb.configure_column('Null values imputation', editable=True, cellEditor='agSelectCellEditor',
                            cellEditorParams={'values': null_values_dropdownlist}, singleClickEdit=True)
        
        #set options for encoding based on "Type" column
        gb.configure_column('Encoding', editable=True, cellEditor='agSelectCellEditor',
                            cellEditorParams={'values': encoding_dropdownlist}, singleClickEdit=True)
        
        gb.configure_column('Scaling', editable=True, cellEditor='agSelectCellEditor',
                            cellEditorParams={'values': scaling_dropdownlist}, singleClickEdit=True)
        

        grid_options = gb.build()
        grid_height = min(len(self.df.columns)*38, 400)
        # grid_width = 0
        st.write(f'<style>div.stAgGrid.e1y1raen0 {{width: 700px !important;}}</style>', unsafe_allow_html=True)

        grid_response = AgGrid(
            st.session_state.dataframe,
            gridOptions=grid_options,
            height=grid_height,
            # width= grid_width
        )

        #Add a button to extract selections
        if st.button("Apply transformation"):
            selections = st.session_state.dataframe[['Features','Encoding', 'Scaling', 'Null values imputation']].to_dict()
            return selections

        self.apply_feature_engineering()

    def apply_feature_engineering(self):
        self.select_date_range()

    def select_date_range(self):
        st.sidebar.subheader("Select Date Range:")
        self.validation_start_date = st.sidebar.text_input("Start Date (dd-mm-yyyy)", "", key="start_date")
        self.validation_end_date = st.sidebar.text_input("End Date (dd-mm-yyyy)", "", key="end_date")

        #check if the date inputs are valid

        if self.validation_start_date and self.validation_end_date:
            try:
                self.validation_start_date = pd.to_datetime(self.validation_start_date, format= '%d-%m-%Y')
                self.validation_end_date = pd.to_datetime(self.validation_end_date, format= '%d-%m-%Y')
                
            except ValueError:
                st.sidebar.error("Invalid date format. Please use 'dd-mm-yyyy' format.")
                return None, None
            
            if self.validation_start_date > self.validation_end_date:
                st.sidebar.error("Start Date cannot be greater than End date.")
                return None, None
            
            return self.validation_start_date, self.validation_end_date
        
    #functionalities on the second screen

    def page2(self):
        st.title("Forecast Accelerator")
        st.write("Analysis of the models")

        self.display_all_models_metrics()

        self.validation_data = pd.read_excel(r"C:\Users\govin\predictions_24_05_2023.xlsx")
        # self.validation_data = self.df[(self.df[self.select_date_column] >= self.validation_start_date) & 
        #                                (self.df[self.select_date_column] <= self.validation_end_date)]

        st.subheader("Validation Data Overview")
        st.dataframe(self.validation_data.head())

        filter_test_data = self.filter_test_data_func(self.validation_data)

        self.plot_line_chart(filter_test_data)

        test_file = st.sidebar.file_uploader("Upload your Test data", type=['xlsx'])
        if test_file is not None:
            self.test_data = pd.read_excel(test_file)

        self.select_model_func()

        # home_button()

        if st.button("Home"):
            st.experimental_set_query_params(page="page1")

    def display_all_models_metrics(self):
        
        data = []

        for model_name, metrics in self.models.items():
            row = [model_name] + [round(value, 2) if isinstance(value, float) else value for value in metrics.values()]
            data.append(row)

        columns = ['Model Names'] + list(self.models[list(self.models.keys())[0]].keys())
        metric_df = pd.DataFrame(data, columns=columns)

        highest_adj_r2_index = metric_df['Adjusted R2'].idxmax()

        metric_df_styled = metric_df.style.applymap(lambda x: 'background-color: yellow', subset=pd.IndexSlice[highest_adj_r2_index, ['Model Names', 'Adjusted R2', 'RMSE', 'MAPE']])
        table_style = """
        <style>
        table {
            border-collapse: collapse;
            border: 3px solid black;
            width: 100%;
        }
        th, td {
            border: 2px solid black;
            padding: 8px;
            text-align: left;
        }
        </style>
        """
        st.markdown(table_style, unsafe_allow_html= True)
       
        st.write(metric_df_styled)


    def get_best_model(self):
        best_model = max(self.models, key= lambda model_name: self.models[model_name]['Adjusted R2'])
        return best_model
    

    def select_model_func(self):
        st.sidebar.subheader('Select the ML Model')
        self.best_model = self.get_best_model()
        model_names = list(self.models.keys())
        selected_model = st.sidebar.selectbox('Select a model:', model_names, index = model_names.index(self.best_model))
        
        if st.sidebar.button("Apply Model"):
            self.apply_selected_model(selected_model)

        st.sidebar.write(f'You selected: {selected_model}')
        
        return selected_model
    

    def apply_selected_model(self, selected_model):
        st.write(f'{selected_model} applied')
        pass



    def filter_test_data_func(self, validation_data):
        q1 = validation_data['y_test'].quantile(0.05)
        q3 = validation_data['y_test'].quantile(0.95)
        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        filter_test_data = validation_data[(validation_data['y_test'] >= lower_bound) & (validation_data['y_test'] <= upper_bound)]
        return filter_test_data

    def plot_line_chart(self, filter_test_data):
        st.subheader('Applying Best Model on the validated dataset')
        filter_test_data.rename(columns={'X_test': 'Date'}, inplace=True)
        filter_test_data['Date'] = pd.to_datetime(filter_test_data['Date'], errors='coerce')
        filter_test_data.set_index('Date', inplace=True)

        fig, ax = plt.subplots()
        ax.plot(filter_test_data["y_test"], label="Validation data")
        ax.plot(filter_test_data["y_pred"], label="Predicted data")
        ax.set_xlabel("Date")
        ax.set_ylabel("Values")
        ax.set_title("Validation vs Predicted")
        ax.legend()

        st.pyplot(fig)


if __name__ == "__main__":
    app = ForecastApp()
    app.run()
